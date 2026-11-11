#!/usr/bin/env python3
"""Gradient of a caustic image loss with respect to the height map of a glass lens.

The scene follows the caustic optimisation tutorial of Mitsuba 3. Example:
    python caustic.py --integrator reslrb_attached --spp 32
"""
import argparse
import os
import time

import drjit as dr
import mitsuba as mi
import numpy as np

mi.set_variant("cuda_ad_rgb", "llvm_ad_rgb")
import reslrb  # registers the integrators
from validate import save_gradient

T = mi.ScalarTransform4f


def lens_mesh(n):
    """A flat n x n grid on [-1, 1]^2 with texture coordinates."""
    u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing="ij")
    u, v = u.ravel(), v.ravel()
    i, j = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="ij")
    v00 = (i * n + j).ravel()
    faces = np.stack([np.stack([v00, v00 + n, v00 + 1], 1),
                      np.stack([v00 + 1, v00 + n, v00 + n + 1], 1)], 1).reshape(-1, 3)

    mesh = mi.Mesh("lens", n * n, len(faces), has_vertex_texcoords=True)
    params = mi.traverse(mesh)
    params["vertex_positions"] = np.stack([2 * (u - 0.5), 2 * (v - 0.5), 0 * u], 1).ravel().astype(np.float32)
    params["vertex_texcoords"] = np.stack([u, v], 1).ravel().astype(np.float32)
    params["faces"] = faces.ravel().astype(np.uint32)
    params.update()
    return mesh


def load_scene(integrator, lens_file, resolution):
    bayer = np.zeros((32, 32, 3), dtype=np.float32)
    bayer[::2, ::2, 2] = 2.2
    bayer[::2, 1::2, 1] = 2.2
    bayer[1::2, 1::2, 0] = 2.2

    glass = {"type": "ref", "id": "glass"}
    return mi.load_dict({
        "type": "scene",
        "integrator": integrator,
        "sensor": {
            "type": "perspective",
            "near_clip": 1, "far_clip": 1000, "fov": 45,
            "to_world": T().look_at(target=[0, -20, 0], origin=[0, -4.65, 0], up=[0, 0, 1]),
            "film": {"type": "hdrfilm", "width": resolution, "height": resolution,
                     "pixel_format": "rgb", "rfilter": {"type": "gaussian"}},
        },
        "glass": {"type": "dielectric", "id": "glass", "ext_ior": "air", "int_ior": 1.5,
                  "specular_reflectance": {"type": "spectrum", "value": 0}},
        "receiver": {
            "type": "obj", "filename": "scenes/caustic/meshes/rectangle.obj",
            "to_world": T().look_at(target=[0, 1, 0], origin=[0, -7, 0], up=[0, 0, 1]).scale((5, 5, 5)),
            "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": (1, 1, 1)}},
        },
        "slab": {"type": "obj", "filename": "scenes/caustic/meshes/slab.obj",
                 "to_world": T().rotate(axis=(1, 0, 0), angle=90), "bsdf": glass},
        "lens": {"type": "ply", "filename": lens_file,
                 "to_world": T().rotate(axis=(1, 0, 0), angle=90), "bsdf": glass},
        "emitter-shape": {
            "type": "obj", "filename": "scenes/caustic/meshes/rectangle.obj",
            "to_world": T().look_at(target=[0, 0, 0], origin=[0, 5, 0], up=[0, 0, 1]),
            "bsdf": {"type": "diffuse", "reflectance": {"type": "spectrum", "value": 0}},
            "emitter": {"type": "directionalarea",
                        "radiance": {"type": "bitmap", "bitmap": mi.Bitmap(bayer),
                                     "raw": True, "filter_type": "nearest"}},
        },
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--integrator", default="reslrb_attached", help="reslrb_attached or ptracer (naive AD)")
    parser.add_argument("--spp", type=int, default=32)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=1024, help="image and height map resolution")
    parser.add_argument("--target", default="assets/target.png")
    parser.add_argument("--out", default="figures/caustic")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    n = args.resolution
    lens_file = os.path.join(args.out, f"lens_{n}.ply")
    if not os.path.isfile(lens_file):
        lens_mesh(n).write_ply(lens_file)

    integrator = {"type": args.integrator, "max_depth": args.max_depth, "hide_emitters": False}
    scene = load_scene(integrator, lens_file, n)
    params = mi.traverse(scene)

    target = mi.Bitmap(args.target).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, False)
    if list(target.size()) != [n, n]:
        target = target.resample([n, n])
    target = mi.TensorXf(target)

    # Displace the lens along its normals by a height map, here zero.
    heightmap = mi.load_dict({"type": "bitmap", "bitmap": mi.Bitmap(dr.zeros(mi.TensorXf, (n, n))), "raw": True})
    height = mi.traverse(heightmap)
    positions = dr.unravel(mi.Vector3f, params["lens.vertex_positions"])
    normals = dr.unravel(mi.Vector3f, params["lens.vertex_normals"])
    si = dr.zeros(mi.SurfaceInteraction3f, dr.width(positions))
    si.uv = dr.unravel(mi.Point2f, params["lens.vertex_texcoords"])

    dr.enable_grad(height["data"])
    height.update()
    params["lens.vertex_positions"] = dr.ravel(positions + heightmap.eval_1(si) * normals)
    params.update()

    start = time.perf_counter()
    image = mi.render(scene, params, seed=0, spp=args.spp)
    loss = dr.mean(dr.square(image - target))
    dr.backward(loss)
    grad = np.array(dr.grad(height["data"])).squeeze()
    elapsed = time.perf_counter() - start

    path = os.path.join(args.out, f"grad_{args.integrator}_{args.spp}spp.png")
    save_gradient(path, grad, 99)
    mi.util.write_bitmap(os.path.join(args.out, f"render_{args.integrator}.png"), image)
    print(f"{args.integrator} @ {args.spp} spp: {elapsed:.2f} s, saved {path}")


if __name__ == "__main__":
    main()
