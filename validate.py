#!/usr/bin/env python3
"""Gradient of an L2 image loss with respect to the floor albedo of the living room.

Run target.py first. Example:
    python validate.py --integrator reslrb --spp 32 --max-depth 16
"""
import argparse
import os
import time

import drjit as dr
import mitsuba as mi
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

mi.set_variant("cuda_ad_rgb", "llvm_ad_rgb")
import reslrb  # registers the integrators

FLOOR = "FloorBSDF.brdf_0.diffuse_reflectance.data"
CMAP = LinearSegmentedColormap.from_list("rdgrbu", ["#d62728", "#888888", "#1f77b4"])


def save_gradient(path, grad, percentile):
    if grad.ndim == 3:
        grad = 0.299 * grad[..., 0] + 0.587 * grad[..., 1] + 0.114 * grad[..., 2]
    vmax = np.percentile(np.abs(grad), percentile) or 1e-8
    rgb = CMAP(np.clip(grad / vmax, -1, 1) * 0.5 + 0.5)[..., :3]
    Image.fromarray((rgb * 255).astype(np.uint8)).save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--integrator", default="reslrb", help="reslrb or ptracer (naive AD)")
    parser.add_argument("--spp", type=int, default=32)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--scene", default="scenes/living-room/scene.xml")
    parser.add_argument("--out", default="figures/living-room")
    args = parser.parse_args()

    scene = mi.load_file(args.scene)
    params = mi.traverse(scene)
    integrator = mi.load_dict({"type": args.integrator, "max_depth": args.max_depth,
                               "hide_emitters": True, "rr_depth": 100})

    # The floor texture is downsampled to 640x360 so that the gradient is easy to look at.
    texture = np.clip(np.array(params[FLOOR]), 0, 1)
    texture = Image.fromarray((texture * 255).astype(np.uint8)).resize((640, 360), Image.LANCZOS)
    params[FLOOR] = mi.TensorXf(np.array(texture).astype(np.float32) / 255.0)
    params.update()
    dr.enable_grad(params[FLOOR])
    params.update()

    target = mi.TensorXf(mi.Bitmap(os.path.join(args.out, "target.exr")))

    start = time.perf_counter()
    image = mi.render(scene, params, seed=0, spp=args.spp, integrator=integrator)
    loss = dr.mean(dr.square(image - target))
    dr.backward(loss)
    grad = np.array(dr.grad(params[FLOOR]))
    elapsed = time.perf_counter() - start

    path = os.path.join(args.out, f"grad_{args.integrator}.png")
    save_gradient(path, grad, 98)
    print(f"{args.integrator}: {elapsed:.2f} s, saved {path}")


if __name__ == "__main__":
    main()
