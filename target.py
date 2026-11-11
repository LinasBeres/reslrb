#!/usr/bin/env python3
"""Render the living room and blur it into the target image of the validation and sweep.

Also writes render.png, the path-traced reference shown in the figure.
"""
import argparse
import os

import mitsuba as mi
import numpy as np

mi.set_variant("cuda_ad_rgb", "llvm_ad_rgb")


def blur(image, sigma):
    radius = int(3 * sigma)
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    out = np.array(image, dtype=np.float32)
    out = np.apply_along_axis(np.convolve, 1, out, kernel, mode="same")
    out = np.apply_along_axis(np.convolve, 0, out, kernel, mode="same")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="scenes/living-room/scene.xml")
    parser.add_argument("--spp", type=int, default=512)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--render-spp", type=int, default=1024)
    parser.add_argument("--out", default="figures/living-room")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    scene = mi.load_file(args.scene, integrator="ptracer", spp=args.spp, max_depth=args.max_depth)
    image = np.array(mi.render(scene, seed=0, spp=args.spp))

    mi.Bitmap(image).write(os.path.join(args.out, "primal.exr"))
    mi.Bitmap(blur(image, args.sigma)).write(os.path.join(args.out, "target.exr"))

    scene = mi.load_file(args.scene, integrator="path", resx=1280, resy=720)
    render = mi.render(scene, seed=0, spp=args.render_spp)
    mi.util.write_bitmap(os.path.join(args.out, "render.png"), render)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
