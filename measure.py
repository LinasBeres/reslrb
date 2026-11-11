#!/usr/bin/env python3
"""One gradient step of the living room, run by sweep.py in a fresh process.

Prints a JSON line with the peak memory (MB) or the times of the repeats (s).
On CUDA the memory is the peak GPU memory, read with NVML. On the LLVM backend
(macOS, or Linux without a GPU) it is the peak resident memory of the process.
"""
import argparse
import json
import resource
import sys
import threading
import time

import drjit as dr
import mitsuba as mi

mi.set_variant("cuda_ad_rgb", "llvm_ad_rgb")
import reslrb  # registers the integrators

FLOOR = "FloorBSDF.brdf_0.diffuse_reflectance.data"


class GPUMemory:
    def __init__(self):
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.used = lambda: pynvml.nvmlDeviceGetMemoryInfo(handle).used

    def _poll(self):
        while self.running:
            self.peak = max(self.peak, self.used())
            time.sleep(0.0001)

    def start(self):
        self.baseline = self.peak = self.used()
        self.running = True
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()
        return (self.peak - self.baseline) / 2**20


class CPUMemory:
    # ru_maxrss is in bytes on macOS and in kilobytes on Linux.
    scale = 1 if sys.platform == "darwin" else 1024

    def start(self):
        self.baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * self.scale

    def stop(self):
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * self.scale
        return (peak - self.baseline) / 2**20


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--integrator", required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--spp", type=int, default=32)
    parser.add_argument("--mode", choices=["memory", "time"], required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--scene", default="scenes/living-room/scene.xml")
    parser.add_argument("--target", default="figures/living-room/target.exr")
    args = parser.parse_args()

    scene = mi.load_file(args.scene)
    params = mi.traverse(scene)
    target = mi.TensorXf(mi.Bitmap(args.target))
    integrator = mi.load_dict({"type": args.integrator, "max_depth": args.depth,
                               "rr_depth": 250, "hide_emitters": True})

    def step():
        params[FLOOR] = dr.detach(params[FLOOR])
        dr.enable_grad(params[FLOOR])
        params.update()
        image = mi.render(scene, params, spp=args.spp, integrator=integrator)
        dr.backward(dr.sum(dr.square(dr.ravel(image) - dr.ravel(target))))
        dr.eval(dr.grad(params[FLOOR]))
        dr.sync_thread()

    if args.mode == "memory":
        monitor = GPUMemory() if mi.variant().startswith("cuda") else CPUMemory()
        monitor.start()
        step()
        print(json.dumps({"memory_mb": monitor.stop()}))
    else:
        step()
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            step()
            times.append(time.perf_counter() - start)
        print(json.dumps({"times": times}))


if __name__ == "__main__":
    main()
