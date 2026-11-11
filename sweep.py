#!/usr/bin/env python3
"""Memory and time per gradient step against the maximum path length.

Every measurement runs in its own process (measure.py), so that one running
out of memory does not affect the others. Once an integrator fails at some
depth, the larger depths are skipped. Run target.py first.
"""
import argparse
import json
import os
import subprocess
import sys


def measure(integrator, depth, spp, mode, repeats):
    cmd = [sys.executable, "measure.py", "--integrator", integrator, "--depth", str(depth),
           "--spp", str(spp), "--mode", mode, "--repeats", str(repeats)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None
    return json.loads(result.stdout.strip().splitlines()[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--integrators", nargs="+", default=["ptracer", "reslrb"])
    parser.add_argument("--depths", nargs="+", type=int, default=[4, 8, 16, 32, 64, 128])
    parser.add_argument("--spp", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", default="figures/sweep/sweep.json")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    results = {"depths": args.depths, "spp": args.spp, "runs": {}}
    for integrator in args.integrators:
        runs = []
        failed = False
        for depth in args.depths:
            memory = None if failed else measure(integrator, depth, args.spp, "memory", 1)
            timing = None if memory is None else measure(integrator, depth, args.spp, "time", args.repeats)
            failed = timing is None
            if failed:
                runs.append({"depth": depth, "ok": False})
                print(f"{integrator:8s} depth {depth:4d}: failed")
                continue
            times = sorted(timing["times"])
            median = times[len(times) // 2]
            runs.append({"depth": depth, "ok": True, "memory_mb": memory["memory_mb"],
                         "times": timing["times"], "time_median": median})
            print(f"{integrator:8s} depth {depth:4d}: {memory['memory_mb']:8.1f} MB {median * 1000:8.1f} ms")
        results["runs"][integrator] = runs

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
