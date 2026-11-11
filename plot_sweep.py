#!/usr/bin/env python3
"""Plot the memory and time bars of the paper from the output of sweep.py.

Example:
    python plot_sweep.py figures/sweep/sweep.json
    python plot_sweep.py assets/sweep_paper.json   # the measurements of the paper
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, MultipleLocator

COLORS = {"ptracer": "#d62728", "reslrb": "#ff7f0e"}
BACKGROUND = "#f2f2f4"


def plot(data, memory, path, gpu_gb):
    depths = data["depths"]
    names = list(data["runs"])
    x = np.arange(len(depths))
    width = 0.18
    offsets = (np.arange(len(names)) - (len(names) - 1) / 2) * width

    def value(run):
        return run["memory_mb"] / 1024 if memory else run["time_median"]

    ok_values = [value(r) for runs in data["runs"].values() for r in runs if r["ok"]]
    ymax = gpu_gb + 2 if memory else max(ok_values) * 1.05

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_facecolor(BACKGROUND)
    for name, offset in zip(names, offsets):
        for k, run in enumerate(data["runs"][name]):
            label = name if k == 0 else None
            if run["ok"]:
                ax.bar(x[k] + offset, value(run), width, color=COLORS.get(name), zorder=3, label=label)
            else:
                ax.bar(x[k] + offset, ymax * 1.5, width, color=COLORS.get(name), hatch="///",
                       alpha=0.6, zorder=3, label=label)
                ax.text(x[k] + offset, ymax * 0.98, "OOM", ha="center", va="top", fontsize=8,
                        fontweight="bold", rotation=90, zorder=5)

    ax.set_ylim(0, ymax)
    ax.set_xticks(x)
    ax.grid(axis="y", linestyle="--", color="white", zorder=0)
    ax.set_axisbelow(True)
    if memory:
        ax.set_xticklabels([])
        ax.set_ylabel("Peak memory usage (GB)")
        ax.yaxis.set_major_locator(MultipleLocator(4))
        ax.axhline(gpu_gb, color="#d62728", linestyle="--", lw=1.5, zorder=5)
        ax.text(-0.45, gpu_gb + 0.2, f"{gpu_gb:g}GB Limit", color="#d62728", fontsize=10,
                fontweight="bold", va="bottom", zorder=6)
        legend = ax.legend(loc="center left", fontsize=11)
        legend.get_frame().set_facecolor(BACKGROUND)
        legend.get_frame().set_edgecolor("black")
    else:
        ax.set_xticklabels([str(d) for d in depths])
        ax.set_xlabel("Path length")
        ax.set_ylabel("Computation time (s)")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep", help="JSON written by sweep.py")
    parser.add_argument("--gpu-gb", type=float, default=16, help="memory limit drawn on the plot")
    parser.add_argument("--out", default="figures/sweep")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with open(args.sweep) as f:
        data = json.load(f)
    plot(data, True, os.path.join(args.out, "memory.pdf"), args.gpu_gb)
    plot(data, False, os.path.join(args.out, "timing.pdf"), args.gpu_gb)


if __name__ == "__main__":
    main()
