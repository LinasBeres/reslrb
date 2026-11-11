#!/usr/bin/env bash
# Regenerates the figures of the paper into figures/.
# Run scripts/download_scenes.sh first. The paper used an NVIDIA RTX 5080 (16 GB).
set -euo pipefail
cd "$(dirname "$0")/.."

# Figure 1: caustic gradients
python caustic.py --integrator ptracer --spp 32
python caustic.py --integrator reslrb_attached --spp 32
python caustic.py --integrator reslrb_attached --spp 1024

# Figure 5: gradient validation in the living room
python target.py
python validate.py --integrator ptracer --spp 32 --max-depth 16
python validate.py --integrator reslrb --spp 32 --max-depth 16

# Figure 6: memory and time against path length
python sweep.py
python plot_sweep.py figures/sweep/sweep.json
