#!/bin/bash

# Initialize Conda (adjust path if necessary)
source /home/leo/miniconda3/etc/profile.d/conda.sh

# Activate the desired environment
conda activate env_isaaclab

# Now you can run commands within the activated environment
python scripts/rsl_rl/train.py --task Isaac-Slope-Unitree-Go2-v0 --num_envs 5

# Deactivate the environment when done (optional)
conda deactivates