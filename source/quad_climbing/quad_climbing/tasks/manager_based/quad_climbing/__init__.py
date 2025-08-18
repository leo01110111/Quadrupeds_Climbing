# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Locomotion environments with velocity-tracking commands.

These environments are based on the `legged_gym` environments provided by Rudin et al.

Reference:
    https://github.com/leggedrobotics/legged_gym
"""

from quad_climbing.tasks.manager_based.quad_climbing.slope_env_cfg import LocomotionSlopeEnvCfg # the dot indicates that we're searching for the module in the file's directory not in the root directory or PYTHONPATH

from . import go2