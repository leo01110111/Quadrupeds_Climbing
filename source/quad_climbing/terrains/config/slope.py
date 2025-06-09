# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen

from isaaclab.terrains import TerrainGeneratorCfg

SLOPE_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0), #size of each subterrain
    border_width=20.0, 
    num_rows=10,
    num_cols=20, 
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.20, #a slope need to be steeper than 0.75 radians in order to be considered a slop
    use_cache=False,
    sub_terrains={
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=1.0, slope_range=(0.0, 0.785), platform_width=2.0, border_width=0.25
        )
    }
)
