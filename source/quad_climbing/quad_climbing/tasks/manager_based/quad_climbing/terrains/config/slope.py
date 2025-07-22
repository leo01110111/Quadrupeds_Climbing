# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen

from isaaclab.terrains import TerrainGeneratorCfg

SLOPE_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0), #size of each subterrain
    border_width=0.25, 
    num_rows=10,
    num_cols=10, 
    horizontal_scale= 0.07,
    vertical_scale= 0.07,
    use_cache=False,
    sub_terrains={
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=1.0, slope_range=(0.0, 0.785), platform_width=2.0, border_width=0.25
        )
    }
)
