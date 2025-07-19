# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
import numpy as np

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter, TerrainGeneratorCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def pyramid_max_height(env : ManagerBasedRLEnv, cfg: TerrainGeneratorCfg, importer : TerrainImporter) -> torch.tensor:
    """This program computes the heights of every slope the robot is on"""

    from isaaclab.markers.config import FRAME_MARKER_CFG
    from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers

    marker_cfg : VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path = "/Debug_Vis/Height_Marker"
    )
    marker_positions = torch.ones((cfg.num_rows, 3), dtype=torch.float32, device = importer.device)
    marker = VisualizationMarkers(marker_cfg)

    levels = [i for i in range(cfg.num_rows)]
    lower, upper = cfg.sub_terrains["hf_pyramid_slope"].slope_range
    hill_heights = torch.zeros((cfg.num_rows), dtype = torch.float32, device = importer.device)

    for level in levels:
        difficulty = (level+1)/cfg.num_rows
    
        #calculate slope
        slope = lower + (upper - lower) * difficulty

        # switch parameters to discrete units
        # -- horizontal scale
        width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
        length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
        # -- height
        # we want the height to be slope * 1/2 of the width since the terrain is a pyramid
        height_max = slope * cfg.size[0] / 2 / cfg.vertical_scale
        # -- center of the terrain
        center_x = int(width_pixels / 2)
        center_y = int(length_pixels / 2)

        # create a meshgrid of the terrain
        x = np.arange(0, width_pixels)
        y = np.arange(0, length_pixels)
        xx, yy = np.meshgrid(x, y, sparse=True)
        # offset the meshgrid to the center of the terrain
        xx = (center_x - np.abs(center_x - xx)) / center_x
        yy = (center_y - np.abs(center_y - yy)) / center_y
        # reshape the meshgrid to be 2D
        xx = xx.reshape(width_pixels, 1)
        yy = yy.reshape(1, length_pixels)
        # create a sloped surface
        
        hf_raw = np.zeros((width_pixels, length_pixels))
        hf_raw = torch.tensor(hf_raw, device=importer.device)
        xx = torch.tensor(xx, device=importer.device)
        yy = torch.tensor(yy, device=importer.device)
        hf_raw = xx * yy
        hf_raw = height_max*hf_raw #scalar times a matrix
        
        # create a flat platform at the center of the terrain
        platform_width = int(cfg.sub_terrains["hf_pyramid_slope"].platform_width / cfg.horizontal_scale)
        # get the height of the platform at the corner of the platform
        x_pf = width_pixels // 2 - platform_width // 2
        y_pf = length_pixels // 2 - platform_width // 2
        z_pf = hf_raw[x_pf, y_pf]
        hill_heights[level] = z_pf*cfg.vertical_scale

        x = -36 + level*8
        tmp = torch.tensor(hill_heights[level], dtype = torch.float32)
        terrain_center = torch.tensor([x, 0],device = importer.device, dtype=torch.float32)
        pos = torch.cat((terrain_center, tmp.unsqueeze(0)))  # (2,) + (1,) → (3,)
        marker_positions[level] = pos

    marker.visualize(translations=marker_positions)

    #debug
    #print(f"levels, {level.shape}: {level.cpu().numpy()}")
    #print(f"slope, {slope.shape}: {slope}")
    print(f"height_max, {height_max.shape}: {height_max}")
    #print(f"z_pf shape: ", z_pf.shape)
    #print(f"z_pf: ", z_pf)
    #print(f"xx, {xx.shape}: {xx}")
    #print(f"yy, {yy.shape}: {yy}")
    #print(f"hf_raw {hf_raw.shape}:",hf_raw)

    return hill_heights

def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    # compute the heights the robots climbed
    robot_height = asset.data.root_pos_w[env_ids, 2]

    max_height = env.scene.env_origins[env_ids, 2]

    # robots that walked far enough progress to harder terrains
    move_up = robot_height > max_height * 100 #0.8
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = robot_height < max_height * 0.5 
    move_down *= ~move_up

    #debug
    """
    print("Leveling Debug: bool format: move down, move up")
    for i in range(len(env_ids)):
        print(f"robot_z {robot_height[i]} vs max_height {max_height[i]} = {move_down[i]}, {move_up[i]}")
        """

    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())
