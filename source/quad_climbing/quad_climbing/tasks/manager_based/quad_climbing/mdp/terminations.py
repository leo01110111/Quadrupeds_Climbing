# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_specific_terminations(
        env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Chooses a termination function depending on where the robot is
    """
    # extract the used quantities (to enable type-hinting)
    terrain: TerrainImporter = env.scene.terrain
    terrain_origins = terrain.terrain_origins
    env_origins = terrain.env_origins

    subterrain_dict = env.scene.terrain.cfg.terrain_generator.sub_terrains
    
    available_terrains = {'hf_pyramid_slope', 'random_rough'}

    num_cols = terrain.cfg.terrain_generator.num_cols
    col_index = 0
    col_separaton = []
    #find the terrain types and their proportions
    #find the y value boundaries of all the sub-terrains
    for col in range(num_cols):
        separation = terrain_origins[0,col,1]
        col_separaton.append(separation.item())

    terminate = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    for key in subterrain_dict.keys():
        if key not in available_terrains:
            raise Exception("No termination function associated with terrain")
        elif key == "hf_pyramid_slope": #the order of the if statements should match the insertion order of the subterrains
            proportion =  terrain.cfg.terrain_generator.sub_terrains["hf_pyramid_slope"].proportion
            col_index = int(num_cols*proportion)
            ids = []
            for id in range(env.num_envs):
                if env_origins[id, 1] <= col_separaton[col_index-1]:
                    ids.append(id)
            terminate[ids] = _robot_at_top(env, ids, asset_cfg)
        """elif key == "random_rough":
            proportion =  terrain.cfg.terrain_generator.sub_terrains["random_rough"].proportion
            col_index += int(num_cols*proportion)
            ids = []
            for id in range(env.num_envs):
                lower_bound = int(col_index - 1 -  num_cols * proportion)
                if env_origins[id, 1] <= col_separaton[col_index - 1] and env_origins[id, 1] > col_separaton[lower_bound]:
                    ids.append(id)
            #we dont have any terrain specific terminations for flat"""
    return terminate


def _robot_at_top(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset : RigidObject = env.scene[asset_cfg.name]
    robot_height = asset.data.root_pos_w[env_ids, 2]
    #print("robot_height: ", robot_height)
    hill_height = env.scene.env_origins[env_ids, 2]
    #print("hill height:", hill_height)
    return robot_height >= hill_height


def terrain_out_of_bounds(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_buffer: float = 3.0
) -> torch.Tensor:
    """Terminate when the actor move too close to the edge of the terrain.

    If the actor moves too close to the edge of the terrain, the termination is activated. The distance
    to the edge of the terrain is calculated based on the size of the terrain and the distance buffer.
    """
    if env.scene.cfg.terrain.terrain_type == "plane":
        return False  # we have infinite terrain because it is a plane
    elif env.scene.cfg.terrain.terrain_type == "generator":
        # obtain the size of the sub-terrains
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size
        n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols
        border_width = terrain_gen_cfg.border_width
        # compute the size of the map
        map_width = n_rows * grid_width + 2 * border_width
        map_height = n_cols * grid_length + 2 * border_width

        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]

        # check if the agent is out of bounds
        x_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 0]) > 0.5 * map_width - distance_buffer
        y_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 1]) > 0.5 * map_height - distance_buffer
        return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
    else:
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")
