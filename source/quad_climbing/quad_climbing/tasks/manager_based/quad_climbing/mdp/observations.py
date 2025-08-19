import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

def joint_effort(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint applied effort of the robot.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their effort returned.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with this observation.

    Returns:
        The joint effort (N or N-m) for joint_names in asset_cfg, shape is [num_env,num_joints].
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.applied_torque[:, asset_cfg.joint_ids]

def foot_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    foot_contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = foot_contact_sensor.data.net_forces_w_history  #(num_envs, history, amt of bodies, xyz)
    #print(f"Foot contact force, {net_contact_forces.shape}: {net_contact_forces[:, :, sensor_cfg.body_ids,:]}") #(num_envs, history * 4 feet * xyz force components)
    foot_contact_states = torch.any(
        torch.max(torch.norm(net_contact_forces[:,:,sensor_cfg.body_ids,:], dim=-1), dim=1)[0].unsqueeze(1) > 1.0, dim=1 
    )  #Chooses the max contact force in history and sees if it passes a threshold. It passes the contact state to a (num_envs, 4 feet) tensor
    print(f"foot contact states {foot_contact_states.shape}: {foot_contact_states}")
    return foot_contact_states

