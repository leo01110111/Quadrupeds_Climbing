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
    #print("Body names:", sensor_cfg.body_names) #the body names is indeed that of feet
    #print("Body_ids (hopefully of foot):", sensor_cfg.body_ids) #verified
    foot_contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = foot_contact_sensor.data.net_forces_w_history  #(num_envs, history, amt of bodies, xyz)
    #print(f"Foot contact force, {net_contact_forces.shape}: {net_contact_forces[:, :, sensor_cfg.body_ids,:]}")
    net_contact_forces_flat = net_contact_forces.view(net_contact_forces.shape[0], -1) #(num_envs, history * 4 feet * xyz force components)
    return net_contact_forces_flat