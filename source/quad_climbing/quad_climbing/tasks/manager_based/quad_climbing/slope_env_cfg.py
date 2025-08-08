# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
import isaacsim.core.utils.prims as prim_utils


##
# Pre-defined configs
##
from .terrains.config.slope import SLOPE_TERRAIN_CFG

##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg( #configures the terrian importer which is a class that handles terrain meshes and imports them into the simulator. A terrian mesh comprises of sub terrain which can be stairs, slopes... arranged in a grid with num_cols nd num_ros. The trerrain origins are the positions of the subterrain where the robot should be spawned.
        prim_path="/World/ground", #where the primitive should be located
        terrain_type="generator", #means that the terrain will be procedurally generated rather than loaded from a file
        terrain_generator=SLOPE_TERRAIN_CFG, #this is a generator config object used to generate the terrain. Params include the tpes of terrain and properties of them
        max_init_terrain_level=5, #max initial how arwe difficulty of the terrain
        collision_group=-1, #means that the terrain will collide with everything
        physics_material=sim_utils.RigidBodyMaterialCfg( #defines the physical properties of the terrain.
            friction_combine_mode="multiply", #how the fric coeffs are cominded whn obj is on both surfaces
            restitution_combine_mode="multiply", #determines how the bouniciness (restitution) is the prodct of two objs restitution values
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg( #config for the visual appearance of the terrain such as its textures and shaders
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False, #if set to true it shos wireframs and collision shapes
    )

    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_1k.hdr",  # Using 1k instead of 4k resolution
        ),
    )

##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    
    pose_command = mdp.UniformPoseCommandCfg(
        class_type=mdp.WorldCentricPoseCommand,  # Use our world-centric command class
        asset_name="robot",  # the asset we're commanding
        body_name="base",    # the body to control
        make_quat_unique=False,  # don't enforce unique quaternion representation
        resampling_time_range=(1000.0, 1000.0),  # very long resampling time to maintain consistent commands
        debug_vis=True,      # enable visualization
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-1.0, 1.0),      # position ranges in world-frame meters
            pos_y=(-1.0, 1.0),      # position ranges in world-frame meters
            pos_z=(0, 0),       # height range in world frame
            roll=(0.0, 0.0),        # keeping roll level
            pitch=(0.0, 0.0),       # keeping pitch level
            yaw=(-math.pi, math.pi)  # full range of yaw motion in world frame
        ),
    )
    
    """#Command specifications for the MDP.
    base_velocity = mdp.WorldCentricVelocityCommandCfg( #the config for giving commands in the mdp
        asset_name="robot", #the asset we're commanding
        resampling_time_range=(1000.0, 1000.0), #min and max time between resampling a new command. Here we make it so that it basically doesnt happen
        rel_standing_envs=0, #percent of envs that stand still
        rel_heading_envs=1.0, #percent that receive a heading
        heading_control_stiffness=1, # stiffness at which the robot keeps its heading
        debug_vis=True,
        ranges=mdp.WorldCentricVelocityCommandCfg.Ranges( #remember that these are commands in respect  to the robot frame except for the heading
            magnitude = (0.0, 3.0), ang_vel_z=(-1.5, 1.5), heading=(math.pi/4, math.pi/4)
        ),
    )"""

    """base_velocity = mdp.UniformVelocityCommandCfg( #the config for giving commands in the mdp
        asset_name="robot", #the asset we're commanding
        resampling_time_range=(1000.0, 1000.0), #min and max time between resampling a new command. Here we make it so that it basically doesnt happen
        rel_standing_envs=0, #percent of envs that stand still
        rel_heading_envs=1.0, #percent that receive a heading
        heading_command=True, #if true the robot receives a heading
        heading_control_stiffness=1, # stiffness at which the robot keeps its heading
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges( #remember that these are commands in respect  to the robot frame except for the heading
            lin_vel_x=(1, 1.5), lin_vel_y=(0, 0), ang_vel_z=(-1.5, 1.5), heading=(math.pi/4, math.pi/4)
        ),
    )"""


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True) #


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        pose_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})
        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-1.5, n_max=1.5))
        joint_torque = ObsTerm(func=mdp.joint_effort, noise=Unoise(n_min=-0.25, n_max=0.25))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05),)
        actions = ObsTerm(func=mdp.last_action)
      
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1, 1),
            "dynamic_friction_range": (1, 1),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_state_curriculum,
        mode="reset",
        params={
            "pose_range": {"x": (-3.3,-3.3), "y": (-3.3, -3.3), "z":(0,0), "yaw": (math.pi/4, math.pi/4)}, # prev at -0.65
            "velocity_range": {
                "x": (0, 0),
                "y": (0, 0),
                "z": (0, 0),
                "roll": (0, 0),
                "pitch": (0, 0),
                "yaw": (0, 0),
            },
            "spawn_at_base": True
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)
    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={"std": 2.0, "command_name": "pose_command"},
    )
    position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={"std": 0.2, "command_name": "pose_command"},
    )
    orientation_tracking = RewTerm(
        func=mdp.heading_command_error_abs,
        weight=-0.2,
        params={"command_name": "pose_command"},
    )
    
    # -- penalties
    #dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    #dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    #action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    """feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )"""
    """undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    )"""

    """undesired_body_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )"""
    # -- optional penalties
    #flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    #dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    at_top = DoneTerm(
        func=mdp.robot_at_top,
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class LocomotionSlopeEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=2048, env_spacing=2.5)  # Reduced number of environments
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 30.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation #4 time steps for every rendered frame
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
    """if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
       """

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training

