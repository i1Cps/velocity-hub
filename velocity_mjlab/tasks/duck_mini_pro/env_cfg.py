"""Duck Mini Pro velocity training environment.

The actor interface matches the policy deployed on the physical robot:
39 observations, 10 joint-position actions, and a 50 Hz control loop.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    TerrainHeightSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from tasks.play_visuals import set_play_atmosphere, set_play_terrain_material

from .robot_cfg import (
    DUCK_MINI_PRO_ACTION_SCALE,
    FOOT_GEOMS,
    SERVO_JOINT_PATTERN,
    get_duck_mini_pro_robot_cfg,
)
from .website_robot_cfg import (
    DUCK_MINI_PRO_WEBSITE_ACTION_SCALE,
    get_duck_mini_pro_website_robot_cfg,
)


_BASE_BODY = "trunk"
_SERVO_JOINTS = (SERVO_JOINT_PATTERN,)
_FOOT_BODIES = r"^(left_foot_2|right_foot_2)$"
# Match the right-first body order resolved by the contact sensor.
_FOOT_SITES = ("right_foot", "left_foot")


def make_env_cfg(
    play: bool = False,
    website: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create the flat-ground Duck Mini Pro velocity environment."""

    if website:
        robot = get_duck_mini_pro_website_robot_cfg()
        action_scale = DUCK_MINI_PRO_WEBSITE_ACTION_SCALE
    else:
        robot = get_duck_mini_pro_robot_cfg()
        action_scale = DUCK_MINI_PRO_ACTION_SCALE

    # Sensors

    foot_height_scan = TerrainHeightSensorCfg(
        name="foot_height_scan",
        frame=tuple(
            ObjRef(type="site", name=name, entity="robot") for name in _FOOT_SITES
        ),
        ray_alignment="yaw",
        max_distance=1.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        debug_vis=not play,
        viz=TerrainHeightSensorCfg.VizCfg(
            show_rays=True,
            hit_color=(1.0, 0.0, 1.0, 0.8),
            hit_sphere_color=(1.0, 0.0, 1.0, 1.0),
        ),
    )

    feet_ground_contact = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=_FOOT_BODIES,
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    # Observations
    #
    # Keep this order stable: it defines the deployed 39-value actor input.
    # Keep random noise zero-mean; persistent calibration offsets are modeled separately.

    actor_terms = {
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.06, n_max=0.06),
            delay_min_lag=0,
            delay_max_lag=3,
            delay_update_period=64,
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.10, n_max=0.10),
            delay_min_lag=0,
            delay_max_lag=3,
            delay_update_period=64,
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.08, n_max=0.08),
            params={
                "biased": True,
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=_SERVO_JOINTS,
                ),
            },
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-2.5, n_max=2.5),
            delay_min_lag=0,
            delay_max_lag=1,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=_SERVO_JOINTS,
                ),
            },
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
        "command": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "twist"},
        ),
    }

    # The critic sees clean, undelayed state plus privileged foot information.
    critic_terms = {
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
        ),
        "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=_SERVO_JOINTS,
                ),
            },
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=_SERVO_JOINTS,
                ),
            },
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
        "command": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "twist"},
        ),
        "base_lin_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
        ),
        "foot_height": ObservationTermCfg(
            func=mdp.foot_height,
            params={"sensor_name": foot_height_scan.name},
        ),
        "foot_air_time": ObservationTermCfg(
            func=mdp.foot_air_time,
            params={"sensor_name": feet_ground_contact.name},
        ),
        "foot_contact": ObservationTermCfg(
            func=mdp.foot_contact,
            params={"sensor_name": feet_ground_contact.name},
        ),
        "foot_contact_forces": ObservationTermCfg(
            func=mdp.foot_contact_forces,
            params={"sensor_name": feet_ground_contact.name},
        ),
    }

    observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=not play,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    # Actions and commands

    actions = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=_SERVO_JOINTS,
            scale=action_scale,
            use_default_offset=True,
        )
    }

    commands: dict[str, CommandTermCfg] = {
        "twist": UniformVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(3.0, 8.0),
            rel_standing_envs=0.1,
            rel_heading_envs=0.0,
            heading_command=False,
            heading_control_stiffness=0.5,
            debug_vis=True,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.5, 0.5),
                lin_vel_y=(-0.3, 0.3),
                ang_vel_z=(-0.75, 0.75),
                heading=None,
            ),
        )
    }

    # Domain randomization and resets

    events = {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (0.00, 0.01),
                    "yaw": (-3.14, 3.14),
                },
                "velocity_range": {},
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=_SERVO_JOINTS,
                ),
            },
        ),
        # Pushes help discover a gait without base linear velocity in the actor.
        "push_robot": EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.0, 3.0),
            params={
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                },
            },
        ),
        # The physical feet use 90A TPU soles.
        "foot_friction": EventTermCfg(
            func=dr.geom_friction,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOMS),
                "operation": "abs",
                "ranges": (0.7, 1.5),
                "shared_random": True,
            },
        ),
        "encoder_bias": EventTermCfg(
            func=dr.encoder_bias,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "bias_range": (-0.02, 0.02),
            },
        ),
        "base_com": EventTermCfg(
            func=dr.body_com_offset,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=(_BASE_BODY,)),
                "operation": "add",
                "ranges": {
                    0: (-0.025, 0.025),
                    1: (-0.025, 0.025),
                    2: (-0.03, 0.03),
                },
            },
        ),
        "dof_armature_randomization": EventTermCfg(
            func=dr.joint_armature,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=_SERVO_JOINTS,
                ),
                "operation": "scale",
                "ranges": (0.9, 1.1),
            },
        ),
    }

    # Rewards

    rewards = {
        "track_linear_velocity": RewardTermCfg(
            func=mdp.track_linear_velocity,
            weight=2.0,
            params={"command_name": "twist", "std": 0.3},
        ),
        "track_angular_velocity": RewardTermCfg(
            func=mdp.track_angular_velocity,
            weight=2.0,
            params={"command_name": "twist", "std": 0.7},
        ),
        "upright": RewardTermCfg(
            func=mdp.upright,
            weight=1.0,
            params={
                "std": math.sqrt(0.2),
                "asset_cfg": SceneEntityCfg("robot", body_names=(_BASE_BODY,)),
            },
        ),
        "pose": RewardTermCfg(
            func=mdp.variable_posture,
            weight=1.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=_SERVO_JOINTS,
                ),
                "command_name": "twist",
                "std_standing": {".*": 0.05},
                "std_walking": {
                    r".*hip_pitch.*": 0.40,
                    r".*hip_roll.*": 0.15,
                    r".*hip_yaw.*": 0.15,
                    r".*knee.*": 0.35,
                    r".*ankle.*": 0.25,
                },
                "std_running": {
                    r".*hip_pitch.*": 0.5,
                    r".*hip_roll.*": 0.2,
                    r".*hip_yaw.*": 0.2,
                    r".*knee.*": 0.6,
                    r".*ankle.*": 0.35,
                },
                "walking_threshold": 0.05,
                "running_threshold": 0.1,
            },
        ),
        "body_ang_vel": RewardTermCfg(
            func=mdp.body_angular_velocity_penalty,
            weight=-0.05,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=(_BASE_BODY,))
            },
        ),
        "angular_momentum": RewardTermCfg(
            func=mdp.angular_momentum_penalty,
            weight=-0.02,
            params={"sensor_name": "robot/root_angmom"},
        ),
        # The policy has no action clipping, so joint limits prevent gain exploits.
        "dof_pos_limits": RewardTermCfg(
            func=mdp.joint_pos_limits,
            weight=-1.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=_SERVO_JOINTS,
                ),
            },
        ),
        "action_rate_l2": RewardTermCfg(
            func=mdp.action_rate_l2,
            weight=-0.1,
        ),
        "air_time": RewardTermCfg(
            func=mdp.feet_air_time,
            weight=3.0,
            params={
                "sensor_name": feet_ground_contact.name,
                "threshold_min": 0.08,
                "threshold_max": 0.25,
                "command_name": "twist",
                "command_threshold": 0.01,
            },
        ),
        "foot_clearance": RewardTermCfg(
            func=mdp.feet_clearance,
            weight=-2.0,
            params={
                "target_height": 0.02,
                "command_name": "twist",
                "command_threshold": 0.01,
                "height_sensor_name": foot_height_scan.name,
                "asset_cfg": SceneEntityCfg("robot", site_names=_FOOT_SITES),
            },
        ),
        "foot_swing_height": RewardTermCfg(
            func=mdp.feet_swing_height,
            weight=-0.25,
            params={
                "sensor_name": feet_ground_contact.name,
                "target_height": 0.02,
                "command_name": "twist",
                "command_threshold": 0.01,
                "height_sensor_name": foot_height_scan.name,
            },
        ),
        "foot_slip": RewardTermCfg(
            func=mdp.feet_slip,
            weight=-1.0,
            params={
                "sensor_name": feet_ground_contact.name,
                "command_name": "twist",
                "command_threshold": 0.01,
                "asset_cfg": SceneEntityCfg("robot", site_names=_FOOT_SITES),
            },
        ),
        "soft_landing": RewardTermCfg(
            func=mdp.soft_landing,
            weight=-1e-5,
            params={
                "sensor_name": feet_ground_contact.name,
                "command_name": "twist",
                "command_threshold": 0.01,
            },
        ),
    }

    # Terminations and curriculum

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=mdp.bad_orientation,
            params={"limit_angle": math.radians(70.0)},
        ),
    }

    # This stage is intentionally explicit: it records the command range used
    # by the successful training run from its first update.
    curriculum = {
        "command_vel": CurriculumTermCfg(
            func=mdp.commands_vel,
            params={
                "command_name": "twist",
                "velocity_stages": [
                    {
                        "step": 0,
                        "lin_vel_x": (-0.5, 0.5),
                    },
                ],
            },
        ),
    }
    if play:
        events.pop("push_robot")
        curriculum = {}

    # Terrain and environment

    terrain = TerrainEntityCfg(
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=0,
    )
    if play:
        set_play_terrain_material(terrain)

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=terrain,
            sensors=(foot_height_scan, feet_ground_contact),
            num_envs=1,
            env_spacing=1.25 if play else 2.0,
            extent=2.0,
            entities={"robot": robot},
            spec_fn=set_play_atmosphere if play else None,
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum=curriculum,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name=_BASE_BODY,
            distance=3.0,
            elevation=-12.0 if play else -5.0,
            azimuth=135.0 if play else 90.0,
            max_extra_envs=3 if play else 2,
        ),
        sim=SimulationCfg(
            nconmax=35,
            njmax=1500,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
                ccd_iterations=100,
            ),
        ),
        decimation=4,
        episode_length_s=int(1e9) if play else 20.0,
    )
