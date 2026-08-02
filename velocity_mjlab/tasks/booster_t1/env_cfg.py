"""Readable, self-contained Booster T1 velocity environment."""

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    GridPatternCfg,
    ObjRef,
    RayCastSensorCfg,
    TerrainHeightSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from tasks.play_visuals import set_play_atmosphere, set_play_terrain_material

from .robot_cfg import FOOT_GEOMS, T1_ACTION_SCALE, get_t1_robot_cfg


_BASE_BODY = "Trunk"
_FOOT_BODIES = r"^(left_foot_link|right_foot_link)$"
_FOOT_SITES = ("left_foot", "right_foot")


def make_env_cfg(*, rough: bool, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the flat or rough Booster T1 velocity environment."""

    # Sensors

    terrain_scan = RayCastSensorCfg(
        name="terrain_scan",
        frame=ObjRef(type="body", name=_BASE_BODY, entity="robot"),
        ray_alignment="yaw",
        pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
        max_distance=5.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        debug_vis=not play,
    )

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

    self_collision = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(
            mode="subtree",
            pattern=_BASE_BODY,
            entity="robot",
        ),
        secondary=ContactMatch(
            mode="subtree",
            pattern=_BASE_BODY,
            entity="robot",
        ),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )

    sensors = [foot_height_scan, feet_ground_contact, self_collision]
    if rough:
        sensors.insert(0, terrain_scan)

    # Observations

    actor_terms = {
        "base_lin_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        ),
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
        "command": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "twist"},
        ),
    }
    if rough:
        actor_terms["height_scan"] = ObservationTermCfg(
            func=envs_mdp.height_scan,
            params={"sensor_name": terrain_scan.name},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            scale=1 / terrain_scan.max_distance,
        )

    critic_terms = dict(actor_terms)
    if rough:
        critic_terms["height_scan"] = ObservationTermCfg(
            func=envs_mdp.height_scan,
            params={"sensor_name": terrain_scan.name},
            scale=1 / terrain_scan.max_distance,
        )
    critic_terms.update(
        {
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
    )

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
            actuator_names=(".*",),
            scale=T1_ACTION_SCALE,
            use_default_offset=True,
        )
    }

    command_ranges = UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.5, 2.0) if play and not rough else (-1.0, 1.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-0.7, 0.7) if play and not rough else (-0.5, 0.5),
        heading=(-math.pi, math.pi),
    )
    commands: dict[str, CommandTermCfg] = {
        "twist": UniformVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(3.0, 8.0),
            rel_standing_envs=0.1,
            rel_heading_envs=0.3,
            rel_forward_envs=0.2,
            heading_command=True,
            heading_control_stiffness=0.5,
            debug_vis=True,
            ranges=command_ranges,
            viz=UniformVelocityCommandCfg.VizCfg(z_offset=1.15),
        )
    }

    # Events

    events = {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (0.01, 0.05),
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
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        "push_robot": EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.0, 3.0),
            params={
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.4, 0.4),
                    "roll": (-0.52, 0.52),
                    "pitch": (-0.52, 0.52),
                    "yaw": (-0.78, 0.78),
                }
            },
        ),
        "foot_friction": EventTermCfg(
            func=dr.geom_friction,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOMS),
                "operation": "abs",
                "ranges": (0.3, 1.2),
                "shared_random": True,
            },
        ),
        "encoder_bias": EventTermCfg(
            func=dr.encoder_bias,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "bias_range": (-0.015, 0.015),
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
    }
    if play:
        events.pop("push_robot")
        events["randomize_terrain"] = EventTermCfg(
            func=envs_mdp.randomize_terrain,
            mode="reset",
            params={},
        )

    # Rewards

    rewards = {
        "track_linear_velocity": RewardTermCfg(
            func=mdp.track_linear_velocity,
            weight=2.0,
            params={"command_name": "twist", "std": math.sqrt(0.25)},
        ),
        "track_angular_velocity": RewardTermCfg(
            func=mdp.track_angular_velocity,
            weight=2.0,
            params={"command_name": "twist", "std": math.sqrt(0.5)},
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
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "command_name": "twist",
                "std_standing": {".*": 0.05},
                "std_walking": {
                    r".*_Hip_Pitch": 0.3,
                    r".*_Hip_Roll": 0.15,
                    r".*_Hip_Yaw": 0.15,
                    r".*_Knee_Pitch": 0.35,
                    r".*_Ankle_Pitch": 0.25,
                    r".*_Ankle_Roll": 0.25,
                    r".*_Shoulder_Pitch": 0.15,
                    r".*_Shoulder_Roll": 0.15,
                    r".*_Elbow_Pitch": 0.15,
                    r".*_Elbow_Yaw": 0.15,
                    "AAHead_yaw": 0.1,
                    "Head_pitch": 0.1,
                    "Waist": 0.1,
                },
                "std_running": {
                    r".*_Hip_Pitch": 0.5,
                    r".*_Hip_Roll": 0.2,
                    r".*_Hip_Yaw": 0.2,
                    r".*_Knee_Pitch": 0.6,
                    r".*_Ankle_Pitch": 0.35,
                    r".*_Ankle_Roll": 0.35,
                    r".*_Shoulder_Pitch": 0.5,
                    r".*_Shoulder_Roll": 0.2,
                    r".*_Elbow_Pitch": 0.35,
                    r".*_Elbow_Yaw": 0.15,
                    "AAHead_yaw": 0.15,
                    "Head_pitch": 0.15,
                    "Waist": 0.15,
                },
                "walking_threshold": 0.05,
                "running_threshold": 1.5,
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
        "dof_pos_limits": RewardTermCfg(
            func=mdp.joint_pos_limits,
            weight=-1.0,
        ),
        "action_rate_l2": RewardTermCfg(
            func=mdp.action_rate_l2,
            weight=-0.1,
        ),
        "air_time": RewardTermCfg(
            func=mdp.feet_air_time,
            weight=0.0,
            params={
                "sensor_name": feet_ground_contact.name,
                "threshold_min": 0.05,
                "threshold_max": 0.5,
                "command_name": "twist",
                "command_threshold": 0.5,
            },
        ),
        "foot_clearance": RewardTermCfg(
            func=mdp.feet_clearance,
            weight=-2.0,
            params={
                "target_height": 0.1,
                "height_sensor_name": foot_height_scan.name,
                "command_name": "twist",
                "command_threshold": 0.05,
                "asset_cfg": SceneEntityCfg("robot", site_names=_FOOT_SITES),
            },
        ),
        "foot_swing_height": RewardTermCfg(
            func=mdp.feet_swing_height,
            weight=-0.25,
            params={
                "sensor_name": feet_ground_contact.name,
                "height_sensor_name": foot_height_scan.name,
                "target_height": 0.1,
                "command_name": "twist",
                "command_threshold": 0.05,
            },
        ),
        "foot_slip": RewardTermCfg(
            func=mdp.feet_slip,
            weight=-0.1,
            params={
                "sensor_name": feet_ground_contact.name,
                "command_name": "twist",
                "command_threshold": 0.05,
                "asset_cfg": SceneEntityCfg("robot", site_names=_FOOT_SITES),
            },
        ),
        "soft_landing": RewardTermCfg(
            func=mdp.soft_landing,
            weight=-1e-5,
            params={
                "sensor_name": feet_ground_contact.name,
                "command_name": "twist",
                "command_threshold": 0.05,
            },
        ),
        "self_collisions": RewardTermCfg(
            func=mdp.self_collision_cost,
            weight=-1.0,
            params={
                "sensor_name": self_collision.name,
                "force_threshold": 10.0,
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
        "out_of_terrain_bounds": TerminationTermCfg(
            func=mdp.out_of_terrain_bounds,
            time_out=True,
        ),
    }

    curriculum = {}
    if rough:
        curriculum["terrain_levels"] = CurriculumTermCfg(
            func=mdp.terrain_levels_vel,
            params={"command_name": "twist"},
        )
    curriculum["command_vel"] = CurriculumTermCfg(
        func=mdp.commands_vel,
        params={
            "command_name": "twist",
            "velocity_stages": [
                {
                    "step": 0,
                    "lin_vel_x": (-1.0, 1.0),
                    "ang_vel_z": (-0.5, 0.5),
                },
                {
                    "step": 5000 * 24,
                    "lin_vel_x": (-1.5, 2.0),
                    "ang_vel_z": (-0.7, 0.7),
                },
                {
                    "step": 10000 * 24,
                    "lin_vel_x": (-2.0, 3.0),
                },
            ],
        },
    )
    if play:
        curriculum = {}

    # Terrain

    terrain_generator = replace(ROUGH_TERRAINS_CFG) if rough else None
    if terrain_generator is not None:
        terrain_generator.curriculum = not play
        if play:
            terrain_generator.num_cols = 5
            terrain_generator.num_rows = 5
            terrain_generator.border_width = 10.0

    terrain = TerrainEntityCfg(
        terrain_type="generator" if rough else "plane",
        terrain_generator=terrain_generator,
        max_init_terrain_level=5,
    )
    if play:
        set_play_terrain_material(terrain)

    # Environment

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=terrain,
            sensors=tuple(sensors),
            num_envs=1,
            extent=2.0,
            entities={"robot": get_t1_robot_cfg()},
            spec_fn=set_play_atmosphere if play else None,
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum=curriculum,
        metrics={
            "mean_action_acc": MetricsTermCfg(func=mdp.mean_action_acc),
        },
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name=_BASE_BODY,
            distance=3.0,
            elevation=-5.0,
            azimuth=90.0,
            max_extra_envs=3 if play else 2,
        ),
        sim=SimulationCfg(
            nconmax=128 if rough else None,
            njmax=1500 if rough else 300,
            contact_sensor_maxmatch=500 if rough else 64,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
                ccd_iterations=500 if rough else 50,
            ),
        ),
        decimation=4,
        episode_length_s=int(1e9) if play else 20.0,
    )
