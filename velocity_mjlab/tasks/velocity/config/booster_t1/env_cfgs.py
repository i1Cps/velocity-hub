"""Booster T1 velocity environment configurations."""

from assets.robots.booster_t1.t1_constants import (
    T1_ACTION_SCALE,
    get_t1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.spec_config import MaterialCfg, TextureCfg


def t1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create T1 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 45

  cfg.scene.entities = {"robot": get_t1_robot_cfg()}

  # Set raycast sensor frame to T1 torso.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "Trunk"

  site_names = ("left_foot", "right_foot")
  geom_names = (
    "left_foot_collision1",
    "left_foot_collision2",
    "left_foot_collision3",
    "left_foot_collision4",
    "right_foot_collision1",
    "right_foot_collision2",
    "right_foot_collision3",
    "right_foot_collision4",
  )

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_foot_link|right_foot_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = T1_ACTION_SCALE

  cfg.viewer.body_name = "Trunk"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("Trunk",)

  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
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
  }
  cfg.rewards["pose"].params["std_running"] = {
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
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("Trunk",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("Trunk",)

  for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.0

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  if play:
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      cfg.scene.terrain.textures = (
        TextureCfg(
          name="groundplane", type="2d", builtin="checker", mark="edge",
          rgb1=(0.55, 0.55, 0.55), rgb2=(0.55, 0.55, 0.55),
          markrgb=(0.0, 0.0, 0.0),
          width=300, height=300,
        ),
      )
      cfg.scene.terrain.materials = (
        MaterialCfg(
          name="groundplane", texuniform=True, texrepeat=(5.0, 5.0),
          reflectance=0.0, texture="groundplane",
          geom_names_expr=("terrain$",),
        ),
      )
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def t1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create T1 flat terrain velocity configuration."""
  cfg = t1_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg
