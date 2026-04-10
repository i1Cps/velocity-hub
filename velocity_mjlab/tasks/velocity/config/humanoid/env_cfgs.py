"""Humanoid velocity environment configurations."""

from assets.robots.humanoid.humanoid_constants import (
    HUMANOID_ACTION_SCALE,
    get_humanoid_robot_cfg
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointEffortActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.spec_config import MaterialCfg, TextureCfg


def humanoid_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create humanoid rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 45

  cfg.scene.entities = {"robot": get_humanoid_robot_cfg()}

  # Set raycast sensor frame to humanoid pelvis.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "humanoid_pelvis"

  site_names = ("humanoid_left_foot", "humanoid_right_foot")
  geom_names = ("humanoid_left_foot", "humanoid_right_foot")

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(humanoid_left_foot|humanoid_right_foot)$",
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
    primary=ContactMatch(mode="subtree", pattern="humanoid_pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="humanoid_pelvis", entity="robot"),
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

  cfg.actions["joint_pos"] = JointEffortActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=HUMANOID_ACTION_SCALE,
  )

  cfg.viewer.body_name = "humanoid"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("humanoid",)

  # Humanoid asset exposes different builtin sensor names than Unitree robots.
  cfg.observations["actor"].terms["base_lin_vel"].params[
    "sensor_name"
  ] = "robot/local_linvel"
  cfg.observations["actor"].terms["base_ang_vel"].params["sensor_name"] = "robot/gyro"

  # Rationale for std values:
  # - Knees/hip_y get the loosest std to allow natural leg bending during stride.
  # - Hip x/z stay tighter to prevent excessive lateral sway and keep gait stable.
  # - Waist roll/pitch stay tight to keep the torso upright and stable.
  # - Shoulders/elbows get moderate freedom for natural arm swing during walking.
  # Running values are ~1.5-2x walking values to accommodate larger motion range.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*hip_y.*": 0.3,
    r".*hip_x.*": 0.15,
    r".*hip_z.*": 0.15,
    r".*knee.*": 0.35,
    # Waist.
    r".*abdomen_z.*": 0.2,
    r".*abdomen_x.*": 0.08,
    r".*abdomen_y.*": 0.1,
    # Arms.
    r".*shoulder1.*": 0.15,
    r".*shoulder2.*": 0.15,
    r".*elbow.*": 0.15,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*hip_y.*": 0.5,
    r".*hip_x.*": 0.2,
    r".*hip_z.*": 0.2,
    r".*knee.*": 0.6,
    # Waist.
    r".*abdomen_z.*": 0.3,
    r".*abdomen_x.*": 0.08,
    r".*abdomen_y.*": 0.2,
    # Arms.
    r".*shoulder1.*": 0.5,
    r".*shoulder2.*": 0.2,
    r".*elbow.*": 0.35,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("humanoid",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("humanoid",)

  for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["angular_momentum"].params["sensor_name"] = "robot/global_angvel"
  cfg.rewards["air_time"].weight = 0.0

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )


  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
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
          markrgb=(0.0, 0.0, 0.0), width=300, height=300,
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


def humanoid_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create humanoid flat terrain velocity configuration."""
  cfg = humanoid_rough_env_cfg(play=play)

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

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg
