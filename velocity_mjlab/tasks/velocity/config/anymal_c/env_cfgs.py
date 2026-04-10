"""ANYmal C velocity environment configurations."""

from assets.robots.anymal_c.c_constants import (
    ANYMAL_C_ACTION_SCALE,
    get_anymal_c_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.spec_config import MaterialCfg, TextureCfg


def anymal_c_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create ANYmal C rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = 45

  cfg.scene.entities = {"robot": get_anymal_c_robot_cfg()}

  # Set raycast sensor frame to anymal base.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "base"

  site_names = ("LF_foot", "RF_foot", "LH_foot", "RH_foot")
  geom_names = ("LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT")

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=r"^(LF|RF|LH|RH)_FOOT$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground_cfg,)

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = ANYMAL_C_ACTION_SCALE

  cfg.viewer.body_name = "base"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 0.7

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base",)

  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*HAA.*": 0.15,
    r".*HFE.*": 0.3,
    r".*KFE.*": 0.3,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*HAA.*": 0.2,
    r".*HFE.*": 0.5,
    r".*KFE.*": 0.5,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("base",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base",)

  for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  # Quadruped legs move horizontally — foot height rewards less relevant.
  cfg.rewards["foot_clearance"].weight = 0.0
  cfg.rewards["foot_swing_height"].weight = -0.125

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.0

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

def anymal_c_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create ANYmal C flat terrain velocity configuration."""
  cfg = anymal_c_rough_env_cfg(play=play)

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
    twist_cmd.ranges.lin_vel_x = (-1.0, 2.0)
    twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)

  return cfg
