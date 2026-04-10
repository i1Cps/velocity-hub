"""Z-Bot constants."""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

ZBOT_XML: Path = (
  Path(__file__).parent / "xmls" / "zbot.xml"
)
assert ZBOT_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, ZBOT_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(ZBOT_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

ZBOT_ACTUATOR_STS3215 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*",),
  stiffness=17.8,
  damping=0.6,
  effort_limit=3.35,
  armature=0.028,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.38),
  joint_pos={
    # Left arm.
    "left_shoulder_yaw": 0.0,
    "left_shoulder_pitch": -0.53,
    "left_elbow": 0.44,
    "left_gripper": 0.0,
    # Right arm.
    "right_shoulder_yaw": 0.0,
    "right_shoulder_pitch": 0.53,
    "right_elbow": -0.44,
    "right_gripper": 0.0,
    # Left leg.
    "left_hip_yaw": 0.0,
    "left_hip_roll": 0.0,
    "left_hip_pitch": -0.73,
    "left_knee": 1.18,
    "left_ankle": 0.46,
    # Right leg.
    "right_hip_yaw": 0.0,
    "right_hip_roll": 0.0,
    "right_hip_pitch": 0.73,
    "right_knee": 1.18,
    "right_ankle": -0.46,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(
    r"^(left|right)_foot_collision\d+$",
  ),
  contype=1,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

ZBOT_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(ZBOT_ACTUATOR_STS3215,),
  soft_joint_pos_limit_factor=0.9,
)


def get_zbot_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FEET_ONLY_COLLISION,),
    spec_fn=get_spec,
    articulation=ZBOT_ARTICULATION,
  )


ZBOT_ACTION_SCALE: dict[str, float] = {
  ".*": 0.2,
}
