"""Booster T1 constants."""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

T1_XML: Path = (
  Path(__file__).parent / "xmls" / "t1.xml"
)
assert T1_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, T1_XML.parent / "assets", meshdir)
  return assets

def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(T1_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

T1_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(".*",),
  stiffness=75.0,
  damping=5.0,
  effort_limit=60.0,
  armature=0.005,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.665),
  joint_pos={
    "AAHead_yaw": 0.0,
    "Head_pitch": 0.0,
    "Left_Shoulder_Pitch": 0.0,
    "Left_Shoulder_Roll": -1.4,
    "Left_Elbow_Pitch": 0.0,
    "Left_Elbow_Yaw": -0.4,
    "Right_Shoulder_Pitch": 0.0,
    "Right_Shoulder_Roll": 1.4,
    "Right_Elbow_Pitch": 0.0,
    "Right_Elbow_Yaw": 0.4,
    "Waist": 0.0,
    "Left_Hip_Pitch": -0.2,
    "Left_Hip_Roll": 0.0,
    "Left_Hip_Yaw": 0.0,
    "Left_Knee_Pitch": 0.4,
    "Left_Ankle_Pitch": -0.2,
    "Left_Ankle_Roll": 0.0,
    "Right_Hip_Pitch": -0.2,
    "Right_Hip_Roll": 0.0,
    "Right_Hip_Yaw": 0.0,
    "Right_Knee_Pitch": 0.4,
    "Right_Ankle_Pitch": -0.2,
    "Right_Ankle_Roll": 0.0,
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

T1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(T1_ACTUATOR,),
  soft_joint_pos_limit_factor=0.9,
)


def get_t1_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FEET_ONLY_COLLISION,),
    spec_fn=get_spec,
    articulation=T1_ARTICULATION,
  )


T1_ACTION_SCALE: dict[str, float] = {}
for a in T1_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    T1_ACTION_SCALE[n] = 0.25 * e / s
