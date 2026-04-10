"""Unitree H1 constants."""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

H1_XML: Path = (
  Path(__file__).parent / "xmls" / "h1.xml"
)
assert H1_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, H1_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(H1_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
# Same motor specs as H1_2, adjusted for H1 joint names.
##

H1_ACTUATOR_M107_24_2 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_yaw",
    ".*_hip_pitch",
    ".*_hip_roll",
    "torso",
  ),
  stiffness=98.7,
  damping=6.3,
  effort_limit=200.0,
  armature=0.025,
)
H1_ACTUATOR_M107_24_1 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_knee",
  ),
  stiffness=157.7,
  damping=10.1,
  effort_limit=300.0,
  armature=0.04,
)
H1_ACTUATOR_GO2HV_1 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_ankle",
    ".*_shoulder_pitch",
    ".*_shoulder_roll",
  ),
  stiffness=19.7,
  damping=1.3,
  effort_limit=40.0,
  armature=0.005,
)
H1_ACTUATOR_GO2HV_2 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_yaw",
    ".*_elbow",
  ),
  stiffness=7.9,
  damping=0.5,
  effort_limit=18.0,
  armature=0.002,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 1.03),
  joint_pos={
    ".*_hip_pitch": -0.0785,
    ".*_knee": 0.41,
    ".*_ankle": -0.307,
    ".*_shoulder_pitch": 0.28,
    ".*_elbow": 0.52,
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

H1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    H1_ACTUATOR_M107_24_2,
    H1_ACTUATOR_M107_24_1,
    H1_ACTUATOR_GO2HV_1,
    H1_ACTUATOR_GO2HV_2,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_h1_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FEET_ONLY_COLLISION,),
    spec_fn=get_spec,
    articulation=H1_ARTICULATION,
  )


H1_ACTION_SCALE: dict[str, float] = {}
for a in H1_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    H1_ACTION_SCALE[n] = 0.25 * e / s
