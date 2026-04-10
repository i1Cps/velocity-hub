"""K-Bot constants."""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

KBOT_XML: Path = (
  Path(__file__).parent / "xmls" / "kbot.xml"
)
assert KBOT_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, KBOT_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(KBOT_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

KBOT_ACTUATOR_ROBSTRIDE_04 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_pitch_.*",
    ".*_knee_.*",
  ),
  stiffness=120.0,
  damping=4.0,
  effort_limit=120.0,
  armature=0.04,
)

KBOT_ACTUATOR_ROBSTRIDE_03 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_roll_.*",
    ".*_hip_yaw_.*",
    ".*_shoulder_pitch_.*",
    ".*_shoulder_roll_.*",
  ),
  stiffness=120.0,
  damping=3.0,
  effort_limit=60.0,
  armature=0.02,
)

KBOT_ACTUATOR_ROBSTRIDE_02 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_ankle_.*",
    ".*_shoulder_yaw_.*",
    ".*_elbow_.*",
  ),
  stiffness=40.0,
  damping=1.0,
  effort_limit=17.0,
  armature=0.0042,
)

KBOT_ACTUATOR_ROBSTRIDE_00 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_wrist_.*",
  ),
  stiffness=25.0,
  damping=0.5,
  effort_limit=14.0,
  armature=0.001,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.81),
  joint_pos={
    # Left leg.
    "dof_left_hip_pitch_04": 0.312,
    "dof_left_hip_roll_03": 0.0,
    "dof_left_hip_yaw_03": 0.0,
    "dof_left_knee_04": 0.669,
    "dof_left_ankle_02": -0.36,
    # Right leg.
    "dof_right_hip_pitch_04": -0.312,
    "dof_right_hip_roll_03": 0.0,
    "dof_right_hip_yaw_03": 0.0,
    "dof_right_knee_04": -0.669,
    "dof_right_ankle_02": 0.36,
    # Right arm.
    "dof_right_shoulder_pitch_03": 0.29823,
    "dof_right_shoulder_roll_03": -0.233,
    "dof_right_shoulder_yaw_02": 0.0,
    "dof_right_elbow_02": 1.290,
    "dof_right_wrist_00": 0.0,
    # Left arm.
    "dof_left_shoulder_pitch_03": -0.43437,
    "dof_left_shoulder_roll_03": 0.233,
    "dof_left_shoulder_yaw_02": 0.0,
    "dof_left_elbow_02": -1.290,
    "dof_left_wrist_00": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(
    r"^(L|R)FootBushing.*_collision_capsule_\d+$",
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

KBOT_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    KBOT_ACTUATOR_ROBSTRIDE_04,
    KBOT_ACTUATOR_ROBSTRIDE_03,
    KBOT_ACTUATOR_ROBSTRIDE_02,
    KBOT_ACTUATOR_ROBSTRIDE_00,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_kbot_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FEET_ONLY_COLLISION,),
    spec_fn=get_spec,
    articulation=KBOT_ARTICULATION,
  )


KBOT_ACTION_SCALE: dict[str, float] = {}
for a in KBOT_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    KBOT_ACTION_SCALE[n] = 0.25 * e / s
