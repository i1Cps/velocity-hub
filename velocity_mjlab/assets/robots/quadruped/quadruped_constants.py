"""Quadruped constants."""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF.
##

QUADRUPED_XML: Path = (
  Path(__file__).parent / "xmls" / "quadruped.xml"
)
assert QUADRUPED_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(QUADRUPED_XML))


##
# Actuator config.
##

EFFORT_LIMIT = 150.0
STIFFNESS = 150.0
DAMPING = 15.0

QUADRUPED_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("quadruped_.*",),
  stiffness=STIFFNESS,
  damping=DAMPING,
  effort_limit=EFFORT_LIMIT,
  armature=0.01,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.55),
  joint_pos={
    "quadruped_front_left_hip": 0.0,
    "quadruped_front_left_ankle": 1.0,
    "quadruped_front_right_hip": 0.0,
    "quadruped_front_right_ankle": -1.0,
    "quadruped_back_left_hip": 0.0,
    "quadruped_back_left_ankle": -1.0,
    "quadruped_back_right_hip": 0.0,
    "quadruped_back_right_ankle": 1.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(
    "quadruped_front_left_foot",
    "quadruped_front_right_foot",
    "quadruped_back_left_foot",
    "quadruped_back_right_foot",
  ),
  contype=1,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(1.0, 0.5, 0.5),
)

##
# Final config.
##

QUADRUPED_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(QUADRUPED_ACTUATOR_CFG,),
  soft_joint_pos_limit_factor=0.9,
)

QUADRUPED_ACTION_SCALE: dict[str, float] = {
  "quadruped_.*": 0.25 * EFFORT_LIMIT / STIFFNESS,
}


def get_quadruped_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FEET_ONLY_COLLISION,),
    spec_fn=get_spec,
    articulation=QUADRUPED_ARTICULATION,
  )
