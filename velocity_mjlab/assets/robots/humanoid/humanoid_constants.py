"""Humanoid constants."""

from pathlib import Path

import mujoco

from mjlab.actuator import XmlMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF.
##

HUMANOID_XML: Path = (
  Path(__file__).parent / "xmls" / "humanoid.xml"
)
assert HUMANOID_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(HUMANOID_XML))


##
# Actuator config.
##

HUMANOID_MOTOR_CFG = XmlMotorActuatorCfg(
  target_names_expr=("humanoid_.*",),
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 1.25),
  joint_pos={
    ".*_hip_y": -0.343,
    ".*_knee": -0.669271,
    "humanoid_right_shoulder1": 0.743,
    "humanoid_right_shoulder2": -0.669,
    "humanoid_left_shoulder1": -0.743,
    "humanoid_left_shoulder2": 0.669,
    ".*_elbow": -0.743,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=("humanoid_right_foot", "humanoid_left_foot"),
  contype=1,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

HUMANOID_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(HUMANOID_MOTOR_CFG,),
  soft_joint_pos_limit_factor=0.9,
)


HUMANOID_ACTION_SCALE: dict[str, float] = {"humanoid_.*": 1.0}


def get_humanoid_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FEET_ONLY_COLLISION,),
    spec_fn=get_spec,
    articulation=HUMANOID_ARTICULATION,
  )
