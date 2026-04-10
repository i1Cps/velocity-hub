"""Duck Mini constants."""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

DUCK_MINI_XML: Path = (
  Path(__file__).parent / "xmls" / "duck_mini.xml"
)
assert DUCK_MINI_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, DUCK_MINI_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(DUCK_MINI_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec

##
# Actuator config.
# Black box servo model: joint frictionloss left at zero since the servo's
# internal PD loop already compensates for gearbox resistance. All behaviour
# is defined by stiffness/damping/forcerange.
##

STS3215 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*",),
  stiffness=17.8,
  damping=0.6,
  effort_limit=4.35,
  armature=0.028,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.22),
  joint_pos={
    # Left leg.
    "left_hip_roll": 0.002,
    "left_hip_yaw": -0.06,
    "left_hip_pitch": 0.63,
    "left_knee": -1.37,
    "left_ankle": -0.79,
    # Right leg.
    "right_hip_roll": -0.002,
    "right_hip_yaw": -0.06,
    "right_hip_pitch": -0.63,
    "right_knee": 1.37,
    "right_ankle": 0.79,
    # Neck.
    "neck_pitch1": 0.0,
    "neck_pitch2": 0.0,
    "neck_yaw": 0.0,
    "neck_roll": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(
    r"^(left|right)_foot_collision",
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

DUCK_MINI_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(STS3215,),
    soft_joint_pos_limit_factor=0.9,
)

def get_duck_mini_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        collisions=(FEET_ONLY_COLLISION,),
        spec_fn=get_spec,
        articulation=DUCK_MINI_ARTICULATION,
    )

DUCK_MINI_ACTION_SCALE: dict[str, float] = {
    ".*": 0.2,
}
