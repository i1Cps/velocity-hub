"""ANYmal C constants."""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

ANYMAL_C_XML: Path = (
  Path(__file__).parent / "xmls" / "c.xml"
)
assert ANYMAL_C_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, ANYMAL_C_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(ANYMAL_C_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

ANYMAL_C_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(".*",),
  stiffness=100.0,
  damping=1.0,
  effort_limit=80.0,
  armature=0.0,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.62),
  joint_pos={
    # Left front leg.
    "LF_HAA": 0.0,
    "LF_HFE": 0.0,
    "LF_KFE": 0.0,
    # Right front leg.
    "RF_HAA": 0.0,
    "RF_HFE": 0.0,
    "RF_KFE": 0.0,
    # Left hind leg.
    "LH_HAA": 0.0,
    "LH_HFE": 0.0,
    "LH_KFE": 0.0,
    # Right hind leg.
    "RH_HAA": 0.0,
    "RH_HFE": 0.0,
    "RH_KFE": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(
    r"^(LF|RF|LH|RH)_FOOT$",
  ),
  contype=1,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.8,),
)

##
# Final config.
##

ANYMAL_C_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(ANYMAL_C_ACTUATOR,),
  soft_joint_pos_limit_factor=0.9,
)


def get_anymal_c_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FEET_ONLY_COLLISION,),
    spec_fn=get_spec,
    articulation=ANYMAL_C_ARTICULATION,
  )


ANYMAL_C_ACTION_SCALE: dict[str, float] = {
  ".*": 0.25 * ANYMAL_C_ACTUATOR.effort_limit / ANYMAL_C_ACTUATOR.stiffness,
}
