"""Spot constants."""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

SPOT_XML: Path = (
  Path(__file__).parent / "xmls" / "spot.xml"
)
assert SPOT_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, SPOT_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(SPOT_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec

##
# Actuator config.
##

SPOT_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(".*",),
  stiffness=500.0,
  damping=1.0,
  effort_limit=80.0,
  armature=0.0,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.46),
  joint_pos={
    # Left front leg.
    "fl_hx": 0.0,
    "fl_hy": 1.04,
    "fl_kn": -1.8,
    # Right front leg.
    "fr_hx": 0.0,
    "fr_hy": 1.04,
    "fr_kn": -1.8,
    # Left hind leg.
    "hl_hx": 0.0,
    "hl_hy": 1.04,
    "hl_kn": -1.8,
    # Right hind leg.
    "hr_hx": 0.0,
    "hr_hy": 1.04,
    "hr_kn": -1.8,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(
    r"^(FL|FR|HL|HR)",
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

SPOT_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(SPOT_ACTUATOR,),
    soft_joint_pos_limit_factor=0.9,
)

def get_spot_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        collisions=(FEET_ONLY_COLLISION,),
        spec_fn=get_spec,
        articulation=SPOT_ARTICULATION,
    )

SPOT_ACTION_SCALE: dict[str, float] = {
    ".*": 0.25 * SPOT_ACTUATOR.effort_limit / SPOT_ACTUATOR.stiffness,
}
