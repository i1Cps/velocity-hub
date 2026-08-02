"""Generic quadruped model and actuator configuration."""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg


QUADRUPED_XML = Path(__file__).parent / "xmls" / "quadruped.xml"
JOINT_PATTERN = r"^quadruped_(front|back)_(left|right)_(hip|ankle)$"


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(QUADRUPED_XML))


QUADRUPED_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(JOINT_PATTERN,),
    stiffness=150.0,
    damping=15.0,
    effort_limit=150.0,
    armature=0.01,
)


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


FOOT_GEOMS = (
    "quadruped_front_left_foot",
    "quadruped_front_right_foot",
    "quadruped_back_left_foot",
    "quadruped_back_right_foot",
)


FEET_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=FOOT_GEOMS,
    contype=1,
    conaffinity=1,
    condim=3,
    priority=1,
    friction=(1.0, 0.5, 0.5),
)


QUADRUPED_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(QUADRUPED_ACTUATOR,),
    soft_joint_pos_limit_factor=0.9,
)


def get_quadruped_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        collisions=(FEET_ONLY_COLLISION,),
        spec_fn=get_spec,
        articulation=QUADRUPED_ARTICULATION,
    )


QUADRUPED_ACTION_SCALE = {
    JOINT_PATTERN: (
        0.25
        * QUADRUPED_ACTUATOR.effort_limit
        / QUADRUPED_ACTUATOR.stiffness
    ),
}
