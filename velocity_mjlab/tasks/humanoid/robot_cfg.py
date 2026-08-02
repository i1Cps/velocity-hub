"""Generic Humanoid model and actuator configuration."""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg


HUMANOID_XML = Path(__file__).parent / "xmls" / "humanoid.xml"


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(HUMANOID_XML))


# These effort limits preserve the original XML motor ceilings:
# 0.4 control range × 100/300/200/25 gear.
HUMANOID_CORE_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "humanoid_abdomen_.*",
        "humanoid_.*_hip_x",
        "humanoid_.*_hip_z",
    ),
    stiffness=40.0,
    damping=5.0,
    effort_limit=40.0,
)

HUMANOID_HIP_PITCH_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=("humanoid_.*_hip_y",),
    stiffness=120.0,
    damping=5.0,
    effort_limit=120.0,
)

HUMANOID_KNEE_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=("humanoid_.*_knee",),
    stiffness=80.0,
    damping=1.0,
    effort_limit=80.0,
)

HUMANOID_ARM_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "humanoid_.*_shoulder1",
        "humanoid_.*_shoulder2",
        "humanoid_.*_elbow",
    ),
    stiffness=10.0,
    damping=1.0,
    effort_limit=10.0,
)


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


FOOT_GEOMS = ("humanoid_left_foot", "humanoid_right_foot")


FEET_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=("humanoid_right_foot", "humanoid_left_foot"),
    contype=1,
    conaffinity=1,
    condim=3,
    priority=1,
    friction=(0.6,),
)


HUMANOID_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        HUMANOID_CORE_ACTUATOR,
        HUMANOID_HIP_PITCH_ACTUATOR,
        HUMANOID_KNEE_ACTUATOR,
        HUMANOID_ARM_ACTUATOR,
    ),
    soft_joint_pos_limit_factor=0.9,
)


def get_humanoid_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        collisions=(FEET_ONLY_COLLISION,),
        spec_fn=get_spec,
        articulation=HUMANOID_ARTICULATION,
    )


HUMANOID_ACTION_SCALE: dict[str, float] = {}
for actuator in HUMANOID_ARTICULATION.actuators:
    assert isinstance(actuator, BuiltinPositionActuatorCfg)
    assert actuator.effort_limit is not None
    for name in actuator.target_names_expr:
        HUMANOID_ACTION_SCALE[name] = (
            0.25 * actuator.effort_limit / actuator.stiffness
        )
