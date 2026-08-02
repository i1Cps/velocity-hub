"""Unitree H1 model and actuator configuration."""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg


H1_XML = Path(__file__).parent / "xmls" / "h1.xml"


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(H1_XML))


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
    target_names_expr=(".*_knee",),
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


HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 1.03),
    joint_pos={
        ".*_hip_pitch": -0.0785,
        ".*_knee": 0.41,
        ".*_ankle": -0.307,
        ".*_shoulder_pitch": 0.28,
        ".*_elbow": 0.52,
    },
    joint_vel={".*": 0.0},
)


FOOT_GEOMS = tuple(
    f"{side}_foot_collision{i}"
    for side in ("left", "right")
    for i in range(1, 4)
)


FEET_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=(r"^(left|right)_foot_collision\d+$",),
    contype=1,
    conaffinity=1,
    condim=3,
    priority=1,
    friction=(0.6,),
)


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
for actuator in H1_ARTICULATION.actuators:
    assert isinstance(actuator, BuiltinPositionActuatorCfg)
    assert actuator.effort_limit is not None
    for name in actuator.target_names_expr:
        H1_ACTION_SCALE[name] = (
            0.25 * actuator.effort_limit / actuator.stiffness
        )
