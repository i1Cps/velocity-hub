"""Simple native actuator used only by the Duck Mini Pro website policy."""

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from .robot_cfg import (
    ACTION_SCALE,
    EFFECTIVE_POSITION_GAIN,
    FEET_ONLY_COLLISION,
    HOME_KEYFRAME,
    MAX_TORQUE,
    SERVO_JOINT_PATTERN,
    get_spec,
)


WEBSITE_POSITION_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(SERVO_JOINT_PATTERN,),
    stiffness=EFFECTIVE_POSITION_GAIN,
    damping=0.9,
    effort_limit=MAX_TORQUE,
    armature=0.00465,
    frictionloss=0.0,
    viscous_damping=0.0,
)

WEBSITE_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(WEBSITE_POSITION_ACTUATOR,),
    soft_joint_pos_limit_factor=0.90,
)


def get_duck_mini_pro_website_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        collisions=(FEET_ONLY_COLLISION,),
        spec_fn=get_spec,
        articulation=WEBSITE_ARTICULATION,
    )


DUCK_MINI_PRO_WEBSITE_ACTION_SCALE = {
    SERVO_JOINT_PATTERN: ACTION_SCALE,
}
