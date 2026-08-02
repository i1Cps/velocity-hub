"""Duck Mini Pro model and measured actuator configuration.

The plant includes the measured +/-0.5 degree gearbox backlash. The actuator
uses the full ST3025 BAM m6 fit from the physical pendulum experiments, with
the firmware configured for P=10, D=0, I=0 and its motion profile disabled.
"""

from pathlib import Path

import mujoco

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from .bam_actuator import DuckMiniProBamActuatorCfg


DUCK_MINI_PRO_XML = Path(__file__).parent / "xmls" / "duck_mini_pro.xml"
ST3025_M6_PARAMS = Path(__file__).parent / "params" / "st3025_m6.json"

SERVO_JOINT_PATTERN = r"^(left|right)_(hip_(yaw|roll|pitch)|knee|ankle)$"

# The deployed policy uses a quarter of the measured torque/gain ratio.
MAX_TORQUE = 3.92
EFFECTIVE_POSITION_GAIN = 13.57
ACTION_SCALE = 0.25 * MAX_TORQUE / EFFECTIVE_POSITION_GAIN


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(DUCK_MINI_PRO_XML))


ST3025_BAM = DuckMiniProBamActuatorCfg(
    target_names_expr=(SERVO_JOINT_PATTERN,),
    json_path=str(ST3025_M6_PARAMS),
    kp_fw=10,
    vin_range=(11.0, 12.1),
    vin_drop_gain_range=(0.0, 0.2),
    vin_min=11.0,
    max_current=2.47,
    delay_min_lag=3,
    delay_max_lag=6,
)


HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.280212),
    joint_pos={
        # Left leg.
        "left_hip_yaw": 0.0300396,
        "left_hip_roll": 0.1400,
        "left_hip_pitch": 1.09051,
        "left_knee": 1.40611,
        "left_ankle": -0.320768,
        # Right leg.
        "right_hip_yaw": -0.0035049,
        "right_hip_roll": -0.1400,
        "right_hip_pitch": -1.03277,
        "right_knee": 1.36127,
        "right_ankle": -0.355119,
    },
    joint_vel={".*": 0.0},
)


FOOT_GEOMS = (
    "left_foot_collision_mesh",
    "right_foot_collision_mesh",
)

FEET_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=(r"^(left|right)_foot_collision",),
    contype=1,
    conaffinity=1,
    condim=3,
    priority=1,
    friction=(0.9,),
)


DUCK_MINI_PRO_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(ST3025_BAM,),
    soft_joint_pos_limit_factor=0.90,
)


def get_duck_mini_pro_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        collisions=(FEET_ONLY_COLLISION,),
        spec_fn=get_spec,
        articulation=DUCK_MINI_PRO_ARTICULATION,
    )


DUCK_MINI_PRO_ACTION_SCALE = {
    SERVO_JOINT_PATTERN: ACTION_SCALE,
}
