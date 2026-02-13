"""MyoSkeleton constants for mjlab integration.

Defines the MyoSkeleton robot (from MyoHub/myosuite) as an mjlab entity
for use in motion imitation / tracking tasks.

The MyoSkeleton is a full-body musculoskeletal model with 80 bodies, a free
root joint, and 46 motor-actuated joints (fingers disabled).  For PD position
control in mjlab we convert the original motor gear ratios into
BuiltinPositionActuator groups with appropriate stiffness / damping /
effort limits.
"""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# Paths.
##

_MYOSKELETON_DIR: Path = MJLAB_SRC_PATH / "asset_zoo" / "robots" / "myoskeleton"
MYOSKELETON_XML: Path = _MYOSKELETON_DIR / "xmls" / "myoskeleton_mjlab.xml"
MYOSKELETON_MOTION_DIR: Path = _MYOSKELETON_DIR / "motions"

assert MYOSKELETON_XML.exists(), f"Missing {MYOSKELETON_XML}"

##
# Finger joints to disable.
##

FINGER_JOINTS: tuple[str, ...] = (
  # Right hand
  "cmc_flexion_r",
  "cmc_abduction_r",
  "mp_flexion_r",
  "ip_flexion_r",
  "mcp2_flexion_r",
  "mcp2_abduction_r",
  "pm2_flexion_r",
  "md2_flexion_r",
  "mcp3_flexion_r",
  "mcp3_abduction_r",
  "pm3_flexion_r",
  "md3_flexion_r",
  "mcp4_flexion_r",
  "mcp4_abduction_r",
  "pm4_flexion_r",
  "md4_flexion_r",
  "mcp5_flexion_r",
  "mcp5_abduction_r",
  "pm5_flexion_r",
  "md5_flexion_r",
  # Left hand
  "cmc_flexion_l",
  "cmc_abduction_l",
  "mp_flexion_l",
  "ip_flexion_l",
  "mcp2_flexion_l",
  "mcp2_abduction_l",
  "pm2_flexion_l",
  "md2_flexion_l",
  "mcp3_flexion_l",
  "mcp3_abduction_l",
  "pm3_flexion_l",
  "md3_flexion_l",
  "mcp4_flexion_l",
  "mcp4_abduction_l",
  "pm4_flexion_l",
  "md4_flexion_l",
  "mcp5_flexion_l",
  "mcp5_abduction_l",
  "pm5_flexion_l",
  "md5_flexion_l",
)

##
# Spec factory.
##


def get_spec() -> mujoco.MjSpec:
  """Load the MyoSkeleton MjSpec with fingers disabled."""
  spec = mujoco.MjSpec.from_file(str(MYOSKELETON_XML))

  # Remove finger joints.
  finger_set = set(FINGER_JOINTS)
  for j in list(spec.joints):
    if j.name in finger_set:
      spec.delete(j)

  return spec


##
# Actuator configuration.
#
# The original MyoSkeleton uses motor actuators with gear ratios.  For PD
# position control we define BuiltinPositionActuator groups with stiffness,
# damping and effort limits calibrated to the original gear ratios.
#
# Approach: effort_limit ~ gear_ratio (torque capacity), and PD gains are
# set using a natural-frequency approach similar to the G1 robot.
##

NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535  # 10 Hz

# Per-group parameters:  (stiffness, damping, effort_limit, armature)
# We use a simplified reflected inertia model: armature ~ gear^2 * base_inertia.
_BASE_INERTIA = 0.5e-5  # Approximate motor rotor inertia.


def _actuator_params(gear: float) -> tuple[float, float, float, float]:
  """Compute (stiffness, damping, effort_limit, armature) from gear ratio."""
  armature = _BASE_INERTIA * gear * gear
  stiffness = armature * NATURAL_FREQ**2
  damping = 2.0 * 2.0 * armature * NATURAL_FREQ  # damping_ratio = 2.0
  effort_limit = gear * 0.5  # Conservative torque scaling.
  return stiffness, damping, effort_limit, armature


# ── Lumbar spine (gear 160 flex/bend, 100 rotation) ─────────────────────────

_S_SPINE_FB, _D_SPINE_FB, _E_SPINE_FB, _A_SPINE_FB = _actuator_params(160)
_S_SPINE_AR, _D_SPINE_AR, _E_SPINE_AR, _A_SPINE_AR = _actuator_params(100)

MYOSKELETON_ACTUATOR_SPINE_FLEX_BEND = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "L5_S1_Flex_Ext",
    "L5_S1_Lat_Bending",
    "L4_L5_Flex_Ext",
    "L4_L5_Lat_Bending",
    "L3_L4_Flex_Ext",
    "L3_L4_Lat_Bending",
    "L2_L3_Flex_Ext",
    "L2_L3_Lat_Bending",
    "L1_L2_Flex_Ext",
    "L1_L2_Lat_Bending",
    "L1_T12_Flex_Ext",
    "L1_T12_Lat_Bending",
  ),
  stiffness=_S_SPINE_FB,
  damping=_D_SPINE_FB,
  effort_limit=_E_SPINE_FB,
  armature=_A_SPINE_FB,
)

MYOSKELETON_ACTUATOR_SPINE_ROTATION = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "L5_S1_axial_rotation",
    "L4_L5_axial_rotation",
    "L3_L4_axial_rotation",
    "L2_L3_axial_rotation",
    "L1_L2_axial_rotation",
    "L1_T12_axial_rotation",
  ),
  stiffness=_S_SPINE_AR,
  damping=_D_SPINE_AR,
  effort_limit=_E_SPINE_AR,
  armature=_A_SPINE_AR,
)

# ── Arms (gear 250 shoulder/elbow/pro_sup, 50 wrist) ────────────────────────

_S_ARM, _D_ARM, _E_ARM, _A_ARM = _actuator_params(250)

MYOSKELETON_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "elbow_flex_r",
    "pro_sup",
    "elbow_flex_l",
    "pro_sup_l",
  ),
  stiffness=_S_ARM,
  damping=_D_ARM,
  effort_limit=_E_ARM,
  armature=_A_ARM,
)

_S_WRIST, _D_WRIST, _E_WRIST, _A_WRIST = _actuator_params(50)

MYOSKELETON_ACTUATOR_WRIST = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "flexion_r",
    "deviation",
    "flexion_l",
    "deviation_l",
  ),
  stiffness=_S_WRIST,
  damping=_D_WRIST,
  effort_limit=_E_WRIST,
  armature=_A_WRIST,
)

# ── Legs ─────────────────────────────────────────────────────────────────────

_S_HIP_FLEX, _D_HIP_FLEX, _E_HIP_FLEX, _A_HIP_FLEX = _actuator_params(275)

MYOSKELETON_ACTUATOR_HIP_FLEX = BuiltinPositionActuatorCfg(
  target_names_expr=("hip_flexion_r", "hip_flexion_l"),
  stiffness=_S_HIP_FLEX,
  damping=_D_HIP_FLEX,
  effort_limit=_E_HIP_FLEX,
  armature=_A_HIP_FLEX,
)

_S_HIP_ADD, _D_HIP_ADD, _E_HIP_ADD, _A_HIP_ADD = _actuator_params(530)

MYOSKELETON_ACTUATOR_HIP_ADD = BuiltinPositionActuatorCfg(
  target_names_expr=("hip_adduction_r", "hip_adduction_l"),
  stiffness=_S_HIP_ADD,
  damping=_D_HIP_ADD,
  effort_limit=_E_HIP_ADD,
  armature=_A_HIP_ADD,
)

_S_HIP_ROT, _D_HIP_ROT, _E_HIP_ROT, _A_HIP_ROT = _actuator_params(600)

MYOSKELETON_ACTUATOR_HIP_ROT = BuiltinPositionActuatorCfg(
  target_names_expr=("hip_rotation_r", "hip_rotation_l"),
  stiffness=_S_HIP_ROT,
  damping=_D_HIP_ROT,
  effort_limit=_E_HIP_ROT,
  armature=_A_HIP_ROT,
)

_S_KNEE, _D_KNEE, _E_KNEE, _A_KNEE = _actuator_params(600)

MYOSKELETON_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=("knee_angle_r", "knee_angle_l"),
  stiffness=_S_KNEE,
  damping=_D_KNEE,
  effort_limit=_E_KNEE,
  armature=_A_KNEE,
)

_S_ANKLE, _D_ANKLE, _E_ANKLE, _A_ANKLE = _actuator_params(500)

MYOSKELETON_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=("ankle_angle_r", "ankle_angle_l"),
  stiffness=_S_ANKLE,
  damping=_D_ANKLE,
  effort_limit=_E_ANKLE,
  armature=_A_ANKLE,
)

_S_FOOT, _D_FOOT, _E_FOOT, _A_FOOT = _actuator_params(50)

MYOSKELETON_ACTUATOR_FOOT = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "subtalar_angle_r",
    "mtp_angle_r",
    "subtalar_angle_l",
    "mtp_angle_l",
  ),
  stiffness=_S_FOOT,
  damping=_D_FOOT,
  effort_limit=_E_FOOT,
  armature=_A_FOOT,
)

##
# Mimic body names (for motion tracking).
# These are the 15 key bodies used for tracking reference motions.
##

MYOSKELETON_TRACKING_BODIES: tuple[str, ...] = (
  "pelvis",
  "thoracic_spine",
  "skull",
  "humerus_l",
  "ulna_l",
  "lunate_l",
  "femur_l",
  "tibia_l",
  "calcn_l",
  "humerus_r",
  "ulna_r",
  "lunate_r",
  "femur_r",
  "tibia_r",
  "calcn_r",
)

##
# Collision configuration.
# The myoskeleton uses capsule collision geoms (class "myo_coll").
##

MYOSKELETON_COLLISION = CollisionCfg(
  geom_names_expr=(".*",),
  contype=0,
  conaffinity=1,
  condim=3,
)

##
# Keyframes.
##

STANDING_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.95),
  joint_pos={".*": 0.0},
  joint_vel={".*": 0.0},
)

##
# Articulation.
##

MYOSKELETON_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    MYOSKELETON_ACTUATOR_SPINE_FLEX_BEND,
    MYOSKELETON_ACTUATOR_SPINE_ROTATION,
    MYOSKELETON_ACTUATOR_ARM,
    MYOSKELETON_ACTUATOR_WRIST,
    MYOSKELETON_ACTUATOR_HIP_FLEX,
    MYOSKELETON_ACTUATOR_HIP_ADD,
    MYOSKELETON_ACTUATOR_HIP_ROT,
    MYOSKELETON_ACTUATOR_KNEE,
    MYOSKELETON_ACTUATOR_ANKLE,
    MYOSKELETON_ACTUATOR_FOOT,
  ),
  soft_joint_pos_limit_factor=0.9,
)


##
# Robot config factory.
##


def get_myoskeleton_robot_cfg() -> EntityCfg:
  """Get a fresh MyoSkeleton robot configuration instance."""
  return EntityCfg(
    init_state=STANDING_KEYFRAME,
    collisions=(MYOSKELETON_COLLISION,),
    spec_fn=get_spec,
    articulation=MYOSKELETON_ARTICULATION,
  )


##
# Action scale (same formula as G1: 0.25 * effort_limit / stiffness).
##

MYOSKELETON_ACTION_SCALE: dict[str, float] = {}
for _a in MYOSKELETON_ARTICULATION.actuators:
  assert isinstance(_a, BuiltinPositionActuatorCfg)
  _e = _a.effort_limit
  _s = _a.stiffness
  assert _e is not None
  for _n in _a.target_names_expr:
    MYOSKELETON_ACTION_SCALE[_n] = 0.25 * _e / _s
