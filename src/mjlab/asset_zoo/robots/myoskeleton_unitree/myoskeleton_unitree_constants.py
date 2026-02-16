"""Unitree-style actuator and contact settings applied to MyoSkeleton.

This module keeps the MyoSkeleton morphology (joints/bodies/collision mesh)
while replacing the actuator parameterization and contact profile on a
lower-body control subset with Unitree G1-like settings.
"""

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.myoskeleton.myoskeleton_constants import (
  STANDING_KEYFRAME,
  get_spec as get_myoskeleton_spec,
)
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  ACTUATOR_5020,
  ACTUATOR_7520_14,
  ACTUATOR_7520_22,
  DAMPING_5020,
  DAMPING_7520_14,
  DAMPING_7520_22,
  STIFFNESS_5020,
  STIFFNESS_7520_14,
  STIFFNESS_7520_22,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

# Unitree-to-MyoSkeleton lower-body joint mapping used by controllers.
MYOSKELETON_UNITREE_JOINT_MAP: dict[str, str] = {
  "left_hip_pitch_joint": "hip_flexion_l",
  "left_hip_roll_joint": "hip_adduction_l",
  "left_hip_yaw_joint": "hip_rotation_l",
  "left_knee_joint": "knee_angle_l",
  "left_ankle_pitch_joint": "ankle_angle_l",
  "left_ankle_roll_joint": "subtalar_angle_l",
  "right_hip_pitch_joint": "hip_flexion_r",
  "right_hip_roll_joint": "hip_adduction_r",
  "right_hip_yaw_joint": "hip_rotation_r",
  "right_knee_joint": "knee_angle_r",
  "right_ankle_pitch_joint": "ankle_angle_r",
  "right_ankle_roll_joint": "subtalar_angle_r",
}

# Functional feet on the MyoSkeleton model (collision geoms only).
MYOSKELETON_UNITREE_FOOT_GEOM_REGEX = r"^(bofoot[12]?_[lr]_coll|foot[123]?_[lr]_coll)$"


# Keep only free root + Unitree-controlled joints in the hybrid model.
MYOSKELETON_UNITREE_CONTROLLED_JOINTS: tuple[str, ...] = (
  "hip_flexion_l",
  "hip_adduction_l",
  "hip_rotation_l",
  "knee_angle_l",
  "ankle_angle_l",
  "subtalar_angle_l",
  "hip_flexion_r",
  "hip_adduction_r",
  "hip_rotation_r",
  "knee_angle_r",
  "ankle_angle_r",
  "subtalar_angle_r",
)


def get_spec():
  """Load MyoSkeleton and remove all non-actuated joints for this hybrid."""
  spec = get_myoskeleton_spec()

  keep = {"myoskeleton_root", *MYOSKELETON_UNITREE_CONTROLLED_JOINTS}
  for joint in list(spec.joints):
    if joint.name not in keep:
      spec.delete(joint)

  return spec

MYOSKELETON_UNITREE_HIP_PITCH_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=("hip_flexion_l", "hip_flexion_r", "hip_rotation_l", "hip_rotation_r"),
  stiffness=STIFFNESS_7520_14,
  damping=DAMPING_7520_14,
  effort_limit=ACTUATOR_7520_14.effort_limit,
  armature=ACTUATOR_7520_14.reflected_inertia,
)

MYOSKELETON_UNITREE_HIP_ROLL_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=("hip_adduction_l", "hip_adduction_r", "knee_angle_l", "knee_angle_r"),
  stiffness=STIFFNESS_7520_22,
  damping=DAMPING_7520_22,
  effort_limit=ACTUATOR_7520_22.effort_limit,
  armature=ACTUATOR_7520_22.reflected_inertia,
)

# Like G1 ankle linkages, we treat ankle pitch/roll as nominally equivalent
# to two coupled 5020 actuators.
MYOSKELETON_UNITREE_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "ankle_angle_l",
    "ankle_angle_r",
    "subtalar_angle_l",
    "subtalar_angle_r",
  ),
  stiffness=STIFFNESS_5020 * 2,
  damping=DAMPING_5020 * 2,
  effort_limit=ACTUATOR_5020.effort_limit * 2,
  armature=ACTUATOR_5020.reflected_inertia * 2,
)

# Foot-centric collision profile: keep all collision geoms active while
# assigning richer contact settings to feet and lower-dimensional contact
# to non-foot geoms.
MYOSKELETON_UNITREE_COLLISION = CollisionCfg(
  geom_names_expr=(".*_coll",),
  condim={MYOSKELETON_UNITREE_FOOT_GEOM_REGEX: 3, ".*_coll": 1},
  priority={MYOSKELETON_UNITREE_FOOT_GEOM_REGEX: 1},
  friction={MYOSKELETON_UNITREE_FOOT_GEOM_REGEX: (0.6,)},
  solimp={MYOSKELETON_UNITREE_FOOT_GEOM_REGEX: (0.9, 0.95, 0.023)},
)

MYOSKELETON_UNITREE_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    MYOSKELETON_UNITREE_HIP_PITCH_YAW,
    MYOSKELETON_UNITREE_HIP_ROLL_KNEE,
    MYOSKELETON_UNITREE_ANKLE,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_myoskeleton_unitree_robot_cfg() -> EntityCfg:
  """Get a MyoSkeleton config with Unitree-like lower-body actuation."""
  return EntityCfg(
    init_state=STANDING_KEYFRAME,
    collisions=(MYOSKELETON_UNITREE_COLLISION,),
    spec_fn=get_spec,
    articulation=MYOSKELETON_UNITREE_ARTICULATION,
  )


MYOSKELETON_UNITREE_ACTION_SCALE: dict[str, float] = {}
for actuator_cfg in MYOSKELETON_UNITREE_ARTICULATION.actuators:
  assert isinstance(actuator_cfg, BuiltinPositionActuatorCfg)
  effort_limit = actuator_cfg.effort_limit
  stiffness = actuator_cfg.stiffness
  assert effort_limit is not None
  for target in actuator_cfg.target_names_expr:
    MYOSKELETON_UNITREE_ACTION_SCALE[target] = 0.25 * effort_limit / stiffness
