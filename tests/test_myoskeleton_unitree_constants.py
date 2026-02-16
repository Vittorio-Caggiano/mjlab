"""Tests for MyoSkeleton Unitree-style constants."""

import re

import mujoco
import numpy as np
import pytest

from mjlab.asset_zoo.robots.myoskeleton_unitree import myoskeleton_unitree_constants as c
from mjlab.asset_zoo.robots.unitree_g1 import g1_constants
from mjlab.entity import Entity


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
  return Entity(c.get_myoskeleton_unitree_robot_cfg()).compile()


def test_hybrid_entity_creation(model: mujoco.MjModel) -> None:
  assert model.nu == 12


def test_all_actuators_use_position_gains(model: mujoco.MjModel) -> None:
  expected_by_joint = {
    "hip_flexion_l": c.MYOSKELETON_UNITREE_HIP_PITCH_YAW,
    "hip_flexion_r": c.MYOSKELETON_UNITREE_HIP_PITCH_YAW,
    "hip_rotation_l": c.MYOSKELETON_UNITREE_HIP_PITCH_YAW,
    "hip_rotation_r": c.MYOSKELETON_UNITREE_HIP_PITCH_YAW,
    "hip_adduction_l": c.MYOSKELETON_UNITREE_HIP_ROLL_KNEE,
    "hip_adduction_r": c.MYOSKELETON_UNITREE_HIP_ROLL_KNEE,
    "knee_angle_l": c.MYOSKELETON_UNITREE_HIP_ROLL_KNEE,
    "knee_angle_r": c.MYOSKELETON_UNITREE_HIP_ROLL_KNEE,
    "ankle_angle_l": c.MYOSKELETON_UNITREE_ANKLE,
    "ankle_angle_r": c.MYOSKELETON_UNITREE_ANKLE,
    "subtalar_angle_l": c.MYOSKELETON_UNITREE_ANKLE,
    "subtalar_angle_r": c.MYOSKELETON_UNITREE_ANKLE,
  }

  for i in range(model.nu):
    actuator = model.actuator(i)
    joint_name = actuator.name
    cfg = expected_by_joint[joint_name]
    assert actuator.gainprm[0] == cfg.stiffness
    assert actuator.biasprm[1] == -cfg.stiffness
    assert actuator.biasprm[2] == -cfg.damping
    assert actuator.forcerange[0] == -cfg.effort_limit
    assert actuator.forcerange[1] == cfg.effort_limit
    assert model.actuator_ctrllimited[i] == 0
    assert model.actuator_forcelimited[i] == 1


def test_contact_profile_is_unitree_like(model: mujoco.MjModel) -> None:
  foot_regex = re.compile(c.MYOSKELETON_UNITREE_FOOT_GEOM_REGEX)

  foot_count = 0
  for i in range(model.ngeom):
    geom = model.geom(i)
    if "_coll" not in geom.name:
      continue

    if foot_regex.match(geom.name):
      foot_count += 1
      assert geom.condim == 3
      assert geom.priority == 1
      assert geom.friction[0] == 0.6
    else:
      assert geom.condim == 1

  assert foot_count > 0


def test_mapped_joints_have_g1_natural_frequency_and_damping(model: mujoco.MjModel) -> None:
  expected_params = {
    "hip_flexion_l": (g1_constants.STIFFNESS_7520_14, g1_constants.DAMPING_7520_14),
    "hip_flexion_r": (g1_constants.STIFFNESS_7520_14, g1_constants.DAMPING_7520_14),
    "hip_rotation_l": (g1_constants.STIFFNESS_7520_14, g1_constants.DAMPING_7520_14),
    "hip_rotation_r": (g1_constants.STIFFNESS_7520_14, g1_constants.DAMPING_7520_14),
    "hip_adduction_l": (g1_constants.STIFFNESS_7520_22, g1_constants.DAMPING_7520_22),
    "hip_adduction_r": (g1_constants.STIFFNESS_7520_22, g1_constants.DAMPING_7520_22),
    "knee_angle_l": (g1_constants.STIFFNESS_7520_22, g1_constants.DAMPING_7520_22),
    "knee_angle_r": (g1_constants.STIFFNESS_7520_22, g1_constants.DAMPING_7520_22),
    "ankle_angle_l": (g1_constants.STIFFNESS_5020 * 2, g1_constants.DAMPING_5020 * 2),
    "ankle_angle_r": (g1_constants.STIFFNESS_5020 * 2, g1_constants.DAMPING_5020 * 2),
    "subtalar_angle_l": (g1_constants.STIFFNESS_5020 * 2, g1_constants.DAMPING_5020 * 2),
    "subtalar_angle_r": (g1_constants.STIFFNESS_5020 * 2, g1_constants.DAMPING_5020 * 2),
  }

  # Proxy for matched actuator dynamics: same kp/kd and implied wn/zeta.
  for i in range(model.nu):
    actuator = model.actuator(i)
    joint_name = actuator.name

    stiffness_expected, damping_expected = expected_params[joint_name]
    np.testing.assert_allclose(actuator.gainprm[0], stiffness_expected)
    np.testing.assert_allclose(-actuator.biasprm[2], damping_expected)

    joint = model.joint(joint_name)
    dof_adr = joint.dofadr[0]
    armature = model.dof_armature[dof_adr]

    omega_n = np.sqrt(actuator.gainprm[0] / armature)
    zeta = (-actuator.biasprm[2]) / (2.0 * np.sqrt(actuator.gainprm[0] * armature))

    np.testing.assert_allclose(omega_n, g1_constants.NATURAL_FREQ, rtol=1e-6)
    np.testing.assert_allclose(zeta, g1_constants.DAMPING_RATIO, rtol=1e-6)


def test_non_actuated_joints_are_removed(model: mujoco.MjModel) -> None:
  joint_names = {model.joint(i).name for i in range(model.njnt)}

  # Keep free root + explicitly controlled joints only.
  expected = {"myoskeleton_root", *c.MYOSKELETON_UNITREE_CONTROLLED_JOINTS}
  assert joint_names == expected

  # 1 free joint + 12 hinge joints.
  assert model.njnt == 13
