"""Diagnostics to explain training gaps between G1 and MyoSkeleton-Unitree.

These tests do two things:
1) verify actuator low-level dynamics (kp/kd/armature/effort) are intentionally
   matched to Unitree templates for mapped joints;
2) surface structural differences (state dimension vs action dimension) that can
   explain why policy learning behavior differs even with matched actuator templates.
"""

import numpy as np

from mjlab.asset_zoo.robots import get_g1_robot_cfg, get_myoskeleton_unitree_robot_cfg
from mjlab.asset_zoo.robots.myoskeleton_unitree import myoskeleton_unitree_constants as c
from mjlab.entity import Entity


def _joint_params(model, joint_name: str) -> tuple[float, float, float, float]:
  """Return (kp, kd, armature, effort_limit) for a 1-DoF actuated joint."""
  actuator_id = model.actuator(joint_name).id
  joint = model.joint(joint_name)
  dof_id = joint.dofadr[0]

  kp = float(model.actuator_gainprm[actuator_id][0])
  kd = float(-model.actuator_biasprm[actuator_id][2])
  armature = float(model.dof_armature[dof_id])
  effort = float(model.actuator_forcerange[actuator_id][1])
  return kp, kd, armature, effort


def _simulate_step_response(
  kp: float,
  kd: float,
  armature: float,
  effort_limit: float,
  target: float = 0.2,
  dt: float = 0.001,
  horizon_s: float = 1.0,
) -> np.ndarray:
  """Simple saturated 2nd-order response for PD position actuator."""
  q = 0.0
  qd = 0.0
  traj = []

  for _ in range(int(horizon_s / dt)):
    tau = kp * (target - q) - kd * qd
    tau = float(np.clip(tau, -effort_limit, effort_limit))
    qdd = tau / armature
    qd += qdd * dt
    q += qd * dt
    traj.append(q)

  return np.asarray(traj, dtype=np.float64)


def test_mapped_joint_step_responses_match_unitree_templates() -> None:
  """Mapped Myo joints should reproduce Unitree-like PD+armature responses."""
  g1 = Entity(get_g1_robot_cfg()).compile()
  myo = Entity(get_myoskeleton_unitree_robot_cfg()).compile()

  # Representative mappings for each actuator family used by the hybrid config.
  pairings = (
    ("left_hip_pitch_joint", "hip_flexion_l"),      # 7520_14 template
    ("left_hip_roll_joint", "hip_adduction_l"),     # 7520_22 template
    ("left_ankle_pitch_joint", "ankle_angle_l"),    # paired 5020 template
  )

  for g1_joint, myo_joint in pairings:
    g1_params = _joint_params(g1, g1_joint)
    myo_params = _joint_params(myo, myo_joint)

    # First verify parameter transfer itself.
    np.testing.assert_allclose(g1_params, myo_params, rtol=1e-9, atol=1e-12)

    g1_traj = _simulate_step_response(*g1_params)
    myo_traj = _simulate_step_response(*myo_params)
    np.testing.assert_allclose(g1_traj, myo_traj, rtol=1e-9, atol=1e-12)


def test_joint_pruning_keeps_all_mapped_unitree_joints() -> None:
  """Hybrid should expose only mapped Unitree joints (plus free root)."""
  myo_model = Entity(get_myoskeleton_unitree_robot_cfg()).compile()

  joint_names = {myo_model.joint(i).name for i in range(myo_model.njnt)}
  assert joint_names == {"myoskeleton_root", *c.MYOSKELETON_UNITREE_CONTROLLED_JOINTS}
  assert myo_model.nu == len(c.MYOSKELETON_UNITREE_JOINT_MAP)
