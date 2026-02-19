"""Tests specific to motion tracking tasks."""

from pathlib import Path

import numpy as np
import pytest

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, MYOSKELETON_ACTION_SCALE
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg


@pytest.fixture(scope="module")
def tracking_task_ids() -> list[str]:
  """Get all tracking task IDs."""
  return [t for t in list_tasks() if "Tracking" in t]


@pytest.fixture(scope="module")
def g1_tracking_task_ids(tracking_task_ids: list[str]) -> list[str]:
  """Get all G1 tracking task IDs."""
  return [t for t in tracking_task_ids if "G1" in t]


def test_tracking_tasks_have_motion_command(tracking_task_ids: list[str]) -> None:
  """All tracking tasks should have a 'motion' command of type MotionCommandCfg."""
  for task_id in tracking_task_ids:
    cfg = load_env_cfg(task_id)

    assert "motion" in cfg.commands, f"Task {task_id} missing 'motion' command"

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg), (
      f"Task {task_id} motion command is not MotionCommandCfg"
    )


def test_tracking_tasks_have_self_collision_sensor(
  tracking_task_ids: list[str],
) -> None:
  """All tracking tasks should have a self_collision sensor."""
  for task_id in tracking_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.sensors is not None, f"Task {task_id} has no sensors"

    sensor_names = {s.name for s in cfg.scene.sensors}
    assert "self_collision" in sensor_names, (
      f"Task {task_id} missing self_collision sensor"
    )


def test_tracking_no_state_estimation_observations() -> None:
  """No-state-estimation tasks remove observations that depend on state estimation."""
  task_id = "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation"

  # Test both training and play modes
  for play_mode in [False, True]:
    cfg = load_env_cfg(task_id, play=play_mode)
    mode_str = "play mode" if play_mode else "training mode"

    assert "actor" in cfg.observations, (
      f"Task {task_id} ({mode_str}) missing policy observations"
    )
    actor_terms = cfg.observations["actor"].terms

    assert "motion_anchor_pos_b" not in actor_terms, (
      f"Task {task_id} ({mode_str}) has motion_anchor_pos_b in policy, "
      "expected it to be removed for no-state-estimation variant"
    )
    assert "base_lin_vel" not in actor_terms, (
      f"Task {task_id} ({mode_str}) has base_lin_vel in policy, "
      "expected it to be removed for no-state-estimation variant"
    )


def test_tracking_play_disables_rsi_randomization() -> None:
  """Tracking play tasks should disable RSI randomization."""
  tracking_tasks = [
    "Mjlab-Tracking-Flat-Unitree-G1",
    "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation",
    "Mjlab-Tracking-Flat-MyoSkeleton",
  ]

  for task_id in tracking_tasks:
    cfg = load_env_cfg(task_id, play=True)

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg), (
      f"Task {task_id} (play mode) motion command is not MotionCommandCfg"
    )

    assert motion_cmd.pose_range == {}, (
      f"Task {task_id} (play mode) has non-empty pose_range={motion_cmd.pose_range}, "
      "expected empty dict for disabled RSI"
    )
    assert motion_cmd.velocity_range == {}, (
      f"Task {task_id} (play mode) has non-empty velocity_range={motion_cmd.velocity_range}, "
      "expected empty dict for disabled RSI"
    )


def test_tracking_play_uses_start_sampling_mode() -> None:
  """Tracking play tasks should use sampling_mode='start'."""
  tracking_tasks = [
    "Mjlab-Tracking-Flat-Unitree-G1",
    "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation",
    "Mjlab-Tracking-Flat-MyoSkeleton",
  ]

  for task_id in tracking_tasks:
    cfg = load_env_cfg(task_id, play=True)

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg), (
      f"Task {task_id} (play mode) motion command is not MotionCommandCfg"
    )

    assert motion_cmd.sampling_mode == "start", (
      f"Task {task_id} (play mode) sampling_mode={motion_cmd.sampling_mode}, expected 'start'"
    )


def test_g1_tracking_has_correct_action_scale(g1_tracking_task_ids: list[str]) -> None:
  """G1 tracking tasks should use G1_ACTION_SCALE."""
  for task_id in g1_tracking_task_ids:
    cfg = load_env_cfg(task_id)

    assert "joint_pos" in cfg.actions, f"Task {task_id} missing 'joint_pos' action"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg), (
      f"Task {task_id} joint_pos action is not JointPositionActionCfg"
    )

    assert joint_pos_action.scale == G1_ACTION_SCALE, (
      f"Task {task_id} action scale mismatch, expected G1_ACTION_SCALE"
    )


# ── MyoSkeleton-specific tests ──────────────────────────────────────────────


@pytest.fixture(scope="module")
def myoskeleton_tracking_task_ids(tracking_task_ids: list[str]) -> list[str]:
  """Get all MyoSkeleton tracking task IDs."""
  return [t for t in tracking_task_ids if "MyoSkeleton" in t]


def test_myoskeleton_tracking_has_correct_action_scale(
  myoskeleton_tracking_task_ids: list[str],
) -> None:
  """MyoSkeleton tracking tasks should use MYOSKELETON_ACTION_SCALE."""
  for task_id in myoskeleton_tracking_task_ids:
    cfg = load_env_cfg(task_id)

    assert "joint_pos" in cfg.actions, f"Task {task_id} missing 'joint_pos' action"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg), (
      f"Task {task_id} joint_pos action is not JointPositionActionCfg"
    )

    assert joint_pos_action.scale == MYOSKELETON_ACTION_SCALE, (
      f"Task {task_id} action scale mismatch, expected MYOSKELETON_ACTION_SCALE"
    )


def test_myoskeleton_tracking_has_correct_motion_file(
  myoskeleton_tracking_task_ids: list[str],
) -> None:
  """MyoSkeleton tracking tasks should reference the standing_motion.npz motion file."""
  for task_id in myoskeleton_tracking_task_ids:
    cfg = load_env_cfg(task_id)

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    assert "standing_motion.npz" in motion_cmd.motion_file, (
      f"Task {task_id} motion_file={motion_cmd.motion_file}, expected standing_motion.npz"
    )


def test_myoskeleton_tracking_has_correct_anchor_body(
  myoskeleton_tracking_task_ids: list[str],
) -> None:
  """MyoSkeleton tracking tasks should use pelvis as anchor body."""
  for task_id in myoskeleton_tracking_task_ids:
    cfg = load_env_cfg(task_id)

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    assert motion_cmd.anchor_body_name == "pelvis", (
      f"Task {task_id} anchor_body_name={motion_cmd.anchor_body_name}, expected 'pelvis'"
    )


def test_myoskeleton_tracking_has_15_tracking_bodies(
  myoskeleton_tracking_task_ids: list[str],
) -> None:
  """MyoSkeleton tracking tasks should track 15 bodies."""
  for task_id in myoskeleton_tracking_task_ids:
    cfg = load_env_cfg(task_id)

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    assert len(motion_cmd.body_names) == 15, (
      f"Task {task_id} has {len(motion_cmd.body_names)} tracking bodies, expected 15"
    )


def test_myoskeleton_task_is_registered() -> None:
  """The MyoSkeleton tracking task should be in the registry."""
  tasks = list_tasks()
  assert "Mjlab-Tracking-Flat-MyoSkeleton" in tasks


def test_myoskeleton_tracking_scene_builds() -> None:
  """The MyoSkeleton scene builds with all required sensors and actuators."""
  from mjlab.scene import Scene

  cfg = load_env_cfg("Mjlab-Tracking-Flat-MyoSkeleton")
  cfg.scene.num_envs = 1

  scene = Scene(cfg.scene, device="cpu")

  # The base tracking config requires these IMU sensors.
  # Scene.__getitem__ raises KeyError if not found.
  try:
    scene["robot/imu_lin_vel"]
  except KeyError:
    pytest.fail("Scene missing 'robot/imu_lin_vel' sensor")
  try:
    scene["robot/imu_ang_vel"]
  except KeyError:
    pytest.fail("Scene missing 'robot/imu_ang_vel' sensor")

  # Verify the model compiles and has actuators.
  model = scene.compile()
  assert model.nu > 0, "Compiled model has no actuators"


# ── Motion data consistency test ──────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[1]  # mjlab/
_SIBLING = _REPO_ROOT.parent  # parent dir containing both mjlab/ and myosuite/
_H5_DIR = _SIBLING / "myosuite/myosuite/envs/myo/mjx"
MODEL_XML = _SIBLING / "myosuite/myosuite/simhive/myo_model/myoskeleton/myoskeleton.xml"

# Default motion used by the tracking task.
H5_FILE = _H5_DIR / "standing_motion.h5"


@pytest.mark.skipif(
  not H5_FILE.exists() or not MODEL_XML.exists(),
  reason="myosuite H5 motion or model XML not available",
)
def test_myoskeleton_npz_matches_h5_playback() -> None:
  """Verify that standing_motion.npz faithfully reproduces the original H5 motion.

  Loads the H5, plays it through the MuJoCo model (same procedure as the
  conversion script), and checks that body positions and joint positions
  recorded in the NPZ match.
  """
  import h5py
  import mujoco

  # ── Load model ──────────────────────────────────────────────────────────
  model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
  data = mujoco.MjData(model)

  all_joint_names = [model.joint(i).name for i in range(model.njnt)]
  FINGER_JOINTS = {
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
  }
  actuated_joints = [
    n for n in all_joint_names if n != "myoskeleton_root" and n not in FINGER_JOINTS
  ]

  # ── Load H5 ─────────────────────────────────────────────────────────────
  with h5py.File(str(H5_FILE), "r") as f:
    motion_name = list(f.keys())[0]
    grp = f[motion_name]
    time_arr = np.array(grp["time"], dtype=np.float64)
    qpos_dict: dict[str, np.ndarray] = {}
    for key in grp["qpos"]:
      qpos_dict[key] = np.array(grp["qpos"][key], dtype=np.float64).squeeze()
  horizon = len(time_arr)

  # ── Load NPZ ────────────────────────────────────────────────────────────
  cfg = load_env_cfg("Mjlab-Tracking-Flat-MyoSkeleton")
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  npz = np.load(motion_cmd.motion_file)
  npz_joint_pos = npz["joint_pos"]
  npz_body_pos = npz["body_pos_w"]
  npz_body_quat = npz["body_quat_w"]

  assert npz_joint_pos.shape[0] == horizon, (
    f"NPZ timesteps ({npz_joint_pos.shape[0]}) != H5 timesteps ({horizon})"
  )

  # ── Play H5 through model and compare ──────────────────────────────────
  h5_joints = [k for k in qpos_dict if k != "myoskeleton_root"]
  valid_joints = [j for j in h5_joints if j in all_joint_names]

  for t in range(horizon):
    # Set root.
    if "myoskeleton_root" in qpos_dict:
      root_data = qpos_dict["myoskeleton_root"]
      data.qpos[:7] = root_data if root_data.ndim == 1 else root_data[t]

    # Set joints from H5.
    for jn in valid_joints:
      idx = all_joint_names.index(jn)
      qadr = model.jnt_qposadr[idx]
      val = qpos_dict[jn]
      data.qpos[qadr] = val[t] if val.ndim > 0 and val.shape[0] == horizon else val

    mujoco.mj_forward(model, data)

    # Compare body positions.
    np.testing.assert_allclose(
      data.xpos,
      npz_body_pos[t],
      atol=1e-4,
      err_msg=f"body_pos_w mismatch at t={t}",
    )

    # Compare body quaternions (allow sign flip).
    for b in range(model.nbody):
      q_sim = data.xquat[b]
      q_npz = npz_body_quat[t, b]
      dot = np.abs(np.dot(q_sim, q_npz))
      assert dot > 0.999, (
        f"body_quat_w mismatch at t={t}, body {model.body(b).name}: dot={dot:.6f}"
      )

    # Compare actuated joint positions.
    for j_idx, jn in enumerate(actuated_joints):
      if jn in all_joint_names:
        model_idx = all_joint_names.index(jn)
        qadr = model.jnt_qposadr[model_idx]
        np.testing.assert_allclose(
          data.qpos[qadr],
          npz_joint_pos[t, j_idx],
          atol=1e-5,
          err_msg=f"joint_pos mismatch at t={t}, joint {jn}",
        )


# ── Standing task tests ──────────────────────────────────────────────────────


def test_standing_task_is_registered() -> None:
  """The MyoSkeleton standing task should be in the registry."""
  tasks = list_tasks()
  assert "Mjlab-Standing-Flat-MyoSkeleton" in tasks


def test_standing_task_commands_are_zero() -> None:
  """Standing task should issue zero velocity commands."""
  from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

  cfg = load_env_cfg("Mjlab-Standing-Flat-MyoSkeleton")

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  assert twist_cmd.rel_standing_envs == 1.0, (
    f"Standing task rel_standing_envs={twist_cmd.rel_standing_envs}, expected 1.0"
  )
  assert twist_cmd.ranges.lin_vel_x == (0.0, 0.0)
  assert twist_cmd.ranges.lin_vel_y == (0.0, 0.0)
  assert twist_cmd.ranges.ang_vel_z == (0.0, 0.0)


def test_standing_task_disables_locomotion_rewards() -> None:
  """Standing task should disable foot clearance, swing height, air time, etc."""
  cfg = load_env_cfg("Mjlab-Standing-Flat-MyoSkeleton")

  for reward_name in [
    "foot_clearance",
    "foot_swing_height",
    "foot_slip",
    "air_time",
    "soft_landing",
  ]:
    assert cfg.rewards[reward_name].weight == 0.0, (
      f"Standing task reward '{reward_name}' weight={cfg.rewards[reward_name].weight}, expected 0.0"
    )


def test_standing_task_has_no_push_perturbation() -> None:
  """Standing task should not push the robot during training."""
  cfg = load_env_cfg("Mjlab-Standing-Flat-MyoSkeleton")
  assert "push_robot" not in cfg.events, (
    "Standing task should not have push_robot event"
  )


def test_standing_task_has_alive_and_upright_rewards() -> None:
  """Standing task must have alive and upright rewards with high weights."""
  cfg = load_env_cfg("Mjlab-Standing-Flat-MyoSkeleton")

  assert "alive" in cfg.rewards
  assert cfg.rewards["alive"].weight >= 10.0, (
    f"alive weight={cfg.rewards['alive'].weight}, expected >= 10.0"
  )

  assert "upright" in cfg.rewards
  assert cfg.rewards["upright"].weight >= 5.0, (
    f"upright weight={cfg.rewards['upright'].weight}, expected >= 5.0"
  )


def test_standing_task_scene_builds() -> None:
  """The MyoSkeleton standing scene builds with all required sensors."""
  from mjlab.scene import Scene

  cfg = load_env_cfg("Mjlab-Standing-Flat-MyoSkeleton")
  cfg.scene.num_envs = 1

  scene = Scene(cfg.scene, device="cpu")
  model = scene.compile()
  assert model.nu > 0, "Compiled model has no actuators"
