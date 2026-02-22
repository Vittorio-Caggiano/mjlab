"""Validate myosuite_mjlab MyoLeg config parity with the original mjlab reference."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REF_ENV = REPO_ROOT / "src/mjlab/tasks/velocity/config/myoleg/env_cfgs.py"
NEW_ENV = (
  REPO_ROOT / "myosuite_mjlab/src/myosuite_mjlab/tasks/velocity/myoleg/env_cfgs.py"
)
REF_MYOSKELETON_RL = REPO_ROOT / "src/mjlab/tasks/velocity/config/myoskeleton/rl_cfg.py"
NEW_RL = REPO_ROOT / "myosuite_mjlab/src/myosuite_mjlab/tasks/velocity/myoleg/rl_cfg.py"


def _extract_reward_weights(src: str) -> dict[str, str]:
  return dict(re.findall(r'cfg\.rewards\["([^"]+)"\]\.weight = ([^\n]+)', src))


def _extract_reward_std(src: str) -> dict[str, str]:
  return dict(
    re.findall(r'cfg\.rewards\["([^"]+)"\]\.params\["std"\] = ([^\n]+)', src)
  )


def _extract_cmd_ranges(src: str, axis: str) -> list[str]:
  pattern = rf"twist_cmd\.ranges\.{axis} = \(([^\)]+)\)"
  return re.findall(pattern, src)


def test_myoleg_env_cfg_matches_reference_reward_and_command_settings() -> None:
  ref_src = REF_ENV.read_text()
  new_src = NEW_ENV.read_text()

  ref_weights = _extract_reward_weights(ref_src)
  new_weights = _extract_reward_weights(new_src)
  for key in (
    "track_linear_velocity",
    "track_angular_velocity",
    "pose",
    "upright",
    "body_ang_vel",
    "action_rate_l2",
    "angular_momentum",
    "soft_landing",
  ):
    assert new_weights[key] == ref_weights[key], key

  ref_std = _extract_reward_std(ref_src)
  new_std = _extract_reward_std(new_src)
  for key in ("track_linear_velocity", "track_angular_velocity"):
    assert new_std[key] == ref_std[key], key

  assert _extract_cmd_ranges(new_src, "lin_vel_x") == _extract_cmd_ranges(
    ref_src, "lin_vel_x"
  )
  assert _extract_cmd_ranges(new_src, "lin_vel_y") == _extract_cmd_ranges(
    ref_src, "lin_vel_y"
  )
  assert _extract_cmd_ranges(new_src, "ang_vel_z") == _extract_cmd_ranges(
    ref_src, "ang_vel_z"
  )


def test_myoleg_rl_cfg_matches_reference_hyperparameters() -> None:
  ref_src = REF_MYOSKELETON_RL.read_text()
  new_src = NEW_RL.read_text()

  for expected in (
    'hidden_dims=(512, 256, 128)',
    'activation="elu"',
    'obs_normalization=True',
    'noise_std_type="log"',
    'entropy_coef=0.01',
    'num_learning_epochs=5',
    'num_mini_batches=4',
    'learning_rate=1.0e-3',
    'schedule="adaptive"',
    'gamma=0.99',
    'lam=0.95',
    'desired_kl=0.01',
    'max_grad_norm=1.0',
  ):
    assert expected in new_src
    assert expected in ref_src

  assert 'experiment_name="myoleg_velocity"' in new_src
