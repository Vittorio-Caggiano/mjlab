"""RL configuration for MyoLegsTorso standing-balance task."""

from __future__ import annotations

from dataclasses import replace

from mjlab.tasks.velocity.config.myoskeleton.rl_cfg import myoskeleton_ppo_runner_cfg


def myolegstorso_balance_ppo_runner_cfg():
  """MyoLegsTorso balance PPO config: reuse MyoSkeleton PPO with a new experiment name."""
  return replace(
    myoskeleton_ppo_runner_cfg(),
    experiment_name="myolegtorso_balance",
  )
