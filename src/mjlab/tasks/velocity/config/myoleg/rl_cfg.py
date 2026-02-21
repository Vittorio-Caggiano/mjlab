"""RL configuration for MyoLeg velocity task (reuses MyoSkeleton PPO config)."""

from dataclasses import replace

from mjlab.tasks.velocity.config.myoskeleton.rl_cfg import myoskeleton_ppo_runner_cfg


def myoleg_ppo_runner_cfg():
  """MyoLeg velocity PPO config: same as MyoSkeleton with experiment_name overridden."""
  return replace(myoskeleton_ppo_runner_cfg(), experiment_name="myoleg_velocity")
