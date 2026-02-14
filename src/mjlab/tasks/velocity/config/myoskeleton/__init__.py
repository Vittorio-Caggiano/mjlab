from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  myoskeleton_flat_env_cfg,
)
from .rl_cfg import myoskeleton_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-MyoSkeleton",
  env_cfg=myoskeleton_flat_env_cfg(),
  play_env_cfg=myoskeleton_flat_env_cfg(play=True),
  rl_cfg=myoskeleton_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
