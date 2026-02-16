from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from mjlab.tasks.velocity.config.myoskeleton.rl_cfg import myoskeleton_ppo_runner_cfg

from .env_cfgs import myoskeleton_unitree_flat_env_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-MyoSkeleton-Unitree",
  env_cfg=myoskeleton_unitree_flat_env_cfg(),
  play_env_cfg=myoskeleton_unitree_flat_env_cfg(play=True),
  rl_cfg=myoskeleton_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
