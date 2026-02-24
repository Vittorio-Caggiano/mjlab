from mjlab.tasks.registry import register_mjlab_task

from mjlab.tasks.balance.config.myolegtorso.env_cfgs import myolegstorso_balance_env_cfg
from mjlab.tasks.balance.config.myolegtorso.rl_cfg import (
  myolegstorso_balance_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Balance-Flat-MyoLegsTorso",
  env_cfg=myolegstorso_balance_env_cfg(),
  play_env_cfg=myolegstorso_balance_env_cfg(play=True),
  rl_cfg=myolegstorso_balance_ppo_runner_cfg(),
)
