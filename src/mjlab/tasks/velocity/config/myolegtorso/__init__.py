from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  myolegstorso_flat_env_cfg,
  myolegstorso_flat_env_cfg_standing_curriculum,
  myolegstorso_flat_env_cfg_synergy,
  myolegstorso_flat_env_cfg_synergy_transfer,
  myolegstorso_flat_env_cfg_trunk_scale,
)
from .rl_cfg import myolegstorso_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-MyoLegsTorso",
  env_cfg=myolegstorso_flat_env_cfg(),
  play_env_cfg=myolegstorso_flat_env_cfg(play=True),
  rl_cfg=myolegstorso_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-MyoLegsTorso-StandingCurriculum",
  env_cfg=myolegstorso_flat_env_cfg_standing_curriculum(),
  play_env_cfg=myolegstorso_flat_env_cfg_standing_curriculum(play=True),
  rl_cfg=myolegstorso_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-MyoLegsTorso-TrunkScale",
  env_cfg=myolegstorso_flat_env_cfg_trunk_scale(),
  play_env_cfg=myolegstorso_flat_env_cfg_trunk_scale(play=True),
  rl_cfg=myolegstorso_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-MyoLegsTorso-Synergy",
  env_cfg=myolegstorso_flat_env_cfg_synergy(),
  play_env_cfg=myolegstorso_flat_env_cfg_synergy(play=True),
  rl_cfg=myolegstorso_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-MyoLegsTorso-SynergyTransfer",
  env_cfg=myolegstorso_flat_env_cfg_synergy_transfer(),
  play_env_cfg=myolegstorso_flat_env_cfg_synergy_transfer(play=True),
  rl_cfg=myolegstorso_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
