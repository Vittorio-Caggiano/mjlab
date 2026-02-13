from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import myoskeleton_flat_tracking_env_cfg
from .rl_cfg import myoskeleton_tracking_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Tracking-Flat-MyoSkeleton",
    env_cfg=myoskeleton_flat_tracking_env_cfg(),
    play_env_cfg=myoskeleton_flat_tracking_env_cfg(play=True),
    rl_cfg=myoskeleton_tracking_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)
