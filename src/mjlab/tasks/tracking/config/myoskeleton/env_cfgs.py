"""MyoSkeleton flat tracking environment configurations."""

from mjlab.asset_zoo.robots import (
  MYOSKELETON_ACTION_SCALE,
  get_myoskeleton_robot_cfg,
)
from mjlab.asset_zoo.robots.myoskeleton.myoskeleton_constants import (
  MYOSKELETON_MOTION_DIR,
  MYOSKELETON_TRACKING_BODIES,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def myoskeleton_flat_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create MyoSkeleton flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_myoskeleton_robot_cfg()}

  # Self-collision sensor using pelvis subtree.
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  # Action scale per actuator group.
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = MYOSKELETON_ACTION_SCALE

  # Motion command configuration.
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "pelvis"
  motion_cmd.body_names = MYOSKELETON_TRACKING_BODIES
  motion_cmd.motion_file = str(MYOSKELETON_MOTION_DIR / "standing_motion.npz")

  # Domain randomization: foot friction geoms and base COM body.
  # The myoskeleton uses capsule collision geoms (class myo_coll).
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = ".*"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("pelvis",)

  # Termination: end-effector body position tracking.
  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "calcn_l",
    "calcn_r",
    "lunate_l",
    "lunate_r",
  )

  # Viewer follows the pelvis.
  cfg.viewer.body_name = "pelvis"

  # Increase contact limits for the complex model.
  cfg.sim.nconmax = 100
  cfg.sim.njmax = 500

  # Apply play mode overrides.
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"

  return cfg
