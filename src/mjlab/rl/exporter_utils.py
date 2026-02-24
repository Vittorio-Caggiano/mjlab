"""Shared utilities for ONNX policy export across RL tasks."""

import onnx
import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction


def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
  """Convert list to CSV string with specified decimal precision."""
  fmt = f"{{:.{decimals}f}}"
  return delimiter.join(
    fmt.format(x)
    if isinstance(x, (int, float))
    else str(x)  # numbers → format, strings → as-is
    for x in arr
  )


def _action_scale_list(action_term) -> list:
  """Extract action scale as a list from an action term (joint_pos or tendon_effort)."""
  scale = action_term._scale
  if isinstance(scale, torch.Tensor):
    return scale[0].cpu().tolist()
  return [scale] if isinstance(scale, (int, float)) else list(scale)


def get_base_metadata(
  env: ManagerBasedRlEnv, run_path: str
) -> dict[str, list | str | float]:
  """Get base metadata common to all RL policy exports.

  Supports both joint_pos (e.g. MyoSkeleton) and tendon_effort (e.g. MyoLeg) tasks.
  """
  robot: Entity = env.scene["robot"]

  if "joint_pos" in env.action_manager.cfg:
    joint_action = env.action_manager.get_term("joint_pos")
    assert isinstance(joint_action, JointPositionAction)
    joint_name_to_ctrl_id = {}
    for actuator in robot.spec.actuators:
      joint_name = actuator.target.split("/")[-1]
      joint_name_to_ctrl_id[joint_name] = actuator.id
    ctrl_ids_natural = [
      joint_name_to_ctrl_id[jname]
      for jname in robot.joint_names
      if jname in joint_name_to_ctrl_id
    ]
    joint_stiffness = env.sim.mj_model.actuator_gainprm[ctrl_ids_natural, 0]
    joint_damping = -env.sim.mj_model.actuator_biasprm[ctrl_ids_natural, 2]
    return {
      "run_path": run_path,
      "joint_names": list(robot.joint_names),
      "joint_stiffness": joint_stiffness.tolist(),
      "joint_damping": joint_damping.tolist(),
      "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
      "command_names": list(env.command_manager.active_terms),
      "observation_names": env.observation_manager.active_terms["actor"],
      "action_scale": _action_scale_list(joint_action),
    }
  # Tendon-effort or tendon-synergy (e.g. MyoLeg, MyoLegsTorso): no joint actuators.
  if "tendon_effort" in env.action_manager.cfg:
    action_term = env.action_manager.get_term("tendon_effort")
    action_scale = _action_scale_list(action_term)
  elif "tendon_synergy" in env.action_manager.cfg:
    action_term = env.action_manager.get_term("tendon_synergy")
    action_scale = [1.0] * action_term.action_dim
  else:
    raise KeyError(
      "Expected joint_pos, tendon_effort, or tendon_synergy action. "
      f"Got: {list(env.action_manager.cfg.keys())}"
    )
  default_joint = (
    robot.data.default_joint_pos[0].cpu().tolist()
    if robot.data.default_joint_pos is not None
    else []
  )
  return {
    "run_path": run_path,
    "joint_names": list(robot.joint_names),
    "joint_stiffness": [],
    "joint_damping": [],
    "default_joint_pos": default_joint,
    "command_names": list(env.command_manager.active_terms),
    "observation_names": env.observation_manager.active_terms["actor"],
    "action_scale": action_scale,
  }


def attach_metadata_to_onnx(
  onnx_path: str, metadata: dict[str, list | str | float]
) -> None:
  """Attach metadata to an ONNX model file.

  Args:
    onnx_path: Path to the ONNX model file.
    metadata: Dictionary of metadata key-value pairs to attach.
  """
  model = onnx.load(onnx_path)

  for k, v in metadata.items():
    entry = onnx.StringStringEntryProto()
    entry.key = k
    entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
    model.metadata_props.append(entry)

  onnx.save(model, onnx_path)
