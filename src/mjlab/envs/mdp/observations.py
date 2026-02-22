"""Useful methods for MDP observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, RayCastSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


##
# Root state.
##


def base_lin_vel(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_lin_vel_b


def base_ang_vel(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_ang_vel_b


def projected_gravity(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.projected_gravity_b


##
# Joint state.
##


def joint_pos_rel(
  env: ManagerBasedRlEnv,
  biased: bool = False,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  jnt_ids = asset_cfg.joint_ids
  joint_pos = asset.data.joint_pos_biased if biased else asset.data.joint_pos
  return joint_pos[:, jnt_ids] - default_joint_pos[:, jnt_ids]


def joint_vel_rel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_vel = asset.data.default_joint_vel
  assert default_joint_vel is not None
  jnt_ids = asset_cfg.joint_ids
  return asset.data.joint_vel[:, jnt_ids] - default_joint_vel[:, jnt_ids]


##
# Muscle / tendon state (for muscle-controlled entities).
##


def tendon_length(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Tendon lengths (muscle length proxy). Shape (num_envs, num_tendons)."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.tendon_len


def tendon_velocity(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Tendon velocities (muscle velocity proxy). Shape (num_envs, num_tendons)."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.tendon_vel


def actuator_force(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Actuator forces (muscle force). Shape (num_envs, num_actuators)."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.actuator_force


##
# Center of mass.
##


def com_pos_w(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Full-body center-of-mass position in world frame. Shape (num_envs, 3)."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.data.subtree_com[:, asset.data.indexing.root_body_id].clone()


def com_lin_vel_w(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Full-body COM linear velocity in world frame. Shape (num_envs, 3)."""
  asset: Entity = env.scene[asset_cfg.name]
  body_ids = asset.data.indexing.body_ids
  mass_arr = asset.data.model.body_mass
  if hasattr(mass_arr, "__getitem__") and mass_arr.ndim >= 1:
    bid = body_ids.cpu().numpy() if isinstance(body_ids, torch.Tensor) else body_ids
    mass = mass_arr[0, bid] if mass_arr.ndim == 2 else mass_arr[bid]
  else:
    mass = mass_arr
  if not isinstance(mass, torch.Tensor):
    mass = torch.as_tensor(
      mass, device=asset.data.body_com_lin_vel_w.device, dtype=torch.float32
    )
  vel = asset.data.body_com_lin_vel_w  # (num_envs, num_bodies, 3)
  mass_2d = mass.unsqueeze(0).unsqueeze(-1)  # (1, num_bodies, 1)
  total_mass = mass.sum()
  com_vel = (vel * mass_2d).sum(dim=1) / torch.clamp(total_mass, min=1e-6)
  return com_vel


##
# Actions.
##


def last_action(env: ManagerBasedRlEnv, action_name: str | None = None) -> torch.Tensor:
  if action_name is None:
    return env.action_manager.action
  return env.action_manager.get_term(action_name).raw_action


##
# Commands.
##


def generated_commands(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return command


##
# Sensors.
##


def builtin_sensor(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Get observation from a built-in sensor by name."""
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, BuiltinSensor)
  return sensor.data


def height_scan(
  env: ManagerBasedRlEnv, sensor_name: str, offset: float = 0.0
) -> torch.Tensor:
  """Height scan from a raycast sensor.

  Returns the height of the sensor frame above each hit point.

  Args:
    env: The environment.
    sensor_name: Name of a RayCastSensor in the scene.
    offset: Constant offset subtracted from heights.

  Returns:
    Tensor of shape [B, N] where B is num_envs and N is num_rays.
  """
  sensor: RayCastSensor = env.scene[sensor_name]
  return sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.hit_pos_w[..., 2] - offset
