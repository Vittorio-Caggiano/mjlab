"""Synergy-based tendon effort actions for high-DOF muscle models.

This module implements a reduced-dimensional action term that mirrors the
`ActionSpaceWrapper` from the StandingBalance project for the MyoLegsTorso
model. A low-dimensional action vector is expanded into the full 290-dimensional
muscle activation vector using a fixed linear mapping over tendons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


# Mapping from synergy action index -> list of tendon indices.
# This mirrors the ActionSpaceWrapper.action_mapping in:
# https://github.com/cherylwang20/StandingBalance/blob/main/MSR_extraction/train_syn.py
#
# - Indices 0–23: grouped trunk muscles (first 210 tendons).
# - Indices 24–103: direct mapping for the remaining 80 tendons (210–289).
_SYNERGY_TO_TENDON_INDICES: Dict[int, List[int]] = {
  # 0–1: psoas major (right / left).
  0: list(range(0, 11)),  # psoas major right
  1: list(range(11, 22)),  # psoas major left
  # 2–3: rectus abdominis (right / left).
  2: [22],  # RA right
  3: [23],  # RA left
  # 4–5: iliocostalis lumborum (pelvis to lumbar, right / left).
  4: [24, 25, 26, 27],  # ILpL right
  5: [28, 29, 30, 31],  # ILpL left
  # 6–7: iliocostalis thoracis (pelvis to thorax, right / left).
  6: [32, 33, 34, 35, 36, 37, 38, 39],  # ILpT right
  7: [40, 41, 42, 43, 44, 45, 46, 47],  # ILpT left
  # 8–9: longissimus thoracis pars thoracis (right / left).
  8: list(range(48, 69)),  # LTpT right
  9: list(range(69, 90)),  # LTpT left
  # 10–11: longissimus thoracis pars lumborum (right / left).
  10: [90, 91, 92, 93, 94],  # LTpL right
  11: [95, 96, 97, 98, 99],  # LTpL left
  # 12–13: quadratus lumborum posterior (right / left).
  12: [100, 101, 102, 103, 104, 105, 106],  # QL_post right
  13: [107, 108, 109, 110, 111, 112, 113],  # QL_post left
  # 14–15: quadratus lumborum mid (right / left).
  14: [114, 115, 116, 117, 118],  # QL_mid right
  15: [119, 120, 121, 122, 123],  # QL_mid left
  # 16–17: quadratus lumborum anterior (right / left).
  16: [124, 125, 126, 127, 128, 129],  # QL_ant right
  17: [130, 131, 132, 133, 134, 135],  # QL_ant left
  # 18–19: multifidus (right / left).
  18: list(range(136, 161)),  # MF right
  19: list(range(161, 186)),  # MF left
  # 20–23: external / internal obliques (right / left).
  20: [186, 187, 188, 189, 190, 191],  # EO right
  21: [192, 193, 194, 195, 196, 197],  # IO right
  22: [198, 199, 200, 201, 202, 203],  # EO left
  23: [204, 205, 206, 207, 208, 209],  # IO left
}

# Add the direct mappings for the remaining 80 tendons (indices 210–289).
for _i in range(24, 104):
  # Synergy index 24 maps to tendon 210, ..., index 103 maps to tendon 289.
  _SYNERGY_TO_TENDON_INDICES[_i] = [210 + (_i - 24)]


@dataclass(kw_only=True)
class SynergyTendonEffortActionCfg(ActionTermCfg):
  """Configuration for synergy-based tendon effort control.

  The action dimension is the number of synergy groups. Each synergy controls
  one or more tendons whose indices are defined in `_SYNERGY_TO_TENDON_INDICES`.
  """

  scale: float = 1.0
  """Scale applied to policy actions (e.g. for curriculum: start low, ramp to 1)."""

  def build(self, env: ManagerBasedRlEnv) -> "SynergyTendonEffortAction":
    return SynergyTendonEffortAction(self, env)


class SynergyTendonEffortAction(ActionTerm):
  """Apply synergy-based tendon effort targets.

  This term expects a low-dimensional action tensor of shape
  (num_envs, num_synergies) and internally expands it to the full tendon
  action vector before applying it to the MuJoCo model.
  """

  cfg: SynergyTendonEffortActionCfg

  def __init__(self, cfg: SynergyTendonEffortActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

    # Discover all tendon targets for this entity in model order.
    mj_model = env.sim.mj_model
    # All tendons that belong to this entity are assumed to be contiguous and
    # ordered as in the combined myoLegsTorso model (total 290 tendons).
    self._tendon_ids = torch.arange(
      mj_model.ntendon, device=env.device, dtype=torch.long
    )
    self._num_tendons = int(mj_model.ntendon)

    if self._num_tendons != 290:
      raise ValueError(
        f"SynergyTendonEffortAction expects exactly 290 tendons, got {self._num_tendons}."
      )

    # Build synergy -> tendon index mapping and validate coverage.
    self._synergy_to_indices: Dict[int, List[int]] = _SYNERGY_TO_TENDON_INDICES
    self._num_synergies = len(self._synergy_to_indices)

    covered = [idx for indices in self._synergy_to_indices.values() for idx in indices]
    if sorted(covered) != list(range(self._num_tendons)):
      raise ValueError(
        "Synergy mapping does not cover all tendons exactly once. "
        f"Expected indices 0..{self._num_tendons - 1}, got {sorted(set(covered))}."
      )

    self._raw_actions = torch.zeros(
      env.num_envs, self._num_synergies, device=env.device, dtype=torch.float32
    )
    self._expanded_actions = torch.zeros(
      env.num_envs, self._num_tendons, device=env.device, dtype=torch.float32
    )

  # Properties.

  @property
  def action_dim(self) -> int:
    """Dimension of the synergy action space."""
    return self._num_synergies

  @property
  def raw_action(self) -> torch.Tensor:
    """Raw synergy actions (before expansion)."""
    return self._raw_actions

  # Core methods.

  def process_actions(self, actions: torch.Tensor) -> None:
    """Expand low-dimensional synergy actions to full tendon efforts."""
    if actions.shape[1] != self._num_synergies:
      raise ValueError(
        f"SynergyTendonEffortAction expected actions of dim {self._num_synergies}, "
        f"got {actions.shape[1]}."
      )
    scale = getattr(self.cfg, "scale", 1.0)
    self._raw_actions[:] = actions * scale
    # Reset expanded buffer.
    self._expanded_actions.zero_()
    # Broadcast each synergy onto its tendon group.
    for syn_idx, tendon_indices in self._synergy_to_indices.items():
      idxs = torch.as_tensor(
        tendon_indices, device=self._expanded_actions.device, dtype=torch.long
      )
      self._expanded_actions[:, idxs] = self._raw_actions[:, syn_idx : syn_idx + 1]

  def apply_actions(self) -> None:
    """Apply expanded tendon efforts to the entity."""
    # The entity exposes a tendon-effort setter via the scene.
    self._entity.set_tendon_effort_target(
      self._expanded_actions, tendon_ids=self._tendon_ids
    )
