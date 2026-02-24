"""Offline fitting pipeline for MyoLegsTorso muscle synergies (PCA/ICA).

This script is inspired by StandingBalance/MSR_extraction/train_syn.py. It:
- Collects tendon-effort activation data from a MyoLegsTorso env.
- Fits a PCA+ICA model to obtain a low-dimensional muscle synergy basis.
- Saves the fitted models and scaler for later use in synergy-based controllers.

NOTE: This script is optional and not wired into training by default.
It is intended as a starting point for MSR-style experiments.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


def collect_tendon_actions(
  env: ManagerBasedRlEnv,
  num_steps: int,
) -> np.ndarray:
  """Collect random tendon-effort actions from the given environment."""
  action_dim = env.action_manager.total_action_dim
  buf: List[np.ndarray] = []

  obs, _ = env.reset()
  del obs  # Unused; we only care about actions here.

  for _ in range(num_steps):
    # Sample random actions in [-1, 1].
    actions = 2.0 * np.random.rand(env.num_envs, action_dim).astype(np.float32) - 1.0
    obs, rew, terminated, truncated, extras = env.step(actions)
    del obs, rew, terminated, truncated, extras
    buf.append(actions.reshape(-1, action_dim))

  return np.concatenate(buf, axis=0)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--task-id",
    default="Mjlab-Velocity-Flat-MyoLegsTorso",
    help="Task ID to use for MSR data collection.",
  )
  parser.add_argument(
    "--num-envs",
    type=int,
    default=64,
    help="Number of parallel envs for data collection.",
  )
  parser.add_argument(
    "--num-steps",
    type=int,
    default=2000,
    help="Number of env steps to sample random actions from.",
  )
  parser.add_argument(
    "--output-dir",
    type=str,
    default="logs/msr/myolegstorso",
    help="Directory to save PCA/ICA/scaler artifacts.",
  )
  args = parser.parse_args()

  try:
    from sklearn.decomposition import FastICA, PCA
    from sklearn.preprocessing import StandardScaler
    import joblib
  except ImportError as exc:  # pragma: no cover - import-time check only.
    raise SystemExit(
      "scikit-learn and joblib are required for MSR fitting.\n"
      "Install them with `uv add scikit-learn joblib` or similar."
    ) from exc

  env_cfg = load_env_cfg(args.task_id)
  env_cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")

  try:
    data = collect_tendon_actions(env, num_steps=args.num_steps)
  finally:
    env.close()

  # Standardize actions before PCA/ICA.
  scaler = StandardScaler()
  data_scaled = scaler.fit_transform(data)

  # Choose a modest synergy dimension; can be tuned.
  n_components = min(35, data_scaled.shape[1])
  pca = PCA(n_components=n_components, whiten=True, random_state=0)
  z_pca = pca.fit_transform(data_scaled)

  ica = FastICA(n_components=n_components, random_state=0, max_iter=1000)
  z_ica = ica.fit_transform(z_pca)
  del z_ica  # Avoid unused variable warning; ica carries learned unmixing.

  out_dir = Path(args.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  joblib.dump(ica, out_dir / f"ica_{n_components}.pkl")
  joblib.dump(pca, out_dir / f"pca_{n_components}.pkl")
  joblib.dump(scaler, out_dir / f"scaler_{n_components}.pkl")

  print(f"Saved MSR models (dim={n_components}) to {out_dir}")


if __name__ == "__main__":
  main()
