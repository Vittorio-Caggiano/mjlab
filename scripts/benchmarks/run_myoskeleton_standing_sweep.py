"""Benchmark sweep for MyoSkeleton standing task.

Runs short training experiments sweeping the key axes identified during
model analysis:

  Axis 1: Stiffness scale — how stiff should the PD controller be?
  Axis 2: Action authority — how much can the policy move joints per step?
  Axis 3: Reward balance  — alive/upright vs pose vs velocity tracking.
  Axis 4: PPO hyperparams — learning rate, entropy, initial noise.

Each case trains for a short window (configurable, default 500 iters) and
logs to W&B + TensorBoard.  Results are saved to a JSON/CSV summary.

Usage:
    python scripts/benchmarks/run_myoskeleton_standing_sweep.py
    python scripts/benchmarks/run_myoskeleton_standing_sweep.py --max-iterations 1000
    python scripts/benchmarks/run_myoskeleton_standing_sweep.py --cases baseline axis1_stiff_0.25x
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Add src to path for imports.
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.myoskeleton.myoskeleton_constants import (
  MYOSKELETON_ARTICULATION,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Case:
  """A single benchmark configuration."""

  name: str
  overrides: tuple[str, ...]
  description: str = ""


@dataclass
class RunResult:
  """Post-run summary for one case."""

  name: str
  description: str
  return_code: int
  duration_s: float
  log_dir: str
  command: str
  final_reward: float | None = None
  notes: str = ""


def _timestamp() -> str:
  return time.strftime("%Y%m%d_%H%M%S")


def _stiffness_overrides(scale: float) -> list[str]:
  """Generate CLI overrides to scale all actuator stiffnesses.

  Damping is scaled by sqrt(stiffness_scale) to maintain the damping ratio.
  """
  overrides: list[str] = []
  damping_scale = math.sqrt(scale)
  for idx, act in enumerate(MYOSKELETON_ARTICULATION.actuators):
    assert isinstance(act, BuiltinPositionActuatorCfg)
    if act.stiffness is not None:
      overrides.extend([
        f"--env.scene.entities.robot.articulation.actuators.{idx}.stiffness",
        str(act.stiffness * scale),
      ])
    if act.damping is not None:
      overrides.extend([
        f"--env.scene.entities.robot.articulation.actuators.{idx}.damping",
        str(act.damping * damping_scale),
      ])
  return overrides


def _action_scale_override(scale: float) -> list[str]:
  """Generate CLI override for uniform action scale."""
  return ["--env.actions.joint_pos.scale", str(scale)]


# ── Case definitions ─────────────────────────────────────────────────────────

# The standing env_cfg already applies 0.5x stiffness and 0.2 action scale
# on top of the base constants.  Overrides here are relative to the
# *registered* config (i.e. after 0.5x stiffness is already applied).


def _all_cases() -> dict[str, Case]:
  """All available sweep cases, keyed by name."""
  cases: dict[str, Case] = {}

  # ── 0. Baseline (registered config, no overrides) ──────────────────────
  cases["baseline"] = Case(
    name="baseline",
    overrides=(),
    description="Registered standing config: 0.5x stiffness, scale=0.2, conservative PPO.",
  )

  # ── Axis 1: Stiffness scale ────────────────────────────────────────────
  # These override the *base* stiffness (before the 0.5x in env_cfg).
  # The env_cfg multiplies by 0.5x, so to get effective 0.25x of base
  # we pass 0.5 * base (the env_cfg then applies 0.5x → 0.25x effective).
  # Simpler: override the final actuator values directly.

  for stiff_mult, label in [
    (0.25, "0.25x"),
    (1.0, "1.0x"),
    (2.0, "2.0x"),
  ]:
    cases[f"axis1_stiff_{label}"] = Case(
      name=f"axis1_stiff_{label}",
      overrides=tuple(_stiffness_overrides(stiff_mult)),
      description=f"Stiffness {label} of base (env_cfg applies 0.5x on top → effective {stiff_mult * 0.5}x).",
    )

  # ── Axis 2: Action authority ───────────────────────────────────────────
  for action_scale, label in [
    (0.05, "0.05"),
    (0.1, "0.1"),
    (0.4, "0.4"),
    (0.8, "0.8"),
  ]:
    cases[f"axis2_scale_{label}"] = Case(
      name=f"axis2_scale_{label}",
      overrides=tuple(_action_scale_override(action_scale)),
      description=f"Uniform action scale {label} rad (~{math.degrees(action_scale):.0f} deg).",
    )

  # ── Axis 3: Reward balance ────────────────────────────────────────────
  cases["axis3_high_alive"] = Case(
    name="axis3_high_alive",
    overrides=(
      "--env.rewards.alive.weight", "30.0",
      "--env.rewards.upright.weight", "15.0",
      "--env.rewards.pose.weight", "3.0",
    ),
    description="Dominant alive/upright, reduced pose weight.",
  )

  cases["axis3_high_pose"] = Case(
    name="axis3_high_pose",
    overrides=(
      "--env.rewards.alive.weight", "10.0",
      "--env.rewards.pose.weight", "15.0",
      "--env.rewards.pose.params.std_standing", '{".*": 0.02}',
    ),
    description="Dominant pose reward with very tight tolerance.",
  )

  cases["axis3_high_stillness"] = Case(
    name="axis3_high_stillness",
    overrides=(
      "--env.rewards.track_linear_velocity.weight", "10.0",
      "--env.rewards.track_angular_velocity.weight", "8.0",
      "--env.rewards.joint_vel.weight", "-0.02",
    ),
    description="Emphasize zero-velocity tracking and penalize joint movement.",
  )

  cases["axis3_no_pose"] = Case(
    name="axis3_no_pose",
    overrides=(
      "--env.rewards.pose.weight", "0.0",
      "--env.rewards.alive.weight", "20.0",
      "--env.rewards.upright.weight", "10.0",
    ),
    description="No pose reward — rely only on alive + upright + velocity.",
  )

  # ── Axis 4: PPO hyperparameters ────────────────────────────────────────
  cases["axis4_high_entropy"] = Case(
    name="axis4_high_entropy",
    overrides=(
      "--agent.algorithm.entropy-coef", "0.02",
      "--agent.actor.init-noise-std", "1.0",
    ),
    description="Higher entropy and noise for more exploration.",
  )

  cases["axis4_low_lr"] = Case(
    name="axis4_low_lr",
    overrides=(
      "--agent.algorithm.learning-rate", "1e-4",
      "--agent.algorithm.entropy-coef", "0.002",
    ),
    description="Very conservative LR + low entropy.",
  )

  cases["axis4_longer_rollout"] = Case(
    name="axis4_longer_rollout",
    overrides=(
      "--agent.num-steps-per-env", "48",
    ),
    description="Longer rollout (48 steps = ~1s) for better value estimates.",
  )

  # ── Combined: best guesses ────────────────────────────────────────────
  combined_a = list(_stiffness_overrides(0.25))
  combined_a.extend(_action_scale_override(0.4))
  combined_a.extend([
    "--env.rewards.alive.weight", "20.0",
    "--env.rewards.upright.weight", "10.0",
  ])
  cases["combined_compliant"] = Case(
    name="combined_compliant",
    overrides=tuple(combined_a),
    description="Low stiffness + high authority + high alive/upright (loco-mujoco style).",
  )

  combined_b = list(_stiffness_overrides(2.0))
  combined_b.extend(_action_scale_override(0.05))
  combined_b.extend([
    "--env.rewards.pose.weight", "12.0",
    "--agent.algorithm.entropy-coef", "0.002",
  ])
  cases["combined_stiff_precise"] = Case(
    name="combined_stiff_precise",
    overrides=tuple(combined_b),
    description="High stiffness + tiny actions + dominant pose (rigid standing).",
  )

  return cases


# ── Runner ───────────────────────────────────────────────────────────────────


def _try_read_final_reward(log_dir: Path, before_ts: float) -> float | None:
  """Try to read the final mean reward from TensorBoard logs."""
  try:
    from tensorboard.backend.event_processing import event_accumulator
  except ImportError:
    return None

  event_files = sorted(log_dir.rglob("events.out.tfevents.*"))
  if not event_files:
    return None

  tag_candidates = ["Train/mean_reward", "Episode_Reward/sum"]
  for tag in tag_candidates:
    for event_file in event_files:
      acc = event_accumulator.EventAccumulator(str(event_file))
      acc.Reload()
      if tag in acc.Tags().get("scalars", []):
        events = acc.Scalars(tag)
        if events:
          return float(events[-1].value)
  return None


def _find_latest_run(experiment_root: Path, before_ts: float) -> Path | None:
  """Find the most recent run directory created after before_ts."""
  runs = [p for p in experiment_root.glob("*") if p.is_dir()]
  eligible = [p for p in runs if p.stat().st_mtime >= before_ts - 5.0]
  if not eligible:
    eligible = runs
  return max(eligible, key=lambda p: p.stat().st_mtime) if eligible else None


def run_case(
  case: Case,
  *,
  max_iterations: int,
  save_interval: int,
  num_envs: int,
  logger: str,
) -> RunResult:
  """Run a single training case."""
  logs_root = Path("logs") / "rsl_rl" / "myoskeleton_standing"

  cmd = [
    sys.executable,
    "-m",
    "mjlab.scripts.train",
    "Mjlab-Standing-Flat-MyoSkeleton",
    "--agent.logger", logger,
    "--agent.experiment-name", "myoskeleton_standing",
    "--agent.run-name", case.name,
    "--agent.max-iterations", str(max_iterations),
    "--agent.save-interval", str(save_interval),
    "--env.scene.num-envs", str(num_envs),
    *case.overrides,
  ]

  env = os.environ.copy()
  env["PYTHONPATH"] = str(Path("src").resolve()) + os.pathsep + env.get("PYTHONPATH", "")

  before_ts = time.time()
  start = time.perf_counter()
  proc = subprocess.run(cmd, env=env, check=False)
  duration = time.perf_counter() - start

  # Try to read final reward.
  run_dir = ""
  final_reward = None
  notes = ""
  if proc.returncode == 0:
    try:
      latest = _find_latest_run(logs_root, before_ts)
      if latest:
        run_dir = str(latest)
        final_reward = _try_read_final_reward(latest, before_ts)
    except Exception as exc:
      notes = f"Post-processing error: {exc}"
  else:
    notes = f"Training failed (rc={proc.returncode})."

  return RunResult(
    name=case.name,
    description=case.description,
    return_code=proc.returncode,
    duration_s=duration,
    log_dir=run_dir,
    command=shlex.join(cmd),
    final_reward=final_reward,
    notes=notes,
  )


def main() -> None:
  all_cases = _all_cases()

  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument(
    "--cases",
    nargs="*",
    default=None,
    help=(
      f"Case names to run (default: all). Available: {', '.join(all_cases.keys())}"
    ),
  )
  parser.add_argument("--max-iterations", type=int, default=500)
  parser.add_argument("--save-interval", type=int, default=100)
  parser.add_argument("--num-envs", type=int, default=2048)
  parser.add_argument("--logger", default="wandb", choices=["wandb", "tensorboard"])
  parser.add_argument("--output-dir", default=None)
  parser.add_argument("--list", action="store_true", help="List all cases and exit.")
  args = parser.parse_args()

  if args.list:
    print(f"{'Name':<30} Description")
    print("-" * 80)
    for name, case in all_cases.items():
      print(f"{name:<30} {case.description}")
    return

  selected = list(all_cases.values())
  if args.cases:
    unknown = set(args.cases) - set(all_cases.keys())
    if unknown:
      parser.error(f"Unknown cases: {unknown}. Use --list to see available cases.")
    selected = [all_cases[n] for n in args.cases]

  output_dir = (
    Path(args.output_dir)
    if args.output_dir
    else Path("logs") / "experiments" / "standing_sweep" / _timestamp()
  )
  output_dir.mkdir(parents=True, exist_ok=True)

  print(f"[sweep] Running {len(selected)} cases, {args.max_iterations} iters each")
  print(f"[sweep] Output: {output_dir}")
  print()

  results: list[RunResult] = []
  for idx, case in enumerate(selected, 1):
    print(f"[{idx}/{len(selected)}] {case.name}: {case.description}")
    result = run_case(
      case,
      max_iterations=args.max_iterations,
      save_interval=args.save_interval,
      num_envs=args.num_envs,
      logger=args.logger,
    )
    results.append(result)
    reward_str = f"{result.final_reward:.2f}" if result.final_reward is not None else "N/A"
    status = "OK" if result.return_code == 0 else "FAIL"
    print(f"  -> {status} | reward={reward_str} | {result.duration_s:.0f}s")
    if result.notes:
      print(f"     {result.notes}")
    print()

  # ── Save summary ─────────────────────────────────────────────────────
  payload = {
    "created_at": _timestamp(),
    "args": {
      "max_iterations": args.max_iterations,
      "num_envs": args.num_envs,
      "cases": [c.name for c in selected],
    },
    "results": [asdict(r) for r in results],
  }
  (output_dir / "summary.json").write_text(json.dumps(payload, indent=2))

  if results:
    with (output_dir / "summary.csv").open("w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
      writer.writeheader()
      for r in results:
        writer.writerow(asdict(r))

  # ── Print ranking ────────────────────────────────────────────────────
  ranked = [r for r in results if r.final_reward is not None]
  ranked.sort(key=lambda r: r.final_reward or 0, reverse=True)
  if ranked:
    print("=" * 60)
    print("RANKING (by final mean reward)")
    print("=" * 60)
    for i, r in enumerate(ranked, 1):
      print(f"  {i}. {r.name:<30} reward={r.final_reward:.2f}")
    print()

  print(f"[sweep] Full results: {output_dir}/summary.json")


if __name__ == "__main__":
  main()
