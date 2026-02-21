"""Benchmark sweep for MyoSkeleton imitation learning (motion tracking).

Runs short training experiments for Mjlab-Tracking-Flat-MyoSkeleton or
Mjlab-Tracking-Flat-MyoSkeleton-Unitree (--task unitree) on the bundled
motion (default: standing_motion.npz), sweeping axes aligned with the
velocity sweep and tracking defaults:

  Axis 1: Stiffness scale — 0.25x (default), 1.0x, 2.0x.
  Axis 2: Reward balance — action_rate, motion error std.
  Axis 3: PPO hyperparams — rollout length, entropy.

Video rendering is enabled by default (disable with --no-video; uses EGL on
headless). Results are saved to JSON/CSV and ranked by final mean reward.

Usage:
    uv run python scripts/benchmarks/run_myoskeleton_tracking_sweep.py
    uv run python scripts/benchmarks/run_myoskeleton_tracking_sweep.py --task unitree
    uv run python scripts/benchmarks/run_myoskeleton_tracking_sweep.py --max-iterations 1000
    uv run python scripts/benchmarks/run_myoskeleton_tracking_sweep.py --cases baseline axis1_stiff_1.0x
    uv run python scripts/benchmarks/run_myoskeleton_tracking_sweep.py --list
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
from dataclasses import asdict, dataclass
from pathlib import Path

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


def _stiffness_overrides(scale: float, *, unitree: bool = False) -> list[str]:
  """CLI overrides to scale all actuator stiffnesses; damping by sqrt(scale)."""
  overrides: list[str] = []
  damping_scale = math.sqrt(scale)
  articulation = MYOSKELETON_ARTICULATION

  for idx, act in enumerate(articulation.actuators):
    assert isinstance(act, BuiltinPositionActuatorCfg)
    if act.stiffness is not None:
      overrides.extend(
        [
          f"--env.scene.entities.robot.articulation.actuators.{idx}.stiffness",
          str(act.stiffness * scale),
        ]
      )
    if act.damping is not None:
      overrides.extend(
        [
          f"--env.scene.entities.robot.articulation.actuators.{idx}.damping",
          str(act.damping * damping_scale),
        ]
      )
  return overrides


def _action_scale_override(scale: float) -> list[str]:
  """Generate CLI override for uniform action scale (joint_pos)."""
  return ["--env.actions.joint_pos.scale", str(scale)]


def _relaxed_termination_overrides(
  *,
  episode_length_s: float = 40,  # 20.0,
  anchor_pos_threshold: float = 0.8,  # 0.4,
  anchor_ori_threshold: float = 1.6,  # 1.0,
  ee_body_pos_threshold: float = 0.8,  # 0.4,
) -> list[str]:
  """CLI overrides to relax terminations and allow more exploration."""
  return [
    "--env.episode_length_s",
    str(episode_length_s),
    "--env.terminations.anchor_pos.params.threshold",
    str(anchor_pos_threshold),
    "--env.terminations.anchor_ori.params.threshold",
    str(anchor_ori_threshold),
    "--env.terminations.ee_body_pos.params.threshold",
    str(ee_body_pos_threshold),
  ]


# ── Case definitions ─────────────────────────────────────────────────────────

TRACKING_TASK_MYOSKELETON = "Mjlab-Tracking-Flat-MyoSkeleton"
TRACKING_TASK_UNITREE = "Mjlab-Tracking-Flat-MyoSkeleton-Unitree"


def _all_cases(*, unitree: bool = False) -> dict[str, Case]:
  """All available sweep cases, keyed by name. Use unitree=True for Unitree task."""
  cases: dict[str, Case] = {}

  # # Baseline = velocity sweep reference: axis1_stiff_0.5x (0.5x stiffness).
  baseline_overrides = _stiffness_overrides(0.5, unitree=unitree)
  # cases["baseline"] = Case(
  #   name="baseline",
  #   overrides=tuple(baseline_overrides),
  #   description="Velocity sweep reference: axis1_stiff_0.5x (0.5x stiffness).",
  # )

  # Relaxed terminations from baseline (0.5x stiff).
  baseline_relaxed = list(baseline_overrides)
  baseline_relaxed.extend(
    _relaxed_termination_overrides(
      episode_length_s=20.0,
      anchor_pos_threshold=0.4,
      anchor_ori_threshold=1.0,
      ee_body_pos_threshold=0.4,
    )
  )
  cases["baseline_relaxed"] = Case(
    name="baseline_relaxed",
    overrides=tuple(baseline_relaxed),
    description="Baseline (0.5x stiff) + relaxed terminations.",
  )

  # Relaxed terminations from baseline (0.5x stiff).
  baseline_relaxed_no_root = list(baseline_overrides)
  baseline_relaxed_no_root.extend(
    _relaxed_termination_overrides(
      episode_length_s=20.0,
      anchor_pos_threshold=0.4,
      anchor_ori_threshold=1.0,
      ee_body_pos_threshold=0.4,
    )
    + [
      "--env.rewards.motion_global_root_pos.weight",
      "0.0",
      "--env.rewards.motion_global_root_ori.weight",
      "0.0",
    ]
  )

  cases["baseline_relaxed_no_root"] = Case(
    name="baseline_relaxed_no_root",
    overrides=tuple(baseline_relaxed_no_root),
    description="Baseline (0.5x stiff) + relaxed terminations + no root tracking.",
  )
  # # Reduce root tracking: ignore root z (reference height may be wrong).
  # baseline_no_root_z = list(baseline_overrides)
  # baseline_no_root_z.extend(
  #   [
  #     "--env.rewards.motion_global_root_pos.params.ignore_z",
  #     "True",
  #   ]
  # )
  # cases["baseline_no_root_z"] = Case(
  #   name="baseline_no_root_z",
  #   overrides=tuple(baseline_no_root_z),
  #   description="0.5x stiff + root position tracking uses only xy (ignore z).",
  # )

  # # Reduce root tracking: lower weight on root pos/ori (less penalty for root error).
  # baseline_low_root = list(baseline_overrides)
  # baseline_low_root.extend(
  #   [
  #     "--env.rewards.motion_global_root_pos.weight",
  #     "0.1",
  #     "--env.rewards.motion_global_root_ori.weight",
  #     "0.1",
  #     "--env.rewards.motion_global_root_pos.params.ignore_z",
  #     "True",
  #   ]
  # )
  # cases["baseline_low_root"] = Case(
  #   name="baseline_low_root",
  #   overrides=tuple(baseline_low_root),
  #   description="0.5x stiff + low root weights (0.1) and ignore root z.",
  # )

  # No root tracking: only body/pose/velocity matter.
  baseline_no_root = list(baseline_overrides)
  baseline_no_root.extend(
    [
      "--env.rewards.motion_global_root_pos.weight",
      "0.0",
      "--env.rewards.motion_global_root_ori.weight",
      "0.0",
    ]
  )
  cases["baseline_no_root"] = Case(
    name="baseline_no_root",
    overrides=tuple(baseline_no_root),
    description="0.5x stiff + no root position/orientation tracking reward.",
  )

  # Axis 1: Stiffness scale (articulation-specific overrides).
  for stiff_mult, label in [
    (0.25, "0.25x"),
    (0.5, "0.5x"),
    # (1.0, "1.0x"),
  ]:
    cases[f"axis1_stiff_{label}"] = Case(
      name=f"axis1_stiff_{label}",
      overrides=tuple(_stiffness_overrides(stiff_mult, unitree=unitree)),
      description=f"Stiffness {label} of base.",
    )

  # # Axis 2: Reward balance.
  # cases["axis2_low_action_penalty"] = Case(
  #   name="axis2_low_action_penalty",
  #   overrides=("--env.rewards.action_rate_l2.weight", "0.0"),
  #   description="No action-rate penalty (allow more aggressive tracking).",
  # )
  # cases["axis2_wide_motion_std"] = Case(
  #   name="axis2_wide_motion_std",
  #   overrides=(
  #     "--env.rewards.motion_global_root_pos.params.std",
  #     "0.5",
  #     "--env.rewards.motion_body_pos.params.std",
  #     "0.5",
  #   ),
  #   description="Wider motion error kernels (easier reward signal).",
  # )
  # cases["axis2_high_root_weight"] = Case(
  #   name="axis2_high_root_weight",
  #   overrides=(
  #     "--env.rewards.motion_global_root_pos.weight",
  #     "1.0",
  #     "--env.rewards.motion_global_root_ori.weight",
  #     "1.0",
  #   ),
  #   description="Emphasize root position and orientation tracking.",
  # )

  # # Axis 3: PPO hyperparams.
  # cases["axis3_longer_rollout"] = Case(
  #   name="axis3_longer_rollout",
  #   overrides=("--agent.num-steps-per-env", "48"),
  #   description="Longer rollout (48 steps) for better value estimates.",
  # )
  # cases["axis3_high_entropy"] = Case(
  #   name="axis3_high_entropy",
  #   overrides=(
  #     "--agent.algorithm.entropy-coef",
  #     "0.02",
  #     "--agent.actor.init-noise-std",
  #     "1.0",
  #   ),
  #   description="Higher entropy and noise for exploration.",
  # )

  # Combined: stiff + low action penalty (alternative to compliant default).
  combined_stiff = list(_stiffness_overrides(1.0, unitree=unitree))
  combined_stiff.extend(("--env.rewards.action_rate_l2.weight", "0.0"))
  cases["combined_stiff_low_penalty"] = Case(
    name="combined_stiff_low_penalty",
    overrides=tuple(combined_stiff),
    description="1.0x stiffness + no action penalty.",
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
  task_id: str,
  max_iterations: int,
  save_interval: int,
  num_envs: int,
  logger: str,
  experiment_name: str = "myoskeleton_tracking",
  video: bool = True,
) -> RunResult:
  """Run a single training case."""
  logs_root = Path("logs") / "rsl_rl" / experiment_name

  cmd = [
    sys.executable,
    "-m",
    "mjlab.scripts.train",
    task_id,
    "--agent.logger",
    logger,
    "--agent.experiment-name",
    experiment_name,
    "--agent.run-name",
    f"tracking_{case.name}",
    "--agent.max-iterations",
    str(max_iterations),
    "--agent.save-interval",
    str(save_interval),
    "--env.scene.num-envs",
    str(num_envs),
    "--video",
    str(video),
    *case.overrides,
  ]

  env = os.environ.copy()
  env["PYTHONPATH"] = (
    str(Path("src").resolve()) + os.pathsep + env.get("PYTHONPATH", "")
  )
  env.setdefault("MUJOCO_GL", "egl")
  env.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

  before_ts = time.time()
  start = time.perf_counter()
  proc = subprocess.run(cmd, env=env, check=False)
  duration = time.perf_counter() - start

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
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument(
    "--cases",
    nargs="*",
    default=None,
    help="Case names to run (default: all). Use --list to see available cases.",
  )
  parser.add_argument(
    "--task",
    choices=["myoskeleton", "unitree"],
    default="myoskeleton",
    help=(
      "Tracking task: myoskeleton (default) or unitree "
      "(Mjlab-Tracking-Flat-MyoSkeleton-Unitree)."
    ),
  )
  parser.add_argument("--max-iterations", type=int, default=500)
  parser.add_argument("--save-interval", type=int, default=100)
  parser.add_argument("--num-envs", type=int, default=2048)
  parser.add_argument("--logger", default="wandb", choices=["wandb", "tensorboard"])
  parser.add_argument(
    "--experiment-name",
    default=None,
    help=(
      "W&B / TensorBoard experiment name. Default: myoskeleton_tracking or "
      "myoskeleton_tracking_unitree when --task unitree."
    ),
  )
  parser.add_argument("--output-dir", default=None)
  parser.add_argument(
    "--no-video",
    action="store_true",
    help="Disable video rendering (default: enabled; uses EGL for headless).",
  )
  parser.add_argument("--list", action="store_true", help="List all cases and exit.")
  args = parser.parse_args()

  use_unitree = args.task == "unitree"
  all_cases = _all_cases(unitree=use_unitree)
  task_id = TRACKING_TASK_UNITREE if use_unitree else TRACKING_TASK_MYOSKELETON
  experiment_name = args.experiment_name or (
    "myoskeleton_tracking_unitree" if use_unitree else "myoskeleton_tracking"
  )

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
    else Path("logs") / "experiments" / "tracking_sweep" / _timestamp()
  )
  output_dir.mkdir(parents=True, exist_ok=True)

  print(f"[sweep] Running {len(selected)} cases, {args.max_iterations} iters each")
  print(f"[sweep] Task: {task_id}")
  print(f"[sweep] Output: {output_dir}")
  print()

  results: list[RunResult] = []
  for idx, case in enumerate(selected, 1):
    print(f"[{idx}/{len(selected)}] {case.name}: {case.description}")
    result = run_case(
      case,
      task_id=task_id,
      max_iterations=args.max_iterations,
      save_interval=args.save_interval,
      num_envs=args.num_envs,
      logger=args.logger,
      experiment_name=experiment_name,
      video=not args.no_video,
    )
    results.append(result)
    reward_str = (
      f"{result.final_reward:.2f}" if result.final_reward is not None else "N/A"
    )
    status = "OK" if result.return_code == 0 else "FAIL"
    print(f"  -> {status} | reward={reward_str} | {result.duration_s:.0f}s")
    if result.notes:
      print(f"     {result.notes}")
    print()

  payload = {
    "created_at": _timestamp(),
    "args": {
      "task_id": task_id,
      "max_iterations": args.max_iterations,
      "num_envs": args.num_envs,
      "experiment_name": experiment_name,
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
