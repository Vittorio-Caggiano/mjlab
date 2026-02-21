"""Minimal benchmark for MyoLeg velocity task (muscle-controlled).

Runs a short training run for Mjlab-Velocity-Flat-MyoLeg to verify training works.
Single baseline case; no sweep axes (tendon effort action, no stiffness overrides).

Usage:
    uv run python scripts/benchmarks/run_myoleg_velocity_sweep.py
    uv run python scripts/benchmarks/run_myoleg_velocity_sweep.py --max-iterations 200
    uv run python scripts/benchmarks/run_myoleg_velocity_sweep.py --no-video
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

VELOCITY_TASK_MYOLEG = "Mjlab-Velocity-Flat-MyoLeg"


@dataclass(frozen=True)
class Case:
  name: str
  overrides: tuple[str, ...]
  description: str = ""


@dataclass
class RunResult:
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


def _all_cases() -> dict[str, Case]:
  return {
    "baseline": Case(
      name="baseline",
      overrides=(),
      description="Registered MyoLeg config: tendon effort, flat terrain.",
    ),
  }


def run_case(
  case: Case,
  *,
  task_id: str,
  max_iterations: int,
  save_interval: int,
  num_envs: int,
  logger: str,
  experiment_name: str,
  video: bool,
) -> RunResult:
  """Run a single training case."""
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
    f"myoleg_{case.name}",
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

  logs_root = Path("logs") / "rsl_rl" / experiment_name
  run_dir = ""
  final_reward = None
  notes = ""
  if proc.returncode == 0:
    runs = [p for p in logs_root.glob("*") if p.is_dir()]
    eligible = [p for p in runs if p.stat().st_mtime >= before_ts - 5.0]
    if eligible:
      latest = max(eligible, key=lambda p: p.stat().st_mtime)
      run_dir = str(latest)
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
    help=f"Case names (default: all). Available: {', '.join(all_cases.keys())}",
  )
  parser.add_argument("--max-iterations", type=int, default=100)
  parser.add_argument("--save-interval", type=int, default=50)
  parser.add_argument("--num-envs", type=int, default=2048)
  parser.add_argument(
    "--logger",
    default="wandb",
    choices=["wandb", "tensorboard"],
    help="Logger to use (default: wandb).",
  )
  parser.add_argument(
    "--experiment-name",
    default="myoleg_velocity",
    help="Experiment name for logs.",
  )
  parser.add_argument("--output-dir", default=None)
  parser.add_argument(
    "--no-video",
    action="store_true",
    help="Disable video rendering (recommended for headless).",
  )
  parser.add_argument("--list", action="store_true", help="List cases and exit.")
  args = parser.parse_args()

  if args.list:
    for name, case in all_cases.items():
      print(f"{name}: {case.description}")
    return

  selected = list(all_cases.values())
  if args.cases:
    unknown = set(args.cases) - set(all_cases.keys())
    if unknown:
      parser.error(f"Unknown cases: {unknown}. Use --list.")
    selected = [all_cases[n] for n in args.cases]

  output_dir = (
    Path(args.output_dir)
    if args.output_dir
    else Path("logs") / "experiments" / "myoleg_velocity_sweep" / _timestamp()
  )
  output_dir.mkdir(parents=True, exist_ok=True)

  print(f"[sweep] Task: {VELOCITY_TASK_MYOLEG}")
  print(f"[sweep] Cases: {len(selected)}, max_iterations={args.max_iterations}")
  print(f"[sweep] Output: {output_dir}")
  print()

  results: list[RunResult] = []
  for idx, case in enumerate(selected, 1):
    print(f"[{idx}/{len(selected)}] {case.name}: {case.description}")
    result = run_case(
      case,
      task_id=VELOCITY_TASK_MYOLEG,
      max_iterations=args.max_iterations,
      save_interval=args.save_interval,
      num_envs=args.num_envs,
      logger=args.logger,
      experiment_name=args.experiment_name,
      video=not args.no_video,
    )
    results.append(result)
    status = "OK" if result.return_code == 0 else "FAIL"
    print(f"  -> {status} | {result.duration_s:.0f}s")
    if result.notes:
      print(f"     {result.notes}")
    print()

  payload = {
    "created_at": _timestamp(),
    "task_id": VELOCITY_TASK_MYOLEG,
    "args": {
      "max_iterations": args.max_iterations,
      "num_envs": args.num_envs,
      "cases": [c.name for c in selected],
    },
    "results": [asdict(r) for r in results],
  }
  (output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
  print(f"[sweep] Results: {output_dir}/summary.json")


if __name__ == "__main__":
  main()
