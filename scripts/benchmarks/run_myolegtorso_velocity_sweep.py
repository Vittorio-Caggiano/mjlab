"""Benchmark sweep for MyoLegTorso velocity task (muscle-controlled).

Runs training for baseline and two trunk-facilitation sweeps:
1. Standing curriculum: upright only for 2500 steps, then forward locomotion.
2. Trunk scale: trunk tendon action scale 0.4, same standing-then-locomotion curriculum.

Usage:
    uv run python scripts/benchmarks/run_myolegtorso_velocity_sweep.py
    uv run python scripts/benchmarks/run_myolegtorso_velocity_sweep.py --cases curriculum trunk_scale
    uv run python scripts/benchmarks/run_myolegtorso_velocity_sweep.py --max-iterations 200 --no-video
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

VELOCITY_TASK_MYOLEGTORSO = "Mjlab-Velocity-Flat-MyoLegsTorso"
TASK_STANDING_CURRICULUM = "Mjlab-Velocity-Flat-MyoLegsTorso-StandingCurriculum"
TASK_TRUNK_SCALE = "Mjlab-Velocity-Flat-MyoLegsTorso-TrunkScale"
TASK_SYNERGY = "Mjlab-Velocity-Flat-MyoLegsTorso-Synergy"


@dataclass(frozen=True)
class Case:
  name: str
  overrides: tuple[str, ...] = ()
  description: str = ""
  task_id: str | None = field(default=None)


@dataclass
class RunResult:
  name: str
  description: str
  return_code: int
  duration_s: float
  log_dir: str
  command: str
  task_id: str = ""
  final_reward: float | None = None
  notes: str = ""


def _timestamp() -> str:
  return time.strftime("%Y%m%d_%H%M%S")


def _all_cases() -> dict[str, Case]:
  return {
    "baseline": Case(
      name="baseline",
      overrides=(),
      description="Registered MyoLegTorso config: tendon effort, flat terrain.",
      task_id=None,
    ),
    "curriculum": Case(
      name="curriculum",
      overrides=(),
      description="Standing only 2500 steps, then forward locomotion.",
      task_id=TASK_STANDING_CURRICULUM,
    ),
    "trunk_scale": Case(
      name="trunk_scale",
      overrides=(),
      description="Trunk tendon scale 0.4, standing 2500 steps then locomotion.",
      task_id=TASK_TRUNK_SCALE,
    ),
    "synergy": Case(
      name="synergy",
      overrides=(),
      description="Synergy-based tendon action space (StandingBalance-style grouping).",
      task_id=TASK_SYNERGY,
    ),
  }


def run_case(
  case: Case,
  *,
  default_task_id: str,
  max_iterations: int,
  save_interval: int,
  num_envs: int,
  logger: str,
  experiment_name: str,
  video: bool,
) -> RunResult:
  """Run a single training case."""
  task_id = case.task_id if case.task_id is not None else default_task_id
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
    f"myolegstorso_{case.name}",
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
    task_id=task_id,
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

  task_ids_used = {c.task_id or VELOCITY_TASK_MYOLEGTORSO for c in selected}
  print(f"[sweep] Tasks: {task_ids_used}")
  print(f"[sweep] Cases: {len(selected)}, max_iterations={args.max_iterations}")
  print(f"[sweep] Output: {output_dir}")
  print()

  results: list[RunResult] = []
  for idx, case in enumerate(selected, 1):
    task_id = case.task_id or VELOCITY_TASK_MYOLEGTORSO
    print(f"[{idx}/{len(selected)}] {case.name} ({task_id}): {case.description}")
    result = run_case(
      case,
      default_task_id=VELOCITY_TASK_MYOLEGTORSO,
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
    "task_id": VELOCITY_TASK_MYOLEGTORSO,
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
