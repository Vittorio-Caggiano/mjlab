"""Run a MyoSkeleton tracking training sequence and summarize minimum iterations.

This utility automates a configurable sequence of training runs for
`Mjlab-Tracking-Flat-MyoSkeleton`, then aggregates metrics from TensorBoard
scalars to estimate the minimum iteration/step count needed to reach a target.

Typical use:

  python scripts/benchmarks/run_myoskeleton_training_sequence.py \
    --motion-file /abs/path/to/motion.npz \
    --target-tag Train/mean_reward \
    --target-threshold 100.0 \
    --target-mode ge
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingCase:
  """Single run configuration in the sequence."""

  name: str
  overrides: tuple[str, ...]


@dataclass
class RunSummary:
  """Per-run summary emitted to JSON/CSV."""

  name: str
  return_code: int
  duration_s: float
  log_dir: str
  command: str
  metric_tag_used: str | None
  final_metric_value: float | None
  min_iteration_hit: int | None
  min_steps_hit: int | None
  notes: str = ""


def _upload_to_wandb(
  *,
  output_dir: Path,
  payload: dict,
  summaries: list[RunSummary],
  project: str,
  entity: str | None,
  run_name: str,
  tags: list[str],
) -> None:
  """Upload sequence summaries/logs to Weights & Biases."""
  try:
    import wandb
  except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
      "W&B upload requested but `wandb` is not available. Install wandb first."
    ) from exc

  run = wandb.init(
    project=project,
    entity=entity,
    name=run_name,
    tags=tags,
    config=payload.get("args", {}),
  )

  table = wandb.Table(
    columns=[
      "name",
      "return_code",
      "duration_s",
      "metric_tag_used",
      "final_metric_value",
      "min_iteration_hit",
      "min_steps_hit",
      "notes",
    ]
  )
  for summary in summaries:
    table.add_data(
      summary.name,
      summary.return_code,
      summary.duration_s,
      summary.metric_tag_used,
      summary.final_metric_value,
      summary.min_iteration_hit,
      summary.min_steps_hit,
      summary.notes,
    )

  best = [
    s for s in summaries if s.min_iteration_hit is not None and s.return_code == 0
  ]
  best.sort(key=lambda s: s.min_iteration_hit or 10**12)

  log_payload: dict[str, float | int | str | None | object] = {
    "sequence/case_count": len(summaries),
    "sequence/results": table,
  }
  if best:
    fastest = best[0]
    log_payload.update(
      {
        "sequence/fastest_case": fastest.name,
        "sequence/fastest_min_iteration": fastest.min_iteration_hit,
        "sequence/fastest_min_steps": fastest.min_steps_hit,
      }
    )
  wandb.log(log_payload)

  artifact = wandb.Artifact("myoskeleton_training_sequence", type="benchmark")
  artifact.add_file(str(output_dir / "summary.json"))
  artifact.add_file(str(output_dir / "summary.csv"))
  for summary in summaries:
    run_log = output_dir / f"{summary.name}.log"
    if run_log.exists():
      artifact.add_file(str(run_log), name=f"logs/{run_log.name}")
  wandb.log_artifact(artifact)
  run.finish()


def _timestamp() -> str:
  return time.strftime("%Y-%m-%d_%H-%M-%S")


def _default_cases() -> list[TrainingCase]:
  """Sequence aligned with the tuning plan phases."""
  return [
    TrainingCase(name="baseline", overrides=()),
    TrainingCase(
      name="phase1_contact_budget",
      overrides=(
        "--env.sim.nconmax",
        "80",
        "--env.sim.njmax",
        "400",
      ),
    ),
    TrainingCase(
      name="phase1_less_exploration",
      overrides=(
        "--agent.actor.init-noise-std",
        "0.6",
        "--agent.num-steps-per-env",
        "16",
      ),
    ),
    TrainingCase(
      name="phase3_ppo_retune",
      overrides=(
        "--agent.algorithm.learning-rate",
        "5e-4",
        "--agent.algorithm.desired-kl",
        "0.005",
        "--agent.algorithm.num-mini-batches",
        "8",
        "--agent.algorithm.num-learning-epochs",
        "3",
      ),
    ),
  ]


def _iter_scalar_events(log_dir: Path, tag: str) -> list[tuple[int, float]]:
  """Load scalar values for a TensorBoard tag as (step, value)."""
  try:
    from tensorboard.backend.event_processing import event_accumulator
  except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
      "TensorBoard is required for scalar parsing. Install `tensorboard`."
    ) from exc

  event_files = sorted(log_dir.rglob("events.out.tfevents.*"))
  if not event_files:
    raise RuntimeError(f"No TensorBoard event files found under {log_dir}")

  all_events: list[tuple[int, float]] = []
  for event_file in event_files:
    acc = event_accumulator.EventAccumulator(str(event_file))
    acc.Reload()
    tags = set(acc.Tags().get("scalars", []))
    if tag not in tags:
      continue
    for ev in acc.Scalars(tag):
      all_events.append((int(ev.step), float(ev.value)))

  all_events.sort(key=lambda x: x[0])
  return all_events


def _discover_tag(log_dir: Path, requested: str, fallbacks: list[str]) -> str | None:
  candidates = [requested] + [f for f in fallbacks if f != requested]
  for tag in candidates:
    try:
      if _iter_scalar_events(log_dir, tag):
        return tag
    except RuntimeError:
      continue
  return None


def _first_hit_iteration(
  events: list[tuple[int, float]], threshold: float, mode: str
) -> int | None:
  if mode == "ge":
    for step, value in events:
      if value >= threshold:
        return step
  else:
    for step, value in events:
      if value <= threshold:
        return step
  return None


def _latest_run_dir(experiment_root: Path, before_ts: float) -> Path:
  runs = [p for p in experiment_root.glob("*") if p.is_dir()]
  if not runs:
    raise RuntimeError(f"No run directories found in {experiment_root}")

  eligible = [p for p in runs if p.stat().st_mtime >= before_ts - 1.0]
  if not eligible:
    eligible = runs
  return max(eligible, key=lambda p: p.stat().st_mtime)


def _run_one_case(
  case: TrainingCase,
  *,
  output_dir: Path,
  motion_file: str,
  max_iterations: int,
  save_interval: int,
  num_envs: int,
  target_tag: str,
  target_threshold: float,
  target_mode: str,
  metric_fallback_tags: list[str],
) -> RunSummary:
  logs_root = Path("logs") / "rsl_rl" / "myoskeleton_tracking_tuning_sequence"
  logs_root.mkdir(parents=True, exist_ok=True)

  cmd = [
    sys.executable,
    "-m",
    "mjlab.scripts.train",
    "Mjlab-Tracking-Flat-MyoSkeleton",
    "--env.commands.motion.motion-file",
    motion_file,
    "--agent.logger",
    "tensorboard",
    "--agent.experiment-name",
    "myoskeleton_tracking_tuning_sequence",
    "--agent.run-name",
    case.name,
    "--agent.max-iterations",
    str(max_iterations),
    "--agent.save-interval",
    str(save_interval),
    "--env.scene.num-envs",
    str(num_envs),
    *case.overrides,
  ]

  run_log = output_dir / f"{case.name}.log"
  env = os.environ.copy()
  env["PYTHONPATH"] = str(Path("src").resolve()) + os.pathsep + env.get(
    "PYTHONPATH", ""
  )

  start = time.perf_counter()
  with run_log.open("w") as f:
    f.write("# COMMAND\n")
    f.write(shlex.join(cmd) + "\n\n")
    f.flush()
    before_ts = time.time()
    proc = subprocess.run(
      cmd,
      stdout=f,
      stderr=subprocess.STDOUT,
      env=env,
      check=False,
    )
  duration = time.perf_counter() - start

  run_dir = ""
  metric_tag_used = None
  final_value = None
  min_iter = None
  min_steps = None
  notes = ""

  if proc.returncode == 0:
    try:
      latest = _latest_run_dir(logs_root, before_ts)
      run_dir = str(latest)

      metric_tag_used = _discover_tag(latest, target_tag, metric_fallback_tags)
      if metric_tag_used is None:
        notes = "No matching scalar tag found in TensorBoard logs."
      else:
        events = _iter_scalar_events(latest, metric_tag_used)
        if events:
          final_value = events[-1][1]
          min_iter = _first_hit_iteration(events, target_threshold, target_mode)
          if min_iter is not None:
            min_steps = min_iter * num_envs
    except Exception as exc:
      notes = f"Post-processing failed: {exc}"
  else:
    notes = "Training process failed. See per-run log file."

  return RunSummary(
    name=case.name,
    return_code=proc.returncode,
    duration_s=duration,
    log_dir=run_dir,
    command=shlex.join(cmd),
    metric_tag_used=metric_tag_used,
    final_metric_value=final_value,
    min_iteration_hit=min_iter,
    min_steps_hit=min_steps,
    notes=notes,
  )


def _parse_cases(case_overrides: list[str]) -> list[TrainingCase]:
  """Parse repeatable --case arguments formatted as name::arg1 arg2 ..."""
  if not case_overrides:
    return _default_cases()

  parsed: list[TrainingCase] = []
  for raw in case_overrides:
    if "::" not in raw:
      raise ValueError(
        f"Invalid --case value '{raw}'. Expected format: name::--arg1 val1 ..."
      )
    name, override_str = raw.split("::", maxsplit=1)
    parsed.append(TrainingCase(name=name, overrides=tuple(shlex.split(override_str))))
  return parsed


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--motion-file", required=True, help="Local path to motion.npz")
  parser.add_argument("--max-iterations", type=int, default=1500)
  parser.add_argument("--save-interval", type=int, default=100)
  parser.add_argument("--num-envs", type=int, default=2048)
  parser.add_argument("--target-tag", default="Train/mean_reward")
  parser.add_argument("--target-threshold", type=float, default=100.0)
  parser.add_argument(
    "--target-mode",
    choices=("ge", "le"),
    default="ge",
    help="Hit criterion: metric >= threshold (ge) or <= threshold (le).",
  )
  parser.add_argument(
    "--fallback-tag",
    action="append",
    default=[
      "Train/mean_reward",
      "Episode_Reward/sum",
      "Episode_Metrics/mpkpe",
    ],
    help="Extra metric tag candidates, checked in order.",
  )
  parser.add_argument(
    "--case",
    action="append",
    default=[],
    help="Repeatable case definition: name::--agent.algorithm.learning-rate 5e-4",
  )
  parser.add_argument(
    "--output-dir",
    default=None,
    help="Output folder (default: logs/experiments/myoskeleton_sequence/<timestamp>).",
  )
  parser.add_argument(
    "--wandb-upload",
    action="store_true",
    help="Upload summary JSON/CSV and logs to Weights & Biases.",
  )
  parser.add_argument(
    "--wandb-project",
    default="mjlab",
    help="W&B project for sequence summary uploads.",
  )
  parser.add_argument(
    "--wandb-entity",
    default=None,
    help="Optional W&B entity/team.",
  )
  parser.add_argument(
    "--wandb-run-name",
    default="myoskeleton_training_sequence",
    help="Run name used when uploading sequence summaries to W&B.",
  )
  parser.add_argument(
    "--wandb-tag",
    action="append",
    default=["myoskeleton", "tracking", "benchmark"],
    help="Repeatable W&B tag for sequence uploads.",
  )
  args = parser.parse_args()

  motion_file = str(Path(args.motion_file).expanduser().resolve())
  if not Path(motion_file).exists():
    raise FileNotFoundError(f"Motion file not found: {motion_file}")

  output_dir = (
    Path(args.output_dir)
    if args.output_dir is not None
    else Path("logs") / "experiments" / "myoskeleton_sequence" / _timestamp()
  )
  output_dir.mkdir(parents=True, exist_ok=True)

  cases = _parse_cases(args.case)
  print(f"[INFO] Running {len(cases)} cases. Output: {output_dir}")

  summaries: list[RunSummary] = []
  for idx, case in enumerate(cases, start=1):
    print(f"[INFO] ({idx}/{len(cases)}) Running case: {case.name}")
    summary = _run_one_case(
      case,
      output_dir=output_dir,
      motion_file=motion_file,
      max_iterations=args.max_iterations,
      save_interval=args.save_interval,
      num_envs=args.num_envs,
      target_tag=args.target_tag,
      target_threshold=args.target_threshold,
      target_mode=args.target_mode,
      metric_fallback_tags=args.fallback_tag,
    )
    summaries.append(summary)
    print(
      f"[INFO] {case.name}: rc={summary.return_code}, "
      f"min_iter={summary.min_iteration_hit}, final={summary.final_metric_value}"
    )

  payload = {
    "created_at": _timestamp(),
    "args": vars(args),
    "cases": [asdict(s) for s in summaries],
  }
  summary_json = output_dir / "summary.json"
  summary_json.write_text(json.dumps(payload, indent=2))

  summary_csv = output_dir / "summary.csv"
  with summary_csv.open("w", newline="") as f:
    writer = csv.DictWriter(
      f,
      fieldnames=list(asdict(summaries[0]).keys()) if summaries else ["name"],
    )
    writer.writeheader()
    for s in summaries:
      writer.writerow(asdict(s))

  best = [s for s in summaries if s.min_iteration_hit is not None and s.return_code == 0]
  best.sort(key=lambda s: s.min_iteration_hit or 10**12)

  print("\n" + "=" * 72)
  print("Training sequence summary")
  print("=" * 72)
  for s in summaries:
    print(
      f"- {s.name:<28} rc={s.return_code:<2} "
      f"min_iter={str(s.min_iteration_hit):<8} final={s.final_metric_value}"
    )
  if best:
    fastest = best[0]
    print("-" * 72)
    print(
      "Fastest-to-threshold: "
      f"{fastest.name} at iteration {fastest.min_iteration_hit} "
      f"(~{fastest.min_steps_hit} env-steps)."
    )
  else:
    print("No run reached the threshold (or metric could not be parsed).")
  print("=" * 72)
  print(f"Saved: {summary_json}")
  print(f"Saved: {summary_csv}")

  if args.wandb_upload:
    _upload_to_wandb(
      output_dir=output_dir,
      payload=payload,
      summaries=summaries,
      project=args.wandb_project,
      entity=args.wandb_entity,
      run_name=args.wandb_run_name,
      tags=args.wandb_tag,
    )
    print("Uploaded sequence artifacts to W&B.")


if __name__ == "__main__":
  main()
