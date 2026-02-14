
"""Run a MyoSkeleton standing tuning sequence with bio-mimetic adjustments.

This utility automates a sequence of training runs for
`Mjlab-Tracking-Flat-MyoSkeleton` on `standing_motion.npz`, sweeping over
stiffness, action scales, and reward weights to find a stable configuration.
Results are logged to Weights & Biases and local TensorBoard/CSV.
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
from typing import Any

# Import constants to calculate overrides
# We need to add src to path to import these
sys.path.append(str(Path(__file__).parents[2] / "src"))

from mjlab.asset_zoo.robots.myoskeleton.myoskeleton_constants import (
    MYOSKELETON_ACTION_SCALE,
    MYOSKELETON_ARTICULATION,
)


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


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def _generate_actuator_overrides(stiffness_scale: float, damping_scale: float | None = None) -> list[str]:
    """Generate CLI overrides for all actuators."""
    overrides = []
    if damping_scale is None:
        # Maintain damping ratio: d ~ sqrt(k)
        damping_scale = math.sqrt(stiffness_scale)

    for idx, act in enumerate(MYOSKELETON_ARTICULATION.actuators):
        if act.stiffness is not None:
             overrides.extend([
                 f"--env.scene.entities.robot.articulation.actuators.{idx}.stiffness",
                 str(act.stiffness * stiffness_scale)
             ])
        if act.damping is not None:
             overrides.extend([
                 f"--env.scene.entities.robot.articulation.actuators.{idx}.damping",
                 str(act.damping * damping_scale)
             ])
    return overrides


def _generate_action_scale_overrides(multiplier: float) -> list[str]:
    """Generate CLI overrides for action scales."""
    # Construct the dictionary as a string
    new_scales = {k: v * multiplier for k, v in MYOSKELETON_ACTION_SCALE.items()}
    # We need to format it as a string acceptable by the parser (likely JSON-style or python-dict style)
    # The help message showed: "{'key': val, ...}"
    # We'll use json.dumps but replace double quotes with single quotes if needed,
    # or just repr() which usually works for python parsers like tyro.
    scale_str = str(new_scales).replace("'", '"') # JSON uses double quotes
    # However, tyro/argparse might prefer python syntax. The error message showed python syntax with single quotes.
    # Let's try standard JSON first, but if it fails, fallback to string representation.
    # actually the help message said: (default: '{'L5...': ...}')
    # Let's use standard JSON string `"{...}"`.
    return [
        "--env.actions.joint_pos.scale",
        json.dumps(new_scales)
    ]


def _default_cases() -> list[TrainingCase]:
    """Sequence aligned with the tuning plan."""
    cases = []

    # 0. Baseline
    cases.append(TrainingCase(name="baseline", overrides=()))

    # 1. Phase 1: Bio-Mimetic / Compliance Sweep
    # Low Stiffness (0.2x), High Action Scale (5.0x)
    compliance_overrides = []
    compliance_overrides.extend(_generate_actuator_overrides(stiffness_scale=0.2))
    compliance_overrides.extend(_generate_action_scale_overrides(multiplier=5.0))
    cases.append(TrainingCase(name="phase1_bio_mimetic", overrides=tuple(compliance_overrides)))

    # Stiff Low Range (2.0x Stiffness, 0.5x Scale)
    stiff_overrides = []
    stiff_overrides.extend(_generate_actuator_overrides(stiffness_scale=2.0))
    stiff_overrides.extend(_generate_action_scale_overrides(multiplier=0.5))
    cases.append(TrainingCase(name="phase1_high_stiffness", overrides=tuple(stiff_overrides)))

    # 2. Phase 2: Action Authority (Just scale, standard stiffness)
    # 2.0x Authority
    cases.append(TrainingCase(
        name="phase2_action_2x",
        overrides=tuple(_generate_action_scale_overrides(multiplier=2.0))
    ))
    # 4.0x Authority
    cases.append(TrainingCase(
        name="phase2_action_4x",
        overrides=tuple(_generate_action_scale_overrides(multiplier=4.0))
    ))

    # 3. Phase 3: Reward Shaping (Root priority + Smoothness)
    # Based on Baseline config
    cases.append(TrainingCase(
        name="phase3_reward_shaping",
        overrides=(
             "--env.rewards.motion_global_root_pos.weight", "2.0",
             "--env.rewards.motion_global_root_ori.weight", "2.0",
             "--env.rewards.action_rate_l2.weight", "-0.1",
        )
    ))

    # 4. Phase 4: PPO Conservative
    cases.append(TrainingCase(
        name="phase4_ppo_conservative",
        overrides=(
            "--agent.algorithm.learning-rate", "3e-4",
            "--agent.algorithm.entropy-coef", "0.02",
        )
    ))

    # 5. Phase 5: MyoSuite Broad Rewards (The "Peak" problem)
    # TrackEnv uses exp(-5.0 * norm2). Flatter kernels (higher std) help gradient discovery.
    # User RELAXED the error scale to 0.5 (std 1.4 ish?). Let's try std=0.7.
    myosuite_reward_overrides = (
        "--env.rewards.motion_global_root_pos.params.std", "0.7",
        "--env.rewards.motion_global_root_ori.params.std", "0.7",
        "--env.rewards.motion_body_pos.params.std", "0.7",
        "--env.rewards.motion_body_ori.params.std", "0.7",
        "--env.rewards.motion_global_root_pos.weight", "2.0",
        "--env.rewards.motion_global_root_ori.weight", "1.0",
    )
    cases.append(TrainingCase(
        name="phase5_myosuite_broad_rewards",
        overrides=myosuite_reward_overrides
    ))

    # 6. Phase 6: MyoSuite Low Stiffness (The "Compliance" approach)
    # User used stiffness 2.0. Our default is ~7000. 0.001x is extreme, let's try 0.01x.
    low_stiff_overrides = []
    low_stiff_overrides.extend(_generate_actuator_overrides(stiffness_scale=0.01))
    low_stiff_overrides.extend(_generate_action_scale_overrides(multiplier=4.0)) # Keep 4x authority
    cases.append(TrainingCase(
        name="phase6_myosuite_low_stiffness",
        overrides=tuple(low_stiff_overrides)
    ))

    # 7. Phase 7: MyoSuite Dynamics + Broad Rewards (Combined)
    combined_overrides = list(myosuite_reward_overrides)
    combined_overrides.extend(_generate_actuator_overrides(stiffness_scale=0.01))
    combined_overrides.extend(_generate_action_scale_overrides(multiplier=4.0))
    cases.append(TrainingCase(
        name="phase7_myosuite_combined_low_stiff",
        overrides=tuple(combined_overrides)
    ))

    # 8. Phase 8: MyoSuite High Authority + Broad Rewards
    authority_overrides = list(myosuite_reward_overrides)
    authority_overrides.extend(_generate_action_scale_overrides(multiplier=10.0))
    cases.append(TrainingCase(
        name="phase8_myosuite_high_authority",
        overrides=tuple(authority_overrides)
    ))

    return cases


def _iter_scalar_events(log_dir: Path, tag: str) -> list[tuple[int, float]]:
    """Load scalar values for a TensorBoard tag as (step, value)."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception as exc:
        print(f"[WARN] TensorBoard not found: {exc}")
        return []

    event_files = sorted(log_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        print(f"[WARN] No event files in {log_dir}")
        return []

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
        if _iter_scalar_events(log_dir, tag):
            return tag
    return None


def _latest_run_dir(experiment_root: Path, before_ts: float) -> Path:
    runs = [p for p in experiment_root.glob("*") if p.is_dir()]
    if not runs:
        raise RuntimeError(f"No run directories found in {experiment_root}")

    # Filter for runs created after the script started
    eligible = [p for p in runs if p.stat().st_mtime >= before_ts - 5.0]
    if not eligible:
         # Fallback to just the latest generic one
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
) -> RunSummary:
    logs_root = Path("logs") / "rsl_rl" / "myoskeleton_standing_tuning"
    logs_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "mjlab.scripts.train",
        "Mjlab-Tracking-Flat-MyoSkeleton",
        "--env.commands.motion.motion-file",
        motion_file,
        "--agent.logger",
        "wandb",
        "--agent.experiment-name",
        "myoskeleton_standing_tuning",
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
    env["PYTHONPATH"] = str(Path("src").resolve()) + os.pathsep + env.get("PYTHONPATH", "")

    start = time.perf_counter()
    with run_log.open("w") as f:
        f.write("# COMMAND\n")
        f.write(shlex.join(cmd) + "\n\n")
        f.flush()
        before_ts = time.time()
        print(f"      Running command for {case.name}...")
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

            # Use fixed tags we care about
            metric_tag_used = _discover_tag(latest, "Train/mean_reward", ["Episode_Reward/sum"])
            if metric_tag_used:
                events = _iter_scalar_events(latest, metric_tag_used)
                if events:
                    final_value = events[-1][1]
                    # Check for "standing" success logic?
                    # For now just use threshold > 50 as a proxy for "not falling immediately"
                    # Baseline was -0.29.
                    for step, val in events:
                        if val > 50.0:
                            min_iter = step
                            break
                    if min_iter:
                        min_steps = min_iter * num_envs
        except Exception as exc:
            notes = f"Post-processing failed: {exc}"
    else:
        notes = "Training process failed."

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-file", default="src/mjlab/asset_zoo/robots/myoskeleton/motions/standing_motion.npz")
    parser.add_argument("--max-iterations", type=int, default=500) # Short runs for tuning
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    motion_file = str(Path(args.motion_file).expanduser().resolve())
    if not Path(motion_file).exists():
        # Fallback search
        alt = Path("src/mjlab/asset_zoo/robots/myoskeleton/motions/standing_motion.npz")
        if alt.exists():
            motion_file = str(alt.resolve())
        else:
            raise FileNotFoundError(f"Motion file not found: {motion_file}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("logs") / "experiments" / "myoskeleton_standing_tuning" / _timestamp()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = _default_cases()
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
        )
        summaries.append(summary)
        print(
            f"[INFO] {case.name}: rc={summary.return_code}, "
            f"final={summary.final_metric_value}"
        )

    # Save results
    payload = {
        "created_at": _timestamp(),
        "args": vars(args),
        "cases": [asdict(s) for s in summaries],
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2))

    with (output_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(summaries[0]).keys()) if summaries else ["name"])
        writer.writeheader()
        for s in summaries:
            writer.writerow(asdict(s))

    print(f"[INFO] Results saved to {output_dir}")

if __name__ == "__main__":
    main()
