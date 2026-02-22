# myosuite_mjlab

Minimal external-style package that registers a MyoLeg velocity task for `mjlab` and provides a small `rsl-rl` scalar-std repro script.

## What this package does

- Registers `Mjlab-Velocity-Flat-MyoLeg` through public `mjlab` task registration APIs.
- Builds MyoLeg robot/env/RL configs without modifying `mjlab` source.
- Includes `scripts/repro_rsl_rl_std.py` to force `noise_std_type="scalar"` and exercise the known std path.

## Install

```bash
cd myosuite_mjlab
uv sync
```

If `mjlab` is not available on PyPI in your environment, install from GitHub before syncing this package.

## Repro

```bash
cd myosuite_mjlab
uv run python scripts/repro_rsl_rl_std.py --num-envs 16 --iters 2
```

Expected behavior:

- Unpatched `rsl-rl`: may fail in `Normal(mean, std)` construction if std becomes invalid.
- Patched/clamped `rsl-rl`: should complete the short run.

## Notes on assets

The task loader resolves the MyoLeg XML from `myosuite` package assets at runtime and can be overridden with:

- `MYOSUITE_MJLAB_MYOLEG_XML=/path/to/myolegs_mjlab.xml`

This keeps assets external to `mjlab` and avoids relying on `MJLAB_SRC_PATH`.
