# Plan: MyoSkeleton morphology with Unitree-style actuation, control, and contact

## Goal

Create a **procedural pipeline** that keeps the MyoSkeleton kinematic/mass model, but replaces its current control assumptions with a Unitree-like stack:

1. Unitree-style joint grouping and command interfaces.
2. Unitree-style actuator parameterization (armature, effort limits, velocity limits, PD behavior).
3. Unitree-style contact setup (foot-focused contact handling and sensing).
4. A verification harness showing that a Unitree baseline and a MyoSkeleton-Unitree hybrid have matched low-level dynamics for equivalent commands and disturbances.

---

## What exists in this repository today

### Unitree side (reference behavior)

- `unitree_g1/g1_constants.py` defines multiple motor classes (5020, 7520-14, 7520-22, 4010), computes reflected inertias from gearbox stages, and derives stiffness/damping from a natural-frequency design.  It then assigns these actuator configs to joint-name groups using `BuiltinPositionActuatorCfg`.  It also defines collision profiles that differentiate foot contacts (`condim=3`, higher priority/friction) from other collisions (`condim=1`).
- Velocity/tracking env configs for G1 and GO1 already implement Unitree-like action and reward wiring.

### MyoSkeleton side (target morphology)

- `myoskeleton_constants.py` loads the MyoSkeleton MJCF, removes finger joints, and defines many actuator groups via simplified gear-derived parameters (`_actuator_params`), then applies broad collision handling (`geom_names_expr=(".*",)`, `condim=3`).
- MyoSkeleton velocity/tracking environment configs are already available, with custom contact sensors, action scaling, and rewards.

---

## Gap analysis: what must change to be "Unitree-like"

### 1) Joint model mismatch

Unitree controllers assume a compact, explicitly named set of actuated joints (e.g., 12 for GO1, grouped by hip/knee/ankle or limb role), while MyoSkeleton has many more DoFs and non-humanoid naming conventions.

**Needed work:**
- Build a deterministic **joint mapping layer**:
  - `unitree_joint_name -> myoskeleton_joint_name` for all controlled channels.
  - Optional many-to-one / one-to-many mapping with weights for redundant DoFs.
- Define a **control subset** in MyoSkeleton for Unitree-style policies (start with legs + trunk, optionally freeze upper body).
- Normalize joint direction/sign conventions and joint limits so Unitree commands map consistently.

### 2) Actuator model mismatch

Unitree constants include physically motivated reflected inertia and per-motor effort/velocity limits. MyoSkeleton currently uses a simplified gear-to-PD mapping and does not mirror Unitree actuator families.

**Needed work:**
- Replace MyoSkeleton actuator grouping for the controlled subset with a new profile:
  - Use Unitree motor classes (or their effective parameters) as templates.
  - Assign one Unitree actuator template per mapped joint group.
- Ensure these properties are transplanted per joint/group:
  - `armature` (reflected inertia)
  - `effort_limit`
  - `stiffness`, `damping`
  - optional velocity saturation behavior (if using explicit actuator wrappers)
- Keep non-controlled MyoSkeleton joints either:
  - passively stabilized (small damping/springs), or
  - locked/frozen to reduce control mismatch during early validation.

### 3) Contact model mismatch

Unitree collision setup is foot-centric (special contact dimensions/priority/friction), while MyoSkeleton currently uses broad collision settings across geoms.

**Needed work:**
- Define Unitree-like **functional feet** on MyoSkeleton (e.g., heel/forefoot sets under each side).
- Build a new collision profile for the hybrid model:
  - foot geoms: Unitree-like high-fidelity contact settings (`condim=3`, tuned friction/priority)
  - non-foot geoms: reduced contact dimensionality and/or self-collision constraints to avoid spurious penalties
- Rewire contact sensors/rewards to the new foot definition so gait phases and contact-driven rewards mirror Unitree tasks.

### 4) Action/observation interface mismatch

Unitree tasks typically assume a fixed action dimensionality and standard proprioceptive channels.

**Needed work:**
- Implement an adapter that exposes the MyoSkeleton hybrid with a Unitree-compatible action interface:
  - fixed action vector ordering by mapped joints
  - Unitree-like action scaling
- Align observation channels and ordering with Unitree policy expectations (joint pos/vel, base state, command, contact indicators).
- Ensure episode resets and randomization use comparable ranges.

---

## Procedural implementation plan

### Phase A — Build the hybrid robot definition

1. Add a new robot constants module (e.g., `myoskeleton_unitree_constants.py`) that:
   - starts from `get_spec()` of MyoSkeleton,
   - applies a `joint_mapping` and `controlled_joint_set`,
   - injects Unitree-style actuator configs onto mapped joints,
   - applies Unitree-like collision/contact profile to designated foot geoms.

2. Add a config factory `get_myoskeleton_unitree_robot_cfg()` returning an `EntityCfg`.

3. Add explicit dictionaries for:
   - joint mapping,
   - actuator template assignment per mapped joint,
   - action scales (computed with same rule as Unitree configs where relevant).

### Phase B — Add Unitree-style env config for hybrid

1. Add `tasks/velocity/config/myoskeleton_unitree/env_cfgs.py`:
   - clone Unitree task layout (commands, observations, rewards, terminations),
   - swap in `get_myoskeleton_unitree_robot_cfg()`.

2. Reuse Unitree-like contact sensors keyed to hybrid foot geoms.

3. Keep a minimal variant first (flat terrain, limited randomization), then expand to rough/randomized.

### Phase C — Verification harness for dynamic matching

Add an evaluation script/test suite that runs **paired rollouts** under matched conditions:

- System U: native Unitree robot (GO1 or G1 baseline).
- System M-U: MyoSkeleton morphology with Unitree-style actuation/control/contact on mapped joints.

Use identical controller form and command sequences on mapped channels.

---

## Test plan: verify "dynamics match"

"Match" should be defined with measurable tolerances, not by visual similarity.

### Test 1: Joint-level closed-loop step response parity

For each mapped joint group:
1. Freeze or stabilize unrelated joints.
2. Apply identical position step commands.
3. Measure rise time, overshoot, settling time, steady-state error.

**Pass criterion:** per-metric relative error below threshold (e.g., <= 10-15%).

### Test 2: Frequency response / chirp parity

1. Apply chirp or multi-sine reference at joint command level.
2. Compare amplitude ratio and phase lag vs frequency.

**Pass criterion:** bounded gain/phase deviation over the control-relevant band.

### Test 3: Contact impulse and stance compliance parity

1. Drop/stance tests with equivalent COM height and posture.
2. Compare normal/tangential contact impulses, peak GRFs, settling behavior.

**Pass criterion:** impulse and peak-force errors within tolerance; no systematic bias in stance deformation.

### Test 4: Multi-joint trajectory tracking parity

1. Run a fixed open-loop reference trajectory (or deterministic PD policy) on mapped joints.
2. Compare tracking RMSE and required actuator effort.

**Pass criterion:** trajectory RMSE and effort profile remain within tolerance bands.

### Test 5: Perturbation rejection parity

1. Apply matched pushes/disturbances to base.
2. Compare recovery time, max orientation deviation, and failure rate.

**Pass criterion:** comparable recovery statistics across seeds.

### Aggregate decision rule

Define a weighted score across tests 1-5.  The hybrid is accepted only if:
- every safety-critical metric is within bound, and
- total score exceeds a preset acceptance threshold.

---

## Practical notes and risks

- Exact one-to-one dynamic equivalence is impossible because morphology/inertia distribution differ; target should be **controller-relevant equivalence** on the mapped channels.
- Contact equivalence is the hardest part; prioritize robust foot geom definition and friction calibration.
- Start with reduced DoFs and flat terrain before enabling full-body control and broad randomization.

---

## Deliverables checklist

1. New hybrid robot constants/config factory.
2. New hybrid velocity env config (Unitree-like interface).
3. Joint/actuator/contact mapping tables.
4. Dynamic matching benchmark tests (1-5 above) with quantitative thresholds.
5. A regression test target that fails if matching metrics drift beyond tolerances.
