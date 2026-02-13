# MJLAB training analysis: G1 vs MyoSkeleton (flat tracking)

## What is currently identical

Both tasks currently use the same base tracking task template and PPO hyperparameters:

- same policy/value network shapes (`512,256,128`)
- same PPO settings (`lr=1e-3`, `epochs=5`, `mini_batches=4`, `entropy_coef=0.005`)
- same rollout horizon (`num_steps_per_env=24`) and max iterations (`30k`)
- same generic reward structure inherited from `make_tracking_env_cfg()`

## Key differences that likely slow MyoSkeleton training

1. **More expensive contact solving budget in MyoSkeleton**
   - MyoSkeleton explicitly raises contact limits to `nconmax=100`, `njmax=500`.
   - Base tracking defaults are lower (`nconmax=35`, `njmax=250`).

2. **Broader friction randomization target**
   - G1 randomizes friction only on foot collision geoms.
   - MyoSkeleton randomizes friction for `".*"` (all geoms), which increases domain variability and can destabilize early learning.

3. **Different morphology / actuator characteristics**
   - MyoSkeleton uses many high-ratio actuator groups and a full-body musculoskeletal morphology.
   - This likely increases control stiffness/sensitivity and makes the same PPO settings less sample-efficient than on G1.

4. **Harder reference trajectory by default**
   - MyoSkeleton locks to `soccer1.npz`, which can be a high-dynamic sequence compared to easier warm-start clips.

## Proposed optimization plan for MyoSkeleton

### Phase 1 — Throughput and stability baseline (fast to try)

1. **Relax simulator/contact cost where possible**
   - Sweep `nconmax`: `[60, 80, 100]`
   - Sweep `njmax`: `[300, 400, 500]`
   - Keep the smallest pair that does not introduce contact artifacts.

2. **Constrain friction randomization to foot geoms first**
   - Replace `geom_names=".*"` with a foot-only regex.
   - Reintroduce whole-body friction randomization later in curriculum.

3. **Reduce exploration noise slightly at start**
   - Lower actor `init_noise_std` from `1.0 -> 0.6`.
   - Keep critic unchanged.

4. **Use shorter update horizon for faster policy iteration**
   - Try `num_steps_per_env=16` (vs 24) while keeping total samples per update comparable by adjusting env count.

### Phase 2 — Accuracy improvements after stable learning

5. **Curriculum for motion difficulty**
   - Start with easier clips / low-velocity segments.
   - Progressively add dynamic clips (e.g., soccer motions) after success threshold.

6. **Reward schedule for early-stage posture tracking**
   - Temporarily upweight pose/orientation terms (`motion_body_pos`, `motion_body_ori`) for the first N iterations.
   - Gradually restore original weights to recover velocity fidelity.

7. **Domain randomization curriculum**
   - First: reduced perturbation ranges for base COM and pushes.
   - Then ramp to full ranges from base config.

### Phase 3 — PPO hyperparameter retuning for MyoSkeleton

8. **Learning-rate / KL target retune**
   - Try `learning_rate in {3e-4, 5e-4, 1e-3}`
   - Try `desired_kl in {0.005, 0.01, 0.02}`

9. **Entropy schedule**
   - Start `entropy_coef=0.01`, decay to `0.003` after convergence starts.

10. **Mini-batch / epoch trade-off**
   - For stability in high-DoF control: compare
     - `(epochs=5, mini_batches=4)` (current)
     - `(epochs=3, mini_batches=8)`

## Minimal experiment matrix

Run each condition with 3 seeds and track:

- env SPS / physics SPS
- episode return
- motion body position/orientation errors
- termination breakdown (`anchor_pos`, `anchor_ori`, `ee_body_pos`)

Suggested sequence:

1. **A/B-1**: contact budget sweep
2. **A/B-2**: foot-only friction randomization
3. **A/B-3**: init noise + rollout horizon
4. **A/B-4**: PPO retune (best config from 1-3)
5. **A/B-5**: curriculum (motion + DR)

## Expected impact

- **Training speed**: primarily from reduced contact solver load and less over-broad randomization early in training.
- **Training accuracy**: primarily from curriculum + PPO retuning to the MyoSkeleton dynamics rather than reusing G1 defaults.
