# ema_KL experiment roadmap

This branch (`ema_KL`) hosts a sequence of TBA-on-Countdown experiments
exploring KL-penalty design choices, building toward EMA reference
policies. Numbering matches the original plan; X is a parallel track.

## 1. KL source for the advantage adjustment

Compare three policies for the per-sample `kl_est` and the K-group mean
that centers it:

| variant | `kl_mean_source` | `kl_per_sample_source` | notes |
| --- | --- | --- | --- |
| all-train | `train` | `train` | symmetric, live train policy |
| mean-only-inference | `inference` | `train` | original asymmetric behavior |
| all-inference | `inference` | `inference` | symmetric, inference snapshot |

Each variant is run twice — IS on (default) and IS off
(`--no-grpo.off-policy.importance-sample`). 6 runs total.

Implemented via `TBConfig.kl_mean_source` and
`TBConfig.kl_per_sample_source` (see `src/zeroband/training/config.py`,
`src/zeroband/training/loss.py`, and the new `Countdown_experiments/`
configs prefixed `TBA_qwen3_klMean*`/`TBA_qwen3_klAllInf*`).

## 2. EMA reference policy

Replace the periodic-reset reference policy (`model_reference`, reset
every `grpo.reference_reset_interval` rollout steps) with an EMA over
the live training policy:

    ref_param ← α · ref_param + (1 − α) · model_param

updated every step (or every k steps) instead of being copied wholesale
on a fixed cadence. The KL penalty `kl_est = log p(x|θ) − log p(x|θ_ref)`
then references a smoother, continuously-updated baseline.

Open knobs: α (decay), update cadence, whether to warm-start from a
hard reset or initialize ref = θ at step 0.

## 3. Approximation to the EMA reference policy

Because EMA over the model means we cannot drop `model_reference` from
GPU between steps without rebuilding it, the next step is to find a
cheap approximation. Possibilities to explore: an EMA of *log-probs*
rather than weights, a low-rank delta, or a precomputed reference per
checkpoint window.

## 4. Same as above with samples from diverse timesteps

Augment the EMA-reference setups with rollouts (or KL-evaluation
samples) drawn from a mix of recent policy versions, not only the
freshest checkpoint. The intent is to broaden the support over which
the KL is measured.

## 5. Same as above with correction based on EMA approximation theory

Layer in the analytical correction term implied by the EMA-as-stationary-
distribution view, to debias the KL estimator under the approximation
chosen in step 3.

## X. Same as step 4 but with OAPL and Kimi

Parallel track: repeat the diverse-timestep variant under OAPL and the
Kimi-K2 sequence-level objectives, to see whether the EMA-reference
benefit transfers across off-policy correction schemes.

## 6. FP4 phase (branch `fp4_KL`): KL reg vs quantized-rollout mismatch

Motivated by the humans& "4-bitter Lesson" blog (July 2026): async RL
policy mismatch = staleness + quantization error, and their published
NVFP4 recipe uses **no KL regularization** (miles defaults `kl_coef=0`).
Hypothesis: the free EMA-KL surrogate with Δ-normalized β (constant
β·c) mitigates the collapse they fix with kernel engineering
(bit-exact 4/6 contracts), and the two approaches compose.

Story: (1) symmetric fake-quant FP4 trains stably on-policy; (2) it
degrades as Δ grows (2×2 controls: {BF16, FP4} × {Δ≈1, Δ large});
(3) approx-KL anchored on the sampler's own vLLM logprobs rescues it.
Discovery arm (asymmetric: BF16 trainer, FP4 sampler): the penalty
contains the quantization gap → soft-QAT effect, measurable as a
shrinking log T_n − log T^Q_n telemetry.

New knobs (all emulation, H100-friendly — QDQ reproduces NVFP4
numerics on BF16 kernels; see `src/zeroband/training/qdq.py`):

- `--ckpt.rollout-quant nvfp4` (+ `-four-over-six`,
  `-skip-last-frac`): QDQ weights in `save_ckpt_for_rollout`.
- `--fake-quant-forward nvfp4`: symmetric arm; trainer linears run
  QDQ+STE forwards (bit-exact with the sampler QDQ by construction).
- `--use-vllm-logprobs --no-recompute-logprobs`: anchor IS ratio and
  approx-KL on the sampler-returned parquet logprobs (preserves
  quantization error in the mismatch terms; recomputing erases it).

Env-file vars: `rollout_quant`, `fake_quant`, `four_over_six`,
`quant_skip_last_frac`, `vllm_logprobs`. Configs:
`Countdown_experiments/TBA_qwen3_fp4*` (wandb project `fp4_KL`).
Smoke first: `TBA_qwen3_fp4sym_smoke` (60 steps, all paths on).

Known gaps: activation quantization not emulated (weight-only, W4A16-
like); per-rollout Δ_i calibration (heterogeneous staleness) still
uses the global `max_async_level` — per-sample Δ stamping is future
work; exact-KL reference stays BF16 by design.

## Operational notes

- Project: wandb `ema_KL` (https://wandb.ai/bartoldson/ema_KL).
- Cluster: LLNL matrix, `pbatch` partition, 2 nodes per job, 4 GPUs/node,
  `--exclusive`. Account: `bridges`. Time limit: 1-00:00:00.
- Launch: `multinode_launch.sh <Countdown_experiments/CONFIG>` from inside
  an `sbatch -N 2 --exclusive` allocation. Single-node smoke runs via
  `launch.sh <CONFIG>` (auto GPU split).
- Eval cadence: `eval_steps=20` (per config) — eval blocks inference for
  ~9 min per evaluation; `eval_steps=5` caused frequent stalls.
- Code change for step 1 introduces a **second precompute forward** when
  `kl_mean_source=train` to populate `batch["train_logprobs"]`. That
  costs ~+15% step time (verified in runs).
