# Overnight status — fp4_KL (night of 2026-07-12 → 13)

Running summary maintained by Claude's hourly loop. Newest entries at the bottom.

## ~23:00 — Setup complete (loop iteration 0)

**Code (branch `fp4_KL`, commits f15c09e + 0d26fa0, pushed):**
- NVFP4 QDQ emulation (`training/qdq.py`, 18 tests) + `--ckpt.rollout-quant nvfp4` + `--fake-quant-forward nvfp4` (symmetric, bit-exact by construction) + `--use-vllm-logprobs` (fixes the TBA overwrite that erased sampler numerics even with recompute off).
- Heterogeneous staleness: `--rl.staleness-offsets 1,4,10,32` (per-DP-rank checkpoint lags; self-throttling via reload poll; retention = async_level+1 ≥ max offset verified). Parquet `step` column (generating ckpt — already recorded!) threaded per-token through data pipeline as `gen_steps`.
- Per-rollout calibration: `--grpo.kl-approx-delta-source per_rollout` → c_i = α/(Δ_i(1−α)) per token; equivalence with global path verified when Δ uniform. Clamp moved to raw log-ratio units (Δ-invariant truncation). Per-Δ stratified telemetry (kl_approx/d01_02..d21_up).
- Includes Brian's in-flight `kl_is` loss change + noKLis configs (were uncommitted; the launcher mapping already referenced them).

**Fable-subagent adversarial review** (full text in session; key points folded into EXPERIMENT_ROADMAP.md §7): vLLM-logprob anchor required for hetero + same anchor for reference arm; ~3.4× dose confound → uniform-dose sweep w∈{0.0045,0.015,0.045} added; β=0 hetero control added; clamp bug found+fixed; per-Δ form-validation telemetry added; deferred: offset rotation, warm-start branching, ≥3 paired seeds, FP4×hetero factorial (FP4 stays a separate homogeneous mini-study).

**Queue (all PENDING on priority, submitted tonight):**
| Job | Config | Time limit |
|---|---|---|
| 276517 | fp4sym_smoke (60 steps, all FP4 paths) | 4h |
| 276520 | bf16_b0_async32 (staleness-only control) | 10h |
| 276521 | hetero_b0_async32 (hetero control; first live test of offsets) | 10h |
| 276522 | homog10_approxKL_b005_vllmLP (reference arm) | 24h |
| 276523 | hetero_approxKL_b005_perRollout (decisive arm) | 24h |

Held back pending smoke: dose-sweep global_w*, fp4sym collapse/rescue arms, fp4asym. Node cap ≤4 enforced 08:45–17:00 by the loop.

## 23:31 — Loop iteration 1

All 5 jobs still PENDING (Priority); pbatch fully allocated (35 nodes busy, 0 idle), no start estimates yet. Nothing running → no logs to check, no action taken. Monitor on the smoke job will fire the moment it starts.

## 00:31 — Loop iteration 2

All 5 jobs still PENDING; cluster still full. No action. Note: queued jobs execute the working tree at start time, so I'm keeping the branch frozen (no further dev commits) until they launch.

## 01:31 — Loop iteration 3

Still all PENDING; no change.

## 02:31 — Loop iteration 4

Still all PENDING. A pbatch job ends ~03:00 which may free 2 nodes; the big wave frees ~08:05–08:35.

## 03:31 — Loop iteration 5

Still all PENDING (03:00 opening went elsewhere). Plan for the 08:45 cap if jobs start in the 08:05–08:35 wave: keep the decisive pair 276522+276523 (24h, 4 nodes), hold/cancel-and-resubmit the smoke and β=0 controls for after 17:00.

## 04:31 — Loop iteration 6

Still all PENDING; no change.

## 05:31 — Loop iteration 7

Still all PENDING; no change.

## 06:31 — Loop iteration 8

Still all PENDING. At the 08:23 iteration I'll scontrol hold the smoke + β=0 controls (276517/276520/276521) so only the decisive pair (276522/276523, 4 nodes) is eligible during the 08:45–17:00 cap window; holds released after 17:00.

## 07:31 — Loop iteration 9

Still all PENDING. Held 276517/276520/276521 (smoke + β=0 controls) pre-emptively: the node-freeing wave lands ~08:05–08:35, possibly before my 08:23 iteration. Only the decisive pair (276522/276523 = exactly 4 nodes) is eligible to start during the day. Holds release after 17:00; if only one pair job gets nodes, I may release the short smoke into the spare slot.

## 09:53 — Loop iteration 10 (cron fired late)

Day window active. Holds intact on smoke+controls; decisive pair still PENDING (Priority) — the 08:05–08:35 wave went to higher-priority jobs. 0 nodes in use, fully cap-compliant. The smoke-job monitor was reaped by the harness; will re-arm when the smoke is released after 17:00.

## ~10:05 — Brian raised the cap to 12 nodes

Released all holds (5 jobs = 10 nodes ≤ 12); loop prompt updated. All five jobs eligible whenever pbatch frees nodes; dose-sweep + FP4 arms still gate on the smoke passing.

## 10:33 — Loop iteration 11

All five back to PENDING (Priority) after hold release; still queued behind higher-priority work. No action.

## 11:33 — Loop iteration 12

Still PENDING. Queue position: ranks 47–51 of 325 pending on pbatch (fair-share priority ~167.8k). Cluster demand is just high; nothing actionable.

## 12:33 — Loop iteration 13

Still PENDING; no change.

## 13:33 — Loop iteration 14

Still PENDING; no change.

## ~14:15 — bridges bank test + 1-node smoke RUNNING

Per Brian's suggestion, probed the bridges bank: priority gain over effml is real but small (169084 vs 167854 → rank 77 vs 79 of ~325), so NOT resubmitting the queued 2-node jobs (they'd lose queue age). The useful discovery: 1-node jobs backfill instantly. Submitted the smoke as a single-node run under bridges (launch.sh auto GPU split, replicate tag "1node"; first attempt failed with 1 visible GPU — fixed with --exclusive). Job 277087 is RUNNING on matrix30 as of ~14:12; monitor armed on its logs. If it passes, the FP4/vllm-logprobs/approx-KL code paths are validated and the gated arms can be submitted (new submissions default to bridges). The queued 2-node smoke 276517 becomes redundant once this passes — will scancel then.

## 14:33 — Loop iteration 15

1-node smoke 277087 RUNNING on matrix30 since 14:31 (earlier start-time note was off; no requeue). Training log in startup phase, no errors. Five 2-node effml jobs still PENDING. Monitor active.

## ~14:45 — Smoke crash root-caused and fixed; resubmitted as 277092

277087 died in startup: `uv run` on matrix30 REMOVED AND RECREATED the shared .venv (interpreter mismatch on that node), dropping the manually-synced flash-attn extra → trainer ImportError. This would have hit the queued 2-node jobs too. Fixes: (1) restored venv with `uv sync --extra fa` (flash_attn 2.7.4.post1 verified); (2) all launcher `uv run` calls now `uv run --no-sync` so a node can never re-sync/nuke the venv mid-job (committed+pushed). Resubmitted 1-node smoke as 277092 under bridges; monitor re-armed.

## 15:33 — Loop iteration 16

Resubmitted smoke 277092 PENDING (earlier instant backfill slot gone). Five 2-node effml jobs unchanged. Monitor armed for 277092's start.

## 16:33 — Loop iteration 17

All six jobs PENDING; no change.

## 17:33 — Loop iteration 18

All six jobs still PENDING; cluster saturated all day. No action.

## 18:33 — Loop iteration 19

1-node smoke 277092 now PENDING (Resources) — front of the schedulable set, starts with the next freed node. Others unchanged.

## ~19:20 — Smoke 277092 healthy; QDQ path verified end-to-end

Running on matrix13 since 18:36. Milestones: fake_quant patched 252 linears (exactly 36×7 ✓); first train step OK (loss −0.55, reward 0.58); vLLM hot-reloaded the step-1 QDQ checkpoint at 19:16. Verified the saved checkpoint on disk: 99.97% of down_proj elements exactly on the NVFP4 grid (0.03% differ by ≤1 scale-ULP from a CPU re-check — GPU/CPU float8 tie rounding; trainer↔sampler contract unaffected since both apply QDQ on GPU). Sampler is now serving NVFP4-grid weights. Remaining: eval at step 20, completion at 60.

## 19:33 — Loop iteration 20

Smoke 277092: trainer step 6/60 (~2.5 min/step; ETA ~22:00, fits 5h limit). QDQ ckpt save→vLLM reload cycling every step. 276522 (reference arm) now PENDING (Resources) — front of the 2-node line. Plan: after smoke passes step-60, scancel redundant 2-node smoke 276517 and submit the gated arms (dose sweep first).

## 20:33 — Loop iteration 21: REFERENCE ARM RUNNING, offsets mechanism verified live

276522 (homog10 reference) running on matrix[20,48] since ~19:35: staleness-offsets code working exactly as designed ("Staleness offset 10 (dp_rank 0): reloading checkpoint step 1 at inference step 11"), zero tracebacks, trainer at step 22. Pace ~2.6 min/step → may truncate near step 550 of 720 at the 24h limit (fine; collapse region is 300–500). 276523 (decisive arm) now Resources — next 2-node opening. Smoke 277092 at step 15/60. Monitors on both running jobs.

## 21:33 — Loop iteration 22: DECISIVE ARM RUNNING, heterogeneity verified in data

Now RUNNING: 277092 smoke (step 24/60), 276522 reference (step 52), 276523 decisive arm (step 10), 276517 2-node smoke (redundant but validates multinode path; keeping — 7 nodes total ≤ 12). Verified in the decisive arm's parquets: trainer batch at step_10 mixes gen ckpt_steps {9,6,0,0} = designed offsets with offset-32 burn-in clamp. Zero errors anywhere. β=0 controls (276520/21) still pending. Smoke step-20 eval ran; scores go to wandb.

## 22:33 — Loop iteration 23: ALL SIX JOBS RUNNING (11/12 nodes)

Every queued job is now on nodes: 2-node smoke (step 24, on pace to finish 60 ~23:30), 1-node smoke (step 33 — slowed by half-node evals, will truncate ~step 40 at its 23:31 limit; harmless, all its validation goals already met), reference arm (step ~80+), decisive arm, both β=0 controls (step 14 each, clean). Zero tracebacks anywhere. Gated arms (dose sweep first: w015 → w0045 → w045, then FP4 arms) will be submitted as the smokes end and free nodes, keeping ≤12.

## 23:33 — Loop iteration 24: bf16_b0 OOM root-caused, fixed, resubmitted (278069)

276520 (bf16_b0) was OOM-killed at ~step 20: it was the only arm with recompute_logprobs on at async_level=32 → the trainer retains async_level+1 CPU policy snapshots (~33×8GB ≈ 264GB) and blew host RAM. Fixed by switching the control to vllm_logprobs=true (also removes the anchor-source inconsistency the design review flagged); committed+pushed; resubmitted as 278069 under bridges. General rule recorded: recompute_logprobs + large async_level is a RAM bomb. Elsewhere: 2-node smoke step 42/60 (~45 min out), 1-node smoke expiring at its 5h limit now (validation complete), other three arms healthy.

## 00:31 — Event-driven check (cron delayed): smoke gate passed, dose sweep submitted

2-node smoke at step 59/60, zero errors end to end (1-node twin TIMEOUTed at 5h/step 40 as expected — benign). Gate cleared → submitted dose-sweep arms w015 (278088) and w0045 (278089) under bridges, 24h each; they take freed nodes as the smokes exit. w045 + FP4 arms staged for later slots (cap ≤12). Running: reference (step ~140 evals), decisive, hetero_b0; bf16_b0 resubmit (278069) pending.

## 00:33 — Loop iteration 25

No change from the 00:31 event-driven check: smoke finishing step 60, three arms healthy, three jobs pending (bf16 resubmit + two dose arms).

## 01:33 — Loop iteration 26: 2-NODE SMOKE COMPLETED CLEAN (exit 0, all 60 steps)

Full Phase-A FP4 machinery now validated on both single-node and multinode paths. Running: reference (~step 170), decisive, hetero_b0 (6 nodes). Pending: bf16_b0 resubmit + w015 + w0045 (6 more nodes; worst-case exactly at the 12 cap, so w045/FP4 arms wait).

## 02:33 — Loop iteration 27

Three arms running clean: hetero_b0 step 107, decisive step 126, reference ~205 (reward 0.75 at step 200). Three jobs still pending — the smoke's freed nodes went to other users; our bridges-priority jobs wait. No action.

## 03:33 — Loop iteration 28

No change: three arms running clean, three pending.

## 04:33 — Loop iteration 29

No change: three arms running (reference ~step 260, decisive ~150s, hetero_b0 ~130s), three pending.

## 05:34 — Loop iteration 30

No change: three arms running clean, three pending.

## 06:33 — Loop iteration 31

All healthy: reference ~step 310, decisive step 224, hetero_b0 step 204 — all past the offset-32 burn-in, so decisive/hetero_b0 batches now span the full Δ∈{1,4,10,32} range. Three jobs still pending.

## 07:33 — Loop iteration 32

No change: three arms running clean, three pending. Brian should be up soon — morning summary is this file top-to-bottom; headline: all Phase-A machinery validated, reference/decisive/hetero-control arms are past the burn-in and producing the comparison data.

## 08:33 — Loop iteration 33: hetero_b0 TIMEOUT at 10h (step 236, clean) — resubmitted 24h as 278160

The β=0 hetero control hit its 10h limit (set back when the 9–5/4-node cap plan existed). 236 clean steps of partial data are on wandb; resubmitted with 24h (278160). Its wandb name will collide-append; treat the new run as the canonical one. Two arms still running (reference ~step 370, decisive ~270); four jobs pending = worst case exactly at the 12-node cap.

## 09:33 — Loop iteration 34

Reference (step ~405, reward 0.856 at 400) and decisive (~290) running clean; four jobs pending.

## 10:33 — Loop iteration 35

No change: two arms running, four pending.

## 11:33 — Loop iteration 36

Reference ~step 465, decisive step 348 — both clean and past the collapse-prone region so far. Four pending.

## 12:33 — Loop iteration 37

No change: reference ~490, decisive ~380, both clean; four pending. Also unified vllm_logprobs across all fp4sym β=0 configs per Brian's question about the recompute path (commit pushed).

## 13:33 — Loop iteration 38

No change: reference step ~505 (reward 0.80 at 500), decisive ~400; four pending.

## 14:33 — Loop iteration 39

No change: two arms running clean (reference ~545, decisive ~430), four pending.

## 15:33 — Loop iteration 40

No change. Reference (20h elapsed) will TIMEOUT ~19:35 around step ~650; decisive ~21:00 around step ~600 — both well past the informative region. Four pending.

## 16:33 — Loop iteration 41

No change: reference step 600+ (reward 0.79), decisive ~470; four pending.

## 17:33 — Loop iteration 42

No change. Reference times out ~19:35 (will reach ~step 650); its freed nodes should flow to our pending jobs (bridges priority did well earlier).

## 18:33 — Loop iteration 43

No change; reference ~1h from timeout.

## 19:33 — Loop iteration 44

Reference at 23:58 elapsed — times out in ~2 min at ~step 685 (nearly full run, ended clean through eval 680). Decisive at ~22.5h continues to ~21:00. Reference's 2 nodes should flow to a pending job shortly.

## 20:33 — Loop iteration 45: REFERENCE ARM DONE (TIMEOUT at step 688/720, clean)

Near-complete reference run in the books — evals every 20 steps through 680 on wandb. Decisive arm times out ~21:00 at ~step 615. Four jobs still pending; freed nodes went to other users this round.

## 21:33 — Loop iteration 46: BOTH MARQUEE ARMS COMPLETE — first results

Decisive arm TIMEOUT at step 561 (clean). Eval curves (wandb):
- Reference (homog Δ=10): peak 0.829 @ 640, final 0.828 @ 680 — REPRODUCES the historical noCenter baseline (0.827–0.831) under the new mechanism+anchor.
- Decisive (hetero {1,4,10,32}, per-rollout c_i): peak 0.816 @ 520; at matched step 520 it reads 0.816 vs reference 0.814 — point-for-point tracking despite 32× staleness spread.
Calibrated-hetero ≈ homogeneous confirmed so far. Pending arms (uniform doses w015/w0045, hetero_b0, bf16_b0) decide whether miscalibration hurts — the other half of the claim.

## 22:33 — Loop iteration 47

Nothing running; four control/dose arms pending. Cluster reabsorbed the freed nodes.

## 23:33 — Loop iteration 48

No change: four pending, nothing running.

## 00:33 — Loop iteration 49

No change: four pending.

## 01:33 — Loop iteration 50

No change: four pending.

## 02:33 — Loop iteration 51

No change: four pending.

## 03:33 — Loop iteration 52

No change: four pending.

## 04:33 — Loop iteration 53

No change: four pending.

## 05:33 — Loop iteration 54

No change: four pending.

## 06:33 — Loop iteration 55

No change: four pending (queued ~30h for the oldest). Cluster has been saturated since Sunday.

## 07:33 — Loop iteration 56

No change: four pending.

## 08:33 — Loop iteration 57

No change: four pending.

## 09:33 — Loop iteration 58

No change: four pending.

## 10:33 — Loop iteration 59

No change: four pending.

## 11:33 — Loop iteration 60

No change: four pending.

## 12:33 — Loop iteration 61

No change: four pending.

## 13:33 — Loop iteration 62

No change: four pending.

## 14:33 — Loop iteration 63

No change: four pending.

## 15:33 — Loop iteration 64

No change: four pending.

## 16:33 — Loop iteration 65

No change: four pending (~40h for oldest). If this persists into tomorrow, consider asking Brian about a reservation or DAT.

## 17:33 — Loop iteration 66

No change: four pending.

## 18:33 — Loop iteration 67

No change: four pending.

## 19:33 — Loop iteration 68

No change: four pending.

## 20:33 — Loop iteration 69

No change: four pending.

## 21:33 — Loop iteration 70

No change: four pending.

## 22:33 — Loop iteration 71

No change: four pending.

## 23:34 — Loop iteration 72

No change: four pending.

## 00:33 — Loop iteration 73

No change: four pending. (Interim report results_fp4.pdf built, committed, pushed at ~23:45; delivered to Brian.)

## 01:33 — Loop iteration 74

No change: four pending.

## 02:33 — Loop iteration 75

No change: four pending.

## 03:33 — Loop iteration 76

No change: four pending.

## 04:33 — Loop iteration 77

No change: four pending.

## 05:33 — Loop iteration 78

No change: four pending.

## 06:33 — Loop iteration 79

No change: four pending (~54h queued for the oldest).

## 07:33 — Loop iteration 80

No change: four pending.

## 08:33 — Loop iteration 81

No change: four pending.

## 09:34 — Loop iteration 82

No change: four pending (ranks ~127–130; queue deepened). Report confirmed current — no new data to plot.

## ~10:20 Jul 16 — Migrated four pending arms to ml4ss bank (Brian-approved after test)

ml4ss verified: 1-node AND 2-node probes ran to completion; association unrestricted. Cancelled bridges jobs 278069/278088/278089/278160 (never started, no data loss) and resubmitted as 280129 (bf16_b0), 280130 (hetero_b0), 280131 (w015), 280132 (w0045) — priority 187,116 vs ~168k on bridges; 280129 already front-of-line (Resources). Expect starts as nodes free.

## 10:33 — Loop iteration 83

Post-migration state: all four arms on ml4ss at 187k priority; bf16_b0 front-of-line (Resources), others right behind. Waiting on node availability.

## 11:33 — Loop iteration 84

No change: four ml4ss arms at front of queue, waiting on 2-node openings.

## 12:33 — Loop iteration 85: bf16_b0 RUNNING (ml4ss migration worked)

280129 started ~11:39 on matrix[11,40]; step 21+, zero errors, trainer RAM 40/503GB — OOM fix confirmed. hetero_b0 (280130) now front-of-line. Monitor armed.

## 13:33 — Loop iteration 86

bf16_b0 at ~2h (step ~60), clean; three arms still queued at front.

## 14:33 — Loop iteration 87

bf16_b0 ~3h in, clean; three queued.

## 15:33 — Loop iteration 88

bf16_b0 step ~105 (reward 0.756 at 100), clean; three queued.

## 16:33 — Loop iteration 89

bf16_b0 ~5h (step ~140), clean; three queued.

## 17:33 — Loop iteration 90: ALL FOUR ml4ss ARMS RUNNING

bf16_b0 ~step 160; hetero_b0, w015, w0045 all started ~17:03, step 8 each, zero errors, offsets cycling. 8/12 nodes. Full dose-sweep + controls now in flight — the calibration comparison completes when these finish (~17:00 tomorrow). Consolidated monitor armed.

## 18:35 — COLLAPSE on bf16_b0 (staleness-only control)

BF16, homogeneous Δ=32, β=0: peak reward 0.756 @ step 100, slow decline to 0.684 @ 160, cliff to 0.188 @ 180 and 0.148 @ 200. First observed collapse — staleness alone (no quantization) breaks training at high async without KL. Run left up: the trajectory is the data. Pushed to Brian. The KL arms (dose sweep, calibrated) now decide the rescue half.

## 18:34 — Loop iteration 91

All four running. bf16_b0 collapsed (0.148 @ 200, see 18:35 entry); hetero_b0/w015/w0045 ~1.5h in, clean.

## 19:33 — Loop iteration 92

All four running, no new events.

## 20:33 — Loop iteration 93

All four running. hetero_b0 at 0.724 @ 80 (healthy so far); w015 at 0.634 @ 77 (slower learning under the heavier uniform dose, as expected from the historical β=0.05 pattern).

## 21:33 — Loop iteration 94

bf16_b0 fully collapsed and staying down (0.072 @ 266). The three hetero arms all past step 100 and healthy: hetero_b0 0.793, w0045 0.762, w015 0.739. Watching the ~170 mark where bf16_b0 fell.

## 22:33 — Loop iteration 95

hetero_b0 (β=0) spiking: 0.79 @ 101 → 0.94 @ 127 — unusually high train reward; the bf16_b0 collapse was preceded by a similar (smaller) overshoot. Dose arms steady ~0.73. Entering the danger zone; watching evals.

## 23:33 — Loop iteration 96

hetero_b0 came back down from its 0.94 spike (0.725 @ 152 — volatile but not collapsed); dose arms steady (w015 0.779, w0045 0.754). All in the 150-200 window now.

## 00:33 — Loop iteration 97

hetero_b0 sliding: 0.94 @ 127 → 0.725 @ 152 → 0.631 @ 176 — downtrend through the danger zone. Dose arms firm (w0045 0.785, w015 0.715). If the slide continues this is the β=0-hetero failure emerging with KL arms unaffected.

## 01:33 — Loop iteration 98

At ~step 200: hetero_b0 0.576 (eroding), w015 0.717, w0045 0.670. The β=0 arm now clearly below both KL arms at matched depth — the separation the rescue claim needs, developing in real time.

## 02:33 — Loop iteration 99

~step 220: hetero_b0 0.553 (still eroding), w015 0.716, w0045 0.803. Gap widening.

## 03:33 — Loop iteration 100: SECOND COLLAPSE — hetero_b0 down, KL arms untouched

hetero_b0 (mixed Δ, β=0) collapsed: 0.553 @ 221 → 0.203 @ 241 — same cliff as bf16_b0 but ~60 steps later (mixed staleness delayed, not prevented, the failure). At the SAME steps the KL'd dose arms read 0.715 (w015) and 0.797 (w0045). Both β=0 controls have now collapsed while every KL-regularized arm (calibrated, w015, w0045, reference) remains stable — the rescue claim's control pair is complete.

## 04:33 — Loop iteration 101

hetero_b0 fully collapsed (0.031 @ 261); dose arms both at 0.80 @ ~265. Cleanest possible contrast.

## 05:33 — Loop iteration 102

All four running; no new events.

## 06:33 — Loop iteration 103

Attribution: the 0.59@300 was a transient (w015 dipped then recovered to 0.785 @ 312). Current: hetero_b0 dead at 0.004 @ 295; w015 0.785 @ 312; w0045 0.793 @ 320. Dose arms sailing.

## 07:33 — Loop iteration 104

All four running; no new events. bf16_b0 ends ~11:39, others ~17:03.

## 08:33 — Loop iteration 105

All four running; no new events.

## 09:33 — Loop iteration 106

All four running; no new events.

## 10:33 — Loop iteration 107

All four running; dose arms both past 400 at 0.72-0.76. bf16_b0 ends ~11:39.

## 11:33 — Loop iteration 108

bf16_b0 6 min from its 24h limit; others ~5.5h out. No new events.

## 12:33 — Loop iteration 109

bf16_b0 finished (24h TIMEOUT, ~step 640, full collapse trajectory recorded). Three hetero arms continue (~19.5h in). No issues.

## 13:33 — Loop iteration 110

Three hetero arms in their final hours; no events.

## 14:33 — Loop iteration 111

Three arms at ~21.5h; dose arms both strong at 500 (0.820, 0.781). ~2.5h to their limits.

## 15:33 — Loop iteration 112

Final ~1.5h for the three hetero arms. No events.

## 16:33 — Loop iteration 113

~30 min to the three arms' limits. Report refresh queued for when they end.

## 17:33 — Loop iteration 114: PHASE-A COMPLETE — final report built and pushed (6bdde20)

All three hetero arms TIMEOUT clean (b0 @ 491, w015 @ 568, w0045 @ 581). Final evals: calibrated 0.816 peak, w0045 0.817, w015 0.801, reference 0.829; both β=0 arms collapsed and stayed down (0.222 / 0.340 final). Report rebuilt with collapse/rescue + dose-calibration figures, delivered to Brian, pushed. Next queue batch: FP4 β=0 ladder + rescue arms.

## 18:33 — Loop iteration 115: FP4 batch submitted (Phase-A part 2)

Six FP4 arms queued under ml4ss (281370–75): β=0 ladder async {1,10,32} — does quantization shift the collapse boundary? — plus rescue arms (β·c=0.0045 at Δ=10 and Δ=32) and the asymmetric soft-QAT arm. 12 nodes exactly at cap. Monitors to be armed as they start.

## 19:33 — Loop iteration 116

FP4 batch (6 jobs) pending; ml4ss priority should move them quickly as nodes free.

## 20:33 — Loop iteration 117

FP4 batch still pending; no change.

## 21:33 — Loop iteration 118: first FP4 arm running

281370 (fp4sym_b0_async1 — near-on-policy FP4, KL off) running since ~20:42, step 11, clean. This is Act 1 of the FP4 story: stable on-policy FP4. 281371 (async10) front-of-line. Monitor armed.

## 22:33 — Loop iteration 119

fp4sym_b0_async1 ~2h in, clean; five queued.

## 23:33 — Loop iteration 120

fp4sym_b0_async1 ~3h in, clean; five queued. Report refreshed+pushed via subagent (f23d13f).

## 00:33 — Loop iteration 121

fp4sym_b0_async10 started ~23:49 (second FP4 ladder rung); async1 ~4h clean. Four queued.

## 01:33 — Loop iteration 122

Two FP4 arms running clean, four queued.

## 02:33 — Loop iteration 123

No change: two running, four queued.

## 03:33 — Loop iteration 124

Five FP4 arms running (10 nodes): async1 β=0 (0.848 @ 100!), async10 β=0, async32 β=0 (the collapse probe, step 11), both rescue arms just started. Asym arm queued. Monitor armed on the three new arms.

## 03:50 — Rescue arm 281374 crashed at startup; resubmitted as (see above)

Inference died in vLLM init with a transformers registry error (Glm4vConfig not found) → trainer TimeoutError on step 0 → job exited. Other 4 arms from the same venv/hour are fine → transient (likely HF-cache contention between concurrent vLLM startups). Resubmitted; if it recurs, will pin/inspect transformers in the venv.

## 04:33 — Loop iteration 125

Five FP4 arms running incl. the asymmetric soft-QAT arm (step 13, correctly NO fake_quant on trainer — BF16 trainer + QDQ sampler only). Resubmitted rescue arm 281576 front-of-line. All clean.

## 05:33 — Loop iteration 126

Five running, rescue resubmit pending. No events.

## 06:33 — Loop iteration 127

All six FP4 arms running (12/12 nodes). Rescue resubmit 281576 cleared startup cleanly (transient confirmed). async32 β=0 collapse probe at step 65, reward 0.74 — approaching the BF16 cliff zone (~170).

## 07:33 — Loop iteration 128

All six running clean.

## 08:00 — fp4asym arm degrading monotonically (FINDING, not bug)

Asym arm (BF16 trainer + QDQ sampler, KL on): reward 0.51@20 → 0.10@100, loss ~20× normal magnitude. Not the staleness collapse pattern — immediate systematic bias. Mechanism: the surrogate's persistent quantization gap enters the advantage-level penalty un-centered → acts as a global negative reward (suppression), not soft-QAT pressure. Score-function path can't deliver the QAT gradient; would need the differentiable (KL-in-loss) path or gap centering. Symmetric arms immune (bit-exact gap=0). Run left up to document the flatline — this is the honest answer to the "provocative QAT reading": at the advantage level, it fails, informatively.

## 08:33 — Loop iteration 129: FP4 SHIFTS THE COLLAPSE BOUNDARY EARLIER

fp4sym_b0_async32 (FP4 + Δ=32, no KL): peaked 0.68 @ 60, collapsing by 100 (0.38) — vs BF16 twin which peaked 0.756 @ 100 and collapsed ~170. Quantization + staleness COMPOUND: the collapse boundary moves ~70 steps earlier and from a lower peak. This is the humans& simulator's central assertion, now measured in a controlled 2×2. Rescue arm at the same Δ (281576, KL on) is the critical comparison — if it stays stable, the surrogate rescues the compounded failure too. Other arms healthy (async1 0.85@100, async10 0.74@100, asym degrading as analyzed).

## 09:33 — Loop iteration 130

All six running. FP4 Δ=32 rescue arm at 0.678 @ 59 — right at the depth where its β=0 twin peaked (0.68 @ 60) before collapsing. The next ~50 steps are the decisive window.

## 10:33 — Loop iteration 131

Watch item: FP4 Δ=32 rescue arm dipped 0.678 @ 59 → 0.561 @ 77 (could be batch noise — other arms show similar swings — or early insufficiency of the BF16-derived dose against the compounded mismatch). β=0 twin fully collapsing (0.177 @ 136). Next hour decides.

## 11:33 — Loop iteration 132

Rescue arm oscillating (0.64 @ 60, 0.53 @ 80) — no cliff yet but slower/noisier than the Δ=10 rescue arm at this depth. β=0 twin at ~0.18. Verdict needs ~step 150+.

## 11:45 — fp4sym_b0_async1 died at step 242: torn parquet read (infra race)

OSError "Couldn't deserialize thrift" in the dataloader — reader/writer race on /p/vast1, most likely at async_level=1 where the trainer consumes files immediately. NOT resubmitting: 242 clean steps at 0.80–0.85 fully establishes "on-policy FP4 is stable without KL" (Act 1). Roadmap note: dataloader should retry corrupt parquet reads (small fix, matters for async-1 configs). Five arms continue.

## 12:33 — Loop iteration 133

Five arms running. FP4 Δ=32 rescue at 0.604 @ 100 — recovered from the 80-dip, no cliff; its β=0 twin was at 0.38 by 100. Rescue holding so far, at reduced altitude vs Δ=10 arms.

## 12:36 — fp4asym flatlined at 0 (step 200) — cancelled to free nodes

Monotone 0.51 → 0.10 → 0.0015 over 200 steps confirms the mechanism analysis (persistent quantization gap through the score-function penalty = global suppression). Negative result fully documented; freed 2 nodes rather than run a dead arm 15 more hours.

## 13:33 — Loop iteration 134

Four arms running. Δ=32 rescue stable ~0.60 through 120 (twin died by here). Holding, slower climb than Δ=10 arms.

## 14:33 — Loop iteration 135

Four arms running clean; no events.

## 15:33 — Loop iteration 136

Four arms running clean; no events.

## 16:33 — Loop iteration 137

Δ=32 rescue climbing again (0.647 @ 180 — twin died at ~90); Δ=10 rescue 0.741 @ 229; Δ=10 β=0 drifting down slightly (0.678 @ 291 vs 0.744 @ 200) — watching for a late FP4+Δ10 slide; Δ=32 β=0 stays collapsed (0.28).

## 17:33 — Loop iteration 138

Δ=10 FP4 β=0 arm oscillating high (0.79–0.89 over 270–300) — overshoot-signature watch continues. Other three arms nominal.

## 18:33 — Loop iteration 139

Four arms running; no events.

## 19:33 — Loop iteration 140

Δ=10 β=0 arm still oscillating 0.76–0.89 (no cliff through 340). Others nominal.

## 20:33 — Loop iteration 141

Four arms running; no events. First two hit 24h limits ~23:50 and ~02:40.

## 21:33 — Loop iteration 142

Four arms running; no events.

## 22:33 — Loop iteration 143

Four arms running; no events.

## 23:33 — Loop iteration 144

Four arms running; Δ=10 β=0 survived to 400 (0.826) — FP4 no-KL boundary is between Δ=10 and 32. First timeout ~23:50.

## 00:33 — Loop iteration 145

Δ=10 β=0 rung finished (24h TIMEOUT, alive at ~step 420, 0.83). Three arms remain: Δ=32 β=0 (collapsed, documenting), Δ=10 rescue, Δ=32 rescue.

## 01:33 — Loop iteration 146

Three arms running; no events.

## 02:33 — Loop iteration 147

Δ=32 β=0 and Δ=10 rescue minutes from their 24h limits; Δ=32 rescue has ~3.5h left.

## 03:33 — Loop iteration 148

Δ=32 β=0 and Δ=10 rescue finished (24h TIMEOUTs, clean). Only the Δ=32 rescue remains (~2.5h left). Final FP4 report refresh once it ends.

## 04:33 — Loop iteration 149

Only the Δ=32 rescue running (~1.5h left).

## 05:33 — Loop iteration 150

Δ=32 rescue ~26 min from its limit. Next iteration: full FP4 report refresh + push.

## 06:33 — Loop iteration 151: ALL FP4 ARMS COMPLETE — final report update in progress

Δ=32 rescue finished its full 24h (no collapse — same-dose rescue confirmed under compounded FP4+staleness). Queue empty; every Phase-A arm has run. Report subagent launched: new FP4 figure, findings 5 (compounding + rescue) and 6 (asymmetric negative result), updated abstract/tables/next-steps; will commit+push.

## 06:45 — FINAL REPORT PUSHED (1b8dffd, 5 pages) — Phase A closed

Correction surfaced during the final data pull: the Δ=32 FP4 rescue arm degraded late (0.56–0.71 through ~320, then 0.16 @ 400) — the unscaled dose DELAYS the compounded failure ~3× past the β=0 boundary but doesn't fully prevent it. Report states this honestly; quantization-aware dose scaling promoted to the top next experiment. Final eval summary: reference 0.829; calibrated 0.816; w0045 0.817; w015 0.801; both BF16/hetero β=0 collapsed; FP4 β=0: Δ1 0.79 (242 steps), Δ10 0.81 (survived), Δ32 collapsed ~100; FP4 rescue: Δ10 0.784 stable, Δ32 delayed-then-degraded; asym flatlined (mechanism documented). All arms done, queue empty, everything committed+pushed. PDF delivered to Brian.

## 07:33 — Loop iteration 152

Queue empty; Phase A closed, report delivered (1b8dffd). Nothing to babysit.

## 08:33 — Loop iteration 153 (FINAL): loop retired

Queue empty for the third consecutive hour; every Phase-A objective in the loop's mandate is complete (all arms run, findings documented, reports pushed through 1b8dffd). Stopping the hourly cron — restart with /loop when the next batch (seeds, dose-scaling, MATH) is ready.

=== END OF PHASE-A OPERATIONS LOG ===

=== OPERATIONS LOG REOPENED: fp4hetero batch (the original hypothesis cell) ===

## Jul 19 ~09:10 — Submitted the staleness-weighted-KL × FP4 test

Brian's call: the original idea (FP4 instability fixed by staleness-WEIGHTED KL, not just KL) was never directly tested — the review's sequencing deferred the factorial. Now sharpened by the uniform-dose late failure at compounded Δ=32. Three arms, all symmetric NVFP4 + offsets {1,4,10,32}: 282348 β=0 control, 282349 uniform w=0.0045, 282350 per-rollout calibrated (decisive). Configs committed+pushed; hourly loop restarted (eddc0e4f).

## 19:51 — fp4hetero iteration 1 (queue cleared this evening)

β=0 control running (step 6, offsets cycling, clean), uniform arm just started, decisive perRollout arm front-of-line. Monitor armed on the two runners.

## 20:51 — fp4hetero iteration 2

Control + uniform running clean; decisive arm still awaiting nodes (Resources).

## 21:51 — fp4hetero iteration 3: ALL THREE ARMS RUNNING

Decisive perRollout arm started 20:55 (step 12, clean). Full comparison in flight; verdicts land through tomorrow evening.

## 22:51 — fp4hetero iteration 4

All three climbing normally: b0 0.581@54, uniform 0.648@44, perRollout 0.507@27. No events.

## 23:51 — fp4hetero iteration 5

All three running; no events.

## 00:51 — fp4hetero iteration 6

All three running; no events.

## 01:51 — fp4hetero iteration 7

β=0 riding high (0.836@101 — familiar pre-collapse pattern), uniform 0.672@89, perRollout 0.580@75. Watching the 150–250 window for the control's collapse.

## 02:51 — fp4hetero iteration 8

All three running; b0 ~0.69–0.84 around step 100 (volatile), KL arms steady. No events.

## 03:51 — fp4hetero iteration 9

Control still climbing at 120 (0.729); collapse window opens shortly. KL arms steady. No events.

## 04:51 — fp4hetero iteration 10

Control at 0.747@140, still pre-collapse. All quiet.

## 05:51 — fp4hetero iteration 11

Control turning: 0.747@140 → 0.555@165 (slide begun, on schedule). KL arms above it: uniform 0.768@150, perRollout 0.697@139.

## 06:51 — fp4hetero iteration 12

Control wobbling (0.62@160, 0.67@180) — sliding but no cliff yet. KL arms steady. No events.

## 07:51 — fp4hetero iteration 13

No stall (trainer at ~199, normal pace); control showing KV-cache preemptions (lengthening generations — early degradation symptom). KL arms nominal.

## 08:51 — fp4hetero iteration 14: CONTROL COLLAPSED (~step 205)

β=0: 0.51@200 → 0.26@212 — cliff on schedule (FP4-hetero boundary ~205, between FP4-homog-32's ~100 and BF16-hetero's ~230, consistent with intermediate effective mismatch). Uniform 0.754@196 and perRollout 0.647@186 both unaffected. Now the decisive late window: does perRollout outlast uniform past ~340?

## 09:51 — fp4hetero iteration 15

perRollout 0.807@200 (leading!), uniform 0.735@200, control collapsed. All running.

## 10:51 — fp4hetero iteration 16

Both KL arms dipped in the 200–226 window (uniform 0.486@226, perRollout 0.638@219) — volatility or the start of uniform's late failure; watching.

## 11:51 — fp4hetero iteration 17

Dips were batch noise: uniform back to 0.770@240; perRollout 0.838@220 — consistently ABOVE uniform at matched steps (unlike BF16 where they tied). Both healthy deep past the control's collapse.

## 12:51 — fp4hetero iteration 18

All three running; no events.

## 13:51 — fp4hetero iteration 19

Both KL arms dipped together at 260 (0.66 each — correlated batch noise); perRollout led at 240 (0.842 vs 0.770). Entering the uniform-failure watch window (~300+).

## 14:51 — fp4hetero iteration 20

Uniform bounced (0.815@280); perRollout soft (0.641@280) — lead trading on batch noise. Both healthy. Real discrimination: 340+ window and final evals.

## 15:51 — fp4hetero iteration 21

All running; uniform entered the late window at 0.662@300. No events.

## 16:51 — fp4hetero iteration 22

Both KL arms ~0.63–0.66 at 300, level. Late window in progress.

## 17:15 — fp4hetero_b0 ended at step 327 (torn parquet read, ≥300 → no resubmit)

Control's data complete: collapse ~205, no recovery through 327. Uniform (320) and perRollout arms continue.

## 17:51 — fp4hetero iteration 23

At 320: perRollout 0.801, uniform 0.713 — both healthy in the late window, perRollout ahead.

## 18:51 — fp4hetero iteration 24

Both through 340 healthy (uniform 0.776, perRollout 0.699) — past the homogeneous failure onset with no degradation in either.

## 19:51 — fp4hetero iteration 25

Uniform arm finished (24h TIMEOUT, healthy to the end). Only the decisive perRollout arm remains (~1h left, at ~360).

## 20:55 — fp4hetero BATCH COMPLETE — final report update launched

Decisive perRollout arm ended healthy at step 381 (24h TIMEOUT, zero errors); uniform ended healthy at 361; control collapsed ~205. Both KL arms survived to horizon where β=0 died — the rescue half of the original hypothesis holds under FP4 heterogeneity. Eval-curve verdict on per-rollout vs uniform being computed by the report subagent (train-reward suggested a per-rollout lead; evals decide).
