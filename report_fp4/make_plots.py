"""Figures for the fp4_KL Phase-A interim report.

Data is pulled from wandb by the report pipeline and cached as JSON
(fp4_report_data.json). Palette: dataviz reference instance, validated
(categorical slots for entity-series; sequential blue ramp for the ordered
Δ strata)."""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "fp4_report_data.json"))

# validated categorical slots (fixed order: entity identity)
C_REF = "#2a78d6"   # slot 1 blue  — homogeneous reference
C_CAL = "#1baf7a"   # slot 2 aqua  — calibrated heterogeneous
C_B0 = "#eda100"    # slot 3 yellow — heterogeneous, KL off
# sequential blue ramp (light→dark) for the ordered Δ strata
SEQ = ["#b7d3f6", "#6da7ec", "#2a78d6", "#184f95"]

TEXT = "#1a1a19"
MUTED = "#6b6a62"
GRID = "#e6e5df"

plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": TEXT,
    "text.color": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _line(ax, series, color, label, endlabel):
    xs, ys = zip(*series)
    ax.plot(xs, ys, color=color, lw=2, label=label, solid_capstyle="round")
    ax.annotate(endlabel, (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points",
                color=color, fontsize=8, fontweight="bold", va="center")


# ---- Figure 1: eval accuracy over training -------------------------------
fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
_line(ax, DATA["homog10_ref"], C_REF, "Homogeneous Δ=10 (reference)", "0.828")
_line(ax, DATA["hetero_perRollout"], C_CAL, "Heterogeneous Δ∈{1,4,10,32}, per-rollout $c_i$", "0.810")
_line(ax, DATA["hetero_b0_partial"], C_B0, "Heterogeneous, KL off (β=0, partial)", "0.749")
ax.set_xlabel("training step")
ax.set_ylabel("Countdown eval accuracy")
ax.set_ylim(0.55, 0.87)
ax.grid(axis="y", color=GRID, lw=0.6)
ax.legend(loc="lower right", frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig("fig_eval_curves.pdf")

# ---- Figure 2: surrogate relative error by Δ stratum ----------------------
fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=200)
labels = {"d01_02": "Δ ∈ [1,2]", "d03_06": "Δ ∈ [3,6]", "d07_20": "Δ ∈ [7,20]", "d21_up": "Δ ≥ 21"}
for i, b in enumerate(("d01_02", "d03_06", "d07_20", "d21_up")):
    pts = DATA["strat"][f"kl_approx/{b}/rel_err_of_means"]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    # light smoothing for readability (window median), raw data in appendix table
    k = 9
    sm = [sorted(ys[max(0, j - k):j + k + 1])[len(ys[max(0, j - k):j + k + 1]) // 2] for j in range(len(ys))]
    ax.plot(xs, sm, color=SEQ[i], lw=2, label=labels[b])
    ax.annotate(labels[b], (xs[-1], sm[-1]), xytext=(4, 0), textcoords="offset points",
                color=SEQ[i], fontsize=8, fontweight="bold", va="center")
ax.set_yscale("log")
ax.set_xlabel("training step (optimizer)")
ax.set_ylabel("rel. error of means\n(approx vs exact EMA-KL)")
ax.grid(axis="y", color=GRID, lw=0.6, which="both")
ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig("fig_delta_strata.pdf")
print("figures written")
