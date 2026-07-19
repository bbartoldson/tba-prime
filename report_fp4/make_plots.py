"""Figures for the fp4_KL Phase-A report (final Phase-A data).

Data cached in fp4_report_data.json (pulled from wandb). Palette: dataviz
reference instance, validated; categorical slots in fixed order per chart;
sequential blue ramp for the ordered Δ strata."""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "fp4_report_data.json"))

C1, C2, C3, C4 = "#2a78d6", "#1baf7a", "#eda100", "#4a3aa7"  # validated categorical slots
SEQ = ["#b7d3f6", "#6da7ec", "#2a78d6", "#184f95"]
TEXT, MUTED, GRID = "#1a1a19", "#6b6a62", "#e6e5df"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
    "text.color": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})


def line(ax, series, color, label, dashed=False):
    xs, ys = zip(*series)
    ax.plot(xs, ys, color=color, lw=2, label=label, ls="--" if dashed else "-",
            solid_capstyle="round")
    ax.annotate(f"{ys[-1]:.2f}", (xs[-1], ys[-1]), xytext=(4, 0),
                textcoords="offset points", color=color, fontsize=8,
                fontweight="bold", va="center")


def base(ax):
    ax.set_xlabel("training step")
    ax.set_ylabel("Countdown eval accuracy")
    ax.set_ylim(0.0, 0.9)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(loc="lower left", frameon=False, fontsize=8)


# ---- Figure 1: collapse without KL, rescue with it -----------------------
fig, ax = plt.subplots(figsize=(6.4, 3.5), dpi=200)
line(ax, DATA["homog10_ref"], C1, "Homogeneous Δ=10, KL (reference)")
line(ax, DATA["global_w0045"], C2, "Heterogeneous, uniform KL w=0.0045")
line(ax, DATA["hetero_b0_full"], C3, "Heterogeneous, KL off (β=0)")
line(ax, DATA["bf16_b0"], C4, "Homogeneous Δ=32, KL off (β=0)")
base(ax)
fig.tight_layout()
fig.savefig("fig_collapse_rescue.pdf")

# ---- Figure 2: calibrated vs uniform doses (all stable arms) -------------
fig, ax = plt.subplots(figsize=(6.4, 3.5), dpi=200)
line(ax, DATA["homog10_ref"], C1, "Homogeneous Δ=10 (reference)")
line(ax, DATA["hetero_perRollout"], C2, "Hetero, per-rollout $c_i$ (calibrated)")
line(ax, DATA["global_w0045"], C3, "Hetero, uniform w=0.0045")
line(ax, DATA["global_w015"], C4, "Hetero, uniform w=0.015")
ax.set_ylim(0.55, 0.87)
ax.set_xlabel("training step")
ax.set_ylabel("Countdown eval accuracy")
ax.grid(axis="y", color=GRID, lw=0.6)
ax.legend(loc="lower right", frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig("fig_dose_calibration.pdf")

# ---- Figure 3: surrogate relative error by Δ stratum ----------------------
fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=200)
labels = {"d01_02": "Δ ∈ [1,2]", "d03_06": "Δ ∈ [3,6]", "d07_20": "Δ ∈ [7,20]", "d21_up": "Δ ≥ 21"}
for i, b in enumerate(("d01_02", "d03_06", "d07_20", "d21_up")):
    pts = DATA["strat"][f"kl_approx/{b}/rel_err_of_means"]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
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

# ---- Figure 4: FP4 ladder and rescue arms --------------------------------
fig, ax = plt.subplots(figsize=(6.4, 3.5), dpi=200)
line(ax, DATA["fp4sym_b0_async10"], C1, "FP4 Δ=10, KL off")
line(ax, DATA["fp4sym_b0_async32"], C2, "FP4 Δ=32, KL off")
line(ax, DATA["fp4sym_resc_async10"], C3, "FP4 Δ=10, KL on")
line(ax, DATA["fp4sym_resc_async32"], C4, "FP4 Δ=32, KL on")
base(ax)
fig.tight_layout()
fig.savefig("fig_fp4.pdf")
print("figures written")
