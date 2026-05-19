"""Generate cleaner figures for results.tex."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb


OUT = Path(__file__).parent
api = wandb.Api()
ALL_RUNS = list(api.runs("bartoldson/ema_KL", per_page=200))


def get_run(name: str):
    rs = [r for r in ALL_RUNS if r.name == name]
    if not rs:
        return None
    return max(rs, key=lambda r: r.summary.get("_step") or 0)


def eval_history(name: str):
    r = get_run(name)
    if r is None:
        return [], []
    rows = list(r.scan_history(keys=["_step", "eval/countdown/score"]))
    pts = [(row["_step"], row["eval/countdown/score"]) for row in rows
           if row.get("eval/countdown/score") is not None]
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]


def metric_history(train_name: str, key: str):
    r = get_run(train_name)
    if r is None:
        return [], []
    rows = list(r.scan_history(keys=["_step", key]))
    pts = [(row["_step"], row[key]) for row in rows if row.get(key) is not None]
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]


def fig_matrix():
    """Eval curves for the 2x2 EMA-reference matrix, replicate-shaded."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    # Each cell has two replicates (iter1 = base name; iter2 = base name + "_iter2").
    cells = [
        ("Exact KL, K-centered",
         "infer_Countdown_experiments_TBA_qwen3_klMeanTrain_emaRef_a09_b005_async10_eval20",
         "infer_Countdown_experiments_TBA_qwen3_klMeanTrain_emaRef_a09_b005_async10_eval20_iter2",
         "tab:red", "-"),
        ("Exact KL, no centering",
         "infer_Countdown_experiments_TBA_qwen3_noCenter_emaRef_a09_b005_async10_eval20",
         "infer_Countdown_experiments_TBA_qwen3_noCenter_emaRef_a09_b005_async10_eval20_iter2",
         "tab:red", "--"),
        ("Approx KL, K-centered",
         "infer_Countdown_experiments_TBA_qwen3_klMeanTrain_emaRef_a09_approxUse_b005_async10_eval20",
         "infer_Countdown_experiments_TBA_qwen3_klMeanTrain_emaRef_a09_approxUse_b005_async10_eval20_iter2",
         "tab:blue", "-"),
        ("Approx KL, no centering",
         "infer_Countdown_experiments_TBA_qwen3_noCenterByAlias_emaRef_a09_approxUse_b005_async10_eval20",
         "infer_Countdown_experiments_TBA_qwen3_noCenter_emaRef_a09_approxUse_b005_async10_eval20_iter2",
         "tab:blue", "--"),
    ]
    for label, n1, n2, color, ls in cells:
        x1, y1 = eval_history(n1)
        x2, y2 = eval_history(n2)
        d1 = dict(zip(x1, y1))
        d2 = dict(zip(x2, y2))
        steps = sorted(set(x1) | set(x2))
        if not steps:
            continue
        means, lo, hi = [], [], []
        for s in steps:
            vals = [v for v in (d1.get(s), d2.get(s)) if v is not None]
            means.append(sum(vals) / len(vals))
            lo.append(min(vals))
            hi.append(max(vals))
        ax.plot(steps, means, label=label, color=color, linestyle=ls, linewidth=1.7)
        ax.fill_between(steps, lo, hi, color=color, alpha=0.15, linewidth=0)
    ax.set_xlabel("training step")
    ax.set_ylabel("Countdown eval accuracy (pass@1)")
    ax.set_title("EMA reference: $2\\times 2$ matrix of \\{exact, approx\\} KL $\\times$ \\{K-centered, no centering\\}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out = OUT / "fig_matrix.pdf"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def _plot_reset(series, title, out_name):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, run_name, color in series:
        x, y = eval_history(run_name)
        if not x:
            continue
        ax.plot(x, y, label=label, color=color, linewidth=1.6)
    ax.set_xlabel("training step")
    ax.set_ylabel("Countdown eval accuracy (pass@1)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(-0.02, 0.9)
    fig.tight_layout()
    out = OUT / out_name
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def fig_reset_is():
    series = [
        ("$T/I$", "infer_Countdown_experiments_TBA_qwen3_klMeanInference_b005_async10_eval20", "tab:blue"),
        ("$T/T$", "infer_Countdown_experiments_TBA_qwen3_klMeanTrain_b005_async10_eval20",     "tab:red"),
        ("$I/I$", "infer_Countdown_experiments_TBA_qwen3_klAllInf_b005_async10_eval20",        "tab:green"),
    ]
    _plot_reset(series, "Periodic-reset reference, IS on", "fig_reset_is.pdf")


def fig_reset_nois():
    series = [
        ("$T/I$", "infer_Countdown_experiments_TBA_qwen3_klMeanInference_b005_async10_eval20_noIS", "tab:blue"),
        ("$T/T$", "infer_Countdown_experiments_TBA_qwen3_klMeanTrain_b005_async10_eval20_noIS",     "tab:red"),
        ("$I/I$", "infer_Countdown_experiments_TBA_qwen3_klAllInf_b005_async10_eval20_noIS",        "tab:green"),
    ]
    _plot_reset(series, "Periodic-reset reference, IS off", "fig_reset_nois.pdf")


def fig_approx_error():
    """MAE and bias of the first-order surrogate, two runs per panel."""
    runs = [
        ("Training loss = exact KL",
         "Countdown_experiments_TBA_qwen3_noCenter_emaRef_a09_b005_async10_eval20",
         "tab:red", "-"),
        ("Training loss = approx KL",
         "Countdown_experiments_TBA_qwen3_noCenterByAlias_emaRef_a09_approxUse_b005_async10_eval20",
         "tab:blue", "-"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for key, ax, title in [
        ("kl_approx/mae", axes[0], "per-token MAE of first-order surrogate vs.\\ exact"),
        ("kl_approx/bias", axes[1], "per-token signed bias of first-order surrogate vs.\\ exact"),
    ]:
        for label, run_name, color, ls in runs:
            x, y = metric_history(run_name, key)
            if not x:
                continue
            ax.plot(x, y, label=label, color=color, linestyle=ls, linewidth=1.5)
        ax.set_xlabel("training step")
        ax.set_ylabel(key)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = OUT / "fig_approx_error.pdf"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def _plot_with_replicates(ax, series, color, ls, label):
    """series: list of run names (replicates). Plots mean line and min/max ribbon."""
    histories = []
    for name in series:
        x, y = eval_history(name)
        if x:
            histories.append(dict(zip(x, y)))
    if not histories:
        return
    steps = sorted(set().union(*[set(h.keys()) for h in histories]))
    means, lo, hi = [], [], []
    for s in steps:
        vals = [h[s] for h in histories if s in h]
        means.append(sum(vals)/len(vals))
        lo.append(min(vals))
        hi.append(max(vals))
    ax.plot(steps, means, color=color, linestyle=ls, linewidth=1.7, label=label)
    if any(h != l for h, l in zip(hi, lo)):
        ax.fill_between(steps, lo, hi, color=color, alpha=0.15, linewidth=0)


def fig_ablation_beta():
    """β ablation on the K-centered EMA cell (A)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    families = [
        ("$\\beta = 0.005$ (baseline)", [
            "infer_Countdown_experiments_TBA_qwen3_klMeanTrain_emaRef_a09_b005_async10_eval20",
            "infer_Countdown_experiments_TBA_qwen3_klMeanTrain_emaRef_a09_b005_async10_eval20_iter2",
        ], "tab:blue", "-"),
        ("$\\beta = 0.01$", [
            "infer_Countdown_experiments_TBA_qwen3_klMeanTrain_emaRef_a09_b01_async10_eval20",
            "infer_Countdown_experiments_TBA_qwen3_klMeanTrain_emaRef_a09_b01_async10_eval20_iter2",
        ], "tab:purple", "-"),
        ("$\\beta = 0.05$", [
            "infer_Countdown_experiments_TBA_qwen3_klMeanTrain_emaRef_a09_b05_async10_eval20",
        ], "tab:red", "-"),
    ]
    for label, runs, color, ls in families:
        _plot_with_replicates(ax, runs, color, ls, label)
    ax.set_xlabel("training step")
    ax.set_ylabel("Countdown eval accuracy (pass@1)")
    ax.set_title("$\\beta$ ablation: EMA exact + K-centered (klMeanTrain)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 0.9)
    fig.tight_layout()
    out = OUT / "fig_ablation_beta.pdf"
    fig.savefig(out); plt.close(fig); print("wrote", out)


def fig_ablation_is():
    """IS ablation on no-centering EMA cells (B and D)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    families = [
        ("Exact, IS on", [
            "infer_Countdown_experiments_TBA_qwen3_noCenter_emaRef_a09_b005_async10_eval20",
            "infer_Countdown_experiments_TBA_qwen3_noCenter_emaRef_a09_b005_async10_eval20_iter2",
        ], "tab:red", "-"),
        ("Exact, IS off", [
            "infer_Countdown_experiments_TBA_qwen3_noCenter_emaRef_a09_b005_async10_eval20_noIS",
        ], "tab:red", "--"),
        ("Approx, IS on", [
            "infer_Countdown_experiments_TBA_qwen3_noCenterByAlias_emaRef_a09_approxUse_b005_async10_eval20",
            "infer_Countdown_experiments_TBA_qwen3_noCenter_emaRef_a09_approxUse_b005_async10_eval20_iter2",
        ], "tab:blue", "-"),
        ("Approx, IS off", [
            "infer_Countdown_experiments_TBA_qwen3_noCenter_emaRef_a09_approxUse_b005_async10_eval20_noIS",
        ], "tab:blue", "--"),
    ]
    for label, runs, color, ls in families:
        _plot_with_replicates(ax, runs, color, ls, label)
    ax.set_xlabel("training step")
    ax.set_ylabel("Countdown eval accuracy (pass@1)")
    ax.set_title("IS ablation: EMA + no centering")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 0.9)
    fig.tight_layout()
    out = OUT / "fig_ablation_is.pdf"
    fig.savefig(out); plt.close(fig); print("wrote", out)


def fig_ablation_alpha():
    """α ablation on no-centering exact (β·c held constant at 0.0045)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    families = [
        ("$\\alpha = 0.1$, $\\beta = 0.405$", [
            "infer_Countdown_experiments_TBA_qwen3_noCenter_emaRef_a01_b405_async10_eval20",
        ], "tab:gray", "-"),
        ("$\\alpha = 0.2$, $\\beta = 0.18$", [
            "infer_Countdown_experiments_TBA_qwen3_noCenter_emaRef_a02_b18_async10_eval20",
        ], "tab:pink", "-"),
        ("$\\alpha = 0.5$, $\\beta = 0.045$", [
            "infer_Countdown_experiments_TBA_qwen3_noCenter_emaRef_a05_b045_async10_eval20",
        ], "tab:cyan", "-"),
        ("$\\alpha = 0.7$, $\\beta = 0.01929$", [
            "infer_Countdown_experiments_TBA_qwen3_noCenter_emaRef_a07_b01929_async10_eval20",
        ], "tab:olive", "-"),
        ("$\\alpha = 0.8$, $\\beta = 0.01125$", [
            "infer_Countdown_experiments_TBA_qwen3_noCenter_emaRef_a08_b01125_async10_eval20",
            "infer_Countdown_experiments_TBA_qwen3_noCenter_emaRef_a08_b01125_async10_eval20_iter2",
        ], "tab:orange", "-"),
        ("$\\alpha = 0.9$, $\\beta = 0.005$ (baseline)", [
            "infer_Countdown_experiments_TBA_qwen3_noCenter_emaRef_a09_b005_async10_eval20",
            "infer_Countdown_experiments_TBA_qwen3_noCenter_emaRef_a09_b005_async10_eval20_iter2",
        ], "tab:red", "-"),
        ("$\\alpha = 0.95$, $\\beta = 0.002368$", [
            "infer_Countdown_experiments_TBA_qwen3_noCenter_emaRef_a095_b002368_async10_eval20",
            "infer_Countdown_experiments_TBA_qwen3_noCenter_emaRef_a095_b002368_async10_eval20_iter2",
        ], "tab:brown", "-"),
        ("$\\alpha = 0.98$, $\\beta = 0.0009183$", [
            "infer_Countdown_experiments_TBA_qwen3_noCenter_emaRef_a098_b0009183_async10_eval20",
        ], "tab:green", "-"),
        ("$\\alpha = 0.99$, $\\beta = 0.0004545$", [
            "infer_Countdown_experiments_TBA_qwen3_noCenter_emaRef_a099_b0004545_async10_eval20",
        ], "tab:purple", "-"),
    ]
    for label, runs, color, ls in families:
        _plot_with_replicates(ax, runs, color, ls, label)
    ax.set_xlabel("training step")
    ax.set_ylabel("Countdown eval accuracy (pass@1)")
    ax.set_title("$\\alpha$ ablation: EMA exact + no centering, $\\beta c$ held constant")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 0.9)
    fig.tight_layout()
    out = OUT / "fig_ablation_alpha.pdf"
    fig.savefig(out); plt.close(fig); print("wrote", out)


def fig_ablation_beta_noIS():
    """β ablation on the no-centering, IS-off EMA cell (B-noIS)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    families = [
        ("$\\beta = 0.005$ (baseline)", [
            "infer_Countdown_experiments_TBA_qwen3_noCenter_emaRef_a09_b005_async10_eval20_noIS",
        ], "tab:blue", "-"),
        ("$\\beta = 0.01$", [
            "infer_Countdown_experiments_TBA_qwen3_noCenter_emaRef_a09_b01_async10_eval20_noIS",
            "infer_Countdown_experiments_TBA_qwen3_noCenter_emaRef_a09_b01_async10_eval20_noIS_iter2",
        ], "tab:red", "-"),
    ]
    for label, runs, color, ls in families:
        _plot_with_replicates(ax, runs, color, ls, label)
    ax.set_xlabel("training step")
    ax.set_ylabel("Countdown eval accuracy (pass@1)")
    ax.set_title("$\\beta$ ablation: EMA exact + no centering, IS off")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 0.9)
    fig.tight_layout()
    out = OUT / "fig_ablation_beta_noIS.pdf"
    fig.savefig(out); plt.close(fig); print("wrote", out)


def fig_ablation_beta_reset():
    """β ablation on the reset-reference T/T cell."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    families = [
        ("$\\beta = 0.005$ (baseline)", [
            "infer_Countdown_experiments_TBA_qwen3_klMeanTrain_b005_async10_eval20",
        ], "tab:blue", "-"),
        ("$\\beta = 0.008$", [
            "infer_Countdown_experiments_TBA_qwen3_klMeanTrain_b008_async10_eval20",
        ], "tab:green", "-"),
        ("$\\beta = 0.01$", [
            "infer_Countdown_experiments_TBA_qwen3_klMeanTrain_b01_async10_eval20",
            "infer_Countdown_experiments_TBA_qwen3_klMeanTrain_b01_async10_eval20_iter2",
        ], "tab:purple", "-"),
        ("$\\beta = 0.05$", [
            "infer_Countdown_experiments_TBA_qwen3_klMeanTrain_b05_async10_eval20",
            "infer_Countdown_experiments_TBA_qwen3_klMeanTrain_b05_async10_eval20_iter2",
        ], "tab:red", "-"),
    ]
    for label, runs, color, ls in families:
        _plot_with_replicates(ax, runs, color, ls, label)
    ax.set_xlabel("training step")
    ax.set_ylabel("Countdown eval accuracy (pass@1)")
    ax.set_title("$\\beta$ ablation: reset reference, $T/T$ (klMeanTrain)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 0.9)
    fig.tight_layout()
    out = OUT / "fig_ablation_beta_reset.pdf"
    fig.savefig(out); plt.close(fig); print("wrote", out)


def _plot_metric_with_replicates(ax, runs, color, label, key):
    """Plot mean line of a training-side metric across replicate runs."""
    histories = []
    for name in runs:
        x, y = metric_history(name, key)
        if x:
            histories.append(dict(zip(x, y)))
    if not histories:
        return
    steps = sorted(set().union(*[set(h.keys()) for h in histories]))
    means = []
    for s in steps:
        vals = [h[s] for h in histories if s in h]
        means.append(sum(vals)/len(vals))
    ax.plot(steps, means, color=color, linewidth=1.4, label=label)


def fig_approx_error_alpha():
    """Per-token MAE and α=0.9-rescaled MAE of the first-order surrogate across α-sweep."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    alphas = [
        ("$\\alpha=0.1$",  ["Countdown_experiments_TBA_qwen3_noCenter_emaRef_a01_b405_async10_eval20"], "tab:gray"),
        ("$\\alpha=0.2$",  ["Countdown_experiments_TBA_qwen3_noCenter_emaRef_a02_b18_async10_eval20"], "tab:pink"),
        ("$\\alpha=0.5$",  ["Countdown_experiments_TBA_qwen3_noCenter_emaRef_a05_b045_async10_eval20"], "tab:cyan"),
        ("$\\alpha=0.7$",  ["Countdown_experiments_TBA_qwen3_noCenter_emaRef_a07_b01929_async10_eval20"], "tab:olive"),
        ("$\\alpha=0.8$",  ["Countdown_experiments_TBA_qwen3_noCenter_emaRef_a08_b01125_async10_eval20",
                            "Countdown_experiments_TBA_qwen3_noCenter_emaRef_a08_b01125_async10_eval20_iter2"], "tab:orange"),
        ("$\\alpha=0.9$",  ["Countdown_experiments_TBA_qwen3_noCenter_emaRef_a09_b005_async10_eval20",
                            "Countdown_experiments_TBA_qwen3_noCenter_emaRef_a09_b005_async10_eval20_iter2"], "tab:red"),
        ("$\\alpha=0.95$", ["Countdown_experiments_TBA_qwen3_noCenter_emaRef_a095_b002368_async10_eval20",
                            "Countdown_experiments_TBA_qwen3_noCenter_emaRef_a095_b002368_async10_eval20_iter2"], "tab:brown"),
        ("$\\alpha=0.98$", ["Countdown_experiments_TBA_qwen3_noCenter_emaRef_a098_b0009183_async10_eval20"], "tab:green"),
        ("$\\alpha=0.99$", ["Countdown_experiments_TBA_qwen3_noCenter_emaRef_a099_b0004545_async10_eval20"], "tab:purple"),
    ]
    for label, runs, color in alphas:
        _plot_metric_with_replicates(axes[0], runs, color, label, "kl_approx/mae")
        _plot_metric_with_replicates(axes[1], runs, color, label, "kl_approx/mae_vs_alpha09")
    for ax, title, ylab in [
        (axes[0], "per-token surrogate MAE (intrinsic)", "kl_approx/mae"),
        (axes[1], "per-token MAE in $\\alpha=0.9$ scale", "kl_approx/mae_vs_alpha09"),
    ]:
        ax.set_xlabel("training step")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = OUT / "fig_approx_error_alpha.pdf"
    fig.savefig(out); plt.close(fig); print("wrote", out)


def fig_ablation_beta_approx():
    """β ablation on the no-centering approx EMA cell (D)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    families = [
        ("$\\beta = 0.0005$", [
            "infer_Countdown_experiments_TBA_qwen3_noCenter_emaRef_a09_approxUse_b0005_async10_eval20",
        ], "tab:green", "-"),
        ("$\\beta = 0.005$ (baseline)", [
            "infer_Countdown_experiments_TBA_qwen3_noCenterByAlias_emaRef_a09_approxUse_b005_async10_eval20",
            "infer_Countdown_experiments_TBA_qwen3_noCenter_emaRef_a09_approxUse_b005_async10_eval20_iter2",
        ], "tab:blue", "-"),
        ("$\\beta = 0.05$", [
            "infer_Countdown_experiments_TBA_qwen3_noCenter_emaRef_a09_approxUse_b05_async10_eval20",
        ], "tab:red", "-"),
    ]
    for label, runs, color, ls in families:
        _plot_with_replicates(ax, runs, color, ls, label)
    ax.set_xlabel("training step")
    ax.set_ylabel("Countdown eval accuracy (pass@1)")
    ax.set_title("$\\beta$ ablation: EMA approx + no centering (cell D)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 0.9)
    fig.tight_layout()
    out = OUT / "fig_ablation_beta_approx.pdf"
    fig.savefig(out); plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    fig_matrix()
    fig_reset_is()
    fig_reset_nois()
    fig_approx_error()
    fig_ablation_beta()
    fig_ablation_is()
    fig_ablation_alpha()
    fig_ablation_beta_noIS()
    fig_ablation_beta_reset()
    fig_ablation_beta_approx()
    fig_approx_error_alpha()
