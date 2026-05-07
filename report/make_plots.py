"""Generate eval-curve and approx-error plots for the empirical companion to main.tex."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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


def fig_ema_matrix():
    fig, ax = plt.subplots(figsize=(8, 5))
    series = [
        ("A: EMA exact + klMeanTrain centering",
         "infer_Countdown_experiments_TBA_qwen3_klMeanTrain_emaRef_a09_b005_async10_eval20",
         "tab:red", "-"),
        ("B: EMA exact + no centering",
         "infer_Countdown_experiments_TBA_qwen3_noCenter_emaRef_a09_b005_async10_eval20",
         "tab:blue", "-"),
        ("C: approx EMA + klMeanTrain centering",
         "infer_Countdown_experiments_TBA_qwen3_klMeanTrain_emaRef_a09_approxUse_b005_async10_eval20",
         "tab:red", "--"),
        ("D: approx EMA + no centering (alias-zero)",
         "infer_Countdown_experiments_TBA_qwen3_noCenterByAlias_emaRef_a09_approxUse_b005_async10_eval20",
         "tab:blue", "--"),
        ("ref: EMA exact + klMeanInf centering (weak)",
         "infer_Countdown_experiments_TBA_qwen3_klMeanInf_emaRef_a09_b005_async10_eval20",
         "tab:gray", ":"),
    ]
    for label, run_name, color, ls in series:
        x, y = eval_history(run_name)
        if not x:
            continue
        ax.plot(x, y, label=label, color=color, linestyle=ls, linewidth=1.6)
    ax.set_xlabel("training step")
    ax.set_ylabel("eval/countdown/score")
    ax.set_title("EMA-reference matrix: $\\{$exact, approx$\\}\\times\\{$klMeanTrain centering, no centering$\\}$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = OUT / "fig_ema_matrix.pdf"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def fig_reset_baselines():
    fig, ax = plt.subplots(figsize=(8, 5))
    series = [
        ("klMeanInf (IS on)",
         "infer_Countdown_experiments_TBA_qwen3_klMeanInference_b005_async10_eval20",
         "tab:blue", "-"),
        ("klMeanTrain (IS on, collapsed)",
         "infer_Countdown_experiments_TBA_qwen3_klMeanTrain_b005_async10_eval20",
         "tab:red", "-"),
        ("klAllInf (IS on)",
         "infer_Countdown_experiments_TBA_qwen3_klAllInf_b005_async10_eval20",
         "tab:green", "-"),
        ("klMeanInf (IS off)",
         "infer_Countdown_experiments_TBA_qwen3_klMeanInference_b005_async10_eval20_noIS",
         "tab:blue", "--"),
        ("klMeanTrain (IS off, collapsed)",
         "infer_Countdown_experiments_TBA_qwen3_klMeanTrain_b005_async10_eval20_noIS",
         "tab:red", "--"),
        ("klAllInf (IS off, stuck)",
         "infer_Countdown_experiments_TBA_qwen3_klAllInf_b005_async10_eval20_noIS",
         "tab:green", "--"),
    ]
    for label, run_name, color, ls in series:
        x, y = eval_history(run_name)
        if not x:
            continue
        ax.plot(x, y, label=label, color=color, linestyle=ls, linewidth=1.4)
    ax.set_xlabel("training step")
    ax.set_ylabel("eval/countdown/score")
    ax.set_title("Reset-reference baseline (no EMA): KL-source $\\times$ IS")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = OUT / "fig_reset_baselines.pdf"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def fig_approx_error():
    """4-panel: MAE, bias, rel_err (log y), rel_err_clamped vs step.

    Each panel shows curves for the runs that have approx-error telemetry.
    """
    runs = [
        ("A (klMeanTrain, exact loss)",
         "Countdown_experiments_TBA_qwen3_klMeanTrain_emaRef_a09_b005_async10_eval20",
         "tab:red", "-"),
        ("B (no centering, exact loss)",
         "Countdown_experiments_TBA_qwen3_noCenter_emaRef_a09_b005_async10_eval20",
         "tab:blue", "-"),
        ("C (klMeanTrain, approx loss)",
         "Countdown_experiments_TBA_qwen3_klMeanTrain_emaRef_a09_approxUse_b005_async10_eval20",
         "tab:red", "--"),
        ("D (no centering, approx loss)",
         "Countdown_experiments_TBA_qwen3_noCenterByAlias_emaRef_a09_approxUse_b005_async10_eval20",
         "tab:blue", "--"),
        ("approxTracked (klMeanInf, exact loss + telemetry)",
         "Countdown_experiments_TBA_qwen3_klMeanInf_emaRef_a09_exactLoss_approxTracked_b005_async10_eval20",
         "tab:gray", ":"),
    ]
    metrics = [
        ("kl_approx/mae", "MAE per token", False),
        ("kl_approx/bias", "signed bias per token", False),
        ("kl_approx/rel_err", "relative error (eps=1e-8)", True),
        ("kl_approx/rel_err_clamped", "relative error (clamp [-10,10])", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (key, title, log_y) in zip(axes.flat, metrics):
        for label, run_name, color, ls in runs:
            x, y = metric_history(run_name, key)
            if not x:
                continue
            if log_y:
                y = [max(v, 1e-10) for v in y]
            ax.plot(x, y, label=label, color=color, linestyle=ls, linewidth=1.3)
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("training step")
        ax.set_ylabel(key)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(loc="upper right", fontsize=7)
    fig.suptitle("First-order EMA approximation: per-token error metrics over training", y=1.00)
    fig.tight_layout()
    out = OUT / "fig_approx_error.pdf"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig_ema_matrix()
    fig_reset_baselines()
    fig_approx_error()
