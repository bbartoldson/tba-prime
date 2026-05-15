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
    """Eval curves for the 2x2 EMA-reference matrix."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    cells = [
        ("Exact KL, K-centered",
         "infer_Countdown_experiments_TBA_qwen3_klMeanTrain_emaRef_a09_b005_async10_eval20",
         "tab:red", "-"),
        ("Exact KL, no centering",
         "infer_Countdown_experiments_TBA_qwen3_noCenter_emaRef_a09_b005_async10_eval20",
         "tab:red", "--"),
        ("Approx KL, K-centered",
         "infer_Countdown_experiments_TBA_qwen3_klMeanTrain_emaRef_a09_approxUse_b005_async10_eval20",
         "tab:blue", "-"),
        ("Approx KL, no centering",
         "infer_Countdown_experiments_TBA_qwen3_noCenterByAlias_emaRef_a09_approxUse_b005_async10_eval20",
         "tab:blue", "--"),
    ]
    for label, run_name, color, ls in cells:
        x, y = eval_history(run_name)
        if not x:
            continue
        ax.plot(x, y, label=label, color=color, linestyle=ls, linewidth=1.7)
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


def fig_reset():
    """Eval curves for the reset-reference baseline."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    series = [
        ("Exact KL, K-centered (inf-mean)",
         "infer_Countdown_experiments_TBA_qwen3_klMeanInference_b005_async10_eval20",
         "tab:blue", "-"),
        ("Exact KL, K-centered (train-mean)",
         "infer_Countdown_experiments_TBA_qwen3_klMeanTrain_b005_async10_eval20",
         "tab:red", "-"),
        ("Inf-snapshot KL, K-centered (inf-mean)",
         "infer_Countdown_experiments_TBA_qwen3_klAllInf_b005_async10_eval20",
         "tab:green", "-"),
        ("Exact KL, K-centered (inf-mean), no IS",
         "infer_Countdown_experiments_TBA_qwen3_klMeanInference_b005_async10_eval20_noIS",
         "tab:blue", "--"),
        ("Exact KL, K-centered (train-mean), no IS",
         "infer_Countdown_experiments_TBA_qwen3_klMeanTrain_b005_async10_eval20_noIS",
         "tab:red", "--"),
        ("Inf-snapshot KL, K-centered (inf-mean), no IS",
         "infer_Countdown_experiments_TBA_qwen3_klAllInf_b005_async10_eval20_noIS",
         "tab:green", "--"),
    ]
    for label, run_name, color, ls in series:
        x, y = eval_history(run_name)
        if not x:
            continue
        ax.plot(x, y, label=label, color=color, linestyle=ls, linewidth=1.4)
    ax.set_xlabel("training step")
    ax.set_ylabel("Countdown eval accuracy (pass@1)")
    ax.set_title("Periodic-reset reference (every 50 steps): KL-source $\\times$ IS")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = OUT / "fig_reset.pdf"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def fig_approx_error():
    """MAE and bias of the first-order surrogate, two runs per panel."""
    runs = [
        ("Exact KL, no centering (loss uses exact)",
         "Countdown_experiments_TBA_qwen3_noCenter_emaRef_a09_b005_async10_eval20",
         "tab:red", "-"),
        ("Approx KL, no centering (loss uses approx)",
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


if __name__ == "__main__":
    fig_matrix()
    fig_reset()
    fig_approx_error()
