"""
Aggregate all results and reproduce report tables/figures.

Reads all result files and produces:
  - results/checkpoint_comparison_table.csv  (Table 4.1 from assignment)
  - results/forgetting_curve.png             (Alpaca metrics ckpt0→ckpt1→ckpt2)
  - results/json_metrics_table.csv           (JSON eval per checkpoint)
  - results/loss_curves.png                  (Stage 1 + Stage 2 training loss)

Usage:
    python src/evaluation/aggregate_results.py --config config.yaml
"""

import argparse
import csv
import json
from pathlib import Path

from src.utils import get_logger, load_config


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_metrics(results_dir: str, checkpoint: int) -> dict:
    """Load automatic Alpaca metrics for a checkpoint."""
    path = Path(results_dir) / f"metrics_ckpt{checkpoint}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_json_eval(results_dir: str, checkpoint: int) -> dict:
    """Load JSON eval summary for a checkpoint."""
    path = Path(results_dir) / f"json_eval_ckpt{checkpoint}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return data.get("summary", {})


def load_judge_summary(results_dir: str) -> dict:
    """Load the judge win-rate summary."""
    path = Path(results_dir) / "judge_summary.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_loss_csv(csv_path: str) -> tuple[list, list]:
    """Load (steps, losses) from a loss CSV."""
    steps, losses = [], []
    path = Path(csv_path)
    if not path.exists():
        return steps, losses
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
    return steps, losses


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def build_comparison_table(checkpoints: list[int], results_dir: str,
                           judge_summary: dict) -> list[dict]:
    """Build the three-checkpoint comparison table (Table 4.1)."""
    rows = []
    ckpt_labels = {0: "Base (ckpt0)", 1: "Stage1-Alpaca (ckpt1)", 2: "Stage2-JSON (ckpt2)"}

    for ckpt in checkpoints:
        metrics = load_metrics(results_dir, ckpt)
        json_eval = load_json_eval(results_dir, ckpt)

        # Judge win rates vs ckpt0 baseline
        judge_win_vs_base = None
        if ckpt > 0:
            pair_key = f"ckpt0v{ckpt}"
            if pair_key in judge_summary:
                # win_rate_b = win rate of the higher checkpoint (b)
                judge_win_vs_base = judge_summary[pair_key].get("win_rate_b", None)

        row = {
            "checkpoint":            ckpt_labels.get(ckpt, f"ckpt{ckpt}"),
            "alpaca_rouge1":         metrics.get("rouge1", ""),
            "alpaca_rouge2":         metrics.get("rouge2", ""),
            "alpaca_rougeL":         metrics.get("rougeL", ""),
            "alpaca_bertscore":      metrics.get("bertscore_f1", ""),
            "alpaca_completion_rate": metrics.get("task_completion_rate", ""),
            "alpaca_avg_length":     metrics.get("avg_output_length", ""),
            "judge_win_rate_vs_ckpt0": judge_win_vs_base if judge_win_vs_base is not None else "",
            "json_validity":         json_eval.get("json_validity_rate", ""),
            "json_schema":           json_eval.get("schema_compliance_rate", ""),
            "json_exact_match":      json_eval.get("exact_match_rate", ""),
            "json_field_f1":         json_eval.get("field_f1_mean", ""),
        }
        rows.append(row)
    return rows


def build_json_table(checkpoints: list[int], results_dir: str) -> list[dict]:
    """Build per-task JSON metrics table."""
    rows = []
    for ckpt in checkpoints:
        json_eval = load_json_eval(results_dir, ckpt)
        per_task = json_eval.get("per_task", {})
        for task, vals in per_task.items():
            rows.append({
                "checkpoint": ckpt,
                "task_type":  task,
                "n":          vals.get("n", ""),
                "json_validity_rate":      vals.get("json_validity_rate", ""),
                "schema_compliance_rate":  vals.get("schema_compliance_rate", ""),
                "exact_match_rate":        vals.get("exact_match_rate", ""),
                "field_f1_mean":           vals.get("field_f1_mean", ""),
            })
    return rows


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def plot_forgetting_curve(checkpoints: list[int], results_dir: str,
                          judge_summary: dict, out_path: str):
    """Plot Alpaca metrics across checkpoints to visualize forgetting."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        return

    rl_vals, bs_vals, wr_vals = [], [], []
    ckpt_labels = []

    for ckpt in checkpoints:
        metrics = load_metrics(results_dir, ckpt)
        rl_vals.append(metrics.get("rougeL", None))
        bs_vals.append(metrics.get("bertscore_f1", None))
        ckpt_labels.append(f"ckpt{ckpt}")

        # Judge win rate vs ckpt0
        if ckpt == 0:
            wr_vals.append(None)
        else:
            pair_key = f"ckpt0v{ckpt}"
            if pair_key in judge_summary:
                wr_vals.append(judge_summary[pair_key].get("win_rate_b", None))
            else:
                wr_vals.append(None)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    x = list(range(len(checkpoints)))

    def plot_metric(ax, values, ylabel, title, color):
        valid = [(xi, v) for xi, v in zip(x, values) if v is not None]
        if valid:
            xs, vs = zip(*valid)
            ax.plot(xs, vs, marker="o", color=color, linewidth=2)
            for xi, v in zip(xs, vs):
                ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=9)
            # Use tight y-axis so small differences are visible
            margin = (max(vs) - min(vs)) * 3 + 0.002
            ax.set_ylim(min(vs) - margin, max(vs) + margin * 2)
        ax.set_xticks(x)
        ax.set_xticklabels(ckpt_labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    plot_metric(axes[0], rl_vals,  "ROUGE-L",      "ROUGE-L (Alpaca)",         "#2196F3")
    plot_metric(axes[1], bs_vals,  "BERTScore F1", "BERTScore F1 (Alpaca)",    "#4CAF50")
    plot_metric(axes[2], wr_vals,  "Win rate vs ckpt0", "Judge Win Rate vs Baseline", "#FF5722")

    # Mark Stage 2 boundary
    if len(checkpoints) >= 3:
        for ax in axes:
            ax.axvline(x=1.5, color="gray", linestyle="--", alpha=0.5, label="Stage 2 →")

    fig.suptitle("Catastrophic Forgetting Analysis: Alpaca Metrics Across Checkpoints",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_loss_curves(cfg: dict, out_path: str):
    """Plot training loss curves for Stage 1 and Stage 2."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        return

    s1_steps, s1_losses = load_loss_csv(cfg["stage1"]["loss_csv"])
    s2_steps, s2_losses = load_loss_csv(cfg["stage2"]["loss_csv"])

    if not s1_steps and not s2_steps:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if s1_steps:
        axes[0].plot(s1_steps, s1_losses, color="#2196F3", linewidth=1.2)
        axes[0].set_title("Stage 1 Training Loss (Alpaca)")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Loss")
        axes[0].grid(alpha=0.3)

    if s2_steps:
        axes[1].plot(s2_steps, s2_losses, color="#4CAF50", linewidth=1.2)
        axes[1].set_title("Stage 2 Training Loss (Teacher JSON)")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Loss")
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Aggregate results and produce tables/figures.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoints", default="0,1,2",
                        help="Comma-separated checkpoint numbers")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("aggregate_results", "logs/aggregate_results.log")
    checkpoints = [int(c.strip()) for c in args.checkpoints.split(",")]
    results_dir = cfg["evaluation"]["results_dir"]
    out_dir = Path(results_dir)
    out_dir.mkdir(exist_ok=True)

    judge_summary = load_judge_summary(results_dir)

    # ── Comparison table ──────────────────────────────────────────────────────
    rows = build_comparison_table(checkpoints, results_dir, judge_summary)
    if rows:
        csv_path = out_dir / "checkpoint_comparison_table.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Saved comparison table → {csv_path}")

    # ── JSON metrics table ────────────────────────────────────────────────────
    json_rows = build_json_table(checkpoints, results_dir)
    if json_rows:
        csv_path = out_dir / "json_metrics_table.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=json_rows[0].keys())
            writer.writeheader()
            writer.writerows(json_rows)
        logger.info(f"Saved JSON metrics table → {csv_path}")

    # ── Forgetting curve figure ───────────────────────────────────────────────
    fig_path = str(out_dir / "forgetting_curve.png")
    plot_forgetting_curve(checkpoints, results_dir, judge_summary, fig_path)
    logger.info(f"Saved forgetting curve → {fig_path}")

    # ── Loss curves figure ────────────────────────────────────────────────────
    loss_fig_path = str(out_dir / "loss_curves.png")
    plot_loss_curves(cfg, loss_fig_path)
    logger.info(f"Saved loss curves → {loss_fig_path}")

    # ── Print summary to log ──────────────────────────────────────────────────
    if rows:
        logger.info("\n=== Checkpoint Comparison Summary ===")
        for row in rows:
            logger.info(
                f"  {row['checkpoint']}: "
                f"ROUGE-L={row['alpaca_rougeL']:.4f if isinstance(row['alpaca_rougeL'], float) else row['alpaca_rougeL']}, "
                f"BERTScore={row['alpaca_bertscore']:.4f if isinstance(row['alpaca_bertscore'], float) else row['alpaca_bertscore']}, "
                f"JSON validity={row['json_validity']:.3f if isinstance(row['json_validity'], float) else row['json_validity']}"
            )

    # ── Forgetting analysis: ckpt1 → ckpt2 ───────────────────────────────────
    if len(checkpoints) >= 3:
        m1 = load_metrics(results_dir, 1)
        m2 = load_metrics(results_dir, 2)
        if m1 and m2:
            delta_rougeL    = m2.get("rougeL", 0) - m1.get("rougeL", 0)
            delta_bertscore = m2.get("bertscore_f1", 0) - m1.get("bertscore_f1", 0)
            logger.info("\n=== Forgetting Analysis (ckpt1 → ckpt2) ===")
            logger.info(f"  ΔROUGE-L:    {delta_rougeL:+.4f}")
            logger.info(f"  ΔBERTScore:  {delta_bertscore:+.4f}")

        j1 = load_json_eval(results_dir, 1)
        j2 = load_json_eval(results_dir, 2)
        if j1 and j2:
            delta_validity = j2.get("json_validity_rate", 0) - j1.get("json_validity_rate", 0)
            logger.info(f"  ΔJSON validity: {delta_validity:+.3f}")

        if "ckpt1v2" in judge_summary:
            wr = judge_summary["ckpt1v2"]
            logger.info(f"  Judge: ckpt2 win rate vs ckpt1: {wr.get('win_rate_b', 0):.3f}")


if __name__ == "__main__":
    main()
