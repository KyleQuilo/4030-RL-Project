"""
plot_results.py

Phase 3 comparative analysis: PPO (base) vs SAC (advanced).

Loads per-episode metrics CSVs from ppo_results/ and sac_results/ and
generates the four required comparison plots:

    1. Learning Speed   — smoothed reward curves overlaid, with shaded raw variance
    2. Loss Convergence — smoothed loss curves overlaid
    3. Final Performance — bar chart of mean ± std over evaluation episodes
    4. Stability / Variance — rolling standard deviation of episode reward

Usage:
    python3 plot_results.py
    python3 plot_results.py --ppo-csv ppo_results/metrics.csv \
                             --sac-csv sac_results/metrics.csv \
                             --out-dir plots/

All figures are saved as high-resolution PNGs. Raw data is plotted as a faint
shaded region; smoothed curves use a rolling window of 50 episodes.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Global plot style
# ---------------------------------------------------------------------------

PPO_COLOR = "#1f77b4"   # blue
SAC_COLOR = "#ff7f0e"   # orange
SMOOTH_WINDOW = 50      # rolling average window (episodes)
ALPHA_RAW = 0.15        # opacity for raw-data shaded region
FONT_SIZE = 12
TITLE_SIZE = 13

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE - 1,
    "figure.dpi": 150,
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_metrics(csv_path):
    """
    Load a per-episode metrics CSV produced by MetricsCallback.

    Expected columns: episode, reward, length, loss, entropy_param.

    Args:
        csv_path (str): Path to the metrics CSV file.

    Returns:
        pd.DataFrame: Loaded metrics with numeric columns. Returns an empty
            DataFrame if the file does not exist or cannot be parsed.
    """
    if not os.path.isfile(csv_path):
        print(f"Warning: metrics file not found: {csv_path}")
        return pd.DataFrame(columns=["episode", "reward", "length", "loss", "entropy_param"])
    df = pd.read_csv(csv_path)
    return df


def smooth(series, window=SMOOTH_WINDOW):
    """
    Apply a centered rolling mean to a pandas Series.

    Args:
        series (pd.Series): Raw per-episode values.
        window (int): Rolling window size in episodes.

    Returns:
        pd.Series: Smoothed values aligned with the original index.
    """
    return series.rolling(window=window, min_periods=1, center=True).mean()


def rolling_std(series, window=SMOOTH_WINDOW):
    """
    Compute a centered rolling standard deviation.

    Args:
        series (pd.Series): Raw per-episode values.
        window (int): Rolling window size in episodes.

    Returns:
        pd.Series: Rolling standard deviation aligned with the original index.
    """
    return series.rolling(window=window, min_periods=1, center=True).std().fillna(0)


# ---------------------------------------------------------------------------
# Plot 1: Learning speed (reward curves)
# ---------------------------------------------------------------------------

def plot_reward_curves(ppo_df, sac_df, out_path):
    """
    Plot smoothed episode reward curves for PPO and SAC overlaid on one axes.

    Shaded regions show the raw data variance (rolling std). A horizontal
    dashed threshold line marks a meaningful performance level if the data
    supports it (mean reward > -100 in the second half of training).

    Args:
        ppo_df (pd.DataFrame): PPO metrics dataframe.
        sac_df (pd.DataFrame): SAC metrics dataframe.
        out_path (str): File path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for df, label, color in [
        (ppo_df, "PPO (base)", PPO_COLOR),
        (sac_df, "SAC (advanced)", SAC_COLOR),
    ]:
        if df.empty:
            continue
        ep = df["episode"]
        raw = df["reward"]
        sm = smooth(raw)
        sd = rolling_std(raw)

        ax.plot(ep, sm, color=color, linewidth=2, label=label)
        ax.fill_between(ep, sm - sd, sm + sd, color=color, alpha=ALPHA_RAW)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Learning Speed: Episode Reward vs Episode")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: Loss convergence
# ---------------------------------------------------------------------------

def plot_loss_curves(ppo_df, sac_df, out_path):
    """
    Plot smoothed training loss curves for PPO and SAC overlaid on one axes.

    PPO loss is the policy gradient loss. SAC loss is the actor loss.
    NaN values (logged before the first SB3 update) are dropped before plotting.

    Args:
        ppo_df (pd.DataFrame): PPO metrics dataframe.
        sac_df (pd.DataFrame): SAC metrics dataframe.
        out_path (str): File path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for df, label, color in [
        (ppo_df, "PPO policy gradient loss", PPO_COLOR),
        (sac_df, "SAC actor loss", SAC_COLOR),
    ]:
        if df.empty:
            continue
        valid = df.dropna(subset=["loss"])
        if valid.empty:
            continue
        ep = valid["episode"]
        sm = smooth(valid["loss"])
        sd = rolling_std(valid["loss"])

        ax.plot(ep, sm, color=color, linewidth=2, label=label)
        ax.fill_between(ep, sm - sd, sm + sd, color=color, alpha=ALPHA_RAW)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Training Loss")
    ax.set_title("Loss Convergence: Training Loss vs Episode")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 3: Final performance (bar chart)
# ---------------------------------------------------------------------------

def plot_final_performance(ppo_eval_rewards, sac_eval_rewards, out_path):
    """
    Bar chart comparing mean ± std final performance over evaluation episodes.

    Uses rewards from the last N evaluation episodes (passed in as lists) to
    represent final trained performance with exploration disabled.

    Args:
        ppo_eval_rewards (list[float]): Per-episode rewards from PPO evaluation.
        sac_eval_rewards (list[float]): Per-episode rewards from SAC evaluation.
        out_path (str): File path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    labels = []
    means = []
    stds = []

    for rewards, label in [(ppo_eval_rewards, "PPO"), (sac_eval_rewards, "SAC")]:
        if rewards:
            labels.append(label)
            means.append(float(np.mean(rewards)))
            stds.append(float(np.std(rewards)))

    if not labels:
        print("No evaluation rewards provided — skipping final performance plot.")
        plt.close(fig)
        return

    colors = [PPO_COLOR if l == "PPO" else SAC_COLOR for l in labels]
    x = np.arange(len(labels))

    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=8, width=0.4,
                  error_kw={"elinewidth": 2})

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Cumulative Reward")
    ax.set_title("Final Performance: Mean ± Std over Evaluation Episodes")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 2,
            f"{mean:.1f}±{std:.1f}",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE - 1,
        )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 4: Stability / variance
# ---------------------------------------------------------------------------

def plot_stability(ppo_df, sac_df, out_path):
    """
    Plot rolling standard deviation of episode reward to compare training stability.

    A lower rolling std indicates more consistent reward across episodes.
    PPO tends toward lower variance due to its clipped surrogate and on-policy
    updates; SAC may show higher early variance before its replay buffer fills.

    Args:
        ppo_df (pd.DataFrame): PPO metrics dataframe.
        sac_df (pd.DataFrame): SAC metrics dataframe.
        out_path (str): File path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for df, label, color in [
        (ppo_df, "PPO (base)", PPO_COLOR),
        (sac_df, "SAC (advanced)", SAC_COLOR),
    ]:
        if df.empty:
            continue
        ep = df["episode"]
        std = rolling_std(df["reward"])
        ax.plot(ep, std, color=color, linewidth=2, label=label)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Rolling Std of Episode Reward")
    ax.set_title(f"Stability: Rolling Reward Std (window={SMOOTH_WINDOW} episodes)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Bonus: Exploration parameter
# ---------------------------------------------------------------------------

def plot_exploration(ppo_df, sac_df, out_path):
    """
    Plot the exploration parameter over training for both agents.

    For PPO this is the entropy loss (negative entropy, should increase as
    policy becomes more deterministic). For SAC this is the entropy coefficient
    alpha (automatically tuned; should stabilize after initial adjustment).

    Args:
        ppo_df (pd.DataFrame): PPO metrics dataframe.
        sac_df (pd.DataFrame): SAC metrics dataframe.
        out_path (str): File path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for df, label, color in [
        (ppo_df, "PPO entropy loss", PPO_COLOR),
        (sac_df, "SAC entropy coef (alpha)", SAC_COLOR),
    ]:
        if df.empty:
            continue
        valid = df.dropna(subset=["entropy_param"])
        if valid.empty:
            continue
        ep = valid["episode"]
        sm = smooth(valid["entropy_param"])
        ax.plot(ep, sm, color=color, linewidth=2, label=label)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Exploration Parameter Value")
    ax.set_title("Exploration Schedule: Entropy Parameter vs Episode")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    """
    Parse command-line arguments for plot_results.py.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Phase 3 comparison plots: PPO vs SAC")
    parser.add_argument(
        "--ppo-csv",
        default="ppo_results/metrics.csv",
        help="Path to PPO training metrics CSV (default: ppo_results/metrics.csv)",
    )
    parser.add_argument(
        "--sac-csv",
        default="sac_results/metrics.csv",
        help="Path to SAC training metrics CSV (default: sac_results/metrics.csv)",
    )
    parser.add_argument(
        "--ppo-eval-csv",
        default=None,
        help="Optional CSV with PPO evaluation episode rewards (column: reward)",
    )
    parser.add_argument(
        "--sac-eval-csv",
        default=None,
        help="Optional CSV with SAC evaluation episode rewards (column: reward)",
    )
    parser.add_argument(
        "--out-dir",
        default="plots/",
        help="Directory to save plot PNGs (default: plots/)",
    )
    return parser.parse_args()


def main():
    """
    Load metrics CSVs and generate all four Phase 3 comparison plots.

    Saves the following PNGs to --out-dir:
        reward_curves.png       — learning speed
        loss_curves.png         — loss convergence
        final_performance.png   — mean ± std bar chart
        stability.png           — rolling reward std
        exploration.png         — entropy parameter schedule
    """
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ppo_df = load_metrics(args.ppo_csv)
    sac_df = load_metrics(args.sac_csv)

    plot_reward_curves(
        ppo_df, sac_df,
        out_path=os.path.join(args.out_dir, "reward_curves.png"),
    )
    plot_loss_curves(
        ppo_df, sac_df,
        out_path=os.path.join(args.out_dir, "loss_curves.png"),
    )
    plot_stability(
        ppo_df, sac_df,
        out_path=os.path.join(args.out_dir, "stability.png"),
    )
    plot_exploration(
        ppo_df, sac_df,
        out_path=os.path.join(args.out_dir, "exploration.png"),
    )

    # Final performance bar chart — load optional eval CSVs if provided
    ppo_eval_rewards = []
    sac_eval_rewards = []

    if args.ppo_eval_csv and os.path.isfile(args.ppo_eval_csv):
        ppo_eval_rewards = pd.read_csv(args.ppo_eval_csv)["reward"].tolist()
    elif not ppo_df.empty:
        # Fall back to last 50 training episodes as a proxy
        ppo_eval_rewards = ppo_df["reward"].tail(50).tolist()

    if args.sac_eval_csv and os.path.isfile(args.sac_eval_csv):
        sac_eval_rewards = pd.read_csv(args.sac_eval_csv)["reward"].tolist()
    elif not sac_df.empty:
        sac_eval_rewards = sac_df["reward"].tail(50).tolist()

    plot_final_performance(
        ppo_eval_rewards, sac_eval_rewards,
        out_path=os.path.join(args.out_dir, "final_performance.png"),
    )

    print(f"\nAll plots saved to {args.out_dir}")


if __name__ == "__main__":
    main()
