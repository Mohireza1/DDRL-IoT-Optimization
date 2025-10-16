# plots.py
from __future__ import annotations
import csv
from typing import List, Tuple
import matplotlib.pyplot as plt


def _load_per_step_csv(path: str) -> Tuple[List[int], List[float]]:
    episodes, rewards = [], []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
    return episodes, rewards


def plot_reward_curve(per_step_csv: str, out_png: str):
    """
    Paper-style 'reward vs episodes' (aggregated by episode mean).
    """
    # Aggregate by episode mean
    by_ep = {}
    with open(per_step_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = int(row["episode"])
            r = float(row["reward"])
            by_ep.setdefault(ep, []).append(r)
    xs = sorted(by_ep.keys())
    ys = [sum(by_ep[e]) / len(by_ep[e]) for e in xs]

    plt.figure(figsize=(8, 4.5))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Average reward (η_EE proxy)")
    plt.title("Training: reward vs. episodes")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)


def plot_metric(per_step_csv: str, metric: str, out_png: str, agg: str = "mean"):
    """
    Plot any step metric from CSV (e.g., E_total_sum, T_off_avg) aggregated per episode.
    agg ∈ {mean, sum, min, max}
    """
    agg_fn = {
        "mean": lambda xs: sum(xs) / len(xs) if xs else 0.0,
        "sum": sum,
        "min": min,
        "max": max,
    }[agg]

    by_ep = {}
    with open(per_step_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = int(row["episode"])
            val = float(row[metric])
            by_ep.setdefault(ep, []).append(val)

    xs = sorted(by_ep.keys())
    ys = [agg_fn(by_ep[e]) for e in xs]

    plt.figure(figsize=(8, 4.5))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Episode")
    plt.ylabel(f"{metric} ({agg} per episode)")
    plt.title(f"{metric} vs. episodes")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
