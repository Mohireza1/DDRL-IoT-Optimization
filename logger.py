# logger.py
from __future__ import annotations
import csv
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class EpisodeLog:
    episode: int
    steps: int = 0
    rewards: List[float] = field(default_factory=list)
    per_step: List[Dict[str, Any]] = field(default_factory=list)

    def add_step(self, r: float, info: Dict[str, Any]):
        self.rewards.append(float(r))
        self.per_step.append(
            {
                "t": int(info.get("t", self.steps)),
                "sum_rates": float(info.get("sum_rates", 0.0)),
                "sum_Po": float(info.get("sum_Po", 0.0)),
                "lyapunov": float(info.get("lyapunov", 0.0)),
                "E_total_sum": float(info.get("E_total_sum", 0.0)),
                "T_off_avg": float(info.get("T_off_avg", 0.0)),
                # add more here if you expose them in env.info
            }
        )
        self.steps += 1

    def episode_reward(self) -> float:
        return float(sum(self.rewards))


class EnvLogger:
    """
    Minimal logger: call start_episode(); on every step call log_step(reward, info);
    then end_episode(). You can save CSVs per-episode and per-step.
    """

    def __init__(self):
        self.episodes: List[EpisodeLog] = []
        self._current: Optional[EpisodeLog] = None

    def start_episode(self, episode_idx: int):
        if self._current is not None:
            raise RuntimeError("Episode already started; call end_episode() first.")
        self._current = EpisodeLog(episode=episode_idx)

    def log_step(self, reward: float, info: Dict[str, Any]):
        if self._current is None:
            raise RuntimeError("No episode started; call start_episode().")
        self._current.add_step(reward, info)

    def end_episode(self):
        if self._current is None:
            raise RuntimeError("No active episode to end.")
        self.episodes.append(self._current)
        self._current = None

    # --- Export helpers ---
    def save_episode_summaries(self, path: str):
        """
        One row per episode: ep_idx, steps, ep_reward, ep_reward_mean, ep_reward_min, ep_reward_max
        """
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "episode",
                    "steps",
                    "reward_sum",
                    "reward_mean",
                    "reward_min",
                    "reward_max",
                ]
            )
            for ep in self.episodes:
                rs = ep.rewards
                writer.writerow(
                    [
                        ep.episode,
                        ep.steps,
                        sum(rs),
                        (sum(rs) / len(rs)) if rs else 0.0,
                        min(rs) if rs else 0.0,
                        max(rs) if rs else 0.0,
                    ]
                )

    def save_per_step(self, path: str):
        """
        All steps flattened: episode, t, reward, sum_rates, sum_Po, lyapunov, E_total_sum, T_off_avg
        """
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "episode",
                    "t",
                    "reward",
                    "sum_rates",
                    "sum_Po",
                    "lyapunov",
                    "E_total_sum",
                    "T_off_avg",
                ]
            )
            for ep in self.episodes:
                for r, row in zip(ep.rewards, ep.per_step):
                    writer.writerow(
                        [
                            ep.episode,
                            row["t"],
                            r,
                            row["sum_rates"],
                            row["sum_Po"],
                            row["lyapunov"],
                            row["E_total_sum"],
                            row["T_off_avg"],
                        ]
                    )
