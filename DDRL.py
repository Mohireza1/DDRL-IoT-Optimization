import gymnasium as gym
from gymnasium import spaces
import numpy as np


# ---------- Paper-consistent primitives ----------


def sinr(p, g_abs, interference, sigma2):
    return (p * g_abs) / (interference + sigma2)


def rate(W, phi):
    return W * np.log2(1.0 + max(phi, 0.0))


def queue_update(q, b, A):
    return max(q - b, 0.0) + A


def rrh_total_power(p_vec_for_rrh):
    # Sum of scheduled transmit powers on all subchannels for a given RRH
    return float(np.sum(p_vec_for_rrh))


def rrh_total_power_model(p_sum_rrh, p_c, p_s, eps):
    eps = max(eps, 1e-12)
    return float(p_c + p_s + p_sum_rrh / eps)


def virtual_power_update(H, P_tilde, P_T):
    # Eq. (17): per-link virtual queue. We use link transmit-component as P_tilde.
    return max(H - P_tilde + P_T, 0.0)


def E_local(xi, f_m, L_l):
    # Eq. (2): E_lo = xi * f_m^2 * L_l
    return float(xi * (f_m**2) * L_l)


def T_local(f_m, psi_lo_m):
    # Paper definition (Sec. C-1): T_lo = f_m / psi_lo_m
    return float(f_m / max(psi_lo_m, 1e-12))


def E_offload(p_m, f_m, psi_off_m):
    # Eq. (3): E_off = (p_m / f_m) * psi_off_m
    return float((p_m / max(f_m, 1e-12)) * psi_off_m)


def T_offload(p_m, f_m, psi_off_m, d_m, r_m):
    # Sec. C-2: T_off = (p_m/f_m)*psi_off_m + d_m / r_m
    t_comp = (p_m / max(f_m, 1e-12)) * psi_off_m
    t_tx = d_m / max(r_m, 1e-12)
    return float(t_comp + t_tx)


def eta_EE(omega, sum_rate, mu, sum_Po):
    denom = max(mu * sum_Po, 1e-20)
    return (omega * sum_rate) / denom


def lyapunov_L(Q, H):
    return float(0.5 * (np.sum(Q**2) + np.sum(H**2)))


class MECEnvDDRL(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, config=None):
        cfg = config or {}

        # System / sim params (paper defaults)
        self.L = int(cfg.get("num_rrh", 2))  # RRHs
        self.M = int(cfg.get("num_devices", 15))  # devices
        self.K = int(cfg.get("num_subch", 4))  # subchannels
        self.W_total = float(cfg.get("bandwidth", 40e6))
        self.W_sub = self.W_total / self.K
        self.sigma2 = float(cfg.get("noise", 1e-13))  # ~ -100 dBm
        # CPU capability f_m (cycles/s)
        self.f_m = np.array(
            cfg.get("f_m", np.full(self.M, 2.0e9)),  # default 2 GHz per device
            dtype=float,
        )

        # Task load L_l (dimensionless task-load term used with f_m^2); allow per-device for flexibility
        self.L_l = np.array(
            cfg.get(
                "L_l", np.full(self.M, 1.0e-9)
            ),  # small scale so energies are in a reasonable range
            dtype=float,
        )

        # Energy coefficient xi (chip/architecture dependent)
        self.xi = float(cfg.get("xi", 1.0e-27))

        # Required compute resources for local/offload branches (psi)
        self.psi_lo = np.array(
            cfg.get("psi_lo", np.full(self.M, 1.0e8)),  # cycles needed locally
            dtype=float,
        )
        self.psi_off = np.array(
            cfg.get("psi_off", np.full(self.M, 5.0e7)),  # cycles needed at MEC path
            dtype=float,
        )

        # Input data size d_m (bits) for offloading latency model
        self.d_m = np.array(
            cfg.get("d_m", np.full(self.M, 1.0e6)),  # 1 Mbit per task by default
            dtype=float,
        )

        # For convenience we’ll also track rolling per-slot r_m and p_m
        self._r_m_last = np.zeros(self.M, dtype=float)
        self._p_m_last = np.zeros(
            self.M, dtype=float
        )  # interpreted as device uplink power for Eq. (3)

        # Constraints
        self.p_max = float(cfg.get("p_max", 2.0))  # per-RRH power cap (per slot)
        self.r_min = float(cfg.get("r_min", 1.0))  # per-link QoS min rate
        Ik_default = np.full(self.K, cfg.get("Ik_value", 0.5), dtype=float)
        self.Ik = np.array(cfg.get("Ik", Ik_default), dtype=float).reshape(self.K)

        # Reward weights
        self.omega = float(cfg.get("omega", 1.0))
        self.mu = float(cfg.get("mu", 1.0))

        # Episode
        self.episode_len = int(cfg.get("episode_len", 200))

        # Discretized power levels (include 0 via "idle" option separately)
        self.num_power_levels = int(cfg.get("num_power_levels", 11))
        self.power_levels = np.linspace(
            0.0, self.p_max, self.num_power_levels, dtype=float
        )

        # Power model for RRH total power (Eq. 11 components)
        self.p_c = float(cfg.get("p_c", 0.1))
        self.p_s = float(cfg.get("p_s", 0.0))
        self.eps = float(cfg.get("epsilon", 0.5))  # PA drain efficiency
        self.P_T = float(cfg.get("P_T", 0.5))  # virtual queue target

        # Internal state
        self.Q = np.zeros((self.L, self.M), dtype=float)  # data queues
        self.H = np.zeros((self.L, self.M), dtype=float)  # virtual power queues
        self.h = np.zeros((self.L, self.M), dtype=float)  # |Rayleigh channel|

        # Observation = [|h|, Q, H]
        obs_dim = self.L * self.M * 3
        self.observation_space = spaces.Box(
            low=0.0, high=np.finfo(np.float32).max, shape=(obs_dim,), dtype=np.float32
        )

        # Joint action: for each (l,k), choose among [idle] + M*P combos
        self._choices_per_lk = 1 + self.M * (
            self.num_power_levels - 1
        )  # exclude zero power from combos
        # MultiDiscrete vector length L*K, each entry in [0, choices-1]
        self.action_space = spaces.MultiDiscrete(
            [self._choices_per_lk] * (self.L * self.K)
        )

        # Penalty
        self.invalid_penalty = float(cfg.get("invalid_penalty", -1.0))

        # RNG / bookkeeping
        self._rng = np.random.default_rng(cfg.get("seed", None))
        self._t = 0
        self._episode_steps = 0
        self._ep_reward_sum = 0.0
        self._ep_reward_min = np.inf
        self._ep_reward_max = -np.inf

        # Initialize channels
        self._draw_channels()

    def _draw_channels(self):
        self.h[:, :] = self._rng.rayleigh(scale=1.0, size=(self.L, self.M))

    def _build_obs(self):
        return np.concatenate([self.h.ravel(), self.Q.ravel(), self.H.ravel()]).astype(
            np.float32
        )

    def _decode_joint_action(self, a_vec):
        """
        Decode MultiDiscrete action into:
          m_sel[l,k] in {-1..M-1}, p_sel[l,k] >= 0
          -1 => idle; else device index with nonzero power
        Mapping:
          idx=0 => idle
          idx>=1 => idx-1 => (m, p_idx_nonzero) with p = power_levels[p_idx]
        """
        m_sel = -np.ones((self.L, self.K), dtype=int)
        p_sel = np.zeros((self.L, self.K), dtype=float)

        for l in range(self.L):
            for k in range(self.K):
                idx = int(a_vec[l * self.K + k])
                if idx == 0:
                    continue  # idle
                idx -= 1
                m = idx // (self.num_power_levels - 1)
                p_idx_nz = (
                    idx % (self.num_power_levels - 1) + 1
                )  # 1..num_power_levels-1
                m_sel[l, k] = m
                p_sel[l, k] = float(self.power_levels[p_idx_nz])
        return m_sel, p_sel

    def _interference_matrix(self, m_sel, p_sel):
        """
        Compute interference seen at each (l,k) receiver from other RRHs on the same k:
          I[l,k] = sum_{lp!=l, scheduled} p_sel[lp,k] * |h[l, m_sel[lp,k]]|
        """
        I = np.zeros((self.L, self.K), dtype=float)
        for k in range(self.K):
            for l in range(self.L):
                if m_sel[l, k] < 0:
                    continue  # even if idle, still check against Ik in C3 below
                itf = 0.0
                for lp in range(self.L):
                    if lp == l:
                        continue
                    if m_sel[lp, k] >= 0:
                        m_lp = m_sel[lp, k]
                        p_lp = p_sel[lp, k]
                        itf += p_lp * abs(self.h[l, m_lp])
                I[l, k] = itf
        return I

    # ---------- gym api ----------

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._episode_steps = 0
        self.Q.fill(0.0)
        self.H.fill(0.0)
        self._draw_channels()

        self._ep_reward_sum = 0.0
        self._ep_reward_min = np.inf
        self._ep_reward_max = -np.inf

        obs = self._build_obs()
        info = {
            "lyapunov": lyapunov_L(self.Q, self.H),
            "t": self._t,
            "episode_steps": self._episode_steps,
        }
        return obs, info

    def step(self, action):
        # Decode joint action
        m_sel, p_sel = self._decode_joint_action(action)

        # -------- Joint constraint checks --------

        # C4 (exclusivity): each (l,k) has at most one device (by construction), AND
        # each device m can be scheduled by at most one RRH across all k.
        # Enforce single association: a device cannot appear in two different l (any k).
        for m in range(self.M):
            used_pairs = [
                (l, k) for l in range(self.L) for k in range(self.K) if m_sel[l, k] == m
            ]
            if len(used_pairs) > 1:
                obs = self._build_obs()
                return (
                    obs,
                    self.invalid_penalty,
                    False,
                    False,
                    {
                        "ok": False,
                        "constraint": "C4_single_association",
                        "device": int(m),
                        "pairs": used_pairs,
                    },
                )

        # C1 (power cap): per RRH total scheduled power ≤ p_max
        for l in range(self.L):
            p_sum = rrh_total_power(p_sel[l, :])
            if p_sum > self.p_max + 1e-12:
                obs = self._build_obs()
                return (
                    obs,
                    self.invalid_penalty,
                    False,
                    False,
                    {
                        "ok": False,
                        "constraint": "C1_power_cap",
                        "l": int(l),
                        "p_sum": float(p_sum),
                        "p_max": float(self.p_max),
                    },
                )

        # Compute interference matrix based on the joint schedule
        I = self._interference_matrix(m_sel, p_sel)

        # C3 (interference threshold per (l,k))
        for l in range(self.L):
            for k in range(self.K):
                if I[l, k] > float(self.Ik[k]) + 1e-12:
                    obs = self._build_obs()
                    return (
                        obs,
                        self.invalid_penalty,
                        False,
                        False,
                        {
                            "ok": False,
                            "constraint": "C3_interference",
                            "l": int(l),
                            "k": int(k),
                            "lhs": float(I[l, k]),
                            "rhs": float(self.Ik[k]),
                        },
                    )

        # C2 (rate minimum) for every scheduled link
        phi = np.zeros((self.L, self.K), dtype=float)
        r = np.zeros((self.L, self.K), dtype=float)
        for l in range(self.L):
            for k in range(self.K):
                m = m_sel[l, k]
                if m < 0:
                    continue
                p = p_sel[l, k]
                g_abs = abs(self.h[l, m])
                phi[l, k] = sinr(p, g_abs, I[l, k], self.sigma2)
                r[l, k] = rate(self.W_sub, phi[l, k])
                if r[l, k] < self.r_min - 1e-12:
                    obs = self._build_obs()
                    return (
                        obs,
                        self.invalid_penalty,
                        False,
                        False,
                        {
                            "ok": False,
                            "constraint": "C2_rate_min",
                            "l": int(l),
                            "k": int(k),
                            "rate": float(r[l, k]),
                            "r_min": float(self.r_min),
                        },
                    )

        # RRH total power + model Po(l) per RRH
        rrh_p_sums = np.array(
            [rrh_total_power(p_sel[l, :]) for l in range(self.L)], dtype=float
        )
        rrh_Po = np.array(
            [
                rrh_total_power_model(rrh_p_sums[l], self.p_c, self.p_s, self.eps)
                for l in range(self.L)
            ],
            dtype=float,
        )

        # Per-link service b and arrivals A; update Q for all (l,m)
        # For scheduled (l,k): affects that device m only (b>0), others b=0
        # Sample arrivals for every (l,m)
        A = self._rng.uniform(0.0, 0.1, size=(self.L, self.M))
        b = np.zeros((self.L, self.M), dtype=float)
        for l in range(self.L):
            for k in range(self.K):
                m = m_sel[l, k]
                if m >= 0:
                    b[l, m] += r[l, k] * 1.0  # Δt = 1

        for l in range(self.L):
            for m in range(self.M):
                self.Q[l, m] = queue_update(self.Q[l, m], b[l, m], A[l, m])

        # Virtual power queues H for scheduled links:
        # Use link transmit component as P_tilde (p_sel/eps); unscheduled links unchanged.
        for l in range(self.L):
            for k in range(self.K):
                m = m_sel[l, k]
                if m >= 0:
                    P_tilde_lm = p_sel[l, k] / max(self.eps, 1e-12)
                    self.H[l, m] = virtual_power_update(
                        self.H[l, m], P_tilde=P_tilde_lm, P_T=self.P_T
                    )

        # Collapse per-(l,k) rates/powers to per-device r_m and p_m (sum over any scheduled subchannels)
        r_m = np.zeros(self.M, dtype=float)
        p_m = np.zeros(self.M, dtype=float)
        for l in range(self.L):
            for k in range(self.K):
                m = m_sel[l, k]
                if m >= 0:
                    r_m[m] += r[l, k]  # total achieved rate for device m this slot
                    p_m[m] += p_sel[
                        l, k
                    ]  # interpret scheduled power as uplink p_m for Eq. (3)

        # Paper Eq. (2)-(4): per-device energies/latencies
        E_lo_m = np.array(
            [E_local(self.xi, self.f_m[m], self.L_l[m]) for m in range(self.M)],
            dtype=float,
        )
        T_lo_m = np.array(
            [T_local(self.f_m[m], self.psi_lo[m]) for m in range(self.M)], dtype=float
        )
        E_off_m = np.array(
            [E_offload(p_m[m], self.f_m[m], self.psi_off[m]) for m in range(self.M)],
            dtype=float,
        )
        T_off_m = np.array(
            [
                T_offload(
                    p_m[m],
                    self.f_m[m],
                    self.psi_off[m],
                    self.d_m[m],
                    max(r_m[m], 1e-12),
                )
                for m in range(self.M)
            ],
            dtype=float,
        )

        E_T_m = E_lo_m + E_off_m  # Eq. (4)
        # Save for info/debug
        self._r_m_last[:] = r_m
        self._p_m_last[:] = p_m

        # Reward = system EE over the slot
        sum_rates = float(np.sum(r))
        sum_Po = float(np.sum(rrh_Po))
        reward = eta_EE(self.omega, sum_rates, self.mu, sum_Po)

        # Stats
        self._ep_reward_sum += reward
        self._ep_reward_min = min(self._ep_reward_min, reward)
        self._ep_reward_max = max(self._ep_reward_max, reward)
        Lphi = lyapunov_L(self.Q, self.H)

        # Advance time: new slot, new channels
        self._episode_steps += 1
        self._t += 1
        self._draw_channels()

        obs = self._build_obs()
        terminated = False
        truncated = self._episode_steps >= self.episode_len
        info = {
            "ok": True,
            "sum_rates": sum_rates,
            "sum_Po": sum_Po,
            "lyapunov": float(Lphi),
            "episode_reward_sum": float(self._ep_reward_sum),
            "episode_reward_min": float(self._ep_reward_min),
            "episode_reward_max": float(self._ep_reward_max),
            "t": self._t,
            "episode_steps": self._episode_steps,
            "r_m": self._r_m_last.copy(),
            "p_m": self._p_m_last.copy(),
            "E_local_sum": float(np.sum(E_lo_m)),
            "E_off_sum": float(np.sum(E_off_m)),
            "E_total_sum": float(np.sum(E_T_m)),
            "T_local_avg": float(np.mean(T_lo_m)),
            "T_off_avg": float(np.mean(T_off_m)),
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass


# ----------------------- Smoke test -----------------------
if __name__ == "__main__":
    env = MECEnvDDRL({"seed": 42})
    obs, info = env.reset()
    print("Reset OK:", obs.shape, info)

    total = 0.0
    invalids = 0
    for i in range(1000):
        for t in range(20):
            a = env.action_space.sample()
            obs, r, term, trunc, info = env.step(a)
            total += r
            if not info.get("ok", False):
                invalids += 1
            if trunc or term:
                obs, info = env.reset()
        # print("Random rollout done. reward_sum=", total, "invalids=", invalids)
        print(
            "Example slot paper-metrics:",
            "E_total_sum=",
            info.get("E_total_sum"),
            "T_off_avg=",
            info.get("T_off_avg"),
        )
