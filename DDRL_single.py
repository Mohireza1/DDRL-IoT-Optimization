# import math
# import numpy as np
# from tqdm import tqdm
# import gymnasium as gym


# def reward(r, p, r_min=1e3, p_max=1):
#     if r < r_min:
#         return -1  # Penalty for low rate
#     if p > p_max:
#         return -1  # Penalty for excessive power usage
#     return r / (p + 1e-9)  # Base energy efficiency


# def channel_fade(scale=0.1):
#     return np.random.rayleigh(scale) ** 2


# def task_arrive(lam=1):
#     return np.random.poisson(lam=1)


# def lyap(q, h):
#     return (q**2 + h**2) / 2


# def lyap_drift(q, h, q_n, h_n):
#     return lyap(q_n, h_n) - lyap(q, h)


# class Task:
#     def __init__(self, mean, var):
#         self.mean = mean
#         self.var = var

#     def load(self):
#         """Ensures that the tasks given to the device vary and allow for a more realistic task simulation"""
#         return np.random.normal(loc=self.mean, scale=self.var)


# class Device:
#     def __init__(self, h, cap, f, p, d, cap_q=100, xi=1e-27):
#         self.q = 0  # Queue size
#         self.h = h  # Stored energy
#         self.cap = cap  # Battery capacity
#         self.f = f  # CPU frequency
#         self.p = p  # Transmit power
#         self.d = d  # Data size
#         self.cap_q = cap_q  # Queue capacity
#         self.b = 0  # Completed/offloaded tasks
#         self.xi = xi  # Energy coefficient (ξ)
#         self.L = 0  # Task load
#         self.task = Task(1e8, 1e8)
#         self.new_task()  # Initialize task load dynamically

#     def update_queue(self, a):
#         self.q = max(self.q - self.b, 0) + a

#     def calculate_tasks_done(self, r, tau=1):
#         self.b = (r * tau) / self.L  # Use self.L here

#     def update_energy(self, used, harvested):
#         self.h = min(max(self.h - used + harvested, 0), self.cap)

#     def new_task(self):
#         self.L = self.task.load()  # Update task load dynamically

#     def consume_energy(w, p, g, noise, f, L, d):

#         r = w * math.log2(1 + (self.p * g / noise))
#         E_loc = self.xi * (self.f**2) * self.L
#         E_off = self.p * (self.d / r)
#         E_total = E_loc + E_off
#         return r, E_loc, E_off, E_total

#     def __str__(self):
#         return f"Queue: {self.q}\n Energy: {self.h}\n Task Load: {self.L}"


# ##### TEST LOOP
# # my_dev = Device(h=1, cap=4, f=200e6, p=1, d=1e6)
# # for i in tqdm(range(100), desc="Test Loop"):
# #     task = task_arrive()
# #     fade = channel_fade()
# #     (
# #         r,
# #         E_loc,
# #         E_off,
# #         E_total,
# #     ) = consume_energy(40e6, my_dev.p, fade, 1e-13, my_dev.f, my_dev.L, my_dev.d)

# #     my_dev.update_energy(
# #         E_total, np.random.uniform(0.01, 0.02)
# #     )  # Random energy harvested
# #     my_dev.calculate_tasks_done(r)  # Update number of completed tasks
# #     my_dev.update_queue(task)  # Update task queue based on arrivals and completions
# #     print(my_dev)


# ##### Evnironment
# class Environment(gym.Env):
#     def __init__(self, max_steps=200):
#         super().__init__()
#         self.device = Device(h=1, cap=4, f=200e6, p=1, d=1e6)
#         self.max_steps = max_steps
#         self.t = 0

#         self.observation_space = gym.spaces.Box(
#             low=np.array([0, 0, 0, 0], dtype=np.float32),
#             high=np.array(
#                 [self.device.cap_q, self.device.cap, 1e5, np.inf], dtype=np.float32
#             ),
#             dtype=np.float32,
#         )
#         self.action_space = gym.spaces.Discrete(2)  # 0=local, 1=offload

#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         self.t = 0
#         self.device = Device(h=1, cap=4, f=200e6, p=1, d=1e6)
#         g = channel_fade()
#         obs = np.array(
#             [self.device.q, self.device.h, self.device.L, g], dtype=np.float32
#         )
#         return obs, {}

#     def step(self, action):
#         self.t += 1
#         done = False

#         g = channel_fade()
#         arrivals = task_arrive()

#         W = 40e6
#         noise = 1e-9

#         r = W * math.log2(1 + (self.device.p * g / noise))

#         if action == 0:  # Local compute
#             E_loc = self.device.xi * (self.device.f**2) * self.device.L
#             E_off = 0
#         else:  # Offload
#             E_loc = 0
#             E_off = self.device.p * (self.device.d / r)

#         E_total = E_loc + E_off

#         harvested = np.random.uniform(0.0001, 0.0002)
#         self.device.update_energy(E_total, harvested)

#         self.device.calculate_tasks_done(r)
#         self.device.update_queue(arrivals)
#         self.device.new_task()

#         rew = reward(r, self.device.p)

#         obs = np.array(
#             [self.device.q, self.device.h, self.device.L, g], dtype=np.float32
#         )

#         if self.device.h <= 0 or self.t >= self.max_steps:
#             done = True

#         info = {"E_total": E_total, "r": r}
#         return obs, rew, done, False, info


# env = Environment()
# obs, _ = env.reset()

# for _ in range(20):
#     action = env.action_space.sample()
#     obs, rew, done, trunc, info = env.step(action)
#     print(
#         f"Step: {env.t}, Action: {action}, Reward: {rew:.4f}, Q: {env.device.q:.2f}, H: {env.device.h:.4f}"
#     )
#     if done:
#         break
import math
import numpy as np
from tqdm import tqdm
import gymnasium as gym

# ---------- Helpers / constants ----------
tau = 1.0  # seconds per time-step (slot)


def channel_fade(scale=0.05):
    # small scale -> moderate average gain
    return np.random.rayleigh(scale) ** 2


def task_arrive(lam=1.0):
    # number of new tasks arriving (tasks/slot)
    return np.random.poisson(lam)


# ---------- Device & Task models ----------
class Task:
    def __init__(self, mean_bits=1e6, std_bits=2e5):
        self.mean = mean_bits
        self.std = std_bits

    def load_bits(self):
        # data size in bits per task; ensure positive
        return max(1e3, np.random.normal(self.mean, self.std))

    def cycles_per_task(self):
        # cycles required per task (use a realistic magnitude)
        return max(1e4, np.random.normal(1e6, 2e5))


class Device:
    def __init__(
        self,
        battery_j=10.0,
        f_hz=1e9,
        tx_power_w=0.1,
        cap_q=1000,
        xi=1e-28,
    ):
        # state
        self.q = 0.0  # tasks in queue (units: tasks)
        self.h = battery_j  # stored energy (Joules)
        self.cap = battery_j  # battery capacity
        # device constants
        self.f = f_hz  # CPU cycles/sec
        self.p = tx_power_w  # transmit power (Watt)
        self.cap_q = cap_q
        self.xi = xi  # CPU energy coefficient (Joule / (cycles^2) scale)
        # current task properties (per-task)
        self.task = Task(mean_bits=1e6, std_bits=2e5)  # default 1 Mbit tasks
        self.d_bits = self.task.load_bits()  # bits per task
        self.L_cycles = self.task.cycles_per_task()  # cycles per task
        # bookkeeping
        self.b = 0.0  # tasks completed in last slot

    def new_task_properties(self):
        self.d_bits = self.task.load_bits()
        self.L_cycles = self.task.cycles_per_task()

    def compute_local_tasks_done(self, tau=tau):
        # tasks completed locally in this slot (tasks/slot)
        return (self.f * tau) / self.L_cycles

    def compute_offload_tasks_done(self, r_bps, tau=tau):
        # tasks offloaded in this slot (r_bps * tau bits moved, divided by bits/task)
        bits_sent = r_bps * tau
        return bits_sent / self.d_bits

    def local_energy_joules(self):
        # local energy per task: xi * (f^2) * cycles_per_task / f?
        # from paper: E_loc = ξ*(f_m)^2 * L  (this yields Joules if xi scaled properly)
        # We want energy per slot for local computing of b_local tasks:
        # E_loc_slot = xi * f^2 * L_cycles * (tasks_done) / f? keep simple: xi * (f^2) * L_cycles_per_task * tasks_done_normalized
        # Simpler, physically plausible: energy per cycle ≈ small constant * cycles. We'll use paper form per task:
        E_per_task = self.xi * (self.f**2) * self.L_cycles
        return E_per_task

    def offload_energy_joules(self, d_bits, r_bps):
        # energy to transmit one task: p * (time to send d_bits) = p * (d_bits / r_bps)
        # If r_bps is extremely small, guard against zeros
        if r_bps <= 1e-12:
            return np.inf
        return self.p * (d_bits / r_bps)

    def update_energy(self, E_used, harvested):
        # E_used and harvested are Joules this slot
        self.h = min(max(self.h - E_used + harvested, 0.0), self.cap)

    def update_queue(self, tasks_done, arrivals):
        self.b = tasks_done
        self.q = max(self.q - tasks_done, 0.0) + arrivals
        # optionally cap queue
        if self.q > self.cap_q:
            self.q = float(self.cap_q)

    def __repr__(self):
        return f"Device(Q={self.q:.2f} tasks, H={self.h:.3f} J, d={self.d_bits:.0f} bits/task, L={self.L_cycles:.0f} cycles/task)"


# ---------- Reward function ----------
def compute_reward(r_bps, E_used):
    # r_bps: bits per second; tau seconds per slot -> bits per slot
    r_slot = r_bps * tau
    # reward = bits per Joule for this slot. If E_used is zero (unlikely), add epsilon.
    eps = 1e-9
    if E_used <= 0:
        return r_slot / (eps)
    return r_slot / (E_used + eps)


# ---------- Gym environment ----------
class MECEnv(gym.Env):
    """
    Single-device MEC toy environment for debugging Phase A/B.
    Actions:
      0 = compute locally
      1 = offload
    Observation: [queue_tasks, battery_J, d_bits, L_cycles, channel_gain]
    """

    metadata = {"render.modes": []}

    def __init__(self, max_steps=200):
        super().__init__()
        # device
        self.device = Device(battery_j=10.0, f_hz=1e9, tx_power_w=0.1, cap_q=1000)
        self.max_steps = max_steps
        self.t = 0

        # observation: q, h, d_bits (scaled), L_cycles (scaled), channel_gain
        obs_low = np.array([0.0, 0.0, 1e3, 1e3, 0.0], dtype=np.float32)
        obs_high = np.array(
            [self.device.cap_q, self.device.cap, 1e9, 1e10, 10.0], dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        # action: discrete local or offload
        self.action_space = gym.spaces.Discrete(2)

        # radio params
        self.W = 5e6  # 5 MHz
        self.noise = 1e-9  # noise power (W)
        self.scale = 0.05  # Rayleigh scale

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.device = Device(battery_j=10.0, f_hz=1e9, tx_power_w=0.1, cap_q=1000)
        # initialize task properties
        self.device.new_task_properties()
        g = channel_fade(self.scale)
        obs = self._make_obs(g)
        return obs, {}

    def _make_obs(self, g):
        return np.array(
            [self.device.q, self.device.h, self.device.d_bits, self.device.L_cycles, g],
            dtype=np.float32,
        )

    def step(self, action):
        self.t += 1
        terminated = False
        truncated = False

        # sample environment randomness
        g = channel_fade(self.scale)
        arrivals = task_arrive(lam=1.0)

        # compute instantaneous rate (Shannon-like)
        snr = (self.device.p * g) / self.noise
        r_bps = self.W * math.log2(1 + snr)

        # compute tasks done and energy used depending on action
        if action == 0:
            # local compute
            tasks_done = self.device.compute_local_tasks_done(tau)
            # local energy: E_per_task * tasks_done
            E_per_task = self.device.local_energy_joules()
            E_loc = E_per_task * tasks_done
            E_off = 0.0
        else:
            # offload
            tasks_done = self.device.compute_offload_tasks_done(r_bps, tau)
            E_loc = 0.0
            E_off_per_task = self.device.offload_energy_joules(
                self.device.d_bits, r_bps
            )
            E_off = E_off_per_task * tasks_done

        E_total = E_loc + E_off

        # harvested energy this slot (make it meaningful)
        harvested = np.random.uniform(0.01, 0.2)  # Joules per slot

        # update device
        self.device.update_energy(E_total, harvested)
        self.device.update_queue(tasks_done, arrivals)
        self.device.new_task_properties()  # new task params for next slot

        # reward: bits per joule (for this slot)
        reward = compute_reward(r_bps, E_total)

        # observation
        obs = self._make_obs(g)

        # termination: battery drained or max steps
        if self.device.h <= 0.0:
            terminated = True
        if self.t >= self.max_steps:
            truncated = True

        info = {
            "r_bps": r_bps,
            "E_total": E_total,
            "tasks_done": tasks_done,
            "arrivals": arrivals,
            "snr": snr,
        }
        return obs, float(reward), terminated, truncated, info


# ---------- quick test run ----------
if __name__ == "__main__":
    env = MECEnv(max_steps=200)
    obs, _ = env.reset()
    for _ in range(20):
        action = env.action_space.sample()
        obs, rew, term, trunc, info = env.step(action)
        print(
            f"Step: {env.t}, Action: {action}, Reward: {rew:.2e}, Q: {env.device.q:.2f}, H: {env.device.h:.3f}, r(Mbps): {info['r_bps']/1e6:.3f}, E(J): {info['E_total']:.4f}"
        )
        if term or trunc:
            break
