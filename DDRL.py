import math
import numpy as np
from tqdm import tqdm


def consume_energy(w, p, g, noise, f, L, d):
    dc = 1e-27  # Device constant

    r = w * math.log2(1 + ((p * g) / (noise)))
    E_loc = dc * (f**2) * L
    E_off = p * (d / r)
    E_total = E_loc + E_off
    return r, E_loc, E_off, E_total


def reward(r, p, r_min=1e3, p_max=1):
    if r < r_min:
        return -1  # Penalty for low rate
    if p > p_max:
        return -1  # Penalty for excessive power usage
    return r / (p + 1e-9)  # Base energy efficiency


def channel_fade(scale=1):
    return np.random.rayleigh(scale) ** 2


def task_arrive(lam=1):
    return np.random.poisson(lam=1)


def lyap(q, h):
    return (q**2 + h**2) / 2


def lyap_drift(q, h, q_n, h_n):
    return lyap(q_n, h_n) - lyap(q, h)


class Task:
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def load(self):
        """Ensures that the tasks given to the device vary and allow for a more realistic task simulation"""
        return np.random.normal(loc=self.mean, scale=self.var)


class Device:
    def __init__(self, h, cap, f, p, d, cap_q=100):
        self.q = 0  # Queue size
        self.h = h  # Stored energy
        self.cap = cap  # Battery capacity
        self.f = f  # CPU frequency
        self.p = p  # Transmit power
        self.d = d  # Data size
        self.cap_q = cap_q  # Queue capacity
        self.b = 0  # Completed/offloaded tasks
        self.task = Task(5000, 2000)
        self.new_task()  # Initialize task load dynamically

    def update_queue(self, a):
        self.q = max(self.q - self.b, 0) + a

    def calculate_tasks_done(self, r, tau=1):
        self.b = (r * tau) / self.L  # Use self.L here

    def update_energy(self, used, harvested):
        self.h = min(max(self.h - used + harvested, 0), self.cap)

    def new_task(self):
        self.L = self.task.load()  # Update task load dynamically

    def __str__(self):
        return f"Queue: {self.q}\n Energy: {self.h}"


##### TEST LOOP
my_dev = Device(h=1, cap=4, f=200e6, p=1, d=1e6)
for i in tqdm(range(100), desc="Test Loop"):
    task = task_arrive()
    fade = channel_fade()
    (
        r,
        E_loc,
        E_off,
        E_total,
    ) = consume_energy(40e6, my_dev.p, fade, 1e-13, my_dev.f, my_dev.L, my_dev.d)

    my_dev.update_energy(
        E_total, np.random.uniform(0.01, 0.02)
    )  # Random energy harvested
    my_dev.calculate_tasks_done(r)  # Update number of completed tasks
    my_dev.update_queue(task)  # Update task queue based on arrivals and completions
    print(my_dev)
