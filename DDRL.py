import gymnasium as gym
from gymnasium import spaces
import numpy as np

### System constants
edge = 15
server = 2
subch = 4

W = 40e6  # bandwidth
sigma2 = 1e-13  # noise power

p_max = 2

# reward params
omega, mu = 1, 1


E_max = 1.0  # max battery
drain_rate = 1e-6  # drain rate for the battery

power_watts = np.arange(
    0, p_max, 0.1
)  # Power should be a continous variable in the state of the device but since we need it to be discerete for RL, it's drawn as closely as possible

X_choice = np.array([0, 1])  # 0=local 1=offload


seed = 0

g = np.zeros((server, edge))  # channel gain
Q = np.zeros((server, edge))  # queue
H = np.zeros((server, edge))  # power
A = np.zeros((server, edge))  # arrivals
U = np.zeros(edge)  # local task OBSERVE
E = np.ones(edge)  # Batteries OBSERVE
phi = np.zeros(edge)  # SINR OBSERVE
prev_ch = np.zeros(edge)  # Previous channel for helping future prediction OBSERVE

# Actions
actions = []
for l in range(server):  # server index
    for m in range(edge):  # device index
        for k in range(subch):  # subchannel index
            for p_idx in range(len(power_watts)):
                for x in X_choice:
                    actions.append((l, m, k, p_idx, int(x)))


### Equations


def interference_ok(p, g, I_k):  # Checks if subchannel interference exceeds its limits
    return p * abs(g) <= I_k


def sinr(p, g, interference, sigma2):
    return (p * abs(g)) / (interference + sigma2)


def rate(W, phi):
    return W * np.log2(1 + phi)


def queue_update(q, b, A):
    return max(q - b, 0) + A


def power_total(p_lm, p_c, p_s, eps):
    eps = max(eps, 1e-20)
    return float(p_c + p_s + abs(p_lm) / eps)


def virtual_power_update(H, P_tilde, P_T):
    return max(H - P_tilde + P_T, 0)


def eta_EE(omega, r, mu, P_o):
    denom = max(mu * P_o, 1e-20)
    return (omega * r) / denom


# assert interference_ok(1.0, 0.5, 1.0) is True
# assert interference_ok(3.0, 0.5, 1.0) is False

# phi = sinr(1.0, 2.0, interference=0.0, sigma2=1e-13)
# assert phi > 1e13

# assert abs(rate(1.0, 0.0) - 0.0) < 1e-12
# assert abs(rate(1.0, 1.0) - np.log2(2.0)) < 1e-12

# assert queue_update(5.0, b=2.0, A=1.0) == 4.0
# assert queue_update(1.0, b=2.0, A=0.5) == 0.5

# assert abs(power_total(0.0, p_c=0.1, p_s=0.05, eps=0.5) - 0.15) < 1e-12
# assert power_total(1.0, p_c=0.1, p_s=0.05, eps=0.5) > 2.0

# assert eta_EE(1.0, r=0.0, mu=1.0, P_o=1.0) == 0.0
# assert abs(eta_EE(1.0, r=10.0, mu=2.0, P_o=5.0) - 1.0) < 1e-12

### Step once

S = {
    "g": g,
    "Q": Q,
    "H": H,
    "A": A,
    "U": U,
    "E": E,
    "phi": phi,
    "prev_ch": prev_ch,
}  # step inputs


def step_once(S, server, edge, k, p, I_k, pc, ps, eps, P_T):
    reward = 0
    info = {}

    # Clamp p for more secure claculation:
    np.clip(p, 0.0, p_max)

    # Draw a random fade for the lmk link and set the device-server fade in the channels grid
    g_lm = np.random.rayleigh(scale=1)
    S["g"][server][edge] = g_lm

    # If interference limit is exceeded, return negatie reward
    if not interference_ok(p, g_lm, I_k):
        reward = -1
        info = {"ok": False, "constraint": "interference"}
        return S, reward, info

    # SINR and rate
    interference = 0.0  # for now
    phi = sinr(p, g_lm, interference, sigma2)
    S["phi"].fill(0.0)
    S["phi"][edge] = phi
    S["prev_ch"][edge] = 0.9 * S["prev_ch"][edge] + 0.1 * S["g"][server, edge]
    r = rate(W, phi)  # rate

    # served bits
    b_lm = r * 1  # 1 is delta time

    # bring in a task and update the queue
    S["A"][server][edge] = np.random.uniform(0, 0.1)
    new_Q = queue_update(S["Q"][server][edge], b_lm, S["A"][server][edge])
    S["Q"][server][edge] = new_Q

    # Power update
    P_o = power_total(p, pc, ps, eps)
    S["E"][edge] -= drain_rate * P_o
    S["E"][edge] = np.clip(S["E"][edge], 0.0, E_max)
    new_H = virtual_power_update(S["H"][server][edge], P_tilde=P_o, P_T=P_T)
    S["H"][server][edge] = new_H

    # Local task done
    alpha = 10  # task scale
    S["U"][edge] = max(S["U"][edge] - r / (W * alpha), 0)

    # Calculate the reward
    reward = eta_EE(omega, r, mu, P_o)

    info = {
        "phi": phi,
        "rate": r,
        "P_o": P_o,
        "battery": S["E"][edge],
        "q_new": new_Q,
        "H_new": new_H,
        "ok": True,
    }

    return S, reward, info


def observe():  # returns the observation space with the indicated parameters
    return np.concatenate([S["phi"], S["E"], S["U"], S["prev_ch"]]).astype(np.float32)


def decode_action(idx: int):
    l, m, k, p_idx, x = actions[idx]
    p = power_watts[p_idx]
    return l, m, k, p, x


### Test
S["A"][0, 0] = 0.02
for t in range(10):
    a_idx = np.random.randint(len(actions))
    l, m, k, p, x = decode_action(a_idx)
    S, r, info = step_once(
        S, server=l, edge=m, k=k, p=p, I_k=1.0, pc=0.1, ps=0.05, eps=0.35, P_T=0.05
    )
    print(t, r, info["ok"])
