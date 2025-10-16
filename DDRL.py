import gymnasium as gym
from gymnasium import spaces
import numpy as np

### System constants
edge = 15
server = 2
subch = 4

W = 40e6  # bandwidth
sigma2 = 1e-13  # noise power

penalty = -1

p_max = 2
r_min = 1
I_k = np.array([0.5] * subch)
print(
    f"Params: p_max={p_max}, r_min={r_min}, I_k[min]={I_k.min():.3g}, I_k[max]={I_k.max():.3g}, subch={subch}"
)


# reward params
omega, mu = 1, 1

# status of subchannels (one user per server per k)
lines = np.zeros((server, subch), dtype=bool)

# status of each slot so the interference can be calculated later
active = np.empty((server, subch), dtype=object)
for l in range(server):
    for k in range(subch):
        active[l, k] = []  # stores tuples (m, p, |g|)


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


def begin_slot():
    lines[:, :] = False
    for s in range(server):
        for k in range(subch):
            active[s, k].clear()


# Fali checks
def check_power(p):
    if p > p_max:
        return {
            "ok": False,
            "constraint": "power",
            "p": float(p),
            "p_max": float(p_max),
        }


def check_exclusive(l, k):
    if lines[l, k]:
        return {"ok": False, "constraint": "busy", "server": int(l), "k": int(k)}


def check_interference(p, g, k):
    lhs = float(p * abs(g))
    rhs = float(I_k[k])
    if lhs > rhs:
        return {
            "ok": False,
            "constraint": "interference",
            "lhs": lhs,
            "rhs": rhs,
            "k": int(k),
        }


def check_rate(r):
    if r < r_min:
        return {
            "ok": False,
            "constraint": "rate",
            "rate": float(r),
            "r_min": float(r_min),
        }


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


def step_once(S, server, edge, k, p, pc, ps, eps, P_T):
    reward = 0
    info = {}

    # Draw a random fade for the lmk link and set the device-server fade in the channels grid
    g_lm = np.random.rayleigh(scale=1)
    S["g"][server, edge] = g_lm

    # check and see if the values are ok
    fail = (
        check_exclusive(server, k) or check_power(p) or check_interference(p, g_lm, k)
    )
    if fail:
        return S, penalty, fail

    # set the line busy
    lines[server, k] = True

    # Clamp p for more safe claculation
    p = float(np.clip(p, 0.0, p_max))

    # SINR and rate
    interference = sum(
        p_i * g_i for (_, p_i, g_i) in active[server][k]
    )  # calculate interference based on the active users
    active[server, k].append((edge, p, abs(g_lm)))
    phi = sinr(p, g_lm, interference, sigma2)
    S["phi"].fill(0.0)
    S["phi"][edge] = phi
    S["prev_ch"][edge] = 0.9 * S["prev_ch"][edge] + 0.1 * S["g"][server, edge]
    r = rate(W, phi)  # rate
    fail = check_rate(r)
    if fail:
        return S, penalty, fail

    # served bits
    b_lm = r * 1  # 1 is delta time

    # bring in a task and update the queue
    S["A"][server, edge] = np.random.uniform(0, 0.1)
    new_Q = queue_update(S["Q"][server, edge], b_lm, S["A"][server, edge])
    S["Q"][server, edge] = new_Q

    # Power update
    P_o = power_total(p, pc, ps, eps)
    S["E"][edge] -= drain_rate * P_o
    S["E"][edge] = np.clip(S["E"][edge], 0.0, E_max)
    new_H = virtual_power_update(S["H"][server, edge], P_tilde=P_o, P_T=P_T)
    S["H"][server, edge] = new_H

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
# S["A"][0, 0] = 0.02
# for t in range(10):
#     a_idx = np.random.randint(len(actions))
#     l, m, k, p, x = decode_action(a_idx)
#     S, r, info = step_once(
#         S, server=l, edge=m, k=k, p=p, I_k=I_k, pc=0.1, ps=0.05, eps=0.35, P_T=0.05
#     )
#     print(t, r, info["ok"])
