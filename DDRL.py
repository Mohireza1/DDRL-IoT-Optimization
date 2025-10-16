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
    S["g"][:] = np.random.rayleigh(scale=1.0, size=(server, edge))


def rrh_total_power(l: int) -> float:
    total = 0.0
    for k in range(subch):
        for _m, p_i, _abs_g in active[l, k]:
            total += p_i
    return float(total)


def current_interference_at(rrh_l: int, k: int) -> float:
    itf = 0.0
    for lp in range(server):
        if lp == rrh_l:
            continue
        for m_prime, p_prime, _abs_g_lp_mprime in active[lp, k]:
            g_rrhl_mprime = S["g"][rrh_l, m_prime]
            itf += p_prime * abs(g_rrhl_mprime)
    return float(itf)


def adding_this_link_breaks_any_receiver(l: int, m: int, k: int, p: float):
    # our own receiver l: must already be within I_k[k] given existing others
    cur_self = current_interference_at(l, k)
    if cur_self > float(I_k[k]):
        return True, {
            "constraint": "C3_interference_self",
            "lhs": float(cur_self),
            "rhs": float(I_k[k]),
            "k": int(k),
            "receiver": int(l),
        }

    # all other receivers on same k: adding our link increases their interference
    for lp in range(server):
        if lp == l:
            continue
        cur = current_interference_at(lp, k)
        add = p * abs(S["g"][lp, m])
        if cur + add > float(I_k[k]):
            return True, {
                "constraint": "C3_interference_other",
                "lhs": float(cur + add),
                "rhs": float(I_k[k]),
                "k": int(k),
                "receiver": int(lp),
            }
    return False, None


# Fail checks
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
    lhs = sum(p_i * abs(g_i) for (_, p_i, g_i) in active[server, k]) + p * abs(g_new)
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
    reward = 0.0
    info = {}

    fail = check_exclusive(server, k)
    if fail:
        return S, penalty, fail

    single_cap = check_power(p)
    if single_cap:
        return S, penalty, single_cap

    # Then total cap for this RRH in the current slot
    used_power = rrh_total_power(server)
    if used_power + p > float(p_max):
        return (
            S,
            penalty,
            {
                "ok": False,
                "constraint": "total power exceeded",
                "used": float(used_power),
                "add": float(p),
                "p_max": float(p_max),
                "server": int(server),
            },
        )

    breaks, info = adding_this_link_breaks_any_receiver(server, edge, k, p)
    if breaks:
        return S, penalty, info

    # Desired link gain
    g_lm = S["g"][server, edge]

    # Interference at OUR receiver (from other RRHs already active on k)
    interference = current_interference_at(server, k)

    p = float(np.clip(p, 0.0, p_max))
    phi_val = sinr(p, g_lm, interference, sigma2)
    S["phi"].fill(0.0)
    S["phi"][edge] = phi_val
    S["prev_ch"][edge] = 0.9 * S["prev_ch"][edge] + 0.1 * g_lm
    r = rate(W, phi_val)

    fail = check_rate(r)
    if fail:
        return S, penalty, fail

    # lock the line and register activity for interference accounting
    lines[server, k] = True
    active[server, k].append((edge, p, abs(g_lm)))

    # Served bits over the slot (Δt = 1 for now)
    b_lm = r * 1.0

    S["A"][server, edge] = np.random.uniform(0, 0.1)
    new_Q = queue_update(S["Q"][server, edge], b_lm, S["A"][server, edge])
    S["Q"][server, edge] = new_Q

    P_o = power_total(p, pc, ps, eps)
    S["E"][edge] -= drain_rate * P_o
    S["E"][edge] = np.clip(S["E"][edge], 0.0, E_max)
    new_H = virtual_power_update(S["H"][server, edge], P_tilde=P_o, P_T=P_T)
    S["H"][server, edge] = new_H

    alpha = 10  # task scale
    S["U"][edge] = max(S["U"][edge] - r / (W * alpha), 0.0)

    reward = eta_EE(omega, r, mu, P_o)

    info = {
        "phi": float(phi_val),
        "rate": float(r),
        "P_o": float(P_o),
        "battery": float(S["E"][edge]),
        "q_new": float(new_Q),
        "H_new": float(new_H),
        "itf_self": float(interference),
        "ok": True,
    }

    return S, reward, info


def observe():  # returns the observation space with the indicated parameters
    return np.concatenate([S["phi"], S["E"], S["U"], S["prev_ch"]]).astype(np.float32)


def decode_action(idx: int):
    l, m, k, p_idx, x = actions[idx]
    p = power_watts[p_idx]
    return l, m, k, p, x
