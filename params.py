"""
Parameter configuration and benchmark calculations for the Competitive Newsvendor simulation
"""

import numpy as np
from scipy.stats import poisson

## parameters 

# profit structure
#C stays for now -> could be decision variable later (set by supplier agent)
C = 3.0

#Comment: RHO not needed without competition, P not fixed anymore -> set by our agent
#RHO = 0.5
#P = 8.0

#Comment: Poisson demand not suitable here -> we need price demand response curve with little randomness
# poisson demand
#LAM = 20
#DEMAND_VAR = LAM
#CRITICAL_FRACTILE = 1 - C / P

A_intercept = 100.0
B_Slope = 3.5
Noise_random = 10.0

# action space 
Q_LOWER = 0
Q_UPPER = 40
#N_ACTIONS = Q_UPPER + 1
def action_space_q() -> np.ndarray:
    """ return discrete order quantities: [Q_LOWER, ..., Q_UPPER]"""
    return np.arange(Q_LOWER, Q_UPPER + 1)

P_LOWER = 0
P_UPPER = 30
P_STEP_SIZE = 0.1

def action_space_p() -> np.ndarray:
    return np.arange(P_LOWER, P_UPPER + P_STEP_SIZE, P_STEP_SIZE)

# ε-greedy
EPS_START = 0.8
EPS_END = 0.05
def epsilon_at(t: int, rounds: int) -> float:
    """ 
    return epsiolon in round t 
    linear decay of epsilon from EPS_START to EPS_END over rounds
    """
    if rounds <= 1:
        return EPS_END
    frac = np.clip(t, 0, rounds - 1) / (rounds - 1)
    return (1 - frac) * EPS_START + frac * EPS_END

# UCB 
UCB_C = 2.0  

''' (Thomson sampling uses RHO and Demand_Var) -> need for change later
# thompson 
TS_PRIOR_MEAN = 0.0
TS_PRIOR_VAR = 1000.0
EFF_SIGMA = np.sqrt(DEMAND_VAR * (1 + RHO**2)) # Poisson variance * (1 + rho^2)
TS_MC_SAMPLES = 300
'''
# simulation
ROUNDS = 365
SEED = 42

#BENCHMARK 1: optimal p,q pair (E[profit(p,q)])
def opt_expected_profit(p, q, n_sims: int = 1000, seed: int = 0) -> float:
    #computes expected profit
    rng = np.random.default_rng(seed)

    mu = A_intercept - B_Slope * p
    d = mu + rng.normal(loc=0.0, scale=Noise_random, size=n_sims)
    d = np.maximum(0.0, d)  

    sales = np.minimum(q, d)
    profit = p * sales - C * q
    return float(np.mean(profit))


def opt_best_pq(n_sims: int = 1000, seed: int = 0):
    # finds the best (p,q)

    prices = action_space_p()
    quantities = action_space_q()

    best_profit = -1e18
    best_p = None
    best_q = None

    # Optional: reuse common random numbers for fairer comparisons across (p,q)
    # but simplest version is below (same seed for each evaluation).
    for p in prices:
        for q in quantities:
            pi = opt_expected_profit(float(p), float(q), n_sims=n_sims, seed=seed)
            if pi > best_profit:
                best_profit = pi
                best_p = float(p)
                best_q = float(q)

    return (best_p, best_q), float(best_profit)
opt_QP, opt_profit = opt_best_pq(n_sims=1000, seed=SEED)



'''

# simple single-agent benchmark (without substitution)
Q_SIMPLE = int(poisson.ppf(CRITICAL_FRACTILE, LAM)) 

# system-monopoly benchmark 
Q_MONOPOLY = int(poisson.ppf(CRITICAL_FRACTILE, 2*LAM)) 

# nash equilibrium benchmark
def compute_equilibrium_qN(lam=LAM, rho=RHO, p=P, c=C,
                           q_max=Q_UPPER, n_sims=200000,
                           seed=42) -> int:
    """
    Computes the symmetric equilibrium order q^N (fixed-point best response) according to Long & Wu (2024)
    """

    rng = np.random.default_rng(seed)
    alpha = 1 - c / p

    # sample once for speed
    D1 = rng.poisson(lam, size=n_sims)
    D2 = rng.poisson(lam, size=n_sims)

    best_q = None
    best_error = np.inf

    for q in range(q_max + 1):
        effective = D1 + rho * np.maximum(0, D2 - q)
        prob = np.mean(effective < q)
        error = abs(prob - alpha)

        if error < best_error:
            best_q = q
            best_error = error

    return int(best_q)

def expected_profit_duopoly():
    """
    Approximates expected profits (mu1, mu2, mu_joint) 
    By using Monte Carlo simulation
    """
    n_sims = 200_000
    rng = np.random.default_rng(SEED)
    D1 = rng.poisson(LAM, size=n_sims)
    D2 = rng.poisson(LAM, size=n_sims)

    # effective demands
    d1_hat = D1 + RHO * np.maximum(0, D2 - Q_NASH)
    d2_hat = D2 + RHO * np.maximum(0, D1 - Q_NASH)

    r1 = P * np.minimum(Q_NASH, d1_hat) - C * Q_NASH
    r2 = P * np.minimum(Q_NASH, d2_hat) - C * Q_NASH

    mu1 = float(np.mean(r1))
    mu2 = float(np.mean(r2))
    return mu1, mu2, mu1 + mu2

# compute once at module load 
Q_NASH = compute_equilibrium_qN()
MU1_NASH, MU2_NASH, MU_JOINT_NASH = expected_profit_duopoly()

# evaluation
R_MIN_JOINT = 0
'''