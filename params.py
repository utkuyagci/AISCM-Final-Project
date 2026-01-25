"""
Parameter configuration and benchmark calculations for the Price-Setting Newsvendor simulation
"""

import numpy as np
from scipy.stats import norm

## parameters 

# profit structure
C = 3.0  # Unit cost (supplier provides at fixed cost)

# price-demand response function: D(p) = A - B*p + epsilon
# where epsilon ~ N(0, NOISE_STD^2)
A_INTERCEPT = 100.0  # Demand intercept (max demand at p=0)
B_SLOPE = 3.5        # Price sensitivity (demand drop per unit price increase)
NOISE_STD = 10.0     # Standard deviation of demand noise

# action spaces
# Price agent action space
P_LOWER = 0.0
P_UPPER = 30.0
P_STEP_SIZE = 0.1
def action_space_p() -> np.ndarray:
    """ return discrete prices: [P_LOWER, P_LOWER+step, ..., P_UPPER]"""
    return np.arange(P_LOWER, P_UPPER + P_STEP_SIZE, P_STEP_SIZE)

# Quantity agent action space  
Q_LOWER = 0
Q_UPPER = 40
def action_space_q() -> np.ndarray:
    """ return discrete order quantities: [Q_LOWER, ..., Q_UPPER]"""
    return np.arange(Q_LOWER, Q_UPPER + 1)

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

# Partner prediction settings
USE_PARTNER_PREDICTION = False  # Toggle between basic and sophisticated approach
PARTNER_WINDOW = 30              # Moving average window for partner prediction
PARTNER_SIMILARITY_THRESHOLD = 2.0  # Distance threshold for weighting in sophisticated mode

# simulation
ROUNDS = 365
SEED = 42

## benchmarks

def expected_sales(q: float, mean_demand: float, std_demand: float) -> float:
    """
    Expected sales E[min(q, D)] where D ~ N(mean_demand, std_demand^2)
    """
    if q <= 0:
        return 0.0
    
    # Standardize
    z = (q - mean_demand) / std_demand if std_demand > 0 else 0
    
    # E[min(q, D)] = q * Φ(z) + (μ - q) * φ(z) + μ * (1 - Φ(z))
    # Simplified: E[min(q, D)] = μ - (μ - q) * Φ(z) - σ * φ(z)
    phi_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    return mean_demand * (1 - phi_z) + q * phi_z - std_demand * pdf_z

def compute_optimal_joint(p_min=P_LOWER, p_max=P_UPPER, p_step=0.5,
                          q_min=Q_LOWER, q_max=Q_UPPER, 
                          n_sims=50000, seed=SEED) -> tuple:
    """
    Compute optimal (p*, q*) via grid search with Monte Carlo simulation
    Returns: (optimal_price, optimal_quantity, expected_profit)
    """
    rng = np.random.default_rng(seed)
    
    best_p = None
    best_q = None
    best_profit = -np.inf
    
    # Grid search over price-quantity pairs
    prices = np.arange(p_min, p_max + p_step, p_step)
    quantities = np.arange(q_min, q_max + 1, 1)
    
    for p in prices:
        # Expected demand at this price
        mean_demand = A_INTERCEPT - B_SLOPE * p
        
        if mean_demand <= 0:
            continue
            
        for q in quantities:
            # Monte Carlo simulation
            demands = mean_demand + rng.normal(0, NOISE_STD, size=n_sims)
            demands = np.maximum(0, demands)  # Demand can't be negative
            
            sales = np.minimum(q, demands)
            profits = p * sales - C * q
            
            avg_profit = np.mean(profits)
            
            if avg_profit > best_profit:
                best_profit = avg_profit
                best_p = p
                best_q = q
    
    return float(best_p), int(best_q), float(best_profit)

def compute_sequential_optimal(leader='price', n_sims=50000, seed=SEED) -> tuple:
    """
    Compute optimal solution when one agent moves first (Stackelberg)
    leader: 'price' (price sets first, quantity responds) or 'quantity' (quantity sets first, price responds)
    Returns: (optimal_price, optimal_quantity, expected_profit)
    """
    rng = np.random.default_rng(seed)
    
    if leader == 'price':
        # Price agent moves first, quantity agent responds optimally
        best_p = None
        best_q = None
        best_profit = -np.inf
        
        for p in np.arange(P_LOWER, P_UPPER + 0.5, 0.5):
            mean_demand = A_INTERCEPT - B_SLOPE * p
            if mean_demand <= 0:
                continue
            
            # Quantity agent's best response given price p
            q_best_profit = -np.inf
            q_best = None
            
            for q in range(Q_LOWER, Q_UPPER + 1):
                demands = mean_demand + rng.normal(0, NOISE_STD, size=n_sims)
                demands = np.maximum(0, demands)
                sales = np.minimum(q, demands)
                profit = np.mean(p * sales - C * q)
                
                if profit > q_best_profit:
                    q_best_profit = profit
                    q_best = q
            
            if q_best_profit > best_profit:
                best_profit = q_best_profit
                best_p = p
                best_q = q_best
        
        return float(best_p), int(best_q), float(best_profit)
    else:
        # Quantity moves first (less realistic, but for comparison)
        # Similar logic but quantity commits first
        raise NotImplementedError("Quantity-first not yet implemented")

# Compute benchmarks at module load
print("Computing optimal benchmarks (this may take a moment)...")
P_OPT, Q_OPT, PROFIT_OPT = compute_optimal_joint()
P_SEQ, Q_SEQ, PROFIT_SEQ = compute_sequential_optimal(leader='price')
print(f"Optimal joint: p={P_OPT:.2f}, q={Q_OPT}, profit={PROFIT_OPT:.2f}")
print(f"Sequential (price-first): p={P_SEQ:.2f}, q={Q_SEQ}, profit={PROFIT_SEQ:.2f}")