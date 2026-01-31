"""
Parameter configuration and benchmark calculations for the Price-Setting Newsvendor simulation
"""

import numpy as np
from scipy.stats import norm

## parameters 

# profit structure
#C = 3.0  # Unit cost (supplier provides at fixed cost)
MANUFACTURING_COST = 2.0
SUPPLIER_TEAM_PLAYER = True  # Default: supplier charges manufacturing cost

# price-demand response function: D(p) = A - B*p + epsilon
# where epsilon ~ N(0, NOISE_STD^2)
A_INTERCEPT = 100.0  # Demand intercept (max demand at p=0)
B_SLOPE = 3.5        # Price sensitivity (demand drop per unit price increase)
NOISE_STD = 10.0     # Standard deviation of demand noise

# action spaces
# Price agent action space
P_LOWER = 0.0
P_UPPER = 30.0
P_STEP_SIZE = 0.5
def action_space_p() -> np.ndarray:
    """ return discrete prices: [P_LOWER, P_LOWER+step, ..., P_UPPER]"""
    return np.arange(P_LOWER, P_UPPER + P_STEP_SIZE, P_STEP_SIZE)

# Quantity agent action space  
Q_LOWER = 0
Q_UPPER = 40
def action_space_q() -> np.ndarray:
    """ return discrete order quantities: [Q_LOWER, ..., Q_UPPER]"""
    return np.arange(Q_LOWER, Q_UPPER + 1)

#Supplier Action Space
C_LOWER = 0.0
C_UPPER = 30.0
SUPPLIER_MAX_COST = 15.0  # Price ceiling to prevent unrealistic supplier costs
C_STEP_SIZE = 1.0  # Larger step size for faster exploration
def action_space_c() -> np.ndarray:

    if SUPPLIER_TEAM_PLAYER:
        return np.array([MANUFACTURING_COST])

    # Use price ceiling instead of C_UPPER to limit supplier exploration
    return np.arange(C_LOWER, SUPPLIER_MAX_COST + C_STEP_SIZE, C_STEP_SIZE)


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

def compute_optimal_joint(cost = None, p_min=P_LOWER, p_max=P_UPPER, p_step=0.5,
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

    supplier_cost = cost if cost is not None else C                                                                     # Fallback to C = 3.0 in case of error
    
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
            profits = p * sales - supplier_cost * q
            
            avg_profit = np.mean(profits)
            
            if avg_profit > best_profit:
                best_profit = avg_profit
                best_p = p
                best_q = q
    
    return float(best_p), int(best_q), float(best_profit)

def compute_sequential_optimal(leader='price', cost = None, n_sims=50000, seed=SEED) -> tuple:
    """
    Compute optimal solution when one agent moves first (Stackelberg)
    leader: 'price' (price sets first, quantity responds) or 'quantity' (quantity sets first, price responds)
    Returns: (optimal_price, optimal_quantity, expected_profit)
    """
    rng = np.random.default_rng(seed)

    supplier_cost = cost if cost is not None else C
    
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
                profit = np.mean(p * sales - supplier_cost * q)
                
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

#Old static benchmark
'''
# Compute benchmarks at module load
print("Computing optimal benchmarks (this may take a moment)...")
P_OPT, Q_OPT, PROFIT_OPT = compute_optimal_joint()
P_SEQ, Q_SEQ, PROFIT_SEQ = compute_sequential_optimal(leader='price')
print(f"Optimal joint: p={P_OPT:.2f}, q={Q_OPT}, profit={PROFIT_OPT:.2f}")
print(f"Sequential (price-first): p={P_SEQ:.2f}, q={Q_SEQ}, profit={PROFIT_SEQ:.2f}")
'''

#New: Pre-Computing dynamic Benchmarks based on supplier price c
# This function can be called to recompute benchmarks
def _compute_benchmarks():
    global PROFIT_OPTIMA_MAP, SEQ_PROFIT_OPTIMA_MAP, P_OPTIMA_MAP, Q_OPTIMA_MAP
    global SUPPLIER_MAX_PROFIT, SUPPLIER_OPTIMAL_COST, SUPPLIER_OPTIMAL_QUANTITY
    global _last_team_player_mode
    
    print("Computing optimal benchmarks (this may take a moment)...")
    PROFIT_OPTIMA_MAP = {}
    SEQ_PROFIT_OPTIMA_MAP = {}
    P_OPTIMA_MAP = {}  # Store optimal prices for each cost
    Q_OPTIMA_MAP = {}  # Store optimal quantities for each cost
    SUPPLIER_MAX_PROFIT = -np.inf
    SUPPLIER_OPTIMAL_COST = None
    SUPPLIER_OPTIMAL_QUANTITY = None

    # Get all possible costs (changes based on SUPPLIER_TEAM_PLAYER)
    if SUPPLIER_TEAM_PLAYER:
        # Team player: supplier always charges manufacturing cost
        # Only need benchmark for this one cost
        retailer_costs_to_compute = np.array([MANUFACTURING_COST])
        # But for supplier optimum calculation, use full range
        supplier_cost_range = np.arange(C_LOWER, C_UPPER + C_STEP_SIZE, C_STEP_SIZE)
    else:
        # Competitive: supplier can choose from limited action space
        # Compute benchmarks for all possible supplier costs
        retailer_costs_to_compute = action_space_c()
        supplier_cost_range = retailer_costs_to_compute

    # Compute retailer optimums for all costs the supplier might actually choose
    for c_val in retailer_costs_to_compute:
        p_opt, q_opt, newsvendor_profit_opt = compute_optimal_joint(cost = c_val, n_sims=5000)
        PROFIT_OPTIMA_MAP[c_val] = newsvendor_profit_opt
        P_OPTIMA_MAP[c_val] = p_opt
        Q_OPTIMA_MAP[c_val] = q_opt

        p_seq, q_seq, newsvendor_profit_seq = compute_sequential_optimal(leader='price', cost = c_val, n_sims=5000)
        SEQ_PROFIT_OPTIMA_MAP[c_val] = newsvendor_profit_seq

    # Compute supplier optimum over full cost range
    for c_val in supplier_cost_range:
        # Need to compute q_opt for this cost if not already done
        if c_val not in PROFIT_OPTIMA_MAP:
            p_opt, q_opt, _ = compute_optimal_joint(cost = c_val, n_sims=5000)
            p_seq, q_seq, _ = compute_sequential_optimal(leader='price', cost = c_val, n_sims=5000)
        else:
            q_opt = Q_OPTIMA_MAP[c_val]
            p_seq, q_seq, _ = compute_sequential_optimal(leader='price', cost = c_val, n_sims=5000)
        
        # Supplier profit from joint optimization case
        suppl_profit_joint = q_opt * (c_val - MANUFACTURING_COST)
        
        # Supplier profit from sequential case (if q_seq differs)
        suppl_profit_seq = q_seq * (c_val - MANUFACTURING_COST)
        
        # Track maximum supplier profit across both cases
        max_suppl_profit_for_c = max(suppl_profit_joint, suppl_profit_seq)
        
        if max_suppl_profit_for_c > SUPPLIER_MAX_PROFIT:
            SUPPLIER_MAX_PROFIT = max_suppl_profit_for_c
            SUPPLIER_OPTIMAL_COST = c_val
            SUPPLIER_OPTIMAL_QUANTITY = q_opt if suppl_profit_joint >= suppl_profit_seq else q_seq

    _last_team_player_mode = SUPPLIER_TEAM_PLAYER
    
    # Print results
    if not SUPPLIER_TEAM_PLAYER:
        print(f"Supplier global optimum: cost={SUPPLIER_OPTIMAL_COST:.2f}, quantity={SUPPLIER_OPTIMAL_QUANTITY}, profit={SUPPLIER_MAX_PROFIT:.2f}")
    
    if MANUFACTURING_COST in PROFIT_OPTIMA_MAP:
        opt_p = P_OPTIMA_MAP[MANUFACTURING_COST]
        opt_q = Q_OPTIMA_MAP[MANUFACTURING_COST]
        print(f"Retailer optimal at manufacturing cost (c={MANUFACTURING_COST:.2f}): price={opt_p:.2f}, quantity={opt_q}")

# Force recomputation if supplier mode changes or first load
_current_team_player = SUPPLIER_TEAM_PLAYER
if 'PROFIT_OPTIMA_MAP' not in globals() or '_last_team_player_mode' not in globals() or _last_team_player_mode != _current_team_player:
    _compute_benchmarks()
