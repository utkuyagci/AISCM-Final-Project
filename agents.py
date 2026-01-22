import numpy as np
from scipy.stats import poisson
from mesa import Agent
import params 

class GreedyAgent(Agent):
    """ε-greedy bandit agent"""

    def __init__(self, model, role = 'quantity'):
        """Initialize agent"""
        super().__init__(model)

        self.role = role
        # Action space depending on the role argument from params
        if role == 'price':
            self.action_space = params.action_space_p()
        else:
            self.action_space = params.action_space_q()

        self.n_actions = len(self.action_space)

        # Bandit statistics
        self.counts = np.zeros(self.n_actions, dtype=int)
        self.average_reward = np.zeros(self.n_actions, dtype=float)

        # Epsilon for ε-greedy (global decay function)
        self.eps = params.epsilon_at(model.t, params.ROUNDS)

        # Selected action + reward
        self.action_idx = None
        self.action = None
        self.reward = 0.0

        # Evaluation
        self.reward_cum = 0.0
        self.regret_cum = 0

    def select_action(self):
        """
        Select an action using ε-greedy
        Update selected action attributes (index and quantity)
        """
        # Exploration
        if self.random.random() < self.eps:
            action_idx = self.random.randrange(self.n_actions)

        # Exploitation
        else:
            action_idx = int(np.argmax(self.average_reward))

        # Store selections
        self.action_idx = action_idx
        self.action = self.action_space[action_idx]


    def update_belief(self):
        """
        Update reward estimates using incremental mean update
        Update ε for the next time step
        """
        a_idx = self.action_idx

        # Reward from environment
        r = self.model.rewards[self]         

        # Incremental update
        n = self.counts[a_idx] + 1
        self.counts[a_idx] = n
        self.average_reward[a_idx] += (r - self.average_reward[a_idx]) / n

        # Update epsilon for the next round t+1
        self.eps = params.epsilon_at(self.model.t + 1, params.ROUNDS)

class UcbAgent(Agent):
    """UCB1 bandit agent"""

    def __init__(self, model, role = 'quantity'):
        super().__init__(model)

        self.role = role
        # Action space depending on the role argument from params
        if role == 'price':
            self.action_space = params.action_space_p()
        else:
            self.action_space = params.action_space_q()

        self.n_actions = len(self.action_space)

        # Estimates
        self.counts = np.zeros(self.n_actions, dtype=int)
        self.means  = np.zeros(self.n_actions, dtype=float)

        # Selected action + reward
        self.action_idx = None
        self.action = None
        self.reward = 0.0

        # Evaluation
        self.reward_cum = 0.0
        self.regret_cum = 0.0

    def select_action(self):
        # Time index for exploration term (avoid log(0))
        t = max(1, int(self.model.t))
        c = float(params.UCB_C)

        # Ensure each arm is tried at least once
        untried = np.flatnonzero(self.counts == 0)
        if len(untried) > 0:
            action_idx = int(self.random.choice(list(untried)))
        else:
            bonus = c * np.sqrt((2.0 * np.log(t)) / self.counts)
            ucb = self.means + bonus
            m = float(np.max(ucb))
            best = np.flatnonzero(ucb == m)
            action_idx = int(self.random.choice(list(best)))

        self.action_idx = action_idx
        self.action = int(self.action_space[action_idx])

    def update_belief(self):
        """
        Update reward estimates using incremental mean update
        Update number of tries for selected action
        """
        a_idx = self.action_idx
        r = float(self.model.rewards[self])
        n = self.counts[a_idx] + 1
        self.counts[a_idx] = n
        self.means[a_idx] += (r - self.means[a_idx]) / n


# Helper function for Thompson sampling 
def exp_sales_poisson(q: int, lam: float):
    """
    Expected sales E[min(q, X)] for X ~ Poisson(lam)
    """
    if q <= 0:
        return 0.0

    # P(X >= q) = 1 - CDF(q-1)
    tail = 1.0 - poisson.cdf(q - 1, lam)

    # sum_{k=0}^{q-1} k * pmf(k)
    k_vals = np.arange(1, q)           
    partial = np.sum(k_vals * poisson.pmf(k_vals, lam))

    return float(q * tail + partial)


class ThompsonAgent(Agent):
    """ Thompson sampling agent """

    def __init__(self, model, alpha0 = 1.0, beta0 = 1.0):
        super().__init__(model)

        # Action space
        self.action_space = params.action_space()
        self.n_actions = len(self.action_space)

        # Gamma prior hyperparams
        self.alpha, self.beta = alpha0, beta0

        # Selected action + reward 
        self.action_idx = None
        self.action = None
        self.reward = 0.0
        self.reward_cum = 0.0
        self.regret_cum = 0.0

    def _expected_profit_for_lambda(self, lam: float):
        qs = self.action_space.astype(int)
        exp_sales = np.array([exp_sales_poisson(int(q), lam) for q in qs], dtype=float)
        return params.P * exp_sales - params.C * qs

    def select_action(self):
        '''
        Action choice: sample λ̂ ~ Gamma(α, β); pick q maximizing
        E[π(q)|λ̂]= P * E[min(q, Poisson(λ̂))] - C * q
        '''
        rng = self.model.rng
        
        # Sample λ 
        lam_hat = float(rng.gamma(shape=self.alpha, scale=1.0 / self.beta))
        exp_profit = self._expected_profit_for_lambda(lam_hat)

        #Choose arm with highest expected value
        m = float(np.max(exp_profit))
        best = np.flatnonzero(exp_profit == m)
        self.action_idx = int(rng.choice(best))
        self.action = int(self.action_space[self.action_idx])

    def update_belief(self):
        # Effective demand observation
        is_first = (self is self.model.agents[0])
        d_eff_obs = self.model.d1_eff if is_first else self.model.d2_eff
        x = int(d_eff_obs) 

        # Gamma–Poisson conjugate update
        self.alpha += x
        self.beta  += 1.0
