

import numpy as np
from mesa import Agent, Model
from mesa.datacollection import DataCollector

import params
from agents import JointPQGreedyAgent, JointPQUcbAgent, JointPQThompsonAgent


class JointPQGreedyAgent(Agent):
    """
    One agent that chooses a (price, quantity) pair jointly.
    ε-greedy over discrete (p,q) pairs; learns expected reward per pair.
    """

    def __init__(self, model):

        super().__init__(model)

        # Joint action space: list of (p,q) pairs
        prices = params.action_space_p()
        quantities = params.action_space_q()
        self.action_space = [(float(p), int(q)) for p in prices for q in quantities]
        self.n_actions = len(self.action_space)

        # Bandit estimates: incremental mean reward per arm
        self.Q = np.zeros(self.n_actions, dtype=float)
        self.N = np.zeros(self.n_actions, dtype=int)

        # Selected action
        self.action_idx = 0
        self.p, self.q = self.action_space[self.action_idx]

        # Reward tracking (same style as your existing agents)
        self.reward = 0.0
        self.reward_cum = 0.0

    def select_action(self):
        """Choose a (p,q) pair via ε-greedy."""
        t = self.model.t
        eps = params.epsilon_at(t, params.ROUNDS)
        self.model.current_eps = eps  # keep in model for logging

        rng = self.model.rng
        if rng.random() < eps:
            self.action_idx = int(rng.integers(0, self.n_actions))
        else:
            best = np.flatnonzero(self.Q == self.Q.max())
            self.action_idx = int(rng.choice(best))

        self.p, self.q = self.action_space[self.action_idx]

    def update_belief(self):
        """
        Update reward estimate for the last chosen (p,q).
        Uses the realized reward computed by the model.
        """
        r = float(self.reward)
        a = int(self.action_idx)
        self.N[a] += 1
        self.Q[a] += (r - self.Q[a]) / self.N[a]


class SingleAgentPQModel(Model):
    """
    Single-agent model: one firm chooses price p and quantity q jointly each round.
    No scheduler; activation uses model.agents.shuffle_do(...) like your model.py.
    """

    def __init__(self, agent_type: str = "greedy", seed: int = params.SEED):
        super().__init__()

        self.agent_type = agent_type
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.current_eps = params.epsilon_at(self.t, params.ROUNDS)

        # Logging variables
        self.p = 0.0
        self.q = 0
        self.expected_demand = 0.0
        self.realized_demand = 0.0
        self.sales = 0.0

        # Rewards dict (same pattern as your NewsVendorModel)
        self.rewards = {}

        # Create exactly one agent (selectable)
        agent_type_l = (agent_type or "greedy").lower()
        agent_map = {
            "Greedy": JointPQGreedyAgent,
            "UCB": JointPQUcbAgent,
            "Thompson": JointPQThompsonAgent,
            "ts": JointPQThompsonAgent,
        }
        if agent_type_l not in agent_map:
            raise ValueError(f"Unknown agent_type='{agent_type}'. Choose from {sorted(agent_map.keys())}.")
        agent_map[agent_type_l].create_agents(self, n=1)
        # Now you have self.agents (AgentSet) with one agent

        # Optional: DataCollector (similar to your model.py style)
        self.datacollector = DataCollector(
            model_reporters={
                "t": lambda m: m.t,
                "epsilon": lambda m: float(m.current_eps),
                "p": lambda m: float(m.p),
                "q": lambda m: int(m.q),
                "expected_demand": lambda m: float(m.expected_demand),
                "realized_demand": lambda m: float(m.realized_demand),
                "sales": lambda m: float(m.sales),
                "profit": lambda m: float(m.rewards.get(m.agents[0], 0.0)),
            },
            agent_reporters={
                "action_idx": lambda a: int(a.action_idx),
                "reward": lambda a: float(a.reward),
                "reward_cum": lambda a: float(a.reward_cum),
            },
        )

    def market_response(self, action_idx: int):
        """
        Market response for chosen joint action.
        Returns (profit, realized_demand).
        """
        agent = self.agents[0]
        p, q = agent.action_space[int(action_idx)]

        expected = params.A_intercept - (params.B_Slope * p)
        realized = expected + self.rng.normal(0.0, params.Noise_random)
        realized = max(0.0, realized)  # Demand can't be negative

        sales = min(float(q), float(realized))
        profit = (p * sales) - (params.C * float(q))

        # Save for logging
        self.p, self.q = float(p), int(q)
        self.expected_demand = float(expected)
        self.realized_demand = float(realized)
        self.sales = float(sales)

        return float(profit), float(realized)

    def step(self):
        # 1) Agent chooses (p,q)
        self.agents.shuffle_do("select_action")
        a_idx = int(self.agents[0].action_idx)

        # 2) Environment realizes demand and profit
        profit, realized_d = self.market_response(a_idx)

        # 3) Store reward (consistent with your model.py style)
        agent = self.agents[0]
        self.rewards[agent] = profit
        agent.reward = profit
        agent.reward_cum += profit

        # 4) Collect data
        self.datacollector.collect(self)

        # 5) Learn
        self.agents.shuffle_do("update_belief")

        self.t += 1


# Quick usage example (optional):
# from single_agent_pq import SingleAgentPQModel
# m = SingleAgentPQModel()
# for _ in range(params.ROUNDS):
#     m.step()
# df_model = m.datacollector.get_model_vars_dataframe()
# df_agent = m.datacollector.get_agent_vars_dataframe()
