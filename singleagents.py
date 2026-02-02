

import numpy as np
from mesa import Agent, Model
from mesa.datacollection import DataCollector

import params
from agents import JointPQGreedyAgent, JointPQUcbAgent, JointPQThompsonAgent


def regret(model, agent=None):
    """
    Calculate per-round regret vs optimal joint profit for single-agent model.
    """
    agent = model.agents[0]
    retailer_profit = model.rewards.get(agent, 0.0)

    current_c = params.MANUFACTURING_COST
    optimal_retailer_profit = params.PROFIT_OPTIMA_MAP.get(current_c, 0.0)

    return optimal_retailer_profit - retailer_profit


class SingleAgentPQModel(Model):
    """
    Single-agent model: one firm chooses price p and quantity q jointly each round.
    
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

        # Rewards dict 
        self.rewards = {}

        # Create exactly one agent (selectable)
        agent_type_l = (agent_type or "greedy").lower()
        agent_map = {
            "greedy": JointPQGreedyAgent,
            "ucb": JointPQUcbAgent,
            "thompson": JointPQThompsonAgent,
            "ts": JointPQThompsonAgent,
        }
        if agent_type_l not in agent_map:
            raise ValueError(f"Unknown agent_type='{agent_type}'. Choose from {sorted(agent_map.keys())}.")
        agent_map[agent_type_l].create_agents(self, n=1)
        

    
        self.datacollector = DataCollector(
            model_reporters={
                "Regret": regret,
                "t": lambda m: m.t,
                "p": lambda m: float(m.p),
                "q": lambda m: int(m.q),
                "e": lambda m: params.epsilon_at(m.t, params.ROUNDS),
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

        expected = params.A_INTERCEPT - (params.B_SLOPE * p)
        realized = expected + self.rng.normal(0.0, params.NOISE_STD)
        realized = max(0.0, realized)  # Demand can't be negative

        sales = min(float(q), float(realized))
        profit = (p * sales) - (params.MANUFACTURING_COST * float(q))

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

        # 3) Store reward 
        agent = self.agents[0]
        self.rewards[agent] = profit
        agent.reward = profit
        agent.reward_cum += profit

        # 4) Collect data
        self.datacollector.collect(self)

        # 5) Learn
        self.agents.shuffle_do("update_belief")

        self.t += 1


