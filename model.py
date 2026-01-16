import numpy as np

from mesa import Model
from mesa.datacollection import DataCollector

from agents import GreedyAgent, UcbAgent, ThompsonAgent
import params

def create_agents(model, agent_types): 
    ''' create agents given list of agent types'''
    for i in range(len(agent_types)):
        if agent_types[i] == "greedy": 
            GreedyAgent.create_agents(model,n=1)
        elif agent_types[i] == "ucb":
            UcbAgent.create_agents(model,n=1)
        elif agent_types[i] == "thompson":
            ThompsonAgent.create_agents(model, n=1)

def regret(model):
    a0, a1 = model.agents[0], model.agents[1]
    r1 = model.rewards.get(a0, 0.0)
    r2 = model.rewards.get(a1, 0.0)
    return [params.MU1_NASH - r1, params.MU2_NASH - r2]

class NewsVendorModel(Model):

    def __init__(self, agent_type=['greedy', 'greedy']):
        super().__init__()

        # Simulation & learning 
        self.rng = np.random.default_rng(params.SEED) 
        self.t = 0 
        self.current_eps = params.epsilon_at(self.t, params.ROUNDS) 
        self.d1_eff, self.d2_eff = 0, 0

        # Agent structure & rewards
        self.action_space = params.action_space() 
        self.num_agents = len(agent_type)
        self.rewards = {}
        create_agents(self, agent_type)

        # Initialization of data collector
        self.datacollector = DataCollector(
            model_reporters={
                "Regret": regret,
            },
            agent_reporters={
                "Order Quantity": "action",
                "Reward": "reward",
                "Cummulative Reward": "reward_cum",
                "Cummulative Regret": "regret_cum"
            },
        )


    # Helper function for demand realization
    def sample_demands(self) -> tuple:
        '''Helper function for demand realization'''
        d1 = self.rng.poisson(params.LAM)
        d2 = self.rng.poisson(params.LAM)
        return d1, d2

    def market_response(self, a_idx_1, a_idx_2) -> tuple:
        '''Market response to agent actions'''
        # Agent decisions
        q1 = float(self.action_space[a_idx_1])
        q2 = float(self.action_space[a_idx_2])

        # (Effective) demand realization
        d1, d2 = self.sample_demands()
        d1_eff = d1 + params.RHO * max(0.0, d2 - q2)
        d2_eff = d2 + params.RHO * max(0.0, d1 - q1)

        # Agents receive rewards
        sales1, sales2 = min(q1, d1_eff), min(q2, d2_eff)
        r1 = params.P * sales1 - params.C * q1
        r2 = params.P * sales2 - params.C * q2
        return r1, r2, d1_eff, d2_eff

    def step(self):
        # 1) Agents choose actions
        self.agents.shuffle_do("select_action")

        # 2) Demand realizes 
        a1_idx, a2_idx = self.agents[0].action_idx, self.agents[1].action_idx
        r1, r2, d1_eff, d2_eff = self.market_response(a1_idx, a2_idx)
        self.d1_eff, self.d2_eff = d1_eff, d2_eff

        # 3) Agents receive rewards
        self.rewards[self.agents[0]], self.rewards[self.agents[1]]  = r1, r2
        self.agents[0].reward, self.agents[1].reward = r1, r2 

        self.agents[0].reward_cum = self.agents[0].reward_cum + r1
        self.agents[1].reward_cum = self.agents[1].reward_cum + r2

        self.agents[0].regret_cum = self.agents[0].regret_cum + regret(self)[0]
        self.agents[1].regret_cum = self.agents[1].regret_cum + regret(self)[1]

        # 4) Collect data (one time per round, after market response)
        self.datacollector.collect(self)

        # 5) Agents learn 
        self.agents.shuffle_do("update_belief")
        self.t += 1