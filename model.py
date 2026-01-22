import numpy as np

from mesa import Model
from mesa.datacollection import DataCollector

from agents import GreedyAgent, UcbAgent, ThompsonAgent
import params

def create_agents(model, agent_types): 
    ''' create agents given list of agent types'''
    for i in range(len(agent_types)):

        current_role = 'price' if i == 0 else 'quantity'                    # We need to know which agent it is for the action space (initialization in agents.py -> init function)

        if agent_types[i] == "greedy": 
            GreedyAgent.create_agents(model,n=1, role = current_role)
        elif agent_types[i] == "ucb":
            UcbAgent.create_agents(model,n=1, role = current_role)
        elif agent_types[i] == "thompson":
            ThompsonAgent.create_agents(model, n=1, role = current_role)

#Calculates Regret based on nash equilibrium (not applicable here)
'''
def regret(model):
    a0, a1 = model.agents[0], model.agents[1]
    r1 = model.rewards.get(a0, 0.0)
    r2 = model.rewards.get(a1, 0.0)
    return [params.MU1_NASH - r1, params.MU2_NASH - r2]
'''
class NewsVendorModel(Model):

    def __init__(self, agent_type=['greedy', 'greedy']):
        super().__init__()

        # Simulation & learning 
        self.rng = np.random.default_rng(params.SEED) 
        self.t = 0 
        self.current_eps = params.epsilon_at(self.t, params.ROUNDS) 
        self.d1_eff, self.d2_eff = 0, 0

        # Agent structure & rewards
        #self.action_space = params.action_space()                                  Comment: Doesn't make sense to have a global action space since the 2 agents get their own (p and q) in our context
        self.num_agents = len(agent_type)
        self.rewards = {}
        create_agents(self, agent_type)

        # Initialization of data collector
        self.datacollector = DataCollector(
            model_reporters={
                "Regret": lambda m: 0,                                              # Dont have a regret at the moment (commented out because of fals calculation for our case)
                "Demand": lambda m: m.d1_eff
            },
            agent_reporters={
                "Order Quantity": "action",
                "Reward": "reward",
                "Cummulative Reward": "reward_cum",
                "Cummulative Regret": "regret_cum"
            },
        )

    #Helper Function not needed at all -> include logic in market response function
    '''
    # Helper function for demand realization
    def sample_demands(self) -> tuple:
        #Helper function for demand realization
        d1 = self.rng.poisson(params.LAM)
        d2 = self.rng.poisson(params.LAM)
        return d1, d2

    def market_response(self, a_idx_1, a_idx_2) -> tuple:
        #Market response to agent actions
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
    '''
    #New Market Response Function
    def market_response(self, price_idx, quantity_idx) -> tuple:

        p = float(params.action_space_p()[price_idx])
        q = float(params.action_space_q()[quantity_idx])

        expected_demand = params.A_intercept - (params.B_Slope * p)
        realized_demand = expected_demand + self.rng.normal(0, params.Noise_random)

        realized_demand = max(0, realized_demand)                                                                       # Demand cant be negative

        sales = min(q, realized_demand)

        profit = (p * sales) - (params.C * q)

        return profit, profit, realized_demand, realized_demand


    def step(self):
        # 1) Agents choose actions
        self.agents.shuffle_do("select_action")

        # 2) Demand realizes 
        price_agent_idx = self.agents[0].action_idx                                                                     # Assumption: Agent 0 is ALWAYS Price -> Agent 1 always quantity (as defined in create_agents in model.py)
        quantity_agent_idx = self.agents[1].action_idx

        r1, r2, d1_realized, d2_realized = self.market_response(price_agent_idx, quantity_agent_idx)                    # Returns Profit & realized Demand (same for both agents)

        self.d1_eff, self.d2_eff = d1_realized, d2_realized                                                             # Relevant for Data Collection & Thompson Sampling (if we do it later)

        # 3) Agents receive rewards
        self.rewards[self.agents[0]] = r1
        self.rewards[self.agents[1]]  = r2

        self.agents[0].reward = r1
        self.agents[1].reward = r2

        self.agents[0].reward_cum = self.agents[0].reward_cum + r1
        self.agents[1].reward_cum = self.agents[1].reward_cum + r2

        # 4) Collect data (one time per round, after market response)
        self.datacollector.collect(self)

        # 5) Agents learn 
        self.agents.shuffle_do("update_belief")
        self.t += 1