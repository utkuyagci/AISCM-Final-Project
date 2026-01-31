import numpy as np

from mesa import Model
from mesa.datacollection import DataCollector

from agents import GreedyAgent, UcbAgent
import params

def create_agents(model, agent_types): 
    ''' 
    Create two agents: one for price, one for quantity
    agent_types[0] controls price, agent_types[1] controls quantity
    '''
    roles = ['price', 'quantity', 'supplier']
    
    for i in range(len(agent_types)):
        role = roles[i]
        agent_type = agent_types[i]
        
        if agent_type == "greedy":
            GreedyAgent(model, role=role)
        elif agent_type == "ucb":
            UcbAgent(model, role=role)
        # Thompson commented out for now
        # elif agent_type == "thompson":
        #     ThompsonAgent(model, role=role)

def regret(model, agent = None):

    '''
    """Calculate per-round regret vs optimal joint profit"""
    # Both agents get same profit in cooperative setting
    profit = model.rewards.get(model.agents[0], 0.0)
    return params.PROFIT_OPT - profit
    '''

    supplier = model.agents[2]
    retailer_profit = model.rewards.get(model.agents[0], 0.0)

    role = agent.role if agent is not None else 'price'                                                                 #Important Case Distinction: Can be called by DataCollector -> agent = None -> return Newsvendor regret

    if role in ['price', 'quantity']:
        current_c = supplier.action
        optimal_retailer_profit = params.PROFIT_OPTIMA_MAP.get(current_c, 0.0)

        return optimal_retailer_profit - retailer_profit

    elif role == 'supplier':
        supplier_profit = model.rewards.get(supplier, 0.0)
        optimal_supplier_profit = params.SUPPLIER_MAX_PROFIT

        return optimal_supplier_profit - supplier_profit

    return 0

class NewsVendorModel(Model):
    """Price-setting newsvendor with two coordinating agents"""

    def __init__(self, agent_type=['greedy', 'greedy', 'greedy']):
        super().__init__()

        # Simulation & learning 
        self.rng = np.random.default_rng(params.SEED) 
        self.t = 0 
        self.current_eps = params.epsilon_at(self.t, params.ROUNDS)
        
        # Track demand realization (for analysis)
        self.demand_realized = 0.0

        # Agent structure & rewards
        self.num_agents = len(agent_type)
        self.rewards = {}
        create_agents(self, agent_type)

        # Initialization of data collector
        self.datacollector = DataCollector(
            model_reporters={
                "Regret": regret,                                                                                       #Newsvendor regret -> for q and p agent, NOT for supplier agent
                "Demand": lambda m: m.demand_realized
            },
            agent_reporters={
                "Action": "action",  # Will be price or quantity depending on agent
                "Reward": "reward",
                "Cummulative Reward": "reward_cum",
                "Cummulative Regret": "regret_cum",
                "Role": "role"
            },
        )

    def market_response(self, price_idx, quantity_idx, cost_idx) -> tuple:
        """
        Market response for price-setting newsvendor
        Returns: (profit, demand_realized)
        """
        # Get actual price and quantity from indices
        p = float(params.action_space_p()[price_idx])
        q = float(params.action_space_q()[quantity_idx])
        c = float(params.action_space_c()[cost_idx])
        
        # Demand function: D = A - B*p + noise
        expected_demand = params.A_INTERCEPT - params.B_SLOPE * p
        noise = self.rng.normal(0, params.NOISE_STD)
        realized_demand = expected_demand + noise
        
        # Demand can't be negative
        realized_demand = max(0.0, realized_demand)
        
        # Sales and profit
        sales = min(q, realized_demand)
        profit = p * sales - c * q

        supplier_profit = q * (c - params.MANUFACTURING_COST)
        
        return profit, realized_demand, supplier_profit

    def step(self):
        """Execute one round of the simulation"""
        # 1) Agents choose actions simultaneously
        self.agents.shuffle_do("select_action")

        # 2) Market responds to (price, quantity) pair
        price_agent = self.agents[0]  # First agent is price setter
        quantity_agent = self.agents[1]  # Second agent is quantity setter
        supplier_agent = self.agents[2]
        
        profit, demand, suppl_profit = self.market_response(price_agent.action_idx, quantity_agent.action_idx, supplier_agent.action_idx)
        self.demand_realized = demand

        # 3) Both agents (q and p) receive same profit (cooperative setting)
        self.rewards[price_agent] = profit
        self.rewards[quantity_agent] = profit

        self.rewards[supplier_agent] = suppl_profit
        
        price_agent.reward = profit
        quantity_agent.reward = profit

        supplier_agent.reward = suppl_profit
        
        price_agent.reward_cum += profit
        quantity_agent.reward_cum += profit
        supplier_agent.reward_cum += suppl_profit
        
        # Calculate regret for retailer agents only (not supplier)

        price_agent.regret_cum += regret(self, price_agent)
        quantity_agent.regret_cum += regret(self, quantity_agent)

        # Supplier regret not tracked (focuses on profit instead)
        # supplier_agent.regret_cum += regret(self, supplier_agent)

        # 4) Update partner history (for coordination)
        # Only price and quantity agents coordinate with each other
        price_agent.partner_history.append(quantity_agent.action)
        quantity_agent.partner_history.append(price_agent.action)
        
        # Supplier doesn't use partner prediction (acts independently)
        # supplier_agent.partner_history is not updated

        # 5) Collect data
        self.datacollector.collect(self)

        # 6) Agents learn from outcomes
        self.agents.shuffle_do("update_belief")
        
        self.t += 1