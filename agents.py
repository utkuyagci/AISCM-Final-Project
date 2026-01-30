import numpy as np
from mesa import Agent
import params 

class GreedyAgent(Agent):
    """ε-greedy bandit agent with optional partner prediction"""

    def __init__(self, model, role='quantity'):
        """Initialize agent with role (price or quantity)"""
        super().__init__(model)
        
        self.role = role
        
        # Action space based on role
        if role == 'price':
            self.action_space = params.action_space_p()
        elif role == 'quantity':
            self.action_space = params.action_space_q()
        elif role == 'supplier':
            self.action_space = params.action_space_c()


        
        self.n_actions = len(self.action_space)
        
        # Standard bandit statistics
        self.counts = np.zeros(self.n_actions, dtype=int)
        self.average_reward = np.zeros(self.n_actions, dtype=float)
        
        # Partner tracking
        self.partner_history = []
        self.partner_prediction = None
        
        # Sophisticated approach: track (action_idx, partner_action, reward, time)
        if params.USE_PARTNER_PREDICTION:
            self.action_history = []
        
        # Epsilon for ε-greedy
        self.eps = params.epsilon_at(model.t, params.ROUNDS)
        
        # Selected action + reward
        self.action_idx = None
        self.action = None
        self.reward = 0.0
        
        # Evaluation
        self.reward_cum = 0.0
        self.regret_cum = 0
    
    def predict_partner_action(self):
        """Estimate partner's likely action from recent history"""
        if len(self.partner_history) == 0:
            # Default predictions
            if self.role == 'price':
                return 50.0  # Default quantity guess
            else:
                return 15.0  # Default price guess
        
        # Use moving average
        window = min(params.PARTNER_WINDOW, len(self.partner_history))
        recent = self.partner_history[-window:]
        return float(np.mean(recent))
    
    def _compute_contextual_reward(self, action_idx, predicted_partner):
        """
        Sophisticated approach: Estimate reward for action given predicted partner
        Weights past rewards by similarity of partner actions
        """
        if not hasattr(self, 'action_history') or len(self.action_history) == 0:
            return self.average_reward[action_idx]
        
        # Find all times we tried this action
        relevant_history = [
            (partner_act, reward) 
            for (act_idx, partner_act, reward, _) in self.action_history 
            if act_idx == action_idx
        ]
        
        if len(relevant_history) == 0:
            return self.average_reward[action_idx]
        
        # Weight by similarity to predicted partner action
        weighted_sum = 0.0
        total_weight = 0.0
        
        for past_partner, past_reward in relevant_history:
            # Similarity weight: closer partner actions get higher weight
            distance = abs(past_partner - predicted_partner)
            weight = 1.0 / (1.0 + distance / params.PARTNER_SIMILARITY_THRESHOLD)
            
            weighted_sum += weight * past_reward
            total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return self.average_reward[action_idx]

    def select_action(self):
        """Select action using ε-greedy with optional partner prediction"""
        # Update partner prediction
        self.partner_prediction = self.predict_partner_action()
        
        # Exploration
        if self.random.random() < self.eps:
            action_idx = self.random.randrange(self.n_actions)
        else:
            # Exploitation
            if params.USE_PARTNER_PREDICTION and hasattr(self, 'action_history'):
                # Sophisticated: Use contextual rewards
                contextual_rewards = np.array([
                    self._compute_contextual_reward(i, self.partner_prediction)
                    for i in range(self.n_actions)
                ])
                action_idx = int(np.argmax(contextual_rewards))
            else:
                # Basic: Use average rewards
                action_idx = int(np.argmax(self.average_reward))
        
        self.action_idx = action_idx
        self.action = self.action_space[action_idx]

    def update_belief(self):
        """Update reward estimates and history"""
        a_idx = self.action_idx
        r = self.model.rewards[self]
        
        # Update standard statistics (always)
        n = self.counts[a_idx] + 1
        self.counts[a_idx] = n
        self.average_reward[a_idx] += (r - self.average_reward[a_idx]) / n
        
        # Update action history for sophisticated approach
        if params.USE_PARTNER_PREDICTION and hasattr(self, 'action_history'):
            # Get partner's action this round
            partner_action = self.partner_history[-1] if len(self.partner_history) > 0 else 0.0
            self.action_history.append((a_idx, partner_action, r, self.model.t))
        
        # Update epsilon
        self.eps = params.epsilon_at(self.model.t + 1, params.ROUNDS)

class UcbAgent(Agent):
    """UCB1 bandit agent with optional partner prediction"""

    def __init__(self, model, role='quantity'):
        super().__init__(model)
        
        self.role = role
        
        # Action space based on role
        if role == 'price':
            self.action_space = params.action_space_p()
        elif role == 'quantity':
            self.action_space = params.action_space_q()
        elif role == 'supplier':
            self.action_space = params.action_space_c()
        
        self.n_actions = len(self.action_space)
        
        # UCB statistics
        self.counts = np.zeros(self.n_actions, dtype=int)
        self.means = np.zeros(self.n_actions, dtype=float)
        
        # Partner tracking
        self.partner_history = []
        self.partner_prediction = None
        
        # Sophisticated approach
        if params.USE_PARTNER_PREDICTION:
            self.action_history = []
        
        # Selected action + reward
        self.action_idx = None
        self.action = None
        self.reward = 0.0
        
        # Evaluation
        self.reward_cum = 0.0
        self.regret_cum = 0.0
    
    def predict_partner_action(self):
        """Estimate partner's likely action from recent history"""
        if len(self.partner_history) == 0:
            if self.role == 'price':
                return 50.0
            else:
                return 15.0
        
        window = min(params.PARTNER_WINDOW, len(self.partner_history))
        recent = self.partner_history[-window:]
        return float(np.mean(recent))
    
    def _compute_contextual_reward(self, action_idx, predicted_partner):
        """Same weighting logic as GreedyAgent"""
        if not hasattr(self, 'action_history') or len(self.action_history) == 0:
            return self.means[action_idx]
        
        relevant_history = [
            (partner_act, reward) 
            for (act_idx, partner_act, reward, _) in self.action_history 
            if act_idx == action_idx
        ]
        
        if len(relevant_history) == 0:
            return self.means[action_idx]
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for past_partner, past_reward in relevant_history:
            distance = abs(past_partner - predicted_partner)
            weight = 1.0 / (1.0 + distance / params.PARTNER_SIMILARITY_THRESHOLD)
            weighted_sum += weight * past_reward
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else self.means[action_idx]

    def select_action(self):
        """UCB selection with optional partner prediction"""
        self.partner_prediction = self.predict_partner_action()
        
        t = max(1, int(self.model.t))
        c = float(params.UCB_C)
        
        # Ensure each arm tried at least once
        untried = np.flatnonzero(self.counts == 0)
        if len(untried) > 0:
            action_idx = int(self.random.choice(list(untried)))
        else:
            if params.USE_PARTNER_PREDICTION and hasattr(self, 'action_history'):
                # Sophisticated: Contextual means + confidence bonus
                contextual_means = np.array([
                    self._compute_contextual_reward(i, self.partner_prediction)
                    for i in range(self.n_actions)
                ])
                bonus = c * np.sqrt((2.0 * np.log(t)) / self.counts)
                ucb = contextual_means + bonus
            else:
                # Basic: Standard UCB
                bonus = c * np.sqrt((2.0 * np.log(t)) / self.counts)
                ucb = self.means + bonus
            
            m = float(np.max(ucb))
            best = np.flatnonzero(ucb == m)
            action_idx = int(self.random.choice(list(best)))
        
        self.action_idx = action_idx
        self.action = self.action_space[action_idx]

    def update_belief(self):
        """Update UCB statistics"""
        a_idx = self.action_idx
        r = float(self.model.rewards[self])
        
        # Update standard UCB statistics
        n = self.counts[a_idx] + 1
        self.counts[a_idx] = n
        self.means[a_idx] += (r - self.means[a_idx]) / n
        
        # Update action history
        if params.USE_PARTNER_PREDICTION and hasattr(self, 'action_history'):
            partner_action = self.partner_history[-1] if len(self.partner_history) > 0 else 0.0
            self.action_history.append((a_idx, partner_action, r, self.model.t))


# Thompson Sampling agent commented out for now - needs redesign for price-demand setting
# Will be implemented later with appropriate demand distribution modeling
