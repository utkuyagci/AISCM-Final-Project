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


class ThompsonAgent(Agent):
    """
    Thompson Sampling agent using Gaussian (Normal-Gamma) priors.
    Suitable for continuous reward distributions (profit in newsvendor setting).
    """

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
        
        # Thompson Sampling with Normal-Gamma prior
        # For each action a, we model reward ~ N(μ_a, 1/τ_a)
        # Prior: μ_a ~ N(μ_0, 1/(κ_0 * τ_a)), τ_a ~ Gamma(α_0, β_0)
        
        # Prior hyperparameters (weakly informative)
        self.mu_0 = 0.0  # Prior mean
        self.kappa_0 = 0.001  # Prior precision scaling
        self.alpha_0 = 1.0  # Gamma shape
        self.beta_0 = 1.0  # Gamma rate
        
        # Posterior hyperparameters (updated with data)
        self.mu_n = np.full(self.n_actions, self.mu_0, dtype=float)
        self.kappa_n = np.full(self.n_actions, self.kappa_0, dtype=float)
        self.alpha_n = np.full(self.n_actions, self.alpha_0, dtype=float)
        self.beta_n = np.full(self.n_actions, self.beta_0, dtype=float)
        
        # Track observations for updates
        self.counts = np.zeros(self.n_actions, dtype=int)
        self.sum_rewards = np.zeros(self.n_actions, dtype=float)
        self.sum_sq_rewards = np.zeros(self.n_actions, dtype=float)
        
        # Partner tracking
        self.partner_history = []
        self.partner_prediction = None
        
        # Sophisticated approach with partner prediction
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
    
    def _compute_contextual_posterior(self, action_idx, predicted_partner):
        """
        Compute posterior parameters using only observations with similar partner actions.
        Similar to contextual approach in Greedy/UCB agents.
        """
        if not hasattr(self, 'action_history') or len(self.action_history) == 0:
            return (self.mu_n[action_idx], self.kappa_n[action_idx], 
                    self.alpha_n[action_idx], self.beta_n[action_idx])
        
        # Find relevant history for this action
        relevant_history = [
            (partner_act, reward) 
            for (act_idx, partner_act, reward, _) in self.action_history 
            if act_idx == action_idx
        ]
        
        if len(relevant_history) == 0:
            return (self.mu_n[action_idx], self.kappa_n[action_idx], 
                    self.alpha_n[action_idx], self.beta_n[action_idx])
        
        # Weight observations by similarity to predicted partner
        weighted_rewards = []
        total_weight = 0.0
        
        for past_partner, past_reward in relevant_history:
            distance = abs(past_partner - predicted_partner)
            weight = 1.0 / (1.0 + distance / params.PARTNER_SIMILARITY_THRESHOLD)
            weighted_rewards.append((weight, past_reward))
            total_weight += weight
        
        if total_weight < 1e-6:
            return (self.mu_n[action_idx], self.kappa_n[action_idx], 
                    self.alpha_n[action_idx], self.beta_n[action_idx])
        
        # Compute weighted statistics
        n_eff = total_weight
        weighted_mean = sum(w * r for w, r in weighted_rewards) / total_weight
        weighted_var = sum(w * (r - weighted_mean)**2 for w, r in weighted_rewards) / total_weight
        
        # Update posterior hyperparameters with weighted observations
        kappa_n = self.kappa_0 + n_eff
        mu_n = (self.kappa_0 * self.mu_0 + n_eff * weighted_mean) / kappa_n
        alpha_n = self.alpha_0 + n_eff / 2.0
        beta_n = self.beta_0 + 0.5 * n_eff * weighted_var + \
                 0.5 * (self.kappa_0 * n_eff / kappa_n) * (weighted_mean - self.mu_0)**2
        
        return (mu_n, kappa_n, alpha_n, beta_n)

    def select_action(self):
        """
        Thompson Sampling: Sample from posterior predictive distribution for each action,
        choose action with highest sampled value.
        """
        self.partner_prediction = self.predict_partner_action()
        
        # Use model's numpy RNG for gamma distribution
        rng = self.model.rng
        
        sampled_means = np.zeros(self.n_actions)
        
        for i in range(self.n_actions):
            if params.USE_PARTNER_PREDICTION and hasattr(self, 'action_history'):
                # Use contextual posterior
                mu_n, kappa_n, alpha_n, beta_n = self._compute_contextual_posterior(i, self.partner_prediction)
            else:
                # Use standard posterior
                mu_n = self.mu_n[i]
                kappa_n = self.kappa_n[i]
                alpha_n = self.alpha_n[i]
                beta_n = self.beta_n[i]
            
            # Sample precision from Gamma(alpha_n, beta_n)
            tau = rng.gamma(alpha_n, 1.0 / beta_n)
            
            # Sample mean from N(mu_n, 1/(kappa_n * tau))
            if tau > 0:
                std = np.sqrt(1.0 / (kappa_n * tau))
                sampled_mean = rng.normal(mu_n, std)
            else:
                sampled_mean = mu_n
            
            sampled_means[i] = sampled_mean
        
        # Choose action with highest sampled mean
        m = float(np.max(sampled_means))
        best = np.flatnonzero(sampled_means == m)
        action_idx = int(self.random.choice(list(best)))
        
        self.action_idx = action_idx
        self.action = self.action_space[action_idx]

    def update_belief(self):
        """Update posterior using Normal-Gamma conjugate update"""
        a_idx = self.action_idx
        r = float(self.model.rewards[self])
        
        # Update sufficient statistics
        n = self.counts[a_idx]
        self.counts[a_idx] = n + 1
        self.sum_rewards[a_idx] += r
        self.sum_sq_rewards[a_idx] += r * r
        
        # Compute sample mean and variance
        n_new = n + 1
        sample_mean = self.sum_rewards[a_idx] / n_new
        
        if n_new > 1:
            sample_var = (self.sum_sq_rewards[a_idx] - n_new * sample_mean**2) / (n_new - 1)
            sample_var = max(sample_var, 1e-6)  # Numerical stability
        else:
            sample_var = 1.0
        
        # Normal-Gamma conjugate update
        self.kappa_n[a_idx] = self.kappa_0 + n_new
        self.mu_n[a_idx] = (self.kappa_0 * self.mu_0 + n_new * sample_mean) / self.kappa_n[a_idx]
        self.alpha_n[a_idx] = self.alpha_0 + n_new / 2.0
        self.beta_n[a_idx] = self.beta_0 + 0.5 * n_new * sample_var + \
                             0.5 * (self.kappa_0 * n_new / self.kappa_n[a_idx]) * (sample_mean - self.mu_0)**2
        
        # Update action history for partner prediction
        if params.USE_PARTNER_PREDICTION and hasattr(self, 'action_history'):
            partner_action = self.partner_history[-1] if len(self.partner_history) > 0 else 0.0
            self.action_history.append((a_idx, partner_action, r, self.model.t))


class LLMAgent(Agent):
    """
    LLM-based decision agent using OpenRouter API.
    Uses prompt engineering to make pricing/quantity decisions based on recent history.
    """

    def __init__(self, model, role='quantity', api_key=None, llm_model='deepseek/deepseek-chat'):
        super().__init__(model)
        
        self.role = role
        self.llm_model_name = llm_model
        
        # Action space based on role
        if role == 'price':
            self.action_space = params.action_space_p()
        elif role == 'quantity':
            self.action_space = params.action_space_q()
        elif role == 'supplier':
            self.action_space = params.action_space_c()
        
        self.n_actions = len(self.action_space)
        
        # Initialize OpenRouter API
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        self.api_key = api_key or os.getenv('api_key')
        if not self.api_key:
            raise ValueError("API key required for LLM agent. Set 'api_key' in .env file.")
        
        # OpenRouter uses OpenAI-compatible API
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Track history for prompt context
        self.history = []  # (action, reward, partner_action)
        self.partner_history = []
        self.partner_prediction = None
        
        # Selected action
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
        
        window = min(5, len(self.partner_history))
        recent = self.partner_history[-window:]
        return float(np.mean(recent))
    
    def _build_prompt(self):
        """Build prompt for LLM based on role and history"""
        
        if self.role == 'price':
            role_desc = "pricing agent"
            action_name = "price"
            partner_name = "quantity"
            min_val = float(self.action_space[0])
            max_val = float(self.action_space[-1])
            step = float(self.action_space[1] - self.action_space[0]) if len(self.action_space) > 1 else 0.5
        elif self.role == 'quantity':
            role_desc = "quantity agent"
            action_name = "quantity"
            partner_name = "price"
            min_val = float(self.action_space[0])
            max_val = float(self.action_space[-1])
            step = 1.0
        else:  # supplier
            role_desc = "supplier agent"
            action_name = "cost"
            partner_name = "retailers"
            min_val = float(self.action_space[0])
            max_val = float(self.action_space[-1])
            step = 1.0
        
        # Build history context
        history_text = ""
        if len(self.history) > 0:
            recent_history = self.history[-5:]  # Last 5 rounds
            history_text = "Recent history:\n"
            for i, (act, rew, partner_act) in enumerate(recent_history, 1):
                history_text += f"Round {i}: {action_name}={act:.2f}, {partner_name}={partner_act:.2f}, profit=${rew:.2f}\n"
        else:
            history_text = "This is the first round (no history yet).\n"
        
        # Add partner prediction if enabled
        partner_context = ""
        if params.USE_PARTNER_PREDICTION and self.partner_prediction is not None:
            partner_context = f"\nYour partner's predicted {partner_name} for next round: {self.partner_prediction:.2f}\n"
        
        # Build prompt with better context
        coordination_note = """
IMPORTANT: You are working with a partner agent. Your profit depends on BOTH your decision AND your partner's decision:
- Price agent sets the selling price
- Quantity agent decides how much to order
- Profit = (Price × Quantity_Sold) - Costs
- Both agents receive the SAME profit from the joint outcome

This is a coordination game. Your goal is to choose actions that work well TOGETHER with your partner."""

        if self.role == 'price':
            strategy_hint = """Strategy tips:
- Higher prices increase margin but may reduce demand
- If profit is increasing, consider testing slightly higher prices
- If profit is decreasing, consider lowering prices to boost demand
- Your profit depends on BOTH your price AND your partner's quantity choice
- Consider: if quantity is high, demand needs to be high (so price shouldn't be too high)
- Consider: if quantity is low, you might be able to charge a premium"""
            if params.USE_PARTNER_PREDICTION:
                strategy_hint += f"\n- Your partner is predicted to order {self.partner_prediction:.1f} units"
                strategy_hint += "\n- Choose a price that maximizes profit given this expected quantity"
        elif self.role == 'quantity':
            strategy_hint = """Strategy tips:
- Order enough to meet demand without excessive leftover inventory
- If profit is increasing and demand seems high, consider ordering more
- If profit is decreasing, you may be overordering (excess inventory cost)
- Your profit depends on BOTH your quantity AND your partner's price choice
- Consider: if price is high, demand will be lower (so don't overorder)
- Consider: if price is low, demand will be higher (so order more to avoid stockouts)"""
            if params.USE_PARTNER_PREDICTION:
                strategy_hint += f"\n- Your partner is predicted to set price at ${self.partner_prediction:.2f}"
                strategy_hint += "\n- Choose a quantity that matches expected demand at this price"
        else:
            strategy_hint = """Strategy tips:
- Balance your cost against what retailers can afford
- Higher costs reduce retailer profits"""
        
        prompt = f"""You are a {role_desc} in a newsvendor supply chain simulation. 

{coordination_note}

{history_text}{partner_context}

{strategy_hint}

Based on this history{' and partner prediction' if params.USE_PARTNER_PREDICTION else ''}, what {action_name} should you choose for the next round to maximize JOINT profit with your partner?

Requirements:
- Your {action_name} must be between {min_val} and {max_val}
- It must be a multiple of {step} (e.g., {min_val}, {min_val + step}, {min_val + 2*step}, ...)
- Respond with ONLY the numerical value, nothing else
- Do not include any explanation, currency symbols, or additional text

{action_name.capitalize()}:"""
        
        return prompt
    
    def select_action(self):
        """Use LLM to select action"""
        self.partner_prediction = self.predict_partner_action()
        
        prompt = self._build_prompt()
        
        try:
            import requests
            
            # OpenRouter API call
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.llm_model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 50
            }
            
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            
            response_json = response.json()
            response_text = response_json['choices'][0]['message']['content'].strip()
            
            # Log for debugging
            print(f"[{self.role}] LLM response: {response_text}")
            
            # Parse numerical value
            # Remove common extras (currency symbols, commas, etc.)
            cleaned = response_text.replace('$', '').replace(',', '').strip()
            suggested_value = float(cleaned)
            
            # Find closest valid action
            closest_idx = int(np.argmin(np.abs(self.action_space - suggested_value)))
            action_idx = closest_idx
            
            print(f"[{self.role}] Suggested: {suggested_value:.2f} → Selected: {self.action_space[action_idx]:.2f}")
            
        except Exception as e:
            # Fallback: random exploration if LLM fails
            print(f"[{self.role}] LLM error: {e}. Using random fallback.")
            action_idx = self.random.randrange(self.n_actions)
        
        self.action_idx = action_idx
        self.action = self.action_space[action_idx]
    
    def update_belief(self):
        """Update history with latest outcome"""
        r = float(self.model.rewards[self])
        partner_action = self.partner_history[-1] if len(self.partner_history) > 0 else 0.0
        
        self.history.append((float(self.action), r, partner_action))

# -------------------------
# Joint (price, quantity) agents
# -------------------------

def joint_pq_action_space():
    
    ps = params.action_space_p()
    qs = params.action_space_q()
    return [(float(p), int(q)) for p in ps for q in qs]


class JointPQGreedyAgent(Agent):
    

    def __init__(self, model):
        super().__init__(model)
        self.action_space = joint_pq_action_space()
        self.n_actions = len(self.action_space)

        self.Q = np.zeros(self.n_actions, dtype=float)
        self.N = np.zeros(self.n_actions, dtype=int)

        self.action_idx = 0
        self.p, self.q = self.action_space[self.action_idx]

        self.reward = 0.0
        self.reward_cum = 0.0

    def select_action(self):
        t = self.model.t
        eps = params.epsilon_at(t, params.ROUNDS)
        self.model.current_eps = eps

        rng = self.model.rng
        if rng.random() < eps:
            self.action_idx = int(rng.integers(0, self.n_actions))
        else:
            best = np.flatnonzero(self.Q == self.Q.max())
            self.action_idx = int(rng.choice(best))

        self.p, self.q = self.action_space[self.action_idx]

    def update_belief(self):
        r = float(self.reward)
        a = int(self.action_idx)
        self.N[a] += 1
        n = self.N[a]
        self.Q[a] += (r - self.Q[a]) / n
        self.eps = params.epsilon_at(self.model.t + 1, params.ROUNDS)


class JointPQUcbAgent(Agent):
    
    def __init__(self, model, c: float = None):
        super().__init__(model)
        self.action_space = joint_pq_action_space()
        self.n_actions = len(self.action_space)

        self.Q = np.zeros(self.n_actions, dtype=float)
        self.N = np.zeros(self.n_actions, dtype=int)

        self.c = float(params.UCB_C if c is None else c)

        self.action_idx = 0
        self.p, self.q = self.action_space[self.action_idx]

        self.reward = 0.0
        self.reward_cum = 0.0

    def select_action(self):
        t = int(self.model.t)
        rng = self.model.rng

        # Ensure each action tried once
        untried = np.flatnonzero(self.N == 0)
        if untried.size > 0:
            self.action_idx = int(rng.choice(untried))
        else:
            bonus = self.c * np.sqrt(2 * np.log(t) / self.N)
            ucb = self.Q + bonus
            best = np.flatnonzero(ucb == ucb.max())
            self.action_idx = int(rng.choice(best))

        self.p, self.q = self.action_space[self.action_idx]

    def update_belief(self):
        r = float(self.reward)
        a = int(self.action_idx)
        self.N[a] += 1
        n = self.N[a]
        self.Q[a] += (r - self.Q[a]) / n


class JointPQThompsonAgent(Agent):
    

    def __init__(self, model):
        super().__init__(model)
        self.action_space = joint_pq_action_space()
        self.n_actions = len(self.action_space)

        # Thompson Sampling with Normal-Gamma prior (no partner prediction)
        # For each action a, we model reward ~ N(μ_a, 1/τ_a)
        # Prior: μ_a ~ N(μ_0, 1/(κ_0 * τ_a)), τ_a ~ Gamma(α_0, β_0)

        # Prior hyperparameters (weakly informative)
        self.mu_0 = 0.0  # Prior mean
        self.kappa_0 = 0.001  # Prior precision scaling
        self.alpha_0 = 1.0  # Gamma shape
        self.beta_0 = 1.0  # Gamma rate

        # Posterior hyperparameters (updated with data)
        self.mu_n = np.full(self.n_actions, self.mu_0, dtype=float)
        self.kappa_n = np.full(self.n_actions, self.kappa_0, dtype=float)
        self.alpha_n = np.full(self.n_actions, self.alpha_0, dtype=float)
        self.beta_n = np.full(self.n_actions, self.beta_0, dtype=float)

        # Track observations for updates
        self.counts = np.zeros(self.n_actions, dtype=int)
        self.sum_rewards = np.zeros(self.n_actions, dtype=float)
        self.sum_sq_rewards = np.zeros(self.n_actions, dtype=float)

        self.action_idx = 0
        self.p, self.q = self.action_space[self.action_idx]

        self.reward = 0.0
        self.reward_cum = 0.0

    def select_action(self):
        """
        Thompson Sampling: Sample from posterior predictive distribution for each action,
        choose action with highest sampled value.
        """
        rng = self.model.rng

        sampled_means = np.zeros(self.n_actions)

        for i in range(self.n_actions):
            mu_n = self.mu_n[i]
            kappa_n = self.kappa_n[i]
            alpha_n = self.alpha_n[i]
            beta_n = self.beta_n[i]

            # Sample precision from Gamma(alpha_n, beta_n)
            tau = rng.gamma(alpha_n, 1.0 / beta_n)

            # Sample mean from N(mu_n, 1/(kappa_n * tau))
            if tau > 0:
                std = np.sqrt(1.0 / (kappa_n * tau))
                sampled_mean = rng.normal(mu_n, std)
            else:
                sampled_mean = mu_n

            sampled_means[i] = sampled_mean

        # Choose action with highest sampled mean
        m = float(np.max(sampled_means))
        best = np.flatnonzero(sampled_means == m)
        self.action_idx = int(rng.choice(best))
        self.p, self.q = self.action_space[self.action_idx]

    def update_belief(self):
        """Update posterior using Normal-Gamma conjugate update"""
        a_idx = int(self.action_idx)
        r = float(self.reward)

        # Update sufficient statistics
        n = self.counts[a_idx]
        self.counts[a_idx] = n + 1
        self.sum_rewards[a_idx] += r
        self.sum_sq_rewards[a_idx] += r * r

        # Compute sample mean and variance
        n_new = n + 1
        sample_mean = self.sum_rewards[a_idx] / n_new

        if n_new > 1:
            sample_var = (self.sum_sq_rewards[a_idx] - n_new * sample_mean**2) / (n_new - 1)
            sample_var = max(sample_var, 1e-6)  # Numerical stability
        else:
            sample_var = 1.0


        # Normal-Gamma conjugate update
        self.kappa_n[a_idx] = self.kappa_0 + n_new
        self.mu_n[a_idx] = (self.kappa_0 * self.mu_0 + n_new * sample_mean) / self.kappa_n[a_idx]
        self.alpha_n[a_idx] = self.alpha_0 + n_new / 2.0
        self.beta_n[a_idx] = self.beta_0 + 0.5 * n_new * sample_var + \
                             0.5 * (self.kappa_0 * n_new / self.kappa_n[a_idx]) * (sample_mean - self.mu_0)**2
