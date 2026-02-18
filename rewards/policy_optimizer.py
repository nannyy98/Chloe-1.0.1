"""
Policy Optimization System - Optimizes decision policies based on rewards
"""
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict, deque

from utils.config import Config
from utils.logger import setup_logger
from rewards.reward_system import RewardCalculator
from learning.strategy_ranker import StrategyRanker


class PolicyOptimizer:
    """Optimizes decision policies based on reward signals"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger("policy_optimizer")
        self.reward_calculator = RewardCalculator(config)
        self.strategy_ranker = StrategyRanker(config)
        
        # Policy parameters
        self.learning_rate = config.get("rewards.learning_rate", 0.1)
        self.discount_factor = config.get("rewards.discount_factor", 0.9)
        self.exploration_rate = config.get("rewards.exploration_rate", 0.1)
        
        # Policy state tracking
        self.policy_weights = defaultdict(float)  # Strategy -> weight
        self.action_values = defaultdict(lambda: defaultdict(float))  # State -> Action -> Value
        self.visit_counts = defaultdict(lambda: defaultdict(int))    # State -> Action -> Count
        self.policy_history = deque(maxlen=1000)  # Track policy changes
        self.performance_history = deque(maxlen=100)  # Track performance
        
        # Initialize with default values for known strategies
        for strategy_name in self.strategy_ranker.get_strategy_names():
            self.policy_weights[strategy_name] = 1.0
    
    async def update_policy(self, state: Dict, action: str, reward: float, 
                          next_state: Dict = None) -> Dict[str, Any]:
        """Update policy based on state, action, reward"""
        try:
            state_key = self._hash_state(state)
            
            # Update action value using temporal difference learning
            old_value = self.action_values[state_key][action]
            td_target = reward
            if next_state:
                next_state_key = self._hash_state(next_state)
                max_next_value = max(self.action_values[next_state_key].values()) if self.action_values[next_state_key] else 0.0
                td_target += self.discount_factor * max_next_value
            
            td_error = td_target - old_value
            new_value = old_value + self.learning_rate * td_error
            self.action_values[state_key][action] = new_value
            
            # Update visit count
            self.visit_counts[state_key][action] += 1
            
            # Update policy weights based on action values
            self._update_policy_weights(state_key)
            
            # Log policy update
            policy_update = {
                "state": state,
                "action": action,
                "reward": reward,
                "td_error": td_error,
                "new_value": new_value,
                "timestamp": datetime.now().isoformat()
            }
            
            self.policy_history.append(policy_update)
            
            self.logger.info(f"Policy updated for state {state_key[:10]}..., action: {action}, reward: {reward:.3f}")
            
            return policy_update
            
        except Exception as e:
            self.logger.error(f"Error updating policy: {e}")
            return {"error": str(e), "status": "failed"}
    
    def _hash_state(self, state: Dict) -> str:
        """Convert state dictionary to hashable string key"""
        if not state:
            return "empty_state"
        
        # Create a deterministic string representation
        items = sorted(state.items())
        return str(hash(str(items)))
    
    def _update_policy_weights(self, state_key: str):
        """Update policy weights based on action values"""
        if state_key in self.action_values:
            actions = self.action_values[state_key]
            if actions:
                # Normalize action values to create probability distribution
                values = list(actions.values())
                if values:
                    min_val = min(values)
                    max_val = max(values) if max(values) != min_val else min_val + 1
                    
                    # Normalize to 0-1 range
                    normalized_values = [(v - min_val) / (max_val - min_val) for v in values]
                    
                    # Update weights for corresponding actions
                    for i, (action, _) in enumerate(actions.items()):
                        self.policy_weights[action] = normalized_values[i] if len(normalized_values) > i else 0.0
    
    async def select_action(self, state: Dict, method: str = "epsilon_greedy") -> str:
        """Select action based on current policy"""
        state_key = self._hash_state(state)
        
        # Get available strategies
        available_strategies = self.strategy_ranker.get_strategy_names()
        
        if method == "epsilon_greedy":
            return self._epsilon_greedy_action(state_key, available_strategies)
        elif method == "softmax":
            return self._softmax_action(state_key, available_strategies)
        elif method == "ucb":
            return self._ucb_action(state_key, available_strategies)
        elif method == "greedy":
            return self._greedy_action(state_key, available_strategies)
        else:
            # Default to greedy
            return self._greedy_action(state_key, available_strategies)
    
    def _epsilon_greedy_action(self, state_key: str, strategies: List[str]) -> str:
        """Epsilon-greedy action selection"""
        if np.random.random() < self.exploration_rate:
            # Explore: random action
            return np.random.choice(strategies)
        else:
            # Exploit: best known action
            return self._greedy_action(state_key, strategies)
    
    def _softmax_action(self, state_key: str, strategies: List[str]) -> str:
        """Softmax action selection based on action values"""
        if not self.action_values[state_key]:
            return np.random.choice(strategies)
        
        # Get action values
        values = [self.action_values[state_key].get(strategy, 0.0) for strategy in strategies]
        
        # Apply softmax with temperature
        temperature = 0.5
        exp_values = [np.exp(v / temperature) for v in values]
        total = sum(exp_values)
        
        if total == 0:
            return np.random.choice(strategies)
        
        # Calculate probabilities
        probs = [ev / total for ev in exp_values]
        
        # Sample according to probabilities
        return np.random.choice(strategies, p=probs)
    
    def _ucb_action(self, state_key: str, strategies: List[str]) -> str:
        """UCB (Upper Confidence Bound) action selection"""
        if not self.action_values[state_key]:
            return np.random.choice(strategies)
        
        total_visits = sum(self.visit_counts[state_key].values())
        if total_visits == 0:
            return np.random.choice(strategies)
        
        ucb_values = []
        for strategy in strategies:
            q_value = self.action_values[state_key].get(strategy, 0.0)
            visits = self.visit_counts[state_key][strategy]
            
            if visits == 0:
                ucb_values.append(float('inf'))  # Prefer unvisited actions
            else:
                confidence = np.sqrt(np.log(total_visits) / visits)
                ucb_value = q_value + 2 * confidence  # UCB formula
                ucb_values.append(ucb_value)
        
        # Select action with highest UCB value
        best_idx = np.argmax(ucb_values)
        return strategies[best_idx]
    
    def _greedy_action(self, state_key: str, strategies: List[str]) -> str:
        """Pure greedy action selection"""
        if not self.action_values[state_key]:
            return np.random.choice(strategies)
        
        # Get action values and select the best one
        best_action = strategies[0]
        best_value = self.action_values[state_key].get(best_action, float('-inf'))
        
        for strategy in strategies[1:]:
            value = self.action_values[state_key].get(strategy, float('-inf'))
            if value > best_value:
                best_action = strategy
                best_value = value
        
        return best_action
    
    def get_policy_info(self) -> Dict[str, Any]:
        """Get current policy information"""
        return {
            "policy_weights": dict(self.policy_weights),
            "action_values": {
                state: dict(actions) 
                for state, actions in list(self.action_values.items())[-10:]  # Last 10 states
            },
            "visit_counts": {
                state: dict(counts) 
                for state, counts in list(self.visit_counts.items())[-10:]  # Last 10 states
            },
            "exploration_rate": self.exploration_rate,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics based on policy updates"""
        if not self.policy_history:
            return {
                "total_updates": 0,
                "average_reward": 0.0,
                "exploration_rate": self.exploration_rate,
                "recent_improvement": 0.0
            }
        
        recent_updates = list(self.policy_history)[-50:]  # Last 50 updates
        rewards = [update["reward"] for update in recent_updates]
        
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        
        # Calculate improvement trend
        if len(rewards) >= 10:
            first_half = rewards[:len(rewards)//2]
            second_half = rewards[len(rewards)//2:]
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)
            improvement = second_avg - first_avg
        else:
            improvement = 0.0
        
        return {
            "total_updates": len(self.policy_history),
            "average_reward": avg_reward,
            "exploration_rate": self.exploration_rate,
            "recent_improvement": improvement,
            "total_states_explored": len(self.action_values)
        }
    
    async def optimize_for_task(self, task: str, context: Dict = None) -> Dict[str, Any]:
        """Optimize policy specifically for a task"""
        # Create state representation
        state = {
            "task": task,
            "context": context or {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Select optimal strategy
        selected_strategy = await self.select_action(state)
        
        # Return optimization result
        return {
            "task": task,
            "selected_strategy": selected_strategy,
            "state_representation": state,
            "optimization_method": "rl_based",
            "confidence": 0.8  # RL-based selection confidence
        }
    
    def update_exploration_rate(self, new_rate: float):
        """Update exploration rate"""
        self.exploration_rate = max(0.01, min(0.99, new_rate))
        self.logger.info(f"Updated exploration rate to {self.exploration_rate}")
    
    def decay_exploration(self, decay_factor: float = 0.995):
        """Decay exploration rate over time"""
        self.exploration_rate *= decay_factor
        self.exploration_rate = max(0.01, self.exploration_rate)  # Minimum exploration
    
    def export_policy(self) -> Dict[str, Any]:
        """Export policy for persistence"""
        return {
            "policy_weights": dict(self.policy_weights),
            "action_values": dict(self.action_values),
            "visit_counts": dict(self.visit_counts),
            "policy_history": list(self.policy_history),
            "exploration_rate": self.exploration_rate,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "export_timestamp": datetime.now().isoformat()
        }
    
    def import_policy(self, policy_data: Dict[str, Any]):
        """Import policy from persisted data"""
        if "policy_weights" in policy_data:
            self.policy_weights.update(policy_data["policy_weights"])
        if "action_values" in policy_data:
            self.action_values.update({k: defaultdict(float, v) for k, v in policy_data["action_values"].items()})
        if "visit_counts" in policy_data:
            self.visit_counts.update({k: defaultdict(int, v) for k, v in policy_data["visit_counts"].items()})
        if "policy_history" in policy_data:
            self.policy_history.extend(policy_data["policy_history"])
        if "exploration_rate" in policy_data:
            self.exploration_rate = policy_data["exploration_rate"]
        if "learning_rate" in policy_data:
            self.learning_rate = policy_data["learning_rate"]
        if "discount_factor" in policy_data:
            self.discount_factor = policy_data["discount_factor"]
        
        self.logger.info("Imported policy from data")


# Example usage
if __name__ == "__main__":
    # This would be run with proper configuration
    pass