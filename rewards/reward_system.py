"""
Reward System - Implementation of task completion reward calculation
"""
import asyncio
import json
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime
from enum import Enum

from utils.config import Config
from utils.logger import setup_logger
from agents.ollama_agent import OllamaAgent


class RewardType(Enum):
    """Types of rewards"""
    TASK_SUCCESS = "task_success"
    TASK_QUALITY = "task_quality"
    EFFICIENCY = "efficiency"
    LEARNING_GAIN = "learning_gain"
    NOVELTY = "novelty"
    COMPLETION_SPEED = "completion_speed"
    ERROR_REDUCTION = "error_reduction"


class RewardCalculator:
    """Calculates rewards for task completions based on multiple factors"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger("reward_calculator")
        self.ollama_agent = OllamaAgent(config)
        
        # Reward weights configuration
        self.weights = {
            "success_bonus": config.get("rewards.success_bonus", 1.0),
            "quality_bonus": config.get("rewards.quality_bonus", 0.5),
            "efficiency_bonus": config.get("rewards.efficiency_bonus", 0.3),
            "learning_bonus": config.get("rewards.learning_bonus", 0.4),
            "novelty_bonus": config.get("rewards.novelty_bonus", 0.2),
            "speed_bonus": config.get("rewards.speed_bonus", 0.2),
            "error_penalty": config.get("rewards.error_penalty", -0.5),
            "complexity_multiplier": config.get("rewards.complexity_multiplier", 1.2)
        }
        
        # Track historical rewards for learning
        self.reward_history = []
    
    async def calculate_reward(self, task: str, result: Dict, 
                             expected_outcome: str = None,
                             execution_time: float = None,
                             strategy_used: str = None,
                             previous_performance: Dict = None) -> Dict[str, Any]:
        """Calculate comprehensive reward for a task completion"""
        try:
            # Calculate individual reward components
            success_reward = await self._calculate_success_reward(task, result)
            quality_reward = await self._calculate_quality_reward(task, result, expected_outcome)
            efficiency_reward = await self._calculate_efficiency_reward(execution_time)
            novelty_reward = await self._calculate_novelty_reward(task, strategy_used)
            learning_reward = await self._calculate_learning_reward(task, result, previous_performance)
            
            # Combine rewards with weights
            total_reward = (
                success_reward * self.weights["success_bonus"] +
                quality_reward * self.weights["quality_bonus"] +
                efficiency_reward * self.weights["efficiency_bonus"] +
                learning_reward * self.weights["learning_bonus"] +
                novelty_reward * self.weights["novelty_bonus"]
            )
            
            # Apply penalties
            if result.get("error"):
                total_reward += self.weights["error_penalty"]
            
            # Apply complexity multiplier based on task difficulty
            complexity_multiplier = await self._calculate_complexity_multiplier(task)
            total_reward *= complexity_multiplier
            
            # Create reward breakdown
            reward_breakdown = {
                "success_component": success_reward * self.weights["success_bonus"],
                "quality_component": quality_reward * self.weights["quality_bonus"],
                "efficiency_component": efficiency_reward * self.weights["efficiency_bonus"],
                "learning_component": learning_reward * self.weights["learning_bonus"],
                "novelty_component": novelty_reward * self.weights["novelty_bonus"],
                "error_penalty": self.weights["error_penalty"] if result.get("error") else 0,
                "complexity_multiplier": complexity_multiplier,
                "total_raw_reward": total_reward / complexity_multiplier
            }
            
            # Create reward record
            reward_record = {
                "task": task,
                "strategy_used": strategy_used,
                "execution_time": execution_time,
                "result": result,
                "expected_outcome": expected_outcome,
                "reward_score": total_reward,
                "breakdown": reward_breakdown,
                "timestamp": datetime.now().isoformat(),
                "reward_type": self._classify_reward_type(total_reward)
            }
            
            # Store in history
            self.reward_history.append(reward_record)
            if len(self.reward_history) > 1000:  # Keep only recent rewards
                self.reward_history.pop(0)
            
            self.logger.info(f"Calculated reward for task: {task[:50]}..., Score: {total_reward:.3f}")
            
            return reward_record
            
        except Exception as e:
            self.logger.error(f"Error calculating reward: {e}")
            # Return fallback reward
            return self._generate_fallback_reward(task, result)
    
    async def _calculate_success_reward(self, task: str, result: Dict) -> float:
        """Calculate reward based on task success"""
        if result.get("status") == "success":
            return 1.0
        elif "error" in str(result).lower():
            return 0.0
        else:
            # Partial success - check if there's meaningful output
            result_str = str(result.get("result", "")) or str(result.get("output", ""))
            if len(result_str) > 10:  # Has some substantial output
                return 0.5
            else:
                return 0.2
    
    async def _calculate_quality_reward(self, task: str, result: Dict, 
                                     expected_outcome: str = None) -> float:
        """Calculate reward based on result quality"""
        if result.get("status") != "success":
            return 0.1  # Low quality if not successful
        
        result_content = str(result.get("result", "")) or str(result.get("output", ""))
        
        if not result_content:
            return 0.0
        
        # If we have expected outcome, compare quality
        if expected_outcome:
            quality_prompt = f"""
            Compare the actual result to the expected outcome and rate quality from 0.0 to 1.0:
            
            Task: {task}
            Expected: {expected_outcome}
            Actual: {result_content}
            
            Quality Score: 0.0-1.0
            """
            
            try:
                quality_result = await self.ollama_agent.execute({
                    "prompt": quality_prompt,
                    "temperature": 0.1,
                    "max_tokens": 50
                })
                
                if quality_result["status"] == "success":
                    # Extract number from response
                    response_text = quality_result["result"]
                    import re
                    numbers = re.findall(r'\d+\.\d+|\d+', response_text)
                    if numbers:
                        return min(1.0, max(0.0, float(numbers[0])))
            except:
                pass
        
        # Fallback: Calculate quality based on content richness
        words = len(result_content.split())
        if words < 5:
            return 0.3  # Too short
        elif words > 100:
            return 0.8  # Good detail
        else:
            return 0.6  # Moderate quality
    
    async def _calculate_efficiency_reward(self, execution_time: float = None) -> float:
        """Calculate reward based on execution efficiency"""
        if execution_time is None:
            return 0.5  # Neutral if no time data
        
        # Define efficiency thresholds (in seconds)
        excellent_time = 2.0  # Very fast
        good_time = 5.0      # Good
        acceptable_time = 15.0  # Acceptable
        slow_time = 30.0     # Slow
        
        if execution_time <= excellent_time:
            return 1.0
        elif execution_time <= good_time:
            return 0.8
        elif execution_time <= acceptable_time:
            return 0.6
        elif execution_time <= slow_time:
            return 0.3
        else:
            return 0.1
    
    async def _calculate_novelty_reward(self, task: str, strategy_used: str = None) -> float:
        """Calculate reward based on novel approach or strategy effectiveness"""
        if not strategy_used:
            return 0.5
        
        # Novelty is higher if this strategy worked well when others might have failed
        # This would typically be calculated against historical data
        # For now, we'll use a simple heuristic
        
        strategy_effectiveness = {
            "ReAct": 0.7,
            "ChainOfThought": 0.6,
            "PlanAndExecute": 0.8
        }
        
        base_effectiveness = strategy_effectiveness.get(strategy_used, 0.5)
        
        # Boost for using less common but effective strategies
        return base_effectiveness
    
    async def _calculate_learning_reward(self, task: str, result: Dict, 
                                      previous_performance: Dict = None) -> float:
        """Calculate reward based on learning and improvement"""
        if not previous_performance:
            # First attempt - neutral learning reward
            return 0.5
        
        # Calculate improvement over previous attempts
        prev_score = previous_performance.get("reward_score", 0.0)
        current_success = result.get("status") == "success"
        prev_success = previous_performance.get("result", {}).get("status") == "success"
        
        if current_success and not prev_success:
            # Improvement from failure to success - big bonus
            return 1.0
        elif current_success and prev_success:
            # Both successful - check for quality improvement
            return 0.8
        elif not current_success and not prev_success:
            # Still failing - small reward for trying
            return 0.2
        else:
            # Worse performance - penalty
            return 0.1
    
    async def _calculate_complexity_multiplier(self, task: str) -> float:
        """Calculate complexity multiplier based on task difficulty"""
        complexity_prompt = f"""
        Rate the complexity of this task from 1.0 (simple) to 2.0 (very complex):
        
        Task: {task}
        
        Complexity Multiplier: 1.0-2.0
        """
        
        try:
            complexity_result = await self.ollama_agent.execute({
                "prompt": complexity_prompt,
                "temperature": 0.1,
                "max_tokens": 50
            })
            
            if complexity_result["status"] == "success":
                response_text = complexity_result["result"]
                import re
                numbers = re.findall(r'\d+\.\d+|\d+', response_text)
                if numbers:
                    return min(2.0, max(1.0, float(numbers[0])))
        except:
            pass
        
        # Fallback complexity calculation
        task_lower = task.lower()
        complexity_indicators = ["analyze", "calculate", "compare", "evaluate", "design", "implement", "optimize"]
        complexity_score = sum(1 for indicator in complexity_indicators if indicator in task_lower)
        
        if complexity_score >= 3:
            return 1.8
        elif complexity_score >= 2:
            return 1.5
        elif complexity_score >= 1:
            return 1.2
        else:
            return 1.0
    
    def _classify_reward_type(self, reward_score: float) -> RewardType:
        """Classify the type of reward based on score"""
        if reward_score >= 0.9:
            return RewardType.TASK_SUCCESS
        elif reward_score >= 0.7:
            return RewardType.TASK_QUALITY
        elif reward_score >= 0.5:
            return RewardType.EFFICIENCY
        elif reward_score >= 0.3:
            return RewardType.LEARNING_GAIN
        else:
            return RewardType.ERROR_REDUCTION
    
    def _generate_fallback_reward(self, task: str, result: Dict) -> Dict[str, Any]:
        """Generate fallback reward when calculation fails"""
        success = result.get("status") == "success"
        reward_score = 1.0 if success else 0.1
        
        return {
            "task": task,
            "strategy_used": "unknown",
            "execution_time": None,
            "result": result,
            "expected_outcome": None,
            "reward_score": reward_score,
            "breakdown": {
                "success_component": reward_score,
                "quality_component": 0.0,
                "efficiency_component": 0.0,
                "learning_component": 0.0,
                "novelty_component": 0.0,
                "error_penalty": 0.0,
                "complexity_multiplier": 1.0,
                "total_raw_reward": reward_score
            },
            "timestamp": datetime.now().isoformat(),
            "reward_type": RewardType.TASK_SUCCESS if success else RewardType.ERROR_REDUCTION
        }
    
    def get_reward_statistics(self) -> Dict[str, Any]:
        """Get statistics about reward distribution"""
        if not self.reward_history:
            return {
                "total_rewards": 0,
                "average_reward": 0.0,
                "best_reward": 0.0,
                "worst_reward": 0.0,
                "success_rate": 0.0
            }
        
        scores = [r["reward_score"] for r in self.reward_history]
        successes = sum(1 for r in self.reward_history if r["reward_score"] >= 0.7)
        
        return {
            "total_rewards": len(self.reward_history),
            "average_reward": sum(scores) / len(scores),
            "best_reward": max(scores),
            "worst_reward": min(scores),
            "success_rate": successes / len(self.reward_history) if self.reward_history else 0.0,
            "recent_average": sum(scores[-20:]) / min(20, len(scores))
        }
    
    def export_reward_data(self) -> Dict[str, Any]:
        """Export reward data for analysis"""
        return {
            "reward_history": self.reward_history[-100:],  # Last 100 rewards
            "statistics": self.get_reward_statistics(),
            "export_timestamp": datetime.now().isoformat()
        }


# Example usage
if __name__ == "__main__":
    # This would be run with proper configuration
    pass