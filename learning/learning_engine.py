"""
Learning Engine - Implements closed-loop learning for Chloe AI
"""

import asyncio
import json
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime
from collections import defaultdict

from utils.config import Config
from utils.logger import setup_logger
from memory.memory_system import MemorySystem
from learning.reflection_engine import ReflectionEngine
from learning.strategy_ranker import StrategyRanker
from learning.self_critique import SelfCritiqueEngine
from evaluation.evaluation_system import EvaluationSystem

class LearningEngine:
    """Self-improving learning system with closed feedback loop"""
    
    def __init__(self, config: Config, memory_system: MemorySystem = None):
        self.config = config
        self.memory_system = memory_system
        self.logger = setup_logger("learning_engine")
        self.reflection_engine = ReflectionEngine(config)
        self.strategy_ranker = StrategyRanker(config)
        self.self_critique_engine = SelfCritiqueEngine(config)
        self.evaluation_system = EvaluationSystem(config)
        
        # Learning parameters
        self.learning_rate = config.get("learning.learning_rate", 0.1)
        self.success_threshold = config.get("learning.success_threshold", 0.7)
        self.experience_buffer_size = config.get("learning.experience_buffer_size", 100)
        
        # Performance tracking
        self.performance_history = []
        self.strategy_success_rates = defaultdict(list)
        self.error_patterns = defaultdict(int)
        
        # Initialize learning models
        self._initialize_learning_models()
        
    def _initialize_learning_models(self):
        """Initialize internal learning models"""
        # Simple reinforcement learning model
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.experience_buffer = []
        
        # Strategy preference model
        self.strategy_weights = {
            "reasoning": 0.5,
            "tool_execution": 0.3,
            "learning": 0.2
        }
        
        self.logger.info("Learning engine initialized")
    
    async def record_experience(self, task: str, decision: Dict, result: Dict, actions: List[Dict] = None):
        """Record experience from task execution with reflection"""
        try:
            # Calculate success score
            success_score = await self._evaluate_success(task, decision, result)
            
            # Generate reflection if this is a significant task
            reflection = ""
            if success_score < 0.7 or success_score > 0.9 or len(self.experience_buffer) % 5 == 0:
                # Generate reflection for poor performance, excellent performance, or every 5th task
                if actions:
                    reflection = await self.reflection_engine.generate_reflection(
                        task=task,
                        actions=actions,
                        result=result,
                        success_score=success_score
                    )
                
            # Store comprehensive experience in memory system
            if self.memory_system:
                if actions:
                    # Store detailed task experience with reflection
                    await self.memory_system.store_task_experience(
                        task=task,
                        actions=actions,
                        result=result,
                        reflection=reflection,
                        success_score=success_score,
                        metadata={
                            "processing_time": result.get("metadata", {}).get("processing_time", 0),
                            "confidence": decision.get("confidence", 0),
                            "decision_action": decision.get("action", "unknown")
                        }
                    )
                else:
                    # Store basic experience
                    await self.memory_system.store_experience(
                        task=task,
                        decision=decision,
                        result=result,
                        success_score=success_score,
                        metadata={
                            "processing_time": result.get("metadata", {}).get("processing_time", 0),
                            "confidence": decision.get("confidence", 0)
                        }
                    )
            
            # Add to experience buffer
            experience = {
                "task": task,
                "decision": decision,
                "result": result,
                "actions": actions or [],
                "success_score": success_score,
                "reflection": reflection,
                "timestamp": datetime.now().isoformat()
            }
            
            self.experience_buffer.append(experience)
            
            # Keep buffer size manageable
            if len(self.experience_buffer) > self.experience_buffer_size:
                self.experience_buffer.pop(0)
            
            # Update learning models
            await self._update_models(experience)
            
            # Track performance
            self.performance_history.append({
                "timestamp": datetime.now().isoformat(),
                "task": task[:50],
                "success_score": success_score,
                "action": decision.get("action", "unknown"),
                "has_reflection": bool(reflection)
            })
            
            self.logger.info(f"Recorded experience: {decision.get('action')} - Score: {success_score:.2f}" + \
                           (f" - Reflection: {'Yes' if reflection else 'No'}"))
            
        except Exception as e:
            self.logger.error(f"Error recording experience: {e}")
    
    async def _evaluate_success(self, task: str, decision: Dict, result: Dict) -> float:
        """Evaluate success of task execution (0.0 to 1.0)"""
        score = 0.5  # Base score
        
        # Check for errors
        if "error" in result:
            score -= 0.4
        else:
            score += 0.3
            
        # Check result quality indicators
        if result.get("status") == "success":
            score += 0.2
            
        # Consider confidence
        confidence = decision.get("confidence", 0.5)
        score += (confidence - 0.5) * 0.2
        
        # Check if task was completed (basic check)
        result_content = str(result.get("result", ""))
        if len(result_content) > 10:  # Basic completeness check
            score += 0.1
            
        return max(0.0, min(1.0, score))
    
    async def _update_models(self, experience: Dict):
        """Update internal learning models based on experience"""
        task = experience["task"]
        decision = experience["decision"]
        success_score = experience["success_score"]
        action = decision.get("action", "unknown")
        
        # Update Q-table (simplified reinforcement learning)
        state = self._get_state_representation(task, decision)
        reward = success_score * 2 - 1  # Convert to [-1, 1] range
        
        # Q-learning update
        old_value = self.q_table[state][action]
        self.q_table[state][action] = old_value + self.learning_rate * (reward - old_value)
        
        # Update strategy success rates
        self.strategy_success_rates[action].append(success_score)
        
        # Track error patterns
        if success_score < self.success_threshold:
            error_type = self._classify_error(experience)
            self.error_patterns[error_type] += 1
    
    def _get_state_representation(self, task: str, decision: Dict) -> str:
        """Create simplified state representation"""
        # Extract key features from task and decision
        task_keywords = " ".join(task.lower().split()[:5])  # First 5 words
        action = decision.get("action", "unknown")
        complexity = str(decision.get("confidence", 0.5))
        
        return f"{task_keywords}_{action}_{complexity}"
    
    def _classify_error(self, experience: Dict) -> str:
        """Classify error type for pattern recognition"""
        result = experience["result"]
        decision = experience["decision"]
        
        if "error" in result:
            error_msg = str(result["error"]).lower()
            if "tool" in error_msg:
                return "tool_error"
            elif "memory" in error_msg:
                return "memory_error"
            elif "timeout" in error_msg:
                return "timeout_error"
            else:
                return "other_error"
        elif experience["success_score"] < 0.3:
            return "poor_result"
        else:
            return "low_confidence"
    
    async def get_current_state(self) -> Dict[str, Any]:
        """Get current learning state and metrics"""
        # Calculate recent performance
        recent_experiences = self.experience_buffer[-20:] if self.experience_buffer else []
        recent_success_rate = (
            np.mean([exp["success_score"] for exp in recent_experiences])
            if recent_experiences else 0.0
        )
        
        # Calculate strategy performance
        strategy_performance = {}
        for strategy, scores in self.strategy_success_rates.items():
            if scores:
                recent_scores = scores[-10:]  # Last 10 attempts
                strategy_performance[strategy] = {
                    "recent_success_rate": np.mean(recent_scores),
                    "total_attempts": len(scores),
                    "average_score": np.mean(scores)
                }
        
        return {
            "recent_success_rate": recent_success_rate,
            "total_experiences": len(self.experience_buffer),
            "strategy_performance": strategy_performance,
            "error_patterns": dict(self.error_patterns),
            "performance_trend": self._calculate_performance_trend()
        }
    
    def _calculate_performance_trend(self) -> str:
        """Calculate overall performance trend"""
        if len(self.performance_history) < 10:
            return "insufficient_data"
        
        recent_scores = [p["success_score"] for p in self.performance_history[-10:]]
        older_scores = [p["success_score"] for p in self.performance_history[-20:-10]]
        
        if not older_scores:
            return "new_system"
        
        recent_avg = np.mean(recent_scores)
        older_avg = np.mean(older_scores)
        
        improvement = recent_avg - older_avg
        
        if improvement > 0.1:
            return "improving"
        elif improvement < -0.1:
            return "declining"
        else:
            return "stable"
    
    async def process_experience(self, task_description: str) -> Dict[str, Any]:
        """Process accumulated experience to generate insights"""
        if not self.experience_buffer:
            return {"message": "No experience data available"}
        
        # Analyze patterns
        insights = {
            "total_experiences": len(self.experience_buffer),
            "average_success_rate": np.mean([exp["success_score"] for exp in self.experience_buffer]),
            "best_performing_strategies": self._get_best_strategies(),
            "common_failure_patterns": self._get_failure_patterns(),
            "learning_recommendations": self._generate_recommendations()
        }
        
        return insights
    
    def _get_best_strategies(self) -> List[Dict[str, Any]]:
        """Get best performing strategies"""
        strategies = []
        for strategy, scores in self.strategy_success_rates.items():
            if scores:
                strategies.append({
                    "strategy": strategy,
                    "success_rate": np.mean(scores[-10:]),  # Recent performance
                    "total_attempts": len(scores),
                    "consistency": np.std(scores[-10:]) if len(scores) >= 10 else 0
                })
        
        return sorted(strategies, key=lambda x: x["success_rate"], reverse=True)
    
    def _get_failure_patterns(self) -> List[Dict[str, Any]]:
        """Get common failure patterns"""
        patterns = []
        total_errors = sum(self.error_patterns.values())
        
        for error_type, count in self.error_patterns.items():
            if count > 0:
                patterns.append({
                    "error_type": error_type,
                    "frequency": count,
                    "percentage": (count / total_errors) * 100 if total_errors > 0 else 0
                })
        
        return sorted(patterns, key=lambda x: x["frequency"], reverse=True)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate learning recommendations"""
        recommendations = []
        
        # Check overall performance
        avg_success = np.mean([exp["success_score"] for exp in self.experience_buffer]) if self.experience_buffer else 0
        
        if avg_success < 0.6:
            recommendations.append("Overall performance needs improvement - consider adjusting decision thresholds")
        
        # Check strategy balance
        strategy_counts = defaultdict(int)
        for exp in self.experience_buffer[-50:]:
            strategy_counts[exp["decision"].get("action", "unknown")] += 1
        
        if len(strategy_counts) == 1:
            recommendations.append("Over-reliance on single strategy - diversify approach")
        
        # Check recent trends
        if len(self.performance_history) >= 20:
            recent_trend = self._calculate_performance_trend()
            if recent_trend == "declining":
                recommendations.append("Performance declining - review recent changes")
        
        # Add generic recommendations if none specific
        if not recommendations:
            recommendations.extend([
                "Continue current learning approach",
                "Monitor performance metrics regularly",
                "Consider expanding tool capabilities"
            ])
        
        return recommendations
    
    async def adapt_strategy(self) -> Dict[str, Any]:
        """Adapt decision strategy based on learning"""
        current_state = await self.get_current_state()
        
        adaptations = {
            "strategy_adjustments": {},
            "confidence_updates": {},
            "threshold_modifications": {}
        }
        
        # Adjust strategy weights based on performance
        for strategy, performance in current_state["strategy_performance"].items():
            recent_success = performance["recent_success_rate"]
            
            if recent_success > 0.8:
                # Increase preference for high-performing strategies
                adaptations["strategy_adjustments"][strategy] = "increase_preference"
            elif recent_success < 0.4:
                # Decrease preference for poor-performing strategies
                adaptations["strategy_adjustments"][strategy] = "decrease_preference"
        
        # Adjust confidence thresholds
        overall_success = current_state["recent_success_rate"]
        if overall_success > 0.8:
            adaptations["confidence_updates"]["general"] = "increase_threshold"
        elif overall_success < 0.5:
            adaptations["confidence_updates"]["general"] = "decrease_threshold"
        
        return {
            "adaptations": adaptations,
            "timestamp": datetime.now().isoformat(),
            "current_state": current_state
        }
    
    def get_learning_metrics(self) -> Dict[str, Any]:
        """Get comprehensive learning metrics"""
        return {
            "experience_count": len(self.experience_buffer),
            "performance_history_length": len(self.performance_history),
            "strategy_diversity": len(self.strategy_success_rates),
            "error_pattern_count": len(self.error_patterns),
            "q_table_size": len(self.q_table),
            "recent_performance": (
                np.mean([exp["success_score"] for exp in self.experience_buffer[-10:]])
                if len(self.experience_buffer) >= 10 else 0.0
            )
        }

# Example usage
if __name__ == "__main__":
    # This would be initialized with actual components
    pass