"""
Adaptive Decision Making System - Makes decisions based on learned policies
"""
import asyncio
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

from utils.config import Config
from utils.logger import setup_logger
from agents.ollama_agent import OllamaAgent
from rewards.policy_optimizer import PolicyOptimizer
from learning.strategy_ranker import StrategyRanker
from learning.strategies import get_strategy_by_name


class AdaptiveDecisionMaker:
    """Makes adaptive decisions based on learned policies and context"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger("adaptive_decision_maker")
        self.ollama_agent = OllamaAgent(config)
        self.policy_optimizer = PolicyOptimizer(config)
        self.strategy_ranker = StrategyRanker(config)
        
        # Decision history
        self.decision_history = []
        self.context_importance = {}  # Track importance of different context factors
        
    async def make_decision(self, task: str, context: Dict = None, 
                          available_strategies: List[str] = None) -> Dict[str, Any]:
        """Make adaptive decision about how to approach a task"""
        try:
            # Prepare context for decision making
            decision_context = await self._prepare_decision_context(task, context)
            
            # Get strategy recommendation from policy optimizer
            policy_recommendation = await self.policy_optimizer.optimize_for_task(task, decision_context)
            
            # Get additional recommendations from strategy ranker
            strategy_rankings = await self.strategy_ranker.rank_strategies_for_task(task, context)
            
            # Combine recommendations
            final_decision = await self._combine_recommendations(
                policy_recommendation, 
                strategy_rankings, 
                decision_context
            )
            
            # Execute decision and track result
            decision_result = {
                "task": task,
                "context": context,
                "decision": final_decision,
                "timestamp": datetime.now().isoformat(),
                "decision_type": "adaptive_rl"
            }
            
            self.decision_history.append(decision_result)
            if len(self.decision_history) > 500:
                self.decision_history.pop(0)
            
            self.logger.info(f"Made adaptive decision for task: {task[:50]}..., strategy: {final_decision['selected_strategy']}")
            
            return decision_result
            
        except Exception as e:
            self.logger.error(f"Error making adaptive decision: {e}")
            return await self._make_fallback_decision(task, context)
    
    async def _prepare_decision_context(self, task: str, context: Dict = None) -> Dict[str, Any]:
        """Prepare context for decision making"""
        context = context or {}
        
        # Analyze task characteristics
        task_analysis = await self._analyze_task_characteristics(task)
        
        # Combine with provided context
        decision_context = {
            "task_type": task_analysis.get("type", "general"),
            "complexity": task_analysis.get("complexity", "medium"),
            "domain": task_analysis.get("domain", "general"),
            "requirements": task_analysis.get("requirements", []),
            "constraints": task_analysis.get("constraints", []),
            "provided_context": context,
            "timestamp": datetime.now().isoformat()
        }
        
        return decision_context
    
    async def _analyze_task_characteristics(self, task: str) -> Dict[str, Any]:
        """Analyze task to extract characteristics"""
        analysis_prompt = f"""
        Analyze the following task and extract its characteristics:
        
        Task: {task}
        
        Provide analysis in JSON format:
        {{
            "type": "mathematical|analytical|creative|planning|research|general",
            "complexity": "low|medium|high|very_high",
            "domain": "mathematics|science|technology|business|social|general",
            "requirements": ["requirement1", "requirement2"],
            "constraints": ["constraint1", "constraint2"],
            "reasoning_approach": "step_by_step|iterative|holistic|decompositional"
        }}
        """
        
        try:
            analysis_result = await self.ollama_agent.execute({
                "prompt": analysis_prompt,
                "temperature": 0.3,
                "max_tokens": 500
            })
            
            if analysis_result["status"] == "success":
                try:
                    return json.loads(analysis_result["result"])
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            self.logger.warning(f"Task analysis failed: {e}")
        
        # Fallback analysis
        task_lower = task.lower()
        
        # Determine type
        if any(word in task_lower for word in ["calculate", "compute", "solve", "math", "equation"]):
            task_type = "mathematical"
        elif any(word in task_lower for word in ["analyze", "compare", "evaluate", "assess"]):
            task_type = "analytical"
        elif any(word in task_lower for word in ["create", "design", "develop", "write", "compose"]):
            task_type = "creative"
        elif any(word in task_lower for word in ["plan", "schedule", "organize", "coordinate"]):
            task_type = "planning"
        else:
            task_type = "general"
        
        # Determine complexity
        word_count = len(task.split())
        if word_count < 10:
            complexity = "low"
        elif word_count < 25:
            complexity = "medium"
        else:
            complexity = "high"
        
        return {
            "type": task_type,
            "complexity": complexity,
            "domain": "general",
            "requirements": ["task_completion"],
            "constraints": ["accuracy", "efficiency"],
            "reasoning_approach": "step_by_step"
        }
    
    async def _combine_recommendations(self, policy_rec: Dict, strategy_rankings: List[Tuple[str, float]], 
                                    decision_context: Dict) -> Dict[str, Any]:
        """Combine policy and ranking recommendations"""
        # Get the top-ranked strategy from policy optimizer
        policy_strategy = policy_rec["selected_strategy"]
        
        # Get the top-ranked strategy from strategy ranker
        if strategy_rankings:
            ranker_strategy = strategy_rankings[0][0]
            ranker_confidence = strategy_rankings[0][1]
        else:
            ranker_strategy = "ReAct"  # Default fallback
            ranker_confidence = 0.5
        
        # Decide which recommendation to follow based on context
        # If we have high confidence in the ranking, use it
        # Otherwise, fall back to policy recommendation
        if ranker_confidence > 0.7:
            selected_strategy = ranker_strategy
            decision_reason = f"High confidence ranking ({ranker_confidence:.2f})"
        else:
            selected_strategy = policy_strategy
            decision_reason = "Policy-based recommendation"
        
        # Get strategy instance
        strategy_instance = get_strategy_by_name(self.config, selected_strategy)
        if strategy_instance:
            strategy_details = {
                "name": strategy_instance.name,
                "description": f"{strategy_instance.__class__.__name__} implementation",
                "suitability": ranker_confidence if selected_strategy == ranker_strategy else 0.5
            }
        else:
            strategy_details = {
                "name": selected_strategy,
                "description": "Unknown strategy",
                "suitability": 0.5
            }
        
        return {
            "selected_strategy": selected_strategy,
            "strategy_details": strategy_details,
            "policy_recommendation": policy_rec["selected_strategy"],
            "ranking_recommendation": ranker_strategy if strategy_rankings else "none",
            "decision_reason": decision_reason,
            "confidence": ranker_confidence if ranker_confidence > 0.7 else policy_rec.get("confidence", 0.8),
            "context_considerations": decision_context
        }
    
    async def execute_decision(self, decision_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the decision and return results"""
        task = decision_result["task"]
        selected_strategy = decision_result["decision"]["selected_strategy"]
        context = decision_result.get("context", {})
        
        # Get and execute the selected strategy
        strategy = get_strategy_by_name(self.config, selected_strategy)
        if not strategy:
            return {
                "error": f"Strategy {selected_strategy} not found",
                "status": "failed"
            }
        
        # Execute strategy
        start_time = datetime.now()
        execution_result = await strategy.execute(task, context)
        end_time = datetime.now()
        
        execution_time = (end_time - start_time).total_seconds()
        
        # Create execution record
        execution_record = {
            "task": task,
            "strategy_used": selected_strategy,
            "execution_result": execution_result,
            "execution_time": execution_time,
            "status": "completed",
            "timestamp": datetime.now().isoformat()
        }
        
        return execution_record
    
    async def learn_from_decision(self, task: str, decision: Dict, execution_result: Dict, 
                               reward: float, next_context: Dict = None) -> Dict[str, Any]:
        """Learn from the decision and its outcome"""
        try:
            # Prepare state representations
            current_context = decision.get("context_considerations", {})
            next_context = next_context or {}
            
            # Update policy based on reward
            policy_update = await self.policy_optimizer.update_policy(
                state=current_context,
                action=decision["selected_strategy"],
                reward=reward,
                next_state=next_context
            )
            
            # Update strategy ranker with performance
            success = execution_result.get("status") == "success"
            score = reward  # Use reward as performance score
            self.strategy_ranker.update_strategy_performance(
                decision["selected_strategy"], 
                success, 
                score, 
                execution_result.get("execution_time")
            )
            
            learning_result = {
                "policy_updated": policy_update.get("status") != "failed",
                "strategy_performance_updated": True,
                "reward_applied": reward,
                "learning_occurred": True,
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"Learned from decision for task: {task[:50]}..., reward: {reward:.3f}")
            
            return learning_result
            
        except Exception as e:
            self.logger.error(f"Error in learning from decision: {e}")
            return {"error": str(e), "learning_occurred": False, "status": "failed"}
    
    def get_decision_insights(self) -> Dict[str, Any]:
        """Get insights about decision making patterns"""
        if not self.decision_history:
            return {
                "total_decisions": 0,
                "most_common_strategies": [],
                "decision_accuracy": 0.0,
                "context_factors": {}
            }
        
        # Analyze decision patterns
        strategy_counts = {}
        successful_decisions = 0
        
        for decision in self.decision_history:
            strategy = decision["decision"]["selected_strategy"]
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
            # Check if decision was successful (this would need execution feedback)
            # For now, we'll assume all recorded decisions were attempted
            successful_decisions += 1  # Placeholder
        
        # Sort strategies by frequency
        sorted_strategies = sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "total_decisions": len(self.decision_history),
            "most_common_strategies": sorted_strategies[:5],  # Top 5
            "decision_accuracy": successful_decisions / len(self.decision_history) if self.decision_history else 0.0,
            "recent_trends": self._get_recent_trends()
        }
    
    def _get_recent_trends(self) -> Dict[str, Any]:
        """Get recent trends in decision making"""
        recent_decisions = self.decision_history[-20:] if self.decision_history else []
        
        if not recent_decisions:
            return {"trending_strategies": [], "pattern_changes": 0}
        
        strategy_sequence = [d["decision"]["selected_strategy"] for d in recent_decisions]
        
        # Count strategy transitions
        transitions = {}
        for i in range(len(strategy_sequence) - 1):
            current = strategy_sequence[i]
            next_strategy = strategy_sequence[i + 1]
            transition = f"{current}->{next_strategy}"
            transitions[transition] = transitions.get(transition, 0) + 1
        
        trending_strategies = list(set(strategy_sequence))
        
        return {
            "trending_strategies": trending_strategies,
            "recent_transitions": dict(list(transitions.items())[:10]),  # Top 10 transitions
            "pattern_diversity": len(set(strategy_sequence)) / len(strategy_sequence) if strategy_sequence else 0
        }
    
    async def _make_fallback_decision(self, task: str, context: Dict = None) -> Dict[str, Any]:
        """Make fallback decision when adaptive system fails"""
        # Use default strategy selection
        strategy_rankings = await self.strategy_ranker.rank_strategies_for_task(task, context)
        
        if strategy_rankings:
            selected_strategy = strategy_rankings[0][0]
        else:
            selected_strategy = "ReAct"  # Default fallback
        
        fallback_decision = {
            "task": task,
            "context": context,
            "decision": {
                "selected_strategy": selected_strategy,
                "strategy_details": {"name": selected_strategy, "description": "Fallback selection"},
                "policy_recommendation": selected_strategy,
                "ranking_recommendation": selected_strategy,
                "decision_reason": "Fallback decision",
                "confidence": 0.5
            },
            "timestamp": datetime.now().isoformat(),
            "decision_type": "fallback"
        }
        
        return fallback_decision
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the decision maker"""
        policy_metrics = self.policy_optimizer.get_performance_metrics()
        decision_insights = self.get_decision_insights()
        
        return {
            "policy_metrics": policy_metrics,
            "decision_insights": decision_insights,
            "total_decisions": len(self.decision_history),
            "learning_updates": policy_metrics["total_updates"],
            "exploration_rate": policy_metrics["exploration_rate"]
        }
    
    def export_decision_data(self) -> Dict[str, Any]:
        """Export decision data for analysis"""
        return {
            "decision_history": self.decision_history[-100:],  # Last 100 decisions
            "context_importance": dict(self.context_importance),
            "performance_metrics": self.get_performance_metrics(),
            "export_timestamp": datetime.now().isoformat()
        }


# Example usage
if __name__ == "__main__":
    # This would be run with proper configuration
    pass