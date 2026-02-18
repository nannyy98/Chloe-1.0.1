"""
Decision Engine - Meta-control layer for Chloe AI
Determines whether to think, act, learn, or use tools
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from enum import Enum

from core.reasoning_core import ReasoningCore
from memory.memory_system import MemorySystem
from agents.tool_manager import ToolManager
from learning.learning_engine import LearningEngine
from utils.config import Config
from utils.logger import setup_logger

class ActionType(Enum):
    REASON = "reason"
    TOOL = "tool"
    LEARN = "learn"
    PLAN = "plan"
    EVALUATE = "evaluate"

class DecisionEngine:
    """Meta-control system that decides what approach to take"""
    
    def __init__(self, 
                 reasoning_core: ReasoningCore,
                 memory_system: MemorySystem,
                 tool_manager: ToolManager,
                 learning_engine: LearningEngine,
                 config: Config):
        self.reasoning_core = reasoning_core
        self.memory_system = memory_system
        self.tool_manager = tool_manager
        self.learning_engine = learning_engine
        self.config = config
        self.logger = setup_logger("decision_engine")
        
        # Decision weights and thresholds
        self.decision_weights = {
            "complexity_threshold": 0.7,
            "tool_availability_bonus": 0.2,
            "learning_opportunity_bonus": 0.15,
            "memory_relevance_bonus": 0.1
        }
        
    async def make_decision(self, user_input: str, context: Dict = None) -> Dict[str, Any]:
        """Main decision-making method"""
        try:
            # 1. Get context and history
            context_info = await self._gather_context(user_input, context)
            
            # 2. Analyze decision factors
            decision_factors = await self._analyze_factors(user_input, context_info)
            
            # 3. Evaluate possible actions
            action_scores = await self._evaluate_actions(user_input, decision_factors, context_info)
            
            # 4. Select best action
            best_action = await self._select_action(action_scores, decision_factors)
            
            # 5. Generate detailed decision
            decision = await self._generate_decision_details(best_action, user_input, context_info)
            
            self.logger.info(f"Decision made: {decision['action']} with confidence {decision['confidence']:.2f}")
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error in decision making: {e}")
            return {
                "action": "reason",
                "confidence": 0.5,
                "reasoning": "Fallback to reasoning due to error",
                "error": str(e)
            }
    
    async def _gather_context(self, user_input: str, context: Dict = None) -> Dict[str, Any]:
        """Gather relevant context information"""
        context_info = {
            "user_input": user_input,
            "provided_context": context or {},
            "recent_history": await self.memory_system.get_recent_interactions(5),
            "relevant_knowledge": await self.memory_system.search_knowledge(user_input),
            "available_tools": self.tool_manager.list_available_tools(),
            "learning_state": await self.learning_engine.get_current_state()
        }
        
        return context_info
    
    async def _analyze_factors(self, user_input: str, context_info: Dict) -> Dict[str, float]:
        """Analyze key decision factors"""
        # Get task analysis from reasoning core
        task_analysis = await self.reasoning_core._analyze_task(user_input, context_info)
        
        factors = {
            "task_complexity": self._assess_complexity(task_analysis),
            "tool_relevance": self._assess_tool_relevance(user_input, context_info["available_tools"]),
            "knowledge_relevance": len(context_info["relevant_knowledge"]) > 0,
            "learning_opportunity": self._assess_learning_opportunity(user_input, context_info),
            "urgency": self._assess_urgency(user_input),
            "memory_similarity": self._assess_memory_similarity(context_info["recent_history"], user_input)
        }
        
        return factors
    
    def _assess_complexity(self, task_analysis: Dict) -> float:
        """Assess task complexity (0.0 to 1.0)"""
        complexity_mapping = {"simple": 0.3, "moderate": 0.6, "complex": 0.9}
        return complexity_mapping.get(task_analysis.get("complexity", "moderate"), 0.5)
    
    def _assess_tool_relevance(self, user_input: str, available_tools: List[str]) -> float:
        """Assess how relevant tools are to the task"""
        # Simple keyword matching for now
        relevant_keywords = ["code", "search", "file", "web", "data", "calculate"]
        tool_matches = sum(1 for keyword in relevant_keywords if keyword in user_input.lower())
        return min(tool_matches / len(relevant_keywords), 1.0)
    
    def _assess_learning_opportunity(self, user_input: str, context_info: Dict) -> float:
        """Assess if this is a good learning opportunity"""
        # Check if this is similar to previous tasks but with different outcomes
        recent_tasks = [task.get("input", "") for task in context_info["recent_history"]]
        similarity_to_past = sum(
            1 for task in recent_tasks 
            if self._text_similarity(user_input, task) > 0.7
        ) / len(recent_tasks) if recent_tasks else 0
        
        return 0.3 + (0.7 * similarity_to_past)  # Base + similarity bonus
    
    def _assess_urgency(self, user_input: str) -> float:
        """Assess task urgency"""
        urgent_keywords = ["urgent", "asap", "immediately", "now", "quick"]
        urgency_score = sum(1 for keyword in urgent_keywords if keyword in user_input.lower())
        return min(urgency_score / len(urgent_keywords), 1.0)
    
    def _assess_memory_similarity(self, recent_history: List[Dict], user_input: str) -> float:
        """Assess similarity to recent interactions"""
        if not recent_history:
            return 0.0
            
        similarities = [
            self._text_similarity(user_input, interaction.get("input", ""))
            for interaction in recent_history
        ]
        return sum(similarities) / len(similarities)
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity calculation"""
        # Very basic implementation - can be enhanced with embeddings
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union if union > 0 else 0.0
    
    async def _evaluate_actions(self, user_input: str, factors: Dict, context_info: Dict) -> Dict[ActionType, float]:
        """Evaluate and score all possible actions"""
        scores = {}
        
        # Reasoning score
        scores[ActionType.REASON] = self._calculate_reasoning_score(factors)
        
        # Tool execution score
        scores[ActionType.TOOL] = self._calculate_tool_score(factors, context_info)
        
        # Learning score
        scores[ActionType.LEARN] = self._calculate_learning_score(factors, context_info)
        
        # Planning score
        scores[ActionType.PLAN] = self._calculate_planning_score(factors)
        
        # Evaluation score
        scores[ActionType.EVALUATE] = self._calculate_evaluation_score(factors)
        
        return scores
    
    def _calculate_reasoning_score(self, factors: Dict) -> float:
        """Calculate score for reasoning approach"""
        base_score = 0.5
        score = base_score
        
        # Higher for complex tasks
        score += factors["task_complexity"] * 0.3
        
        # Lower if tools are very relevant
        score -= factors["tool_relevance"] * 0.2
        
        # Higher if there's relevant knowledge
        if factors["knowledge_relevance"]:
            score += 0.2
            
        return max(0.0, min(1.0, score))
    
    def _calculate_tool_score(self, factors: Dict, context_info: Dict) -> float:
        """Calculate score for tool execution"""
        base_score = 0.3
        score = base_score
        
        # Higher if tools are relevant
        score += factors["tool_relevance"] * 0.4
        
        # Lower for very complex tasks (might need reasoning first)
        complexity_penalty = max(0, factors["task_complexity"] - 0.7) * 0.3
        score -= complexity_penalty
        
        # Bonus if tools are available and relevant
        if context_info["available_tools"]:
            score += 0.2
            
        return max(0.0, min(1.0, score))
    
    def _calculate_learning_score(self, factors: Dict, context_info: Dict) -> float:
        """Calculate score for learning approach"""
        base_score = 0.2
        score = base_score
        
        # Higher for learning opportunities
        score += factors["learning_opportunity"] * 0.5
        
        # Higher if this is similar to past tasks
        score += factors["memory_similarity"] * 0.3
        
        # Lower for urgent tasks
        score -= factors["urgency"] * 0.4
        
        return max(0.0, min(1.0, score))
    
    def _calculate_planning_score(self, factors: Dict) -> float:
        """Calculate score for planning approach"""
        base_score = 0.4
        score = base_score
        
        # Higher for complex tasks
        score += factors["task_complexity"] * 0.4
        
        # Lower for simple tasks
        if factors["task_complexity"] < 0.4:
            score -= 0.3
            
        return max(0.0, min(1.0, score))
    
    def _calculate_evaluation_score(self, factors: Dict) -> float:
        """Calculate score for evaluation approach"""
        base_score = 0.3
        score = base_score
        
        # Higher when there's relevant knowledge to evaluate against
        if factors["knowledge_relevance"]:
            score += 0.3
            
        # Higher for moderate complexity tasks
        if 0.4 <= factors["task_complexity"] <= 0.8:
            score += 0.2
            
        return max(0.0, min(1.0, score))
    
    async def _select_action(self, action_scores: Dict[ActionType, float], factors: Dict) -> ActionType:
        """Select the best action based on scores"""
        # Apply some strategic adjustments
        best_action = max(action_scores.items(), key=lambda x: x[1])[0]
        
        # Strategic overrides
        if factors["task_complexity"] > 0.8 and action_scores[ActionType.PLAN] > 0.6:
            return ActionType.PLAN
            
        if factors["learning_opportunity"] > 0.7 and action_scores[ActionType.LEARN] > 0.5:
            return ActionType.LEARN
            
        return best_action
    
    async def _generate_decision_details(self, action: ActionType, user_input: str, context_info: Dict) -> Dict[str, Any]:
        """Generate detailed decision with reasoning"""
        decision_details = {
            "action": action.value,
            "confidence": 0.8,  # Default confidence
            "reasoning": f"Selected {action.value} approach based on task analysis",
            "context_used": {
                "recent_interactions": len(context_info["recent_history"]),
                "relevant_knowledge": len(context_info["relevant_knowledge"]),
                "available_tools": len(context_info["available_tools"])
            }
        }
        
        # Add action-specific details
        if action == ActionType.TOOL:
            decision_details["tool_name"] = await self._select_best_tool(user_input, context_info)
            decision_details["tool_params"] = {"input": user_input}
            
        elif action == ActionType.PLAN:
            decision_details["planning_approach"] = "decomposition"
            
        elif action == ActionType.LEARN:
            decision_details["learning_focus"] = "strategy_optimization"
            
        return decision_details
    
    async def _select_best_tool(self, user_input: str, context_info: Dict) -> str:
        """Select the most appropriate tool for the task"""
        available_tools = context_info["available_tools"]
        if not available_tools:
            return "default_tool"
            
        # Simple selection based on keywords
        tool_keywords = {
            "code_runner": ["code", "program", "script", "function"],
            "web_agent": ["search", "web", "internet", "online"],
            "file_agent": ["file", "document", "read", "write"],
            "data_analysis_agent": ["data", "analyze", "statistics", "numbers"]
        }
        
        best_tool = "default_tool"
        best_score = 0
        
        for tool, keywords in tool_keywords.items():
            if tool in available_tools:
                score = sum(1 for keyword in keywords if keyword in user_input.lower())
                if score > best_score:
                    best_score = score
                    best_tool = tool
                    
        return best_tool

# Example usage
if __name__ == "__main__":
    # This would be initialized with actual components
    pass