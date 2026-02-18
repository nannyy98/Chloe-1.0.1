"""
Self-Critique Mechanism - Implements self-evaluation and improvement for failed tasks
"""
import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from utils.config import Config
from utils.logger import setup_logger
from agents.ollama_agent import OllamaAgent
from learning.strategies import BaseStrategy, get_all_strategies


class SelfCritiqueEngine:
    """Manages self-critique and improvement for task failures"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger("self_critique_engine")
        self.ollama_agent = OllamaAgent(config)
        self.strategies = get_all_strategies(config)
        self.critique_history = []
        self.improvement_suggestions = []
        
    async def critique_failed_task(self, task: str, original_result: Dict, 
                                 original_strategy: str, context: Dict = None) -> Dict[str, Any]:
        """Generate critique for a failed task and suggest improvements"""
        try:
            # Generate detailed critique
            critique = await self._generate_critique(task, original_result, original_strategy, context)
            
            # Identify alternative strategies
            alternative_strategies = await self._identify_alternative_strategies(
                task, original_result, original_strategy, critique
            )
            
            # Generate improvement suggestions
            suggestions = await self._generate_improvement_suggestions(
                task, original_result, original_strategy, critique, alternative_strategies
            )
            
            # Create critique record
            critique_record = {
                "task": task,
                "original_strategy": original_strategy,
                "original_result": original_result,
                "critique": critique,
                "alternative_strategies": alternative_strategies,
                "improvement_suggestions": suggestions,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store critique
            self.critique_history.append(critique_record)
            
            # Keep only recent critiques
            if len(self.critique_history) > 50:
                self.critique_history.pop(0)
            
            self.logger.info(f"Generated critique for failed task using {original_strategy}")
            
            return critique_record
            
        except Exception as e:
            self.logger.error(f"Error generating critique: {e}")
            return self._generate_fallback_critique(task, original_result, original_strategy)
    
    async def _generate_critique(self, task: str, result: Dict, strategy: str, context: Dict = None) -> Dict[str, Any]:
        """Generate detailed critique of the failed attempt"""
        result_str = json.dumps(result, indent=2)
        context_str = json.dumps(context, indent=2) if context else "No additional context"
        
        prompt = f"""
        You are an AI assistant analyzing a failed task attempt. Provide a detailed critique.
        
        Task: {task}
        Strategy Used: {strategy}
        Result: {result_str}
        Context: {context_str}
        
        Analyze what went wrong and provide a structured critique in JSON format:
        {{
            "failure_analysis": {{
                "primary_causes": ["cause1", "cause2"],
                "secondary_factors": ["factor1", "factor2"],
                "strategy_mismatch": "Was the strategy appropriate for this task?",
                "execution_issues": ["issue1", "issue2"]
            }},
            "performance_rating": 0.0-1.0,
            "key_problems": ["problem1", "problem2", "problem3"],
            "what_went_wrong": "Detailed explanation of failures",
            "success_indicators": ["What would success look like?"],
            "critical_factors": ["Most important factors for success"]
        }}
        """
        
        try:
            critique_result = await self.ollama_agent.execute({
                "prompt": prompt,
                "temperature": 0.4,
                "max_tokens": 1000
            })
            
            if critique_result["status"] == "success":
                return json.loads(critique_result["result"])
            else:
                return self._generate_fallback_analysis(result)
                
        except Exception as e:
            self.logger.warning(f"Critique generation failed: {e}")
            return self._generate_fallback_analysis(result)
    
    def _generate_fallback_analysis(self, result: Dict) -> Dict[str, Any]:
        """Generate fallback analysis when LLM fails"""
        error_msg = str(result.get("error", "Unknown error"))
        
        # Simple error classification
        if "timeout" in error_msg.lower():
            primary_causes = ["Time constraints exceeded"]
            key_problems = ["Processing took too long"]
        elif "tool" in error_msg.lower():
            primary_causes = ["Tool execution failed"]
            key_problems = ["Tool integration issues"]
        elif "memory" in error_msg.lower():
            primary_causes = ["Memory limitations"]
            key_problems = ["Resource constraints"]
        else:
            primary_causes = ["General execution failure"]
            key_problems = ["Unknown execution issues"]
        
        return {
            "failure_analysis": {
                "primary_causes": primary_causes,
                "secondary_factors": ["System limitations"],
                "strategy_mismatch": "Unclear",
                "execution_issues": key_problems
            },
            "performance_rating": 0.2,
            "key_problems": key_problems,
            "what_went_wrong": f"Task failed with error: {error_msg}",
            "success_indicators": ["Error-free execution", "Complete task completion"],
            "critical_factors": ["Reliable execution", "Appropriate strategy selection"]
        }
    
    async def _identify_alternative_strategies(self, task: str, result: Dict, 
                                            original_strategy: str, critique: Dict) -> List[Dict[str, Any]]:
        """Identify alternative strategies that might work better"""
        alternative_strategies = []
        
        # Get all strategies except the original one
        other_strategies = [s for s in self.strategies if s.name != original_strategy]
        
        for strategy in other_strategies:
            try:
                # Analyze suitability of alternative strategy
                analysis = await strategy.analyze_task(task)
                suitability = analysis.get("suitability", 0.0)
                
                # Consider the critique insights
                failure_causes = critique.get("failure_analysis", {}).get("primary_causes", [])
                strategy_strengths = self._get_strategy_strengths(strategy.name)
                
                # Calculate if this strategy addresses the failure causes
                match_score = self._calculate_match_score(failure_causes, strategy_strengths)
                
                # Combined score
                combined_score = (0.6 * suitability) + (0.4 * match_score)
                
                if combined_score > 0.3:  # Only suggest if reasonably suitable
                    alternative_strategies.append({
                        "strategy_name": strategy.name,
                        "suitability": suitability,
                        "match_score": match_score,
                        "combined_score": combined_score,
                        "reasoning": analysis.get("reasoning", "Alternative approach"),
                        "estimated_benefit": combined_score - 0.3  # How much better it might be
                    })
                    
            except Exception as e:
                self.logger.warning(f"Error analyzing alternative strategy {strategy.name}: {e}")
                continue
        
        # Sort by combined score
        return sorted(alternative_strategies, key=lambda x: x["combined_score"], reverse=True)
    
    def _get_strategy_strengths(self, strategy_name: str) -> List[str]:
        """Get strengths of a strategy for matching with failure causes"""
        strengths = {
            "ReAct": ["Iterative reasoning", "Action-observation cycle", "Complex problem solving"],
            "ChainOfThought": ["Step-by-step reasoning", "Mathematical/logical tasks", "Clear explanation"],
            "PlanAndExecute": ["Structured approach", "Multi-step tasks", "Systematic execution"]
        }
        return strengths.get(strategy_name, ["General problem solving"])
    
    def _calculate_match_score(self, failure_causes: List[str], strategy_strengths: List[str]) -> float:
        """Calculate how well strategy strengths match failure causes"""
        if not failure_causes or not strategy_strengths:
            return 0.0
        
        # Simple keyword matching for now
        match_count = 0
        failure_text = " ".join(failure_causes).lower()
        
        for strength in strategy_strengths:
            if strength.lower() in failure_text:
                match_count += 1
        
        return min(1.0, match_count / len(strategy_strengths))
    
    async def _generate_improvement_suggestions(self, task: str, result: Dict, 
                                              original_strategy: str, critique: Dict, 
                                              alternative_strategies: List[Dict]) -> List[str]:
        """Generate specific improvement suggestions"""
        prompt = f"""
        Based on this task failure analysis, generate specific improvement suggestions:
        
        Task: {task}
        Original Strategy: {original_strategy}
        Critique: {json.dumps(critique, indent=2)}
        Alternative Strategies: {json.dumps(alternative_strategies, indent=2)}
        
        Provide 5 specific, actionable improvement suggestions in JSON array format:
        [
            "Suggestion 1: Specific actionable improvement",
            "Suggestion 2: Another specific improvement",
            "Suggestion 3: Systematic improvement",
            "Suggestion 4: Process improvement", 
            "Suggestion 5: Knowledge/learning improvement"
        ]
        """
        
        try:
            suggestions_result = await self.ollama_agent.execute({
                "prompt": prompt,
                "temperature": 0.5,
                "max_tokens": 800
            })
            
            if suggestions_result["status"] == "success":
                try:
                    suggestions = json.loads(suggestions_result["result"])
                    if isinstance(suggestions, list):
                        return suggestions[:5]  # Limit to 5 suggestions
                except json.JSONDecodeError:
                    pass
                    
        except Exception as e:
            self.logger.warning(f"Suggestions generation failed: {e}")
        
        # Fallback suggestions
        return self._generate_fallback_suggestions(original_strategy, critique)
    
    def _generate_fallback_suggestions(self, original_strategy: str, critique: Dict) -> List[str]:
        """Generate fallback improvement suggestions"""
        failure_causes = critique.get("failure_analysis", {}).get("primary_causes", [])
        
        suggestions = [
            f"Consider alternative strategy instead of {original_strategy}",
            "Improve error handling and validation",
            "Add more detailed task analysis before execution",
            "Implement better resource management",
            "Enhance monitoring and early termination for problematic tasks"
        ]
        
        # Add strategy-specific suggestions
        if "timeout" in str(failure_causes).lower():
            suggestions.append("Implement timeout handling and progress monitoring")
        if "tool" in str(failure_causes).lower():
            suggestions.append("Improve tool integration and error recovery")
            
        return suggestions[:5]
    
    async def retry_with_alternative_strategy(self, task: str, original_result: Dict,
                                            alternative_strategy: str, context: Dict = None) -> Dict[str, Any]:
        """Retry task with an alternative strategy"""
        try:
            # Get the alternative strategy
            from learning.strategies import get_strategy_by_name
            strategy = get_strategy_by_name(self.config, alternative_strategy)
            
            if not strategy:
                return {
                    "error": f"Alternative strategy {alternative_strategy} not found",
                    "status": "failed"
                }
            
            self.logger.info(f"Retrying task with alternative strategy: {alternative_strategy}")
            
            # Execute with alternative strategy
            start_time = datetime.now()
            result = await strategy.execute(task, context)
            end_time = datetime.now()
            
            execution_time = (end_time - start_time).total_seconds()
            
            # Return enhanced result with retry information
            return {
                "original_result": original_result,
                "retry_result": result,
                "alternative_strategy": alternative_strategy,
                "execution_time": execution_time,
                "improvement": self._calculate_improvement(original_result, result),
                "status": "retry_completed"
            }
            
        except Exception as e:
            self.logger.error(f"Alternative strategy retry failed: {e}")
            return {
                "error": f"Retry with {alternative_strategy} failed: {str(e)}",
                "original_result": original_result,
                "status": "retry_failed"
            }
    
    def _calculate_improvement(self, original_result: Dict, new_result: Dict) -> Dict[str, Any]:
        """Calculate improvement metrics between original and retry results"""
        original_success = original_result.get("status") == "success"
        new_success = new_result.get("status") == "success"
        
        # Simple improvement calculation
        if new_success and not original_success:
            improvement_type = "failure_to_success"
            score_improvement = 0.5
        elif new_success and original_success:
            improvement_type = "success_to_better_success"
            score_improvement = 0.2
        elif not new_success and not original_success:
            improvement_type = "continued_failure"
            score_improvement = -0.1
        else:
            improvement_type = "success_to_failure"
            score_improvement = -0.3
        
        return {
            "type": improvement_type,
            "score_improvement": score_improvement,
            "original_success": original_success,
            "new_success": new_success
        }
    
    def get_critique_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent critique history"""
        return self.critique_history[-limit:]
    
    def get_improvement_statistics(self) -> Dict[str, Any]:
        """Get statistics about improvements and critiques"""
        if not self.critique_history:
            return {
                "total_critiques": 0,
                "successful_improvements": 0,
                "improvement_rate": 0.0,
                "most_common_failures": []
            }
        
        total_critiques = len(self.critique_history)
        successful_improvements = sum(
            1 for critique in self.critique_history 
            if any("success" in str(alt.get("strategy_name", "")).lower() 
                  for alt in critique.get("alternative_strategies", []))
        )
        
        # Extract common failure causes
        failure_causes = []
        for critique in self.critique_history:
            causes = critique.get("critique", {}).get("failure_analysis", {}).get("primary_causes", [])
            failure_causes.extend(causes)
        
        # Count most common causes
        cause_counts = {}
        for cause in failure_causes:
            cause_counts[cause] = cause_counts.get(cause, 0) + 1
        
        most_common = sorted(cause_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_critiques": total_critiques,
            "successful_improvements": successful_improvements,
            "improvement_rate": successful_improvements / total_critiques if total_critiques > 0 else 0.0,
            "most_common_failures": [{"cause": cause, "count": count} for cause, count in most_common]
        }
    
    def _generate_fallback_critique(self, task: str, result: Dict, strategy: str) -> Dict[str, Any]:
        """Generate simple fallback critique when everything fails"""
        return {
            "task": task,
            "original_strategy": strategy,
            "original_result": result,
            "critique": {
                "failure_analysis": {
                    "primary_causes": ["System error", "Execution failure"],
                    "secondary_factors": ["Unknown factors"],
                    "strategy_mismatch": "Unclear",
                    "execution_issues": ["General failure"]
                },
                "performance_rating": 0.1,
                "key_problems": ["Task execution failed"],
                "what_went_wrong": "System was unable to complete the task successfully",
                "success_indicators": ["Error-free execution", "Complete task completion"],
                "critical_factors": ["Reliable execution", "Error handling"]
            },
            "alternative_strategies": [
                {
                    "strategy_name": "ReAct",
                    "suitability": 0.6,
                    "reasoning": "Alternative reasoning approach"
                }
            ],
            "improvement_suggestions": [
                "Try different strategy",
                "Improve error handling",
                "Add better validation"
            ],
            "timestamp": datetime.now().isoformat()
        }


# Example usage
if __name__ == "__main__":
    # This would be run with proper configuration
    pass