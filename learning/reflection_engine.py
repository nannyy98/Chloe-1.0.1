"""
Reflection Engine - Generates post-task lessons and insights
"""
import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

from utils.config import Config
from utils.logger import setup_logger
from agents.ollama_agent import OllamaAgent


class ReflectionEngine:
    """Generates reflections and lessons learned from task experiences"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger("reflection_engine")
        self.ollama_agent = OllamaAgent(config)
        
    async def generate_reflection(self, task: str, actions: List[Dict], result: Dict, success_score: float) -> str:
        """Generate LLM-based reflection on task performance"""
        try:
            # Create reflection prompt
            prompt = self._create_reflection_prompt(task, actions, result, success_score)
            
            # Get reflection from LLM
            reflection_result = await self.ollama_agent.execute({
                "prompt": prompt,
                "temperature": 0.7,
                "max_tokens": 1000
            })
            
            if reflection_result["status"] == "success":
                reflection = reflection_result["result"]
                self.logger.info(f"Generated reflection for task: {task[:50]}...")
                return reflection.strip()
            else:
                self.logger.warning(f"Failed to generate reflection: {reflection_result.get('error')}")
                return self._generate_fallback_reflection(task, success_score)
                
        except Exception as e:
            self.logger.error(f"Error generating reflection: {e}")
            return self._generate_fallback_reflection(task, success_score)
    
    def _create_reflection_prompt(self, task: str, actions: List[Dict], result: Dict, success_score: float) -> str:
        """Create structured prompt for reflection generation"""
        actions_str = json.dumps(actions, indent=2)
        result_str = json.dumps(result, indent=2)
        
        return f"""
        You are an AI assistant analyzing your own performance on a task. 
        Reflect on what happened and extract key lessons learned.

        Task: {task}
        Actions Taken: {actions_str}
        Result: {result_str}
        Success Score: {success_score}/1.0

        Please provide a structured reflection in the following format:

        ## Performance Analysis
        [Brief analysis of what went well and what didn't]

        ## Key Lessons Learned
        - Lesson 1: [specific insight]
        - Lesson 2: [specific insight]
        - Lesson 3: [specific insight]

        ## Improvement Suggestions
        [What could be done differently next time]

        ## Success Factors
        [What contributed to the success/failure]

        Keep it concise but insightful. Focus on actionable insights.
        """
    
    def _generate_fallback_reflection(self, task: str, success_score: float) -> str:
        """Generate simple fallback reflection when LLM fails"""
        if success_score >= 0.8:
            return f"Task '{task[:50]}...' was completed successfully. The approach worked well."
        elif success_score >= 0.5:
            return f"Task '{task[:50]}...' had mixed results. Some aspects worked, others need improvement."
        else:
            return f"Task '{task[:50]}...' was not successful. Need to try a different approach next time."
    
    async def extract_patterns(self, recent_experiences: List[Dict]) -> Dict[str, Any]:
        """Extract common patterns from recent experiences"""
        try:
            # Create pattern extraction prompt
            experiences_text = "\n\n".join([
                f"Task: {exp.get('task', 'Unknown')}\n"
                f"Success: {exp.get('success_score', 0)}\n"
                f"Reflection: {exp.get('reflection', 'No reflection')}"
                for exp in recent_experiences[:10]  # Last 10 experiences
            ])
            
            prompt = f"""
            Analyze these recent task experiences and identify common patterns:

            {experiences_text}

            Please identify:
            1. Common success factors
            2. Recurring challenges
            3. Effective strategies
            4. Areas for improvement

            Provide a concise analysis in JSON format:
            {{
                "success_patterns": ["pattern1", "pattern2"],
                "challenge_patterns": ["challenge1", "challenge2"],
                "effective_strategies": ["strategy1", "strategy2"],
                "improvement_areas": ["area1", "area2"]
            }}
            """
            
            result = await self.ollama_agent.execute({
                "prompt": prompt,
                "temperature": 0.3,
                "max_tokens": 800
            })
            
            if result["status"] == "success":
                try:
                    patterns = json.loads(result["result"])
                    return patterns
                except json.JSONDecodeError:
                    # Return structured fallback
                    return self._extract_fallback_patterns(recent_experiences)
            else:
                return self._extract_fallback_patterns(recent_experiences)
                
        except Exception as e:
            self.logger.error(f"Error extracting patterns: {e}")
            return self._extract_fallback_patterns(recent_experiences)
    
    def _extract_fallback_patterns(self, recent_experiences: List[Dict]) -> Dict[str, Any]:
        """Extract basic patterns without LLM"""
        success_count = sum(1 for exp in recent_experiences if exp.get('success_score', 0) >= 0.7)
        total_count = len(recent_experiences)
        success_rate = success_count / total_count if total_count > 0 else 0
        
        return {
            "success_patterns": ["Tasks with clear instructions tend to succeed"],
            "challenge_patterns": ["Ambiguous tasks require more clarification"],
            "effective_strategies": ["Breaking down complex tasks into steps"],
            "improvement_areas": ["Need better error handling for edge cases"],
            "success_rate": success_rate
        }
    
    async def generate_improvement_plan(self, patterns: Dict[str, Any], current_performance: float) -> Dict[str, Any]:
        """Generate improvement plan based on identified patterns"""
        try:
            prompt = f"""
            Based on these performance patterns and current performance of {current_performance}:
            
            Patterns: {json.dumps(patterns, indent=2)}
            
            Generate a specific improvement plan with:
            1. Priority areas to focus on
            2. Specific actions to take
            3. Expected outcomes
            4. Timeline for implementation
            
            Return in JSON format:
            {{
                "priority_areas": ["area1", "area2"],
                "specific_actions": ["action1", "action2"],
                "expected_outcomes": ["outcome1", "outcome2"],
                "timeline_days": 7
            }}
            """
            
            result = await self.ollama_agent.execute({
                "prompt": prompt,
                "temperature": 0.4,
                "max_tokens": 600
            })
            
            if result["status"] == "success":
                try:
                    plan = json.loads(result["result"])
                    return plan
                except json.JSONDecodeError:
                    return self._generate_fallback_plan(current_performance)
            else:
                return self._generate_fallback_plan(current_performance)
                
        except Exception as e:
            self.logger.error(f"Error generating improvement plan: {e}")
            return self._generate_fallback_plan(current_performance)
    
    def _generate_fallback_plan(self, current_performance: float) -> Dict[str, Any]:
        """Generate simple fallback improvement plan"""
        if current_performance < 0.6:
            return {
                "priority_areas": ["Error handling", "Task understanding"],
                "specific_actions": ["Add more validation", "Improve prompt engineering"],
                "expected_outcomes": ["Reduced error rate", "Better task comprehension"],
                "timeline_days": 5
            }
        else:
            return {
                "priority_areas": ["Optimization", "Advanced features"],
                "specific_actions": ["Fine-tune responses", "Add specialized tools"],
                "expected_outcomes": ["Faster processing", "Enhanced capabilities"],
                "timeline_days": 10
            }


# Example usage
if __name__ == "__main__":
    # This would be run with proper configuration
    pass