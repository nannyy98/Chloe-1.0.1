"""
Strategy Frameworks - Implementation of different reasoning strategies
"""
import asyncio
import json
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from datetime import datetime

from utils.config import Config
from utils.logger import setup_logger
from agents.ollama_agent import OllamaAgent


class BaseStrategy(ABC):
    """Abstract base class for all strategies"""
    
    def __init__(self, config: Config, name: str):
        self.config = config
        self.name = name
        self.logger = setup_logger(f"strategy_{name}")
        self.ollama_agent = OllamaAgent(config)
        self.success_count = 0
        self.total_attempts = 0
        self.performance_history = []
        
    @abstractmethod
    async def execute(self, task: str, context: Dict = None) -> Dict[str, Any]:
        """Execute the strategy for a given task"""
        pass
    
    @abstractmethod
    async def analyze_task(self, task: str, context: Dict = None) -> Dict[str, Any]:
        """Analyze task suitability for this strategy"""
        pass
    
    def update_performance(self, success: bool, score: float = None):
        """Update performance metrics"""
        self.total_attempts += 1
        if success:
            self.success_count += 1
            
        performance = {
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "score": score or (1.0 if success else 0.0)
        }
        self.performance_history.append(performance)
        
        # Keep only recent performance data
        if len(self.performance_history) > 50:
            self.performance_history.pop(0)
    
    def get_success_rate(self) -> float:
        """Calculate current success rate"""
        if self.total_attempts == 0:
            return 0.0
        return self.success_count / self.total_attempts
    
    def get_recent_performance(self, window: int = 10) -> float:
        """Get recent performance score"""
        if not self.performance_history:
            return 0.0
            
        recent = self.performance_history[-window:]
        if not recent:
            return 0.0
            
        return sum(p["score"] for p in recent) / len(recent)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        return {
            "name": self.name,
            "total_attempts": self.total_attempts,
            "success_count": self.success_count,
            "success_rate": self.get_success_rate(),
            "recent_performance": self.get_recent_performance(),
            "performance_history": len(self.performance_history)
        }


class ReActStrategy(BaseStrategy):
    """Reasoning with Action and Observation (ReAct) strategy"""
    
    def __init__(self, config: Config):
        super().__init__(config, "ReAct")
        self.max_steps = 20
        
    async def execute(self, task: str, context: Dict = None) -> Dict[str, Any]:
        """Execute ReAct strategy"""
        try:
            # Implement ReAct loop
            trace = []
            current_state = context or {}
            current_state['input'] = task
            
            for step in range(self.max_steps):
                # 1. REASON: Generate thought
                thought = await self._generate_thought(task, current_state, trace, step)
                trace.append({"step": step, "type": "thought", "content": thought})
                
                # 2. ACT: Determine action
                action = await self._select_action(thought, current_state)
                
                if action["type"] == "finish":
                    return {
                        "strategy": self.name,
                        "final_answer": action["content"],
                        "trace": trace,
                        "steps_taken": step + 1,
                        "confidence": action.get("confidence", 0.8),
                        "status": "completed"
                    }
                
                # 3. OBSERVE: Execute action and observe
                observation = await self._execute_action(action, current_state)
                trace.append({"step": step, "type": "action", "content": action})
                trace.append({"step": step, "type": "observation", "content": observation})
                
                # Update state
                current_state = await self._update_state(current_state, action, observation)
                
                # Check if should finish
                if self._should_finish(thought, observation):
                    final_answer = await self._generate_final_answer(trace)
                    return {
                        "strategy": self.name,
                        "final_answer": final_answer,
                        "trace": trace,
                        "steps_taken": step + 1,
                        "confidence": thought.get("confidence", 0.8),
                        "status": "completed"
                    }
            
            # Max steps reached
            final_answer = await self._generate_final_answer(trace)
            return {
                "strategy": self.name,
                "final_answer": final_answer,
                "trace": trace,
                "steps_taken": self.max_steps,
                "confidence": 0.5,
                "status": "max_steps_reached"
            }
            
        except Exception as e:
            self.logger.error(f"ReAct strategy failed: {e}")
            return {
                "strategy": self.name,
                "error": str(e),
                "status": "failed"
            }
    
    async def analyze_task(self, task: str, context: Dict = None) -> Dict[str, Any]:
        """Analyze if task is suitable for ReAct strategy"""
        prompt = f"""
        Analyze if this task is suitable for the ReAct (Reasoning-Action-Observation) strategy:
        
        Task: {task}
        Context: {context or 'No additional context'}
        
        Consider:
        - Does this require step-by-step reasoning?
        - Are there actions that can be taken and observed?
        - Is this a complex problem that benefits from iterative thinking?
        
        Respond in JSON format like this:
        {{suitability: 0.8, reasoning: "Your explanation here", estimated_steps: "5", key_factors: ["factor1", "factor2"]}}
        """
        
        try:
            result = await self.ollama_agent.execute({
                "prompt": prompt,
                "temperature": 0.3,
                "max_tokens": 500
            })
            
            if result["status"] == "success":
                try:
                    return json.loads(result["result"])
                except json.JSONDecodeError:
                    # Fallback analysis
                    return self._fallback_analysis(task)
            else:
                # Fallback analysis
                return self._fallback_analysis(task)
        except Exception as e:
            self.logger.warning(f"Task analysis failed: {e}")
            return self._fallback_analysis(task)
    
    def _fallback_analysis(self, task: str) -> Dict[str, Any]:
        """Fallback task analysis when LLM fails"""
        task_lower = task.lower()
        
        # Simple heuristic-based analysis
        complexity_indicators = ["calculate", "analyze", "explain", "solve", "implement"]
        action_indicators = ["search", "find", "get", "compute", "process"]
        
        complexity_score = sum(1 for word in complexity_indicators if word in task_lower)
        action_score = sum(1 for word in action_indicators if word in task_lower)
        
        suitability = min(0.9, (complexity_score + action_score) / len(complexity_indicators))
        
        return {
            "suitability": suitability,
            "reasoning": "Heuristic-based analysis",
            "estimated_steps": "5-10",
            "key_factors": ["Complexity indicators", "Action requirements"]
        }
    
    async def _generate_thought(self, task: str, current_state: Dict, trace: List, step: int) -> Dict[str, Any]:
        """Generate reasoning thought for current step"""
        trace_str = json.dumps(trace[-3:], indent=2) if trace else "No previous steps"
        
        prompt = f"""
        You are using the ReAct strategy. Generate your next reasoning step.
        
        Task: {task}
        Current State: {json.dumps(current_state, indent=2)}
        Previous Steps: {trace_str}
        
        Think step by step:
        1. What is the current situation?
        2. What needs to be done?
        3. What action should I take next?
        
        Respond in JSON format:
        {{
            "thought": "Your reasoning about the task and what to do",
            "reason": "Why you chose this approach",
            "next_action": "tool_use|reasoning|finish",
            "confidence": 0.0-1.0
        }}
        """
        
        result = await self.ollama_agent.execute({
            "prompt": prompt,
            "temperature": 0.7,
            "max_tokens": 800
        })
        
        if result["status"] == "success":
            return json.loads(result["result"])
        else:
            return {
                "thought": "Continue with systematic approach",
                "reason": "Fallback reasoning",
                "next_action": "reasoning",
                "confidence": 0.6
            }
    
    async def _select_action(self, thought: Dict, current_state: Dict) -> Dict[str, Any]:
        """Select action based on thought"""
        next_action = thought.get("next_action", "reasoning")
        
        if next_action == "finish":
            return {
                "type": "finish",
                "content": thought.get("thought", "No answer generated"),
                "confidence": thought.get("confidence", 0.5)
            }
        elif next_action == "tool_use":
            return {
                "type": "tool",
                "tool_name": "placeholder",
                "arguments": {},
                "description": "Tool action to be implemented"
            }
        else:
            return {
                "type": "reason",
                "content": thought.get("thought", "Continue reasoning"),
                "next_step": "Continue processing"
            }
    
    async def _execute_action(self, action: Dict, current_state: Dict) -> str:
        """Execute the selected action"""
        if action["type"] == "finish":
            return "Final answer generated"
        elif action["type"] == "tool":
            return f"Executed tool: {action.get('tool_name', 'unknown')}"
        else:
            return f"Continuing reasoning: {action.get('content', 'Continue')}"
    
    async def _update_state(self, current_state: Dict, action: Dict, observation: str) -> Dict:
        """Update state based on action and observation"""
        updated_state = current_state.copy()
        updated_state["last_action"] = action
        updated_state["last_observation"] = observation
        updated_state["step_count"] = updated_state.get("step_count", 0) + 1
        return updated_state
    
    def _should_finish(self, thought: Dict, observation: str) -> bool:
        """Determine if we should finish the loop"""
        thought_content = thought.get("thought", "").lower()
        return ("final answer:" in thought_content or 
                "conclusion:" in thought_content or
                thought.get("next_action") == "finish")
    
    async def _generate_final_answer(self, trace: List) -> str:
        """Generate final answer from the trace"""
        trace_str = json.dumps(trace, indent=2)
        
        prompt = f"""
        Based on the following ReAct trace, provide a final answer:
        
        Reasoning Trace: {trace_str}
        
        Provide a clear, concise final answer.
        """
        
        result = await self.ollama_agent.execute({
            "prompt": prompt,
            "temperature": 0.3,
            "max_tokens": 500
        })
        
        if result["status"] == "success":
            return result["result"]
        else:
            return "Answer generated through ReAct process"


class ChainOfThoughtStrategy(BaseStrategy):
    """Chain of Thought reasoning strategy"""
    
    def __init__(self, config: Config):
        super().__init__(config, "ChainOfThought")
        
    async def execute(self, task: str, context: Dict = None) -> Dict[str, Any]:
        """Execute Chain of Thought strategy"""
        try:
            # Generate step-by-step reasoning
            prompt = f"""
            Solve this task using Chain of Thought reasoning. Think through it step by step.
            
            Task: {task}
            Context: {context or 'No additional context'}
            
            Provide your solution with clear reasoning steps:
            1. [First step and reasoning]
            2. [Second step and reasoning]
            3. [Continue as needed...]
            
            Final Answer: [Your conclusion]
            """
            
            result = await self.ollama_agent.execute({
                "prompt": prompt,
                "temperature": 0.5,
                "max_tokens": 1500
            })
            
            if result["status"] == "success":
                return {
                    "strategy": self.name,
                    "result": result["result"],
                    "reasoning_steps": self._extract_reasoning_steps(result["result"]),
                    "confidence": 0.8,
                    "status": "completed"
                }
            else:
                return {
                    "strategy": self.name,
                    "error": result.get("error", "Unknown error"),
                    "status": "failed"
                }
                
        except Exception as e:
            self.logger.error(f"CoT strategy failed: {e}")
            return {
                "strategy": self.name,
                "error": str(e),
                "status": "failed"
            }
    
    async def analyze_task(self, task: str, context: Dict = None) -> Dict[str, Any]:
        """Analyze if task is suitable for Chain of Thought"""
        # CoT works well for mathematical, logical, and reasoning tasks
        task_lower = task.lower()
        reasoning_keywords = ["calculate", "solve", "prove", "explain", "analyze", "compare", "evaluate"]
        
        suitability = sum(1 for word in reasoning_keywords if word in task_lower) / len(reasoning_keywords)
        suitability = min(0.9, suitability + 0.3)  # Base suitability
        
        return {
            "suitability": suitability,
            "reasoning": "CoT suitable for reasoning and analytical tasks",
            "estimated_steps": "3-7",
            "key_factors": ["Reasoning requirements", "Logical analysis needed"]
        }
    
    def _extract_reasoning_steps(self, response: str) -> List[str]:
        """Extract reasoning steps from response"""
        lines = response.split('\n')
        steps = []
        current_step = ""
        
        for line in lines:
            if line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                if current_step:
                    steps.append(current_step.strip())
                current_step = line
            elif current_step:
                current_step += " " + line
        
        if current_step:
            steps.append(current_step.strip())
            
        return steps if steps else [response]


class PlanAndExecuteStrategy(BaseStrategy):
    """Plan and Execute strategy"""
    
    def __init__(self, config: Config):
        super().__init__(config, "PlanAndExecute")
        
    async def execute(self, task: str, context: Dict = None) -> Dict[str, Any]:
        """Execute Plan and Execute strategy"""
        try:
            # 1. Generate plan
            plan = await self._generate_plan(task, context)
            
            # 2. Execute plan steps
            execution_results = []
            for i, step in enumerate(plan.get("steps", [])):
                step_result = await self._execute_step(step, i, execution_results)
                execution_results.append(step_result)
            
            # 3. Generate final answer
            final_answer = await self._generate_final_answer(plan, execution_results)
            
            return {
                "strategy": self.name,
                "plan": plan,
                "execution_results": execution_results,
                "final_answer": final_answer,
                "confidence": 0.85,
                "status": "completed"
            }
            
        except Exception as e:
            self.logger.error(f"Plan and Execute strategy failed: {e}")
            return {
                "strategy": self.name,
                "error": str(e),
                "status": "failed"
            }
    
    async def analyze_task(self, task: str, context: Dict = None) -> Dict[str, Any]:
        """Analyze if task is suitable for Plan and Execute"""
        # P&E works well for complex, multi-step tasks
        task_lower = task.lower()
        complexity_indicators = ["plan", "steps", "multiple", "several", "various", "complex", "detailed"]
        
        suitability = sum(1 for word in complexity_indicators if word in task_lower) / len(complexity_indicators)
        suitability = min(0.9, suitability + 0.4)  # Higher base suitability
        
        return {
            "suitability": suitability,
            "reasoning": "P&E suitable for complex, multi-step tasks",
            "estimated_steps": "4-12",
            "key_factors": ["Complexity", "Multi-step requirements"]
        }
    
    async def _generate_plan(self, task: str, context: Dict = None) -> Dict[str, Any]:
        """Generate detailed plan for the task"""
        prompt = f"""
        Create a detailed execution plan for this task:
        
        Task: {task}
        Context: {context or 'No additional context'}
        
        Provide a structured plan in JSON format:
        {{
            "task_analysis": "Brief analysis of what needs to be done",
            "steps": [
                {{
                    "step_number": 1,
                    "description": "What to do in this step",
                    "expected_outcome": "What this should accomplish",
                    "resources_needed": ["resource1", "resource2"]
                }}
            ],
            "estimated_completion_time": "time estimate",
            "success_criteria": ["criterion1", "criterion2"]
        }}
        """
        
        result = await self.ollama_agent.execute({
            "prompt": prompt,
            "temperature": 0.4,
            "max_tokens": 1000
        })
        
        if result["status"] == "success":
            try:
                return json.loads(result["result"])
            except json.JSONDecodeError:
                # Return structured fallback
                return {
                    "task_analysis": "Task requires systematic approach",
                    "steps": [
                        {"step_number": 1, "description": "Initial analysis", "expected_outcome": "Understanding"},
                        {"step_number": 2, "description": "Main execution", "expected_outcome": "Core solution"},
                        {"step_number": 3, "description": "Review and finalize", "expected_outcome": "Complete answer"}
                    ],
                    "estimated_completion_time": "moderate",
                    "success_criteria": ["Task completed", "Answer provided"]
                }
        else:
            return self._fallback_plan()
    
    def _fallback_plan(self) -> Dict[str, Any]:
        """Fallback plan when LLM fails"""
        return {
            "task_analysis": "Standard approach",
            "steps": [
                {"step_number": 1, "description": "Analyze requirements", "expected_outcome": "Clear understanding"},
                {"step_number": 2, "description": "Execute main approach", "expected_outcome": "Primary solution"},
                {"step_number": 3, "description": "Verify and refine", "expected_outcome": "Quality result"}
            ],
            "estimated_completion_time": "standard",
            "success_criteria": ["Completion", "Accuracy"]
        }
    
    async def _execute_step(self, step: Dict, step_number: int, previous_results: List) -> Dict[str, Any]:
        """Execute a single step of the plan"""
        prompt = f"""
        Execute this step of the plan:
        
        Step: {json.dumps(step, indent=2)}
        Previous Results: {json.dumps(previous_results[-2:] if previous_results else [], indent=2)}
        
        Provide the execution result and any observations.
        """
        
        result = await self.ollama_agent.execute({
            "prompt": prompt,
            "temperature": 0.3,
            "max_tokens": 500
        })
        
        return {
            "step_number": step_number,
            "step_description": step.get("description", ""),
            "result": result["result"] if result["status"] == "success" else "Step executed",
            "status": "completed" if result["status"] == "success" else "completed_with_issues"
        }
    
    async def _generate_final_answer(self, plan: Dict, execution_results: List) -> str:
        """Generate final answer from plan execution"""
        prompt = f"""
        Based on the executed plan, provide a final answer:
        
        Original Plan: {json.dumps(plan, indent=2)}
        Execution Results: {json.dumps(execution_results, indent=2)}
        
        Provide a comprehensive final answer that addresses the original task.
        """
        
        result = await self.ollama_agent.execute({
            "prompt": prompt,
            "temperature": 0.3,
            "max_tokens": 800
        })
        
        if result["status"] == "success":
            return result["result"]
        else:
            return "Answer generated through plan execution"


# Strategy factory
def get_all_strategies(config: Config) -> List[BaseStrategy]:
    """Get all available strategies"""
    return [
        ReActStrategy(config),
        ChainOfThoughtStrategy(config),
        PlanAndExecuteStrategy(config)
    ]


def get_strategy_by_name(config: Config, name: str) -> Optional[BaseStrategy]:
    """Get strategy by name"""
    strategies = {
        "ReAct": ReActStrategy(config),
        "ChainOfThought": ChainOfThoughtStrategy(config),
        "PlanAndExecute": PlanAndExecuteStrategy(config)
    }
    return strategies.get(name)


# Example usage
if __name__ == "__main__":
    # This would be run with proper configuration
    pass