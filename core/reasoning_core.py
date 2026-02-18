"""
Reasoning Core - The thinking layer of Chloe AI
Implements ReAct (Reason-Act-Observe) loop for task processing
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
# from openai import AsyncOpenAI  # Disabled - using Ollama only
# import anthropic  # Disabled - using Ollama only

from utils.config import Config
from utils.logger import setup_logger

class ReasoningCore:
    """LLM-based reasoning engine implementing ReAct (Reason-Act-Observe) loop"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger("reasoning_core")
        self.max_steps = 20  # Maximum steps in ReAct loop
        self._initialize_clients()
        
    def _initialize_clients(self):
        """Initialize LLM clients - Ollama only"""
        # OpenAI and Anthropic disabled - using Ollama only
        self.openai_client = None
        self.anthropic_client = None
        self.logger.info("Reasoning Core initialized with Ollama only")
    
    async def process(self, user_input: str, context: Dict = None) -> Dict[str, Any]:
        """Main processing method implementing ReAct loop"""
        try:
            # Execute ReAct loop
            result = await self._react_loop(user_input, context)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in reasoning process: {e}")
            return {"error": str(e), "confidence": 0.0}
    
    async def _react_loop(self, user_input: str, context: Dict = None) -> Dict[str, Any]:
        """Implement ReAct (Reason-Act-Observe) loop"""
        # Initialize the trace of thoughts, actions, and observations
        trace = []
        current_state = context or {}
        current_state['input'] = user_input
        
        for step in range(self.max_steps):
            # 1. REASON: Generate thought based on current state
            thought = await self._generate_thought(user_input, current_state, trace, step)
            
            # Add thought to trace
            trace.append({"step": step, "type": "thought", "content": thought})
            
            # 2. ACT: Determine and execute action
            action = await self._select_action(thought, current_state)
            
            if action["type"] == "finish":
                # If finish action, return the final result
                return {
                    "final_answer": action["content"],
                    "trace": trace,
                    "steps_taken": step + 1,
                    "confidence": action.get("confidence", 0.8)
                }
            
            # Execute the action
            observation = await self._execute_action(action, current_state)
            
            # Add action and observation to trace
            trace.append({"step": step, "type": "action", "content": action})
            trace.append({"step": step, "type": "observation", "content": observation})
            
            # 3. OBSERVE: Update state based on observation
            current_state = await self._update_state(current_state, action, observation)
            
            # Check if we have a final answer
            if self._should_finish(thought, observation):
                final_answer = await self._generate_final_answer(trace)
                return {
                    "final_answer": final_answer,
                    "trace": trace,
                    "steps_taken": step + 1,
                    "confidence": thought.get("confidence", 0.8)
                }
        
        # Max steps reached, return best effort answer
        final_answer = await self._generate_final_answer(trace)
        return {
            "final_answer": final_answer,
            "trace": trace,
            "steps_taken": self.max_steps,
            "confidence": 0.5,
            "status": "max_steps_reached"
        }
    
    async def _generate_thought(self, user_input: str, current_state: Dict, trace: List, step: int) -> Dict[str, Any]:
        """Generate reasoning thought for current step"""
        trace_str = json.dumps(trace[-3:], indent=2) if trace else "No previous steps"
        
        prompt = f"""
        You are a helpful AI assistant. You will be given a task. You need to reason about the task and decide what action to take.
        
        Task: {user_input}
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
        
        response = await self._call_llm(prompt, response_format="json")
        thought = json.loads(response)
        
        return thought
    
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
            # For now, return a placeholder for tool usage
            # In a full implementation, this would select a specific tool
            return {
                "type": "tool",
                "tool_name": "placeholder",
                "arguments": {},
                "description": "Tool action to be implemented"
            }
        else:  # reasoning
            return {
                "type": "reason",
                "content": thought.get("thought", "Continue reasoning"),
                "next_step": "Continue processing"
            }
    
    async def _execute_action(self, action: Dict, current_state: Dict) -> str:
        """Execute the selected action and return observation"""
        if action["type"] == "finish":
            return "Final answer generated"
        elif action["type"] == "tool":
            # Placeholder for tool execution
            # In a full implementation, this would call actual tools
            return f"Executed tool: {action.get('tool_name', 'unknown')} with args: {action.get('arguments', {})}"
        else:  # reason
            # Continue reasoning
            return f"Continuing reasoning: {action.get('content', 'Continue')}. State: {json.dumps(current_state.get('input', 'No input'))}"
    
    async def _update_state(self, current_state: Dict, action: Dict, observation: str) -> Dict:
        """Update state based on action and observation"""
        updated_state = current_state.copy()
        updated_state["last_action"] = action
        updated_state["last_observation"] = observation
        updated_state["step_count"] = updated_state.get("step_count", 0) + 1
        
        return updated_state
    
    def _should_finish(self, thought: Dict, observation: str) -> bool:
        """Determine if we should finish the loop"""
        # Finish if the thought indicates completion
        thought_content = thought.get("thought", "")
        if "final answer:" in thought_content.lower() or "conclusion:" in thought_content.lower():
            return True
        
        # Finish if the next action is to finish
        if thought.get("next_action") == "finish":
            return True
        
        return False
    
    async def _generate_final_answer(self, trace: List) -> str:
        """Generate final answer from the trace of thoughts and observations"""
        trace_str = json.dumps(trace, indent=2)
        
        prompt = f"""
        Based on the following reasoning trace, provide a final answer to the original task.
        
        Reasoning Trace: {trace_str}
        
        Provide a clear, concise final answer to the user's original request.
        """
        
        response = await self._call_llm(prompt, response_format="text")
        return response
    
    async def _analyze_task(self, user_input: str, context: Dict = None) -> Dict[str, Any]:
        """Analyze the task to determine type and complexity"""
        prompt = f"""
        Analyze this user task and classify it:
        
        Task: {user_input}
        Context: {context or 'No additional context'}
        
        Provide analysis in JSON format:
        {{
            "task_type": "coding|research|creative|analytical|planning|other",
            "complexity": "simple|moderate|complex",
            "required_approach": "reasoning|tool_execution|learning",
            "estimated_steps": number,
            "key_requirements": ["requirement1", "requirement2"]
        }}
        """
        
        response = await self._call_llm(prompt, response_format="json")
        return json.loads(response)
    
    async def _generate_reasoning(self, user_input: str, task_analysis: Dict, context: Dict = None) -> Dict[str, Any]:
        """Generate detailed reasoning for the task"""
        prompt = f"""
        Generate comprehensive reasoning for this task:
        
        Original Task: {user_input}
        Task Analysis: {json.dumps(task_analysis, indent=2)}
        Context: {context or 'No additional context'}
        
        Provide your reasoning in JSON format:
        {{
            "understanding": "Your understanding of what needs to be done",
            "approach": "Step-by-step approach to solve this",
            "considerations": ["key consideration 1", "key consideration 2"],
            "potential_challenges": ["challenge 1", "challenge 2"],
            "success_criteria": ["criterion 1", "criterion 2"],
            "confidence": 0.0-1.0
        }}
        """
        
        response = await self._call_llm(prompt, response_format="json")
        return json.loads(response)
    
    async def _create_plan(self, reasoning_result: Dict, context: Dict = None) -> List[Dict[str, Any]]:
        """Create detailed execution plan based on reasoning"""
        prompt = f"""
        Create a detailed execution plan based on this reasoning:
        
        Reasoning: {json.dumps(reasoning_result, indent=2)}
        Context: {context or 'No additional context'}
        
        Provide plan as JSON array:
        [
            {{
                "step": 1,
                "action": "description of what to do",
                "type": "reasoning|tool|learning",
                "expected_outcome": "what this step should accomplish",
                "success_criteria": "how to know if successful"
            }}
        ]
        """
        
        response = await self._call_llm(prompt, response_format="json")
        return json.loads(response)
    
    async def _call_llm(self, prompt: str, model: str = "ollama", response_format: str = "text") -> str:
        """Call Ollama LLM with proper error handling"""
        try:
            # Import Ollama agent locally to avoid circular imports
            from agents.ollama_agent import OllamaAgent
            ollama_agent = OllamaAgent(self.config)
            
            if ollama_agent.is_available():
                result = await ollama_agent.execute({
                    "prompt": prompt,
                    "temperature": 0.7,
                    "max_tokens": 2000
                })
                
                if result["status"] == "success":
                    response = result["result"]
                    if response_format == "json":
                        # Validate JSON response
                        try:
                            json.loads(response)
                            # Return original response if it's valid JSON
                            return response
                        except json.JSONDecodeError:
                            self.logger.warning("Ollama returned non-JSON response for JSON format request")
                            # Return structured response instead
                            return json.dumps({"response": response.strip()})
                    return response
                else:
                    raise Exception(f"Ollama error: {result.get('error', 'Unknown error')}")
            else:
                raise Exception("Ollama not available")
                
        except Exception as e:
            self.logger.error(f"Ollama failed: {e}")
            raise Exception(f"LLM provider failed: {str(e)}")
    
    async def generate_code(self, task_description: str, context: Dict = None) -> str:
        """Generate code for programming tasks"""
        prompt = f"""
        Generate Python code for this task:
        
        Task: {task_description}
        Context: {context or 'No additional context'}
        
        Requirements:
        - Include proper error handling
        - Add comments explaining key parts
        - Follow Python best practices
        - Make code modular and reusable
        
        Return only the code without any additional text.
        """
        
        return await self._call_llm(prompt, model="ollama")
    
    async def explain_concept(self, concept: str, depth: str = "intermediate") -> str:
        """Generate explanations for concepts"""
        prompt = f"""
        Explain the concept '{concept}' at {depth} level.
        
        Include:
        - Clear definition
        - Key principles
        - Practical examples
        - Common applications
        - Potential pitfalls
        
        Make it accessible and actionable.
        """
        
        return await self._call_llm(prompt, model="ollama")

# Example usage
if __name__ == "__main__":
    # This would be run after proper configuration setup
    pass