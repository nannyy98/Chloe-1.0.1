"""
Enhanced Reasoning Core with better error handling and fallbacks
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional
# from openai import AsyncOpenAI  # Disabled - using Ollama only
# import anthropic  # Disabled - using Ollama only
import os

from utils.config import Config
from utils.logger import setup_logger

class EnhancedReasoningCore:
    """Enhanced LLM-based reasoning engine with robust error handling"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger("enhanced_reasoning_core")
        self._initialize_clients()
        self.retry_count = 2
        self.timeout = 60
        
    def _initialize_clients(self):
        """Initialize LLM clients with proper error handling"""
        # OpenAI and Anthropic disabled - using Ollama only
        self.openai_client = None
        self.anthropic_client = None
        self.logger.info("Cloud providers disabled - using Ollama only")
        
        # Local fallback model (simulated)
        self.local_model = LocalReasoningModel()
        
        # Ollama integration
        try:
            from agents.ollama_agent import OllamaAgent
            self.ollama_agent = OllamaAgent(self.config)
            # Check availability with debug info
            self.ollama_available = self.ollama_agent.is_available()
            if self.ollama_available:
                self.logger.info(f"Ollama integration available (model: {self.ollama_agent.model})")
            else:
                self.logger.warning("Ollama not available - install and run: ollama serve")
        except Exception as e:
            self.logger.warning(f"Ollama integration failed: {e}")
            self.ollama_agent = None
            self.ollama_available = False
        
        self.logger.info("Enhanced Reasoning Core initialized")
    
    async def process(self, user_input: str, context: Dict = None) -> Dict[str, Any]:
        """Simplified processing with direct Ollama call"""
        start_time = time.time()
        
        try:
            # Direct call to Ollama for reasoning
            if self.ollama_available and self.ollama_agent:
                reasoning_result = await self._direct_reasoning(user_input, context)
            else:
                # Fallback to local model
                task_analysis = await self.local_model.analyze_task(user_input, context)
                reasoning_result = await self.local_model.generate_reasoning(user_input, task_analysis, context)
            
            # Create simple plan
            execution_plan = await self._create_validated_plan(reasoning_result, context)
            
            processing_time = time.time() - start_time
            
            return {
                "task_analysis": reasoning_result.get("task_analysis", {"task_type": "other", "complexity": "simple", "required_approach": "reasoning"}),
                "reasoning": reasoning_result,
                "execution_plan": execution_plan,
                "confidence": reasoning_result.get("confidence", 0.7),
                "processing_time": processing_time,
                "provider_used": reasoning_result.get("provider", "local")
            }
            
        except Exception as e:
            self.logger.error(f"Error in reasoning process: {e}")
            return {
                "error": str(e),
                "confidence": 0.0,
                "fallback_response": await self._generate_fallback_response(user_input),
                "processing_time": time.time() - start_time
            }
    
    async def _robust_task_analysis(self, user_input: str, context: Dict = None) -> Dict[str, Any]:
        """Task analysis with retries and validation"""
        for attempt in range(self.retry_count):
            try:
                analysis = await self._analyze_task(user_input, context)
                if self._validate_analysis(analysis):
                    return analysis
                else:
                    self.logger.warning(f"Invalid analysis on attempt {attempt + 1}")
            except Exception as e:
                self.logger.warning(f"Task analysis failed on attempt {attempt + 1}: {e}")
                if attempt == self.retry_count - 1:
                    raise
        
        # If all attempts fail, use local model
        return await self.local_model.analyze_task(user_input, context)
    
    async def _direct_reasoning(self, user_input: str, context: Dict = None) -> Dict[str, Any]:
        """Direct reasoning call to Ollama with simplified processing"""
        # Create a simple prompt for reasoning
        prompt = f"Explain in simple terms: {user_input}. Respond in JSON format with understanding, approach, and confidence."
        
        try:
            result = await self.ollama_agent.execute({
                "prompt": prompt,
                "temperature": 0.7,
                "max_tokens": 500
            })
            
            if result["status"] == "success":
                response_text = result["result"]
                # Try to parse as JSON, if not, create a simple structure
                try:
                    parsed = json.loads(response_text)
                    parsed["provider"] = "ollama"
                    return parsed
                except json.JSONDecodeError:
                    # Create simple structure from text response
                    return {
                        "understanding": f"Task: {user_input}",
                        "approach": response_text[:200],
                        "considerations": ["Processing request", "Generating response"],
                        "confidence": 0.75,
                        "provider": "ollama"
                    }
            else:
                raise Exception(f"Ollama error: {result.get('error', 'Unknown error')}")
        except Exception as e:
            self.logger.warning(f"Direct reasoning failed: {e}, falling back to local")
            # Fallback to local model
            task_analysis = await self.local_model.analyze_task(user_input, context)
            return await self.local_model.generate_reasoning(user_input, task_analysis, context)
    
    async def _robust_reasoning(self, user_input: str, task_analysis: Dict, context: Dict = None) -> Dict[str, Any]:
        """Reasoning with Ollama as primary provider"""
        providers = []
        # Ollama is the primary and only cloud provider
        if self.ollama_available and self.ollama_agent:
            providers.append(("ollama", self._call_ollama))
        else:
            self.logger.warning("Ollama not available - using local model only")
        providers.append(("local", self.local_model.generate_reasoning))
        
        for provider_name, provider_func in providers:
            try:
                result = await provider_func(user_input, task_analysis, context)
                if self._validate_reasoning(result):
                    result["provider"] = provider_name
                    return result
                else:
                    self.logger.warning(f"Invalid reasoning from {provider_name}")
            except Exception as e:
                self.logger.warning(f"Provider {provider_name} failed: {e}")
                continue
        
        # All providers failed
        raise Exception("All reasoning providers failed")
    
    async def _analyze_task(self, user_input: str, context: Dict = None) -> Dict[str, Any]:
        """Analyze task with structured prompt - use local model to avoid recursion"""
        # Use local model to avoid potential recursion issues
        return await self.local_model.analyze_task(user_input, context)
    
    def _create_analysis_prompt(self, user_input: str, context: Dict = None) -> str:
        """Create structured analysis prompt"""
        return f"""
        Analyze this user task and provide a detailed classification in EXACT JSON format.
        
        Task: {user_input}
        Context: {context or 'No additional context'}
        
        Respond ONLY with valid JSON in this exact format:
        {{
            "task_type": "coding|research|analytical|creative|planning|other",
            "complexity": "simple|moderate|complex",
            "required_approach": "reasoning|tool_execution|learning",
            "estimated_steps": 1-5,
            "key_requirements": ["requirement1", "requirement2"],
            "confidence": 0.0-1.0
        }}
        
        Provide ONLY the JSON, no other text.
        """
    
    # OpenAI and Anthropic methods disabled - using Ollama only
    # async def _call_openai(self, user_input: str, task_analysis: Dict, context: Dict = None) -> Dict[str, Any]:
    #     """Call OpenAI with timeout and error handling"""
    #     pass
    # 
    # async def _call_anthropic(self, user_input: str, task_analysis: Dict, context: Dict = None) -> Dict[str, Any]:
    #     """Call Anthropic with timeout and error handling"""
    #     pass
    
    async def _call_ollama(self, user_input: str, task_analysis: Dict, context: Dict = None) -> Dict[str, Any]:
        """Call Ollama with timeout and error handling"""
        if not self.ollama_agent or not self.ollama_available:
            raise Exception("Ollama not available")
        
        prompt = self._create_reasoning_prompt(user_input, task_analysis, context)
        
        try:
            result = await self.ollama_agent.execute({
                "prompt": prompt,
                "temperature": 0.7,
                "max_tokens": 2000
            })
            
            if result["status"] == "success":
                # Parse JSON response from Ollama
                try:
                    response_text = result["result"]
                    self.logger.debug(f"Ollama response text: {response_text[:200]}...")  # Log first 200 chars
                    parsed_result = json.loads(response_text)
                    parsed_result["provider"] = "ollama"
                    return parsed_result
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Ollama returned non-JSON response: {e}")
                    # If not JSON, create structured response from the text
                    response_text = result["result"].strip()
                    if not response_text:
                        response_text = "No meaningful response generated"
                    
                    # Try to extract some structure from the response
                    if "task" in response_text.lower() or "analysis" in response_text.lower():
                        understanding = response_text[:200]
                        approach = response_text[200:500] if len(response_text) > 200 else response_text
                    else:
                        understanding = f"Task analysis: {task_analysis.get('task_type', 'general')}"
                        approach = response_text[:500]
                    
                    return {
                        "understanding": understanding,
                        "approach": approach,
                        "considerations": ["Local processing", "Resource optimization"],
                        "confidence": 0.75,
                        "provider": "ollama"
                    }
            else:
                raise Exception(f"Ollama error: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            self.logger.error(f"Ollama call failed: {e}")
            raise Exception(f"Ollama request failed: {str(e)}")
    
    def _create_reasoning_prompt(self, user_input: str, task_analysis: Dict, context: Dict = None) -> str:
        """Create structured reasoning prompt"""
        return f"""
        Provide detailed reasoning for this task:
        
        Original Task: {user_input}
        Task Analysis: {json.dumps(task_analysis, indent=2)}
        Context: {context or 'No additional context'}
        
        Respond ONLY with valid JSON in this exact format:
        {{
            "understanding": "Clear understanding of requirements",
            "approach": "Step-by-step approach with justification",
            "considerations": ["key consideration 1", "key consideration 2"],
            "potential_challenges": ["specific challenge 1", "specific challenge 2"],
            "success_criteria": ["measurable criterion 1", "measurable criterion 2"],
            "confidence": 0.85,
            "alternative_approaches": ["approach 1", "approach 2"]
        }}
        """
    
    async def _create_validated_plan(self, reasoning_result: Dict, context: Dict = None) -> List[Dict[str, Any]]:
        """Create plan with validation"""
        try:
            plan = await self._create_plan(reasoning_result, context)
            if self._validate_plan(plan):
                return plan
            else:
                # Generate fallback plan
                return await self._generate_fallback_plan(reasoning_result)
        except Exception as e:
            self.logger.error(f"Plan creation failed: {e}")
            return await self._generate_fallback_plan(reasoning_result)
    
    async def _create_plan(self, reasoning_result: Dict, context: Dict = None) -> List[Dict[str, Any]]:
        """Create execution plan - simplified to avoid complex validation"""
        # Return a simple plan based on reasoning result to avoid complex JSON validation
        return [
            {
                "step": 1,
                "action": reasoning_result.get("approach", "Process the user request"),
                "type": "reasoning",
                "expected_outcome": "Completed task",
                "success_criteria": "Satisfactory response",
                "required_resources": ["computation"]
            }
        ]
    
    async def _call_llm_with_validation(self, prompt: str, expected_format: str = "text") -> str:
        """Call LLM with validation - Ollama primary, local fallback"""
        # Try Ollama first
        if self.ollama_available and self.ollama_agent:
            try:
                result = await self.ollama_agent.execute({
                    "prompt": prompt,
                    "temperature": 0.7,
                    "max_tokens": 2000
                })
                if result["status"] == "success":
                    response = result["result"]
                    if expected_format == "json":
                        try:
                            # Validate that response is valid JSON
                            parsed = json.loads(response)
                            # Return the original response since it's valid JSON
                            return response
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Ollama returned non-JSON response: {e}")
                            # Try to fix common JSON formatting issues
                            fixed_response = self._fix_json_format(response)
                            if fixed_response:
                                try:
                                    json.loads(fixed_response)
                                    self.logger.info("Successfully fixed JSON format")
                                    return fixed_response
                                except json.JSONDecodeError:
                                    pass
                            
                            # Provide structured fallback based on request type
                            if "task analysis" in prompt.lower():
                                return '{"task_type": "other", "complexity": "simple", "required_approach": "reasoning"}'
                            elif "detailed reasoning" in prompt.lower():
                                return '{"understanding": "Basic understanding", "approach": "Direct approach", "confidence": 0.7}'
                            elif "execution plan" in prompt.lower():
                                return '[{"step": 1, "action": "Basic response", "type": "reasoning"}]'
                            else:
                                return '{"response": "Fallback response"}'
                    return response
            except Exception as e:
                self.logger.warning(f"Ollama failed: {e}")
        
        # Fallback to local model
        return await self.local_model.process_prompt(prompt, expected_format)
    
    # OpenAI and Anthropic simple call methods disabled
    # async def _call_openai_simple(self, prompt: str, format_type: str) -> str:
    #     pass
    # 
    # async def _call_anthropic_simple(self, prompt: str, format_type: str) -> str:
    #     pass
    
    async def _generate_fallback_response(self, user_input: str) -> str:
        """Generate basic response when all providers fail"""
        return await self.local_model.generate_fallback(user_input)
    
    async def _generate_fallback_plan(self, reasoning_result: Dict) -> List[Dict[str, Any]]:
        """Generate basic plan when LLM fails"""
        return await self.local_model.generate_fallback_plan(reasoning_result)
    
    def _validate_analysis(self, analysis: Dict) -> bool:
        """Validate task analysis structure"""
        required_fields = ["task_type", "complexity", "required_approach"]
        return all(field in analysis for field in required_fields)
    
    def _validate_reasoning(self, reasoning: Dict) -> bool:
        """Validate reasoning structure"""
        required_fields = ["understanding", "approach", "confidence"]
        return all(field in reasoning for field in required_fields)
    
    def _validate_plan(self, plan: List[Dict]) -> bool:
        """Validate plan structure"""
        if not isinstance(plan, list) or len(plan) == 0:
            return False
        
        required_fields = ["step", "action", "type"]
        return all(all(field in step for field in required_fields) for step in plan)
    
    def _fix_json_format(self, response: str) -> str:
        """Attempt to fix common JSON formatting issues with simple extraction"""
        try:
            import re
            import json
            
            # If it's already valid JSON, return as-is
            try:
                json.loads(response)
                return response
            except:
                pass
            
            # Extract key information using regex patterns
            understanding = re.search(r'["\']?understanding["\']?\s*[:\s]*["\']?([^"\'}\]]+)', response, re.IGNORECASE)
            approach = re.search(r'["\']?approach["\']?\s*[:\s]*["\']?([^"\'}\]]+)', response, re.IGNORECASE)
            confidence = re.search(r'["\']?confidence["\']?\s*[:\s]*([0-9.]+)', response, re.IGNORECASE)
            
            # Create simplified valid JSON
            result = {
                "understanding": understanding.group(1).strip() if understanding else "Basic understanding from response",
                "approach": approach.group(1).strip() if approach else "Extracted approach from response",
                "confidence": float(confidence.group(1)) if confidence else 0.7
            }
            
            return json.dumps(result, ensure_ascii=False)
            
        except Exception as e:
            self.logger.debug(f"JSON fixing failed: {e}")
            return None

class LocalReasoningModel:
    """Simplified local reasoning model for fallback"""
    
    def __init__(self):
        self.task_keywords = {
            "coding": ["code", "program", "function", "script", "python", "algorithm"],
            "research": ["research", "find", "search", "information", "learn", "study"],
            "analytical": ["analyze", "data", "calculate", "compare", "evaluate"],
            "creative": ["create", "write", "design", "imagine", "generate"],
            "planning": ["plan", "strategy", "schedule", "organize", "structure"]
        }
    
    async def analyze_task(self, user_input: str, context: Dict = None) -> Dict[str, Any]:
        """Local task analysis"""
        # Simple keyword-based analysis
        task_type = "other"
        for t_type, keywords in self.task_keywords.items():
            if any(keyword in user_input.lower() for keyword in keywords):
                task_type = t_type
                break
        
        # Basic complexity estimation
        word_count = len(user_input.split())
        if word_count < 10:
            complexity = "simple"
        elif word_count < 30:
            complexity = "moderate"
        else:
            complexity = "complex"
        
        return {
            "task_type": task_type,
            "complexity": complexity,
            "required_approach": "reasoning" if complexity == "simple" else "tool_execution",
            "estimated_steps": 2 if complexity == "simple" else 4,
            "key_requirements": ["understand task", "execute approach"],
            "confidence": 0.6
        }
    
    async def generate_reasoning(self, user_input: str, task_analysis: Dict, context: Dict = None) -> Dict[str, Any]:
        """Local reasoning generation"""
        return {
            "understanding": f"Task involves {task_analysis['task_type']} with {task_analysis['complexity']} complexity",
            "approach": "Break down the problem and address each component systematically",
            "considerations": ["Task requirements", "Available resources"],
            "confidence": 0.7,
            "provider": "local"
        }
    
    async def process_prompt(self, prompt: str, expected_format: str) -> str:
        """Process prompt locally"""
        if expected_format == "json":
            return '{"response": "Local model response", "confidence": 0.5}'
        else:
            return "Local model response to your query"
    
    async def generate_fallback(self, user_input: str) -> str:
        """Generate basic fallback response"""
        return f"I understand you're asking about: {user_input[:50]}... I'll help you with this task."
    
    async def generate_fallback_plan(self, reasoning_result: Dict) -> List[Dict[str, Any]]:
        """Generate basic fallback plan"""
        return [
            {
                "step": 1,
                "action": "Understand the requirements",
                "type": "reasoning",
                "expected_outcome": "Clear understanding of what's needed"
            },
            {
                "step": 2,
                "action": "Execute the main approach",
                "type": "tool",
                "expected_outcome": "Primary task completion"
            }
        ]