"""
Ollama Agent - Local LLM integration for Chloe AI
"""

import requests
import json
import asyncio
from typing import Dict, Any
from utils.config import Config
from utils.logger import setup_logger

class OllamaAgent:
    """Local LLM agent using Ollama with fallback capabilities"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger("ollama_agent")
        
        # Initialize the new LLM Manager for multi-provider support
        from agents.llm_agent import LLMManager
        self.llm_manager = LLMManager(config)
        
        # Keep legacy attributes for backward compatibility
        self.ollama_url = config.get("ollama.url", "http://localhost:11434/api/generate")
        self.model = config.get("ollama.model", "phi")  # Changed to phi as default
        self.timeout = config.get("ollama.timeout", 60)
        
        # Create a new session for this agent
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute LLM request using the new LLM manager with fallback"""
        return await self.llm_manager.execute(params)
    
    async def execute_ollama_only(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute using Ollama only (for backward compatibility)"""
        try:
            prompt = params.get("prompt", "")
            if not prompt:
                return {"error": "No prompt provided", "status": "error"}
            
            # Prepare request data
            request_data = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": params.get("temperature", 0.7),
                    "num_predict": params.get("max_tokens", 2000)
                }
            }
            
            # Make async request with session
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.session.post(
                    self.ollama_url,
                    json=request_data,
                    timeout=self.timeout
                )
            )
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    if not result or "response" not in result:
                        raise ValueError("Invalid response format from Ollama")
                    
                    self.logger.info(f"Ollama response generated successfully")
                    return {
                        "result": result.get("response", ""),
                        "model": self.model,
                        "prompt_tokens": result.get("prompt_eval_count", 0),
                        "completion_tokens": result.get("eval_count", 0),
                        "status": "success"
                    }
                except ValueError as e:
                    self.logger.error(f"Failed to parse Ollama response: {e}")
                    return {"error": f"Invalid response format: {e}", "status": "error"}
                except Exception as e:
                    self.logger.error(f"JSON parsing error: {e}")
                    return {"error": f"JSON parsing failed: {e}", "status": "error"}
            else:
                error_msg = f"Ollama API error: {response.status_code}"
                self.logger.error(error_msg)
                return {"error": error_msg, "status": "error"}
                
        except requests.exceptions.ConnectionError:
            error_msg = "Cannot connect to Ollama server. Make sure it's running: ollama serve"
            self.logger.error(error_msg)
            return {"error": error_msg, "status": "error"}
        except requests.exceptions.Timeout:
            error_msg = f"Ollama request timeout after {self.timeout} seconds"
            self.logger.error(error_msg)
            return {"error": error_msg, "status": "error"}
        except Exception as e:
            error_msg = f"Ollama execution failed: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg, "status": "error"}
    
    async def list_models(self) -> Dict[str, Any]:
        """List available models"""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.session.get("http://localhost:11434/api/tags", timeout=10)
            )
            
            if response.status_code == 200:
                models = response.json().get("models", [])
                return {
                    "models": [model["name"] for model in models],
                    "status": "success"
                }
            else:
                return {"error": f"Failed to list models: {response.status_code}", "status": "error"}
                
        except Exception as e:
            return {"error": f"Cannot list models: {str(e)}", "status": "error"}
    
    def is_available(self) -> bool:
        """Check if Ollama server is available"""
        try:
            response = self.session.get("http://localhost:11434/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            self.logger.debug(f"Ollama availability check failed: {e}")
            return False

# Test function
if __name__ == "__main__":
    config = Config()
    agent = OllamaAgent(config)
    
    if agent.is_available():
        print("✅ Ollama server is available")
        # Test basic prompt
        test_params = {"prompt": "Hello, how are you?"}
        result = asyncio.run(agent.execute(test_params))
        print(f"Test result: {result}")
    else:
        print("❌ Ollama server not available. Run: ollama serve")