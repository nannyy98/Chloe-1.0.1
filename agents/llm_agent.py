"""
Multi-provider LLM Agent for Chloe AI
Supports Ollama, OpenRouter, Groq and other providers with fallback capabilities
"""

import asyncio
import requests
import json
from typing import Dict, Any, List, Optional
from collections import deque
import time
from utils.config import Config
from utils.logger import setup_logger

class LLMManager:
    """Manages multiple LLM providers with fallback capabilities"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger("llm_manager")
        self.providers = {}
        self.primary_provider = config.get("llm_providers.primary", "ollama")
        self.fallback_order = config.get("llm_providers.fallback_order", ["ollama"])
        
        # Initialize available providers
        self._initialize_providers()
        
        # Track provider performance for intelligent routing
        self.provider_stats = {
            provider: {
                'success_count': 0,
                'failure_count': 0,
                'avg_response_time': 0,
                'last_used': None
            }
            for provider in self.providers.keys()
        }
        
        self.logger.info(f"LLM Manager initialized with providers: {list(self.providers.keys())}")
    
    def _initialize_providers(self):
        """Initialize all configured LLM providers"""
        provider_configs = self.config.get("llm_providers.providers", {})
        
        for provider_name, provider_config in provider_configs.items():
            if provider_config.get("enabled", False):
                if provider_name == "ollama":
                    self.providers[provider_name] = OllamaProvider(provider_config)
                elif provider_name == "openrouter":
                    self.providers[provider_name] = OpenRouterProvider(provider_config)
                elif provider_name == "groq":
                    self.providers[provider_name] = GroqProvider(provider_config)
                else:
                    self.logger.warning(f"Unknown provider: {provider_name}")
    
    async def execute(self, params: Dict[str, Any], preferred_provider: str = None) -> Dict[str, Any]:
        """Execute LLM request with fallback mechanism"""
        start_time = time.time()
        
        # Determine provider order
        if preferred_provider and preferred_provider in self.providers:
            provider_order = [preferred_provider] + [
                p for p in self.fallback_order if p != preferred_provider and p in self.providers
            ]
        else:
            provider_order = [p for p in self.fallback_order if p in self.providers]
        
        # Add primary provider first if not already in the list
        if self.primary_provider not in provider_order and self.primary_provider in self.providers:
            provider_order.insert(0, self.primary_provider)
        
        # Try each provider in order
        for provider_name in provider_order:
            provider = self.providers[provider_name]
            
            try:
                self.logger.info(f"Attempting to execute with {provider_name} provider")
                result = await provider.execute(params)
                
                if result["status"] == "success":
                    # Update provider stats
                    response_time = time.time() - start_time
                    self._update_provider_stats(provider_name, success=True, response_time=response_time)
                    
                    # Add provider info to result
                    result["provider_used"] = provider_name
                    return result
                
            except Exception as e:
                self.logger.warning(f"Provider {provider_name} failed: {e}")
                self._update_provider_stats(provider_name, success=False)
                continue
        
        # All providers failed
        return {
            "error": "All LLM providers failed",
            "status": "error",
            "provider_used": "none"
        }
    
    def _update_provider_stats(self, provider_name: str, success: bool, response_time: float = None):
        """Update provider performance statistics"""
        if provider_name in self.provider_stats:
            stats = self.provider_stats[provider_name]
            if success:
                stats['success_count'] += 1
                if response_time is not None:
                    # Update average response time with exponential moving average
                    if stats['avg_response_time'] == 0:
                        stats['avg_response_time'] = response_time
                    else:
                        # Use smoothing factor of 0.3
                        stats['avg_response_time'] = 0.7 * stats['avg_response_time'] + 0.3 * response_time
            else:
                stats['failure_count'] += 1
            
            stats['last_used'] = time.time()
    
    def get_best_provider(self) -> str:
        """Return the currently best performing provider based on stats"""
        best_provider = self.primary_provider
        best_score = float('-inf')
        
        for provider_name, stats in self.provider_stats.items():
            if provider_name not in self.providers:
                continue
                
            total_requests = stats['success_count'] + stats['failure_count']
            if total_requests == 0:
                continue
                
            success_rate = stats['success_count'] / total_requests if total_requests > 0 else 0
            # Score combines success rate and response time (lower is better)
            score = success_rate - (stats['avg_response_time'] / 10 if stats['avg_response_time'] > 0 else 0)
            
            if score > best_score:
                best_score = score
                best_provider = provider_name
        
        return best_provider


class BaseProvider:
    """Base class for LLM providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get("name", "base")
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class OllamaProvider(BaseProvider):
    """Ollama LLM provider implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.url = config.get("url", "http://localhost:11434/api/generate")
        self.model = config.get("model", "phi")
        self.timeout = config.get("timeout", 60)
        
        # Create session for connection reuse
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute request to Ollama"""
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
                    "num_predict": params.get("max_tokens", 2000),
                    "top_p": params.get("top_p", 0.9)
                }
            }
            
            # Make async request
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.session.post(
                    self.url,
                    json=request_data,
                    timeout=self.timeout
                )
            )
            
            if response.status_code == 200:
                result = response.json()
                if result and "response" in result:
                    return {
                        "result": result["response"],
                        "model": self.model,
                        "prompt_tokens": result.get("prompt_eval_count", 0),
                        "completion_tokens": result.get("eval_count", 0),
                        "status": "success"
                    }
                else:
                    return {"error": "Invalid response format from Ollama", "status": "error"}
            else:
                return {"error": f"Ollama API error: {response.status_code}", "status": "error"}
                
        except requests.exceptions.RequestException as e:
            return {"error": f"Ollama connection error: {str(e)}", "status": "error"}
        except Exception as e:
            return {"error": f"Ollama execution failed: {str(e)}", "status": "error"}


class OpenRouterProvider(BaseProvider):
    """OpenRouter LLM provider implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key", "")
        self.model = config.get("model", "meta-llama/llama-3.1-70b-instruct")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.timeout = config.get("timeout", 30)
        
        if not self.api_key:
            self.enabled = False
            return
        self.enabled = True
        
        # Create session for connection reuse
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        })
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute request to OpenRouter"""
        if not self.enabled:
            return {"error": "OpenRouter API key not configured", "status": "error"}
        
        try:
            prompt = params.get("prompt", "")
            if not prompt:
                return {"error": "No prompt provided", "status": "error"}
            
            # Prepare request data
            request_data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": params.get("temperature", 0.7),
                "max_tokens": params.get("max_tokens", 2000),
                "top_p": params.get("top_p", 0.9)
            }
            
            # Make async request
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.session.post(
                    self.base_url,
                    json=request_data,
                    timeout=self.timeout
                )
            )
            
            if response.status_code == 200:
                result = response.json()
                if result and "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    return {
                        "result": content,
                        "model": self.model,
                        "status": "success"
                    }
                else:
                    return {"error": "Invalid response format from OpenRouter", "status": "error"}
            elif response.status_code == 429:
                return {"error": "OpenRouter rate limit exceeded", "status": "error"}
            else:
                error_detail = response.json().get("error", {}).get("message", "Unknown error")
                return {"error": f"OpenRouter API error {response.status_code}: {error_detail}", "status": "error"}
                
        except requests.exceptions.RequestException as e:
            return {"error": f"OpenRouter connection error: {str(e)}", "status": "error"}
        except Exception as e:
            return {"error": f"OpenRouter execution failed: {str(e)}", "status": "error"}


class GroqProvider(BaseProvider):
    """Groq LLM provider implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key", "")
        self.model = config.get("model", "llama3-70b-8192")
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.timeout = config.get("timeout", 15)
        
        if not self.api_key:
            self.enabled = False
            return
        self.enabled = True
        
        # Create session for connection reuse
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        })
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute request to Groq"""
        if not self.enabled:
            return {"error": "Groq API key not configured", "status": "error"}
        
        try:
            prompt = params.get("prompt", "")
            if not prompt:
                return {"error": "No prompt provided", "status": "error"}
            
            # Prepare request data
            request_data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": params.get("temperature", 0.7),
                "max_tokens": params.get("max_tokens", 2000),
                "top_p": params.get("top_p", 0.9)
            }
            
            # Make async request
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.session.post(
                    self.base_url,
                    json=request_data,
                    timeout=self.timeout
                )
            )
            
            if response.status_code == 200:
                result = response.json()
                if result and "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    return {
                        "result": content,
                        "model": self.model,
                        "status": "success"
                    }
                else:
                    return {"error": "Invalid response format from Groq", "status": "error"}
            elif response.status_code == 429:
                return {"error": "Groq rate limit exceeded", "status": "error"}
            else:
                error_detail = response.json().get("error", {}).get("message", "Unknown error")
                return {"error": f"Groq API error {response.status_code}: {error_detail}", "status": "error"}
                
        except requests.exceptions.RequestException as e:
            return {"error": f"Groq connection error: {str(e)}", "status": "error"}
        except Exception as e:
            return {"error": f"Groq execution failed: {str(e)}", "status": "error"}


# For backward compatibility with existing OllamaAgent
class OllamaAgent:
    """Wrapper for backward compatibility"""
    
    def __init__(self, config: Config):
        # Create LLMManager instead
        self.llm_manager = LLMManager(config)
    
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute using the LLM manager"""
        return await self.llm_manager.execute(params, preferred_provider="ollama")
    
    def is_available(self) -> bool:
        """Check if Ollama is available"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False