"""
Configuration management for Chloe AI system using Pydantic BaseSettings
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic_settings import BaseSettings
from pydantic import Field

class Config(BaseSettings):
    """Configuration manager for the AI system using Pydantic BaseSettings"""
    
    # Model settings
    llm_model: str = Field(default="qwen2.5:7b", env="LLM_MODEL")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    
    # Memory settings
    memory_path: str = Field(default="./memory/chroma", env="MEMORY_PATH")
    
    # API settings
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    
    # Other settings
    config_path: str = Field(default="config/config.json", env="CONFIG_PATH")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        # Allow extra fields from the JSON config file
        extra = "allow"
    
    def __init__(self, config_path: str = "config/config.json"):
        super().__init__()
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get specific model configuration"""
        return self.get(f"models.{model_type}", {})
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration"""
        return self.get("api", {})
    
    def get_memory_config(self) -> Dict[str, Any]:
        """Get memory system configuration"""
        return self.get("memory", {})