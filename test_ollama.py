#!/usr/bin/env python3
"""
Ollama Integration Test Script
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.config import Config
from utils.logger import setup_logger
from agents.ollama_agent import OllamaAgent

async def test_ollama_integration():
    """Test Ollama integration"""
    print("🧪 Testing Ollama Integration")
    print("=" * 40)
    
    config = Config()
    logger = setup_logger("ollama_test")
    
    # Initialize Ollama agent
    try:
        ollama_agent = OllamaAgent(config)
        print("✅ Ollama Agent initialized")
    except Exception as e:
        print(f"❌ Failed to initialize Ollama agent: {e}")
        return False
    
    # Check if Ollama server is available
    if ollama_agent.is_available():
        print("✅ Ollama server is running")
    else:
        print("❌ Ollama server not available")
        print("   Please run: ollama serve")
        print("   And install a model: ollama pull llama2")
        return False
    
    # List available models
    print("\n📋 Available models:")
    models_result = await ollama_agent.list_models()
    if models_result["status"] == "success":
        for model in models_result["models"]:
            print(f"   • {model}")
    else:
        print(f"   Error listing models: {models_result.get('error', 'Unknown')}")
    
    # Test basic reasoning
    print("\n🧠 Testing reasoning capabilities:")
    test_prompts = [
        "Explain what machine learning is in simple terms",
        "Write a Python function to calculate factorial",
        "What are the benefits of modular architecture?"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}: {prompt}")
        try:
            result = await ollama_agent.execute({
                "prompt": prompt,
                "temperature": 0.7,
                "max_tokens": 500
            })
            
            if result["status"] == "success":
                print(f"   ✅ Success! Response length: {len(result['result'])} characters")
                print(f"   First 200 chars: {result['result'][:200]}...")
            else:
                print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    # Test integration with reasoning core
    print("\n🔄 Testing integration with reasoning core:")
    try:
        from core.enhanced_reasoning_core import EnhancedReasoningCore
        reasoning_core = EnhancedReasoningCore(config)
        
        test_task = "Explain neural networks"
        print(f"Task: {test_task}")
        
        result = await reasoning_core.process(test_task)
        print(f"Provider used: {result.get('provider_used', 'unknown')}")
        print(f"Confidence: {result.get('confidence', 0):.2f}")
        print(f"Understanding: {result.get('reasoning', {}).get('understanding', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Reasoning core integration failed: {e}")
    
    print("\n" + "=" * 40)
    print("🎉 Ollama integration test completed!")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_ollama_integration())
    sys.exit(0 if success else 1)