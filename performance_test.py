#!/usr/bin/env python3
"""
Performance test for Chloe AI system
"""
import asyncio
import time
from agents.ollama_agent import OllamaAgent
from utils.config import Config

async def performance_test():
    print("🚀 Performance Test for Chloe AI")
    print("=" * 50)
    
    config = Config()
    agent = OllamaAgent(config)
    
    if not agent.is_available():
        print("❌ Ollama server not available")
        return
    
    print("✅ Ollama server is running")
    print(f"🧠 Model: {agent.model}")
    print()
    
    # Test 1: Simple response time
    print("Test 1: Simple response time")
    start_time = time.time()
    result = await agent.execute({
        "prompt": "Hello",
        "max_tokens": 50
    })
    response_time = time.time() - start_time
    
    if result.get("status") == "success":
        print(f"   ✅ Response time: {response_time:.2f} seconds")
        print(f"   📝 Response: {result['result'][:50]}...")
    else:
        print(f"   ❌ Error: {result.get('error')}")
    
    print()
    
    # Test 2: Complex reasoning
    print("Test 2: Complex reasoning")
    start_time = time.time()
    result = await agent.execute({
        "prompt": "Explain what machine learning is in simple terms",
        "max_tokens": 200
    })
    response_time = time.time() - start_time
    
    if result.get("status") == "success":
        print(f"   ✅ Response time: {response_time:.2f} seconds")
        print(f"   📝 Response length: {len(result['result'])} characters")
    else:
        print(f"   ❌ Error: {result.get('error')}")
    
    print()
    
    # Test 3: Multiple requests
    print("Test 3: Multiple requests (3 requests)")
    start_time = time.time()
    
    tasks = []
    for i in range(3):
        task = agent.execute({
            "prompt": f"Answer briefly: What is {i+1}+{i+1}?",
            "max_tokens": 30
        })
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    total_time = time.time() - start_time
    
    successful = sum(1 for r in results if r.get("status") == "success")
    print(f"   ✅ {successful}/3 requests successful")
    print(f"   ⏱️  Total time: {total_time:.2f} seconds")
    print(f"   📊 Average time per request: {total_time/3:.2f} seconds")
    
    print()
    print("🏁 Performance test completed!")

if __name__ == "__main__":
    asyncio.run(performance_test())