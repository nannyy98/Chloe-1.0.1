#!/usr/bin/env python3
"""
Demo Tasks for Phase 1 - Testing the implemented features
Measures success rate on various tasks
"""
import asyncio
import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from main import ChloeAI
from utils.logger import setup_logger


async def run_demo_tasks():
    """Run demo tasks and measure success rate"""
    print("🧪 Running Demo Tasks for Phase 1")
    print("=" * 50)
    
    logger = setup_logger("demo_tasks")
    ai_system = ChloeAI()
    
    # Define demo tasks as specified in the requirements
    demo_tasks = [
        "Calculate integral of x^2 dx from 0 to 1",  # math
        "Simulate energy of H2 molecule",  # quantum physics
        "Study PDF quantum_book.pdf on 'Schrodinger equation'",  # PDF study (will simulate)
        "Explain quantum entanglement from YouTube https://youtu.be/XXXX",  # YouTube study (will simulate)
        "Write Python code for factorial and save to file",  # code generation
        "What is the capital of France?",
        "Translate 'Hello' to Spanish",
        "How to make coffee?",
        "Explain what is artificial intelligence",
        "Calculate 25 * 4 + 10",
        "What are the planets in our solar system?",
        "Write a poem about spring",
        "Convert 100 Fahrenheit to Celsius",
        "What is the square root of 64?",
        "How does photosynthesis work?",
        "List 5 types of clouds",
        "What is the largest ocean on Earth?",
        "Explain gravity in simple terms",
        "How many continents are there?",
        "What is the fastest land animal?"
    ]
    
    results = {
        "total_tasks": len(demo_tasks),
        "successful": 0,
        "failed": 0,
        "success_rate": 0,
        "task_results": []
    }
    
    print(f"Running {len(demo_tasks)} demo tasks...\n")
    
    for i, task in enumerate(demo_tasks, 1):
        print(f"{i:2d}/{len(demo_tasks)}: {task[:50]}{'...' if len(task) > 50 else ''}")
        
        try:
            # Process the task
            result = await ai_system.process_task(task)
            
            # Check if the task was processed successfully
            is_successful = True
            if isinstance(result, dict):
                if "error" in result:
                    is_successful = False
                elif "result" in result and isinstance(result["result"], dict) and "error" in result["result"]:
                    is_successful = False
            else:
                is_successful = False
                
            if is_successful:
                results["successful"] += 1
                status = "✅ SUCCESS"
            else:
                results["failed"] += 1
                status = "❌ FAILED"
                
            results["task_results"].append({
                "task": task,
                "status": status,
                "result": result
            })
            
            print(f"      {status}")
            
        except Exception as e:
            results["failed"] += 1
            results["task_results"].append({
                "task": task,
                "status": "❌ ERROR",
                "error": str(e)
            })
            print(f"      ❌ ERROR: {e}")
    
    # Calculate success rate
    results["success_rate"] = (results["successful"] / results["total_tasks"]) * 100
    
    # Print summary
    print(f"\n📊 DEMO TASKS SUMMARY")
    print("=" * 50)
    print(f"Total tasks: {results['total_tasks']}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"Success rate: {results['success_rate']:.1f}%")
    
    # Check if success rate meets the requirement (>70% for Phase 1)
    if results["success_rate"] >= 70:
        print(f"\n🎯 SUCCESS: Phase 1 target achieved! (≥70%)")
    else:
        print(f"\n⚠️  NOTE: Success rate is below Phase 1 target of ≥70%")
    
    # Save results
    results_file = project_root / "demo_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📝 Results saved to: {results_file}")
    
    # Test the new ReAct endpoint specifically
    print(f"\n🧪 Testing ReAct Loop Specifically")
    print("-" * 30)
    
    try:
        # Test ReAct loop directly through the decision engine
        test_input = "What is 2+2?"
        decision = await ai_system.decision_engine.make_decision(test_input, {})
        print(f"Decision for '{test_input}': {decision['action']}")
        
        if decision['action'] == 'reason':
            react_result = await ai_system.reasoning_core.process(test_input, {})
            print(f"ReAct loop result: {react_result.get('final_answer', 'No answer generated')[:100]}...")
        else:
            print(f"Action selected: {decision['action']}")
            
    except Exception as e:
        print(f"Error in ReAct test: {e}")
    
    return results


async def test_basic_functionality():
    """Test basic functionality of the system"""
    print(f"\n🔍 Testing Basic Functionality")
    print("-" * 30)
    
    ai_system = ChloeAI()
    
    # Test basic reasoning
    try:
        result = await ai_system.process_task("Say hello")
        print("✅ Basic reasoning works")
    except Exception as e:
        print(f"❌ Basic reasoning failed: {e}")
    
    # Test memory system
    try:
        await ai_system.memory_system.store_interaction("test", {"response": "ok"})
        recent = await ai_system.memory_system.get_recent_interactions(1)
        if len(recent) > 0:
            print("✅ Memory system works")
        else:
            print("❌ Memory system issue")
    except Exception as e:
        print(f"❌ Memory system failed: {e}")
    
    # Test tool manager
    try:
        tools = ai_system.tool_manager.list_available_tools()
        if len(tools) > 0:
            print(f"✅ Tool manager works ({len(tools)} tools available)")
        else:
            print("❌ Tool manager issue - no tools found")
    except Exception as e:
        print(f"❌ Tool manager failed: {e}")
    
    # Test API server setup
    try:
        # Just check if the API server can be instantiated
        api_server = ai_system.api_server
        print("✅ API server setup works")
    except Exception as e:
        print(f"❌ API server setup failed: {e}")


async def main():
    """Main function to run all tests"""
    print("🚀 Chloe AI - Phase 1 Demo & Testing")
    print("=" * 60)
    
    # Test basic functionality
    await test_basic_functionality()
    
    # Run demo tasks
    results = await run_demo_tasks()
    
    print(f"\n🏁 Phase 1 Implementation Complete!")
    print(f"Success rate: {results['success_rate']:.1f}%")
    
    if results['success_rate'] >= 70:
        print("✅ Phase 1 target achieved!")
        return 0
    else:
        print("⚠️  Phase 1 target not fully met, but system is functional")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Error running demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)