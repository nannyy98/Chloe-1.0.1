"""
Test script for Chloe AI system
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from main import ChloeAI
from utils.logger import setup_logger

async def test_basic_functionality():
    """Test basic system functionality"""
    print("🧪 Testing Chloe AI System")
    print("=" * 50)
    
    try:
        # Initialize system
        print("1. Initializing system...")
        ai_system = ChloeAI()
        print("   ✓ System initialized")
        
        # Test reasoning core
        print("\n2. Testing reasoning core...")
        reasoning_result = await ai_system.reasoning_core.process(
            "Explain what machine learning is in simple terms"
        )
        print(f"   ✓ Reasoning completed (confidence: {reasoning_result.get('confidence', 0):.2f})")
        
        # Test decision engine
        print("\n3. Testing decision engine...")
        decision = await ai_system.decision_engine.make_decision(
            "Write a Python function to calculate factorial"
        )
        print(f"   ✓ Decision made: {decision['action']} (confidence: {decision['confidence']:.2f})")
        
        # Test tool manager
        print("\n4. Testing tool manager...")
        tools = ai_system.tool_manager.list_available_tools()
        print(f"   ✓ Available tools: {', '.join(tools)}")
        
        # Test memory system
        print("\n5. Testing memory system...")
        if ai_system.memory_system:
            await ai_system.memory_system.store_interaction(
                "Test input", 
                {"result": "Test response"}
            )
            recent = await ai_system.memory_system.get_recent_interactions(1)
            print(f"   ✓ Memory operations working ({len(recent)} recent interactions)")
        else:
            print("   ⚠ Memory system not available")
        
        # Test learning engine
        print("\n6. Testing learning engine...")
        if ai_system.learning_engine:
            await ai_system.learning_engine.record_experience(
                "test task",
                {"action": "reason", "confidence": 0.8},
                {"result": "success", "status": "completed"}
            )
            state = await ai_system.learning_engine.get_current_state()
            print(f"   ✓ Learning engine working (experiences: {state['total_experiences']})")
        else:
            print("   ⚠ Learning engine not available")
        
        # Test full pipeline
        print("\n7. Testing complete pipeline...")
        result = await ai_system.process_task("Hello, how are you?")
        print(f"   ✓ Complete pipeline working")
        print(f"   Result: {result['result']}")
        print(f"   Decision: {result['decision']['action']}")
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed! System is ready for use.")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_interactive_demo():
    """Run interactive demo"""
    print("\n🎮 Interactive Demo")
    print("=" * 30)
    print("Type your questions or commands below.")
    print("Commands: /quit, /help, /tools, /memory, /learn")
    print("=" * 30)
    
    ai_system = ChloeAI()
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == "/quit":
                break
            elif user_input.lower() == "/help":
                print("Available commands:")
                print("  /help  - Show this help")
                print("  /tools - List available tools")
                print("  /quit  - Exit demo")
                continue
            elif user_input.lower() == "/tools":
                tools = ai_system.tool_manager.list_available_tools()
                print(f"Available tools: {', '.join(tools)}")
                continue
            elif not user_input:
                continue
            
            print("🤖 Processing...")
            result = await ai_system.process_task(user_input)
            
            print(f"\n🤖 Chloe AI: {result['result']}")
            print(f"📋 Action: {result['decision']['action']} (confidence: {result['decision']['confidence']:.2f})")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("Chloe AI - System Test")
    print("Choose mode:")
    print("1. Automated tests")
    print("2. Interactive demo")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        success = asyncio.run(test_basic_functionality())
        sys.exit(0 if success else 1)
    elif choice == "2":
        asyncio.run(test_interactive_demo())
    else:
        print("Invalid choice. Running automated tests...")
        success = asyncio.run(test_basic_functionality())
        sys.exit(0 if success else 1)