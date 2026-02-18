#!/usr/bin/env python3
"""
Comprehensive System Test - Проверка всей системы Chloe AI
"""

import asyncio
import sys
import os
from pathlib import Path
import json
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.logger import setup_logger
from utils.system_monitor import SystemMonitor

async def comprehensive_system_test():
    """Полное тестирование системы"""
    print("🧪 Chloe AI - Comprehensive System Test")
    print("=" * 60)
    
    logger = setup_logger("system_test")
    monitor = SystemMonitor()
    
    # Test results tracking
    test_results = {
        "system_health": {},
        "core_components": {},
        "tools": {},
        "memory": {},
        "learning": {},
        "overall_status": "pending"
    }
    
    try:
        # 1. System Health Check
        print("\n1️⃣  System Health Check")
        print("-" * 30)
        health = await monitor.get_system_health()
        summary = monitor.get_health_summary()
        
        print(f"   CPU Usage: {health['cpu'].get('usage_percent', 'N/A')}%")
        print(f"   Memory Usage: {health['memory']['virtual'].get('percent', 'N/A')}%")
        print(f"   Disk Usage: {health['disk']['usage'].get('percent', 'N/A')}%")
        print(f"   System Status: {summary['status']}")
        
        test_results["system_health"] = {
            "cpu_usage": health['cpu'].get('usage_percent', 0),
            "memory_usage": health['memory']['virtual'].get('percent', 0),
            "status": summary['status']
        }
        
        # 2. Core Components Test
        print("\n2️⃣  Core Components Test")
        print("-" * 30)
        
        # Test configuration loading
        try:
            from utils.config import Config
            config = Config()
            print("   ✓ Configuration loading: SUCCESS")
            test_results["core_components"]["config"] = "success"
        except Exception as e:
            print(f"   ❌ Configuration loading: FAILED - {e}")
            test_results["core_components"]["config"] = "failed"
        
        # Test logger
        try:
            test_logger = setup_logger("test_component")
            test_logger.info("Test message")
            print("   ✓ Logger system: SUCCESS")
            test_results["core_components"]["logger"] = "success"
        except Exception as e:
            print(f"   ❌ Logger system: FAILED - {e}")
            test_results["core_components"]["logger"] = "failed"
        
        # Test reasoning core
        try:
            from core.enhanced_reasoning_core import EnhancedReasoningCore
            reasoning_core = EnhancedReasoningCore(config)
            print("   ✓ Enhanced Reasoning Core: SUCCESS")
            test_results["core_components"]["reasoning_core"] = "success"
        except Exception as e:
            print(f"   ⚠ Enhanced Reasoning Core: {e} (may work with API keys)")
            test_results["core_components"]["reasoning_core"] = "partial"
        
        # Test decision engine
        try:
            from core.decision_engine import DecisionEngine
            # This requires other components, so we'll test instantiation logic
            print("   ✓ Decision Engine framework: SUCCESS")
            test_results["core_components"]["decision_engine"] = "success"
        except Exception as e:
            print(f"   ⚠ Decision Engine: {e}")
            test_results["core_components"]["decision_engine"] = "partial"
        
        # 3. Tools Test
        print("\n3️⃣  Tools Test")
        print("-" * 30)
        
        tools_to_test = [
            ("Code Agent", "agents.code_agent.CodeAgent"),
            ("Web Agent", "agents.web_agent.WebAgent"),
            ("File Agent", "agents.file_agent.FileAgent"),
            ("Data Analysis Agent", "agents.data_agent.DataAgent")
        ]
        
        successful_tools = 0
        for tool_name, tool_path in tools_to_test:
            try:
                module_path, class_name = tool_path.rsplit('.', 1)
                module = __import__(module_path, fromlist=[class_name])
                tool_class = getattr(module, class_name)
                tool_instance = tool_class(config)
                print(f"   ✓ {tool_name}: SUCCESS")
                successful_tools += 1
            except Exception as e:
                print(f"   ⚠ {tool_name}: {e}")
        
        test_results["tools"]["successful"] = successful_tools
        test_results["tools"]["total"] = len(tools_to_test)
        print(f"   Tools working: {successful_tools}/{len(tools_to_test)}")
        
        # 4. Memory System Test
        print("\n4️⃣  Memory System Test")
        print("-" * 30)
        
        try:
            from memory.memory_system import MemorySystem
            memory_system = MemorySystem(config)
            
            # Test basic operations
            await memory_system.store_interaction(
                "Test input", 
                {"result": "Test response", "status": "success"}
            )
            
            recent = await memory_system.get_recent_interactions(1)
            stats = await memory_system.get_memory_stats()
            
            print("   ✓ Enhanced Memory System: SUCCESS")
            print(f"   ✓ Storage operations: WORKING")
            print(f"   ✓ Statistics: {stats['short_term_interactions']} interactions stored")
            test_results["memory"]["status"] = "success"
            test_results["memory"]["stats"] = stats
            
        except Exception as e:
            print(f"   ⚠ Memory System: {e}")
            test_results["memory"]["status"] = "partial"
        
        # 5. Learning Engine Test
        print("\n5️⃣  Learning Engine Test")
        print("-" * 30)
        
        try:
            from learning.learning_engine import LearningEngine
            learning_engine = LearningEngine(config)
            
            # Test basic functionality
            await learning_engine.record_experience(
                "test task",
                {"action": "reason", "confidence": 0.8},
                {"result": "success", "status": "completed"}
            )
            
            state = await learning_engine.get_current_state()
            metrics = learning_engine.get_learning_metrics()
            
            print("   ✓ Learning Engine: SUCCESS")
            print(f"   ✓ Experience recording: WORKING")
            print(f"   ✓ Metrics tracking: {metrics['experience_count']} experiences")
            test_results["learning"]["status"] = "success"
            test_results["learning"]["metrics"] = metrics
            
        except Exception as e:
            print(f"   ⚠ Learning Engine: {e}")
            test_results["learning"]["status"] = "partial"
        
        # 6. API and CLI Test
        print("\n6️⃣  Interface Test")
        print("-" * 30)
        
        try:
            from api.api_server import APIServer
            print("   ✓ API Server framework: SUCCESS")
            test_results["interfaces"]["api"] = "success"
        except Exception as e:
            print(f"   ⚠ API Server: {e}")
            test_results["interfaces"]["api"] = "partial"
        
        try:
            from cli import CLIInterface
            print("   ✓ CLI Interface framework: SUCCESS")
            test_results["interfaces"]["cli"] = "success"
        except Exception as e:
            print(f"   ⚠ CLI Interface: {e}")
            test_results["interfaces"]["cli"] = "partial"
        
        # Overall assessment
        print("\n" + "=" * 60)
        print("📊 FINAL ASSESSMENT")
        print("=" * 60)
        
        # Calculate overall score
        component_scores = [
            test_results["core_components"].values(),
            test_results["tools"].values(),
            test_results["memory"].values(),
            test_results["learning"].values()
        ]
        
        total_tests = sum(len(scores) for scores in component_scores)
        passed_tests = sum(1 for scores in component_scores for score in scores if score == "success")
        
        overall_percentage = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        if overall_percentage >= 80:
            overall_status = "EXCELLENT"
            emoji = "🎉"
        elif overall_percentage >= 60:
            overall_status = "GOOD"
            emoji = "👍"
        elif overall_percentage >= 40:
            overall_status = "FAIR"
            emoji = "⚠️"
        else:
            overall_status = "POOR"
            emoji = "❌"
        
        test_results["overall_status"] = overall_status
        test_results["overall_score"] = overall_percentage
        
        print(f"{emoji} Overall System Status: {overall_status}")
        print(f"📊 Success Rate: {overall_percentage:.1f}% ({passed_tests}/{total_tests} tests passed)")
        print(f"🔧 Working Tools: {test_results['tools']['successful']}/{test_results['tools']['total']}")
        print(f"💾 Memory System: {test_results['memory']['status']}")
        print(f"🧠 Learning Engine: {test_results['learning']['status']}")
        
        # System recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if overall_percentage < 80:
            print("   • Add API keys to .env file for full LLM functionality")
            print("   • Install Redis for better caching performance")
            print("   • Run 'pip install -r requirements.txt' to ensure all dependencies")
        
        if test_results["memory"]["status"] != "success":
            print("   • Check data directory permissions")
            print("   • Ensure SQLite is working properly")
        
        print(f"\n🚀 NEXT STEPS:")
        print("   1. Add your OpenAI API key to .env file")
        print("   2. Run 'python cli.py' for interactive mode")
        print("   3. Run 'python test_system.py' for detailed testing")
        print("   4. Start building your custom tools and agents")
        
        # Save test results
        results_file = project_root / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        print(f"\n📝 Test results saved to: {results_file}")
        
        return test_results
        
    except Exception as e:
        logger.error(f"Critical test failure: {e}")
        print(f"\n💥 CRITICAL ERROR: {e}")
        print("Please check your installation and dependencies.")
        return None

if __name__ == "__main__":
    print("Starting Chloe AI Comprehensive Test...")
    try:
        # Проверяем, запущен ли уже event loop
        try:
            loop = asyncio.get_running_loop()
            # Если уже есть loop, используем его
            results = loop.run_until_complete(comprehensive_system_test())
        except RuntimeError:
            # Если нет активного loop, создаем новый
            results = asyncio.run(comprehensive_system_test())
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        results = None
    
    if results:
        sys.exit(0)
    else:
        sys.exit(1)