"""
CLI Interface for Chloe AI
"""

import asyncio
import json
import sys
from typing import Dict, Any
import argparse

from main import ChloeAI
from utils.logger import setup_logger

class CLIInterface:
    """Command-line interface for Chloe AI"""
    
    def __init__(self):
        self.ai_system = None
        self.logger = setup_logger("cli_interface")
        self.session_id = "cli_session"
        
    async def initialize_system(self):
        """Initialize the AI system"""
        print("Initializing Chloe AI system...")
        self.ai_system = ChloeAI()
        print("✓ System initialized successfully")
        
    async def interactive_mode(self):
        """Run interactive CLI mode"""
        if not self.ai_system:
            await self.initialize_system()
            
        print("\n" + "="*50)
        print("🤖 Chloe AI - Interactive Mode")
        print("="*50)
        print("Commands:")
        print("  /help     - Show this help")
        print("  /tools    - List available tools") 
        print("  /memory   - Show memory stats")
        print("  /learn    - Show learning insights")
        print("  /quit     - Exit the program")
        print("  /clear    - Clear conversation history")
        print("="*50)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() == "/quit":
                    print("Goodbye! 👋")
                    break
                    
                elif user_input.lower() == "/help":
                    self._show_help()
                    continue
                    
                elif user_input.lower() == "/tools":
                    await self._show_tools()
                    continue
                    
                elif user_input.lower() == "/memory":
                    await self._show_memory_stats()
                    continue
                    
                elif user_input.lower() == "/learn":
                    await self._show_learning_insights()
                    continue
                    
                elif user_input.lower() == "/clear":
                    await self._clear_history()
                    continue
                
                # Process regular input
                print("🤖 Thinking...")
                try:
                    result = await self.ai_system.process_task(
                        user_input, 
                        {"session_id": self.session_id}
                    )
                except Exception as e:
                    print(f"❌ Error processing task: {e}")
                    self.logger.error(f"Task processing error: {e}")
                    continue
                
                self._display_result(result)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                self.logger.error(f"CLI error: {e}")
    
    def _show_help(self):
        """Show help information"""
        print("\n📚 Available Commands:")
        print("  /help     - Show this help message")
        print("  /tools    - List all available tools and agents")
        print("  /memory   - Display memory system statistics")
        print("  /learn    - Show learning insights and performance metrics")
        print("  /clear    - Clear conversation history")
        print("  /quit     - Exit the program")
        print("\n💡 Examples:")
        print("  'Write a Python function to calculate factorial'")
        print("  'Search for information about machine learning'")
        print("  'Analyze this code for potential bugs'")
    
    async def _show_tools(self):
        """Show available tools"""
        if not self.ai_system:
            print("❌ System not initialized")
            return
            
        tools = self.ai_system.tool_manager.list_available_tools()
        print(f"\n🛠️  Available Tools ({len(tools)}):")
        for tool in tools:
            print(f"  • {tool}")
        
        print("\n🔧 Tool Descriptions:")
        print("  code_runner    - Execute and debug Python code")
        print("  web_agent      - Search web and scrape content") 
        print("  file_agent     - Read, write, and manage files")
        print("  data_analysis  - Analyze data and generate insights")
    
    async def _show_memory_stats(self):
        """Show memory system statistics"""
        if not self.ai_system or not self.ai_system.memory_system:
            print("❌ Memory system not available")
            return
            
        try:
            stats = await self.ai_system.memory_system.get_memory_stats()
            print("\n🧠 Memory System Statistics:")
            print(f"  Short-term interactions: {stats['short_term_interactions']}")
            print(f"  Long-term knowledge: {stats['long_term_knowledge']}")
            print(f"  Learning experiences: {stats['learning_experiences']}")
            
            if 'collections' in stats:
                print(f"  Knowledge vectors: {stats['collections']['knowledge']}")
                print(f"  Experience vectors: {stats['collections']['experiences']}")
                
        except Exception as e:
            print(f"❌ Error getting memory stats: {e}")
    
    async def _show_learning_insights(self):
        """Show learning engine insights"""
        if not self.ai_system or not self.ai_system.learning_engine:
            print("❌ Learning engine not available")
            return
            
        try:
            state = await self.ai_system.learning_engine.get_current_state()
            metrics = self.ai_system.learning_engine.get_learning_metrics()
            
            print("\n📈 Learning Engine Insights:")
            print(f"  Recent success rate: {state['recent_success_rate']:.2%}")
            print(f"  Total experiences: {metrics['experience_count']}")
            print(f"  Performance trend: {state['performance_trend']}")
            
            if state['strategy_performance']:
                print("\n  Strategy Performance:")
                for strategy, perf in state['strategy_performance'].items():
                    print(f"    {strategy}: {perf['recent_success_rate']:.2%} ({perf['total_attempts']} attempts)")
            
            if state['error_patterns']:
                print("\n  Common Error Patterns:")
                for error_type, count in list(state['error_patterns'].items())[:3]:
                    print(f"    {error_type}: {count} occurrences")
                    
        except Exception as e:
            print(f"❌ Error getting learning insights: {e}")
    
    async def _clear_history(self):
        """Clear conversation history"""
        if self.ai_system and self.ai_system.memory_system:
            try:
                await self.ai_system.memory_system.clear_memory("short_term")
                print("✓ Conversation history cleared")
            except Exception as e:
                print(f"❌ Error clearing history: {e}")
        else:
            print("✓ History cleared (no memory system)")
    
    def _display_result(self, result: Dict[str, Any]):
        """Display AI response in a formatted way"""
        print(f"\n🤖 Chloe AI:")
        
        # Display main result content
        if "result" in result:
            result_content = result["result"]
            
            # Handle different result structures
            if isinstance(result_content, dict):
                # Try to extract the actual response content
                if "result" in result_content:
                    content = result_content["result"]
                    if isinstance(content, str):
                        print(f"   {content}")
                    else:
                        print(f"   {json.dumps(content, indent=2, ensure_ascii=False)}")
                elif "output" in result_content:
                    print(f"   {result_content['output']}")
                elif "message" in result_content:
                    print(f"   {result_content['message']}")
                elif "reasoning" in result_content:
                    # Format reasoning response nicely
                    reasoning = result_content["reasoning"]
                    if isinstance(reasoning, dict):
                        print(f"   Understanding: {reasoning.get('understanding', 'N/A')}")
                        print(f"   Approach: {reasoning.get('approach', 'N/A')}")
                        if "considerations" in reasoning:
                            print("   Considerations:")
                            for consideration in reasoning["considerations"]:
                                print(f"     • {consideration}")
                        confidence = reasoning.get('confidence', 0.0)
                        # Ensure confidence is a numeric value before formatting
                        if isinstance(confidence, (int, float)):
                            print(f"\n   Confidence: {confidence:.2%}")
                        elif isinstance(confidence, str):
                            # Convert string to float if needed
                            try:
                                confidence_float = float(confidence)
                                print(f"\n   Confidence: {confidence_float:.2%}")
                            except (ValueError, TypeError):
                                print(f"\n   Confidence: {confidence} (could not format as percentage)")
                        else:
                            print(f"\n   Confidence: {confidence} (unknown type)")
                    else:
                        print(f"   {reasoning}")
                elif "task_analysis" in result_content:
                    # Format task analysis response
                    analysis = result_content["task_analysis"]
                    if isinstance(analysis, dict):
                        print(f"   Task Type: {analysis.get('task_type', 'unknown')}")
                        print(f"   Complexity: {analysis.get('complexity', 'unknown')}")
                        print(f"   Approach: {analysis.get('required_approach', 'unknown')}")
                        if "key_requirements" in analysis:
                            print("   Key Requirements:")
                            for req in analysis["key_requirements"]:
                                print(f"     • {req}")
                    else:
                        print(f"   {analysis}")
                else:
                    # Fallback to formatted JSON for complex responses
                    print(f"   {json.dumps(result_content, indent=2, ensure_ascii=False)}")
            else:
                # Simple string response
                print(f"   {result_content}")
        
        # Display decision info
        if "decision" in result:
            decision = result["decision"]
            print(f"\n📋 Decision: {decision.get('action', 'unknown')}")
            if "confidence" in decision:
                confidence = decision['confidence']
                if isinstance(confidence, (int, float)):
                    print(f"   Confidence: {confidence:.2%}")
                elif isinstance(confidence, str):
                    # Convert string to float if needed
                    try:
                        confidence_float = float(confidence)
                        print(f"   Confidence: {confidence_float:.2%}")
                    except (ValueError, TypeError):
                        print(f"   Confidence: {confidence} (could not format as percentage)")
                else:
                    print(f"   Confidence: {confidence} (unknown type)")
        
        # Display any errors
        if "error" in result:
            print(f"\n❌ Error: {result['error']}")

async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Chloe AI CLI Interface")
    parser.add_argument("--mode", choices=["interactive", "api"], 
                       default="interactive", help="Run mode")
    parser.add_argument("--host", default="0.0.0.0", help="API host")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    
    args = parser.parse_args()
    
    cli = CLIInterface()
    
    if args.mode == "interactive":
        await cli.interactive_mode()
    elif args.mode == "api":
        if not cli.ai_system:
            await cli.initialize_system()
        print(f"Starting API server on {args.host}:{args.port}")
        cli.ai_system.start_api_server()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nGoodbye! 👋")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)