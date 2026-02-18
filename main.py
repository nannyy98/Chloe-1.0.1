"""
Chloe AI - Adaptive Cognitive System
Main entry point for the modular AI architecture
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.enhanced_reasoning_core import EnhancedReasoningCore as ReasoningCore
from core.decision_engine import DecisionEngine
from memory.memory_system import MemorySystem
from learning.learning_engine import LearningEngine
from agents.tool_manager import ToolManager
from api.api_server import APIServer
from utils.config import Config
from utils.logger import setup_logger

class ChloeAI:
    """Main AI system orchestrator"""
    
    def __init__(self, config_path: str = "config/config.json"):
        """Initialize the complete AI system"""
        self.config = Config(config_path)
        self.logger = setup_logger("chloe_ai")
        
        # Initialize core components
        self.reasoning_core = ReasoningCore(self.config)
        self.memory_system = MemorySystem(self.config)
        self.tool_manager = ToolManager(self.config)
        self.learning_engine = LearningEngine(self.config)
        self.decision_engine = DecisionEngine(
            reasoning_core=self.reasoning_core,
            memory_system=self.memory_system,
            tool_manager=self.tool_manager,
            learning_engine=self.learning_engine,
            config=self.config
        )
        
        # Initialize API server
        self.api_server = APIServer(self.decision_engine, self.config)
        
        self.logger.info("Chloe AI system initialized successfully")
    
    async def process_task(self, user_input: str, context: dict = None) -> dict:
        """Process a user task through the complete cognitive pipeline with experience learning"""
        try:
            # 1. Get relevant past experiences
            similar_experiences = await self.memory_system.get_similar_task_experiences(user_input, limit=3)
            
            # 2. Decision Engine determines the approach (with experience context)
            decision_context = context or {}
            if similar_experiences:
                decision_context["similar_experiences"] = similar_experiences
                
            decision = await self.decision_engine.make_decision(user_input, decision_context)
            
            # 3. Execute the determined approach
            actions_taken = []
            if decision["action"] == "reason":
                result = await self.reasoning_core.process(user_input, context)
                actions_taken.append({
                    "type": "reasoning",
                    "details": "ReAct loop processing",
                    "timestamp": "now"
                })
            elif decision["action"] == "tool":
                result = await self.tool_manager.execute_tool(
                    decision["tool_name"], 
                    decision["tool_params"]
                )
                actions_taken.append({
                    "type": "tool_execution",
                    "tool_name": decision["tool_name"],
                    "parameters": decision["tool_params"],
                    "timestamp": "now"
                })
            elif decision["action"] == "learn":
                result = await self.learning_engine.process_experience(user_input)
                actions_taken.append({
                    "type": "learning",
                    "details": "Experience analysis",
                    "timestamp": "now"
                })
            else:
                result = {"error": "Unknown action type"}
                actions_taken.append({
                    "type": "error",
                    "details": "Unknown action type",
                    "timestamp": "now"
                })
            
            # 4. Store comprehensive experience with reflection
            await self.learning_engine.record_experience(
                user_input, decision, result, actions_taken
            )
            
            # 5. Update memory
            await self.memory_system.store_interaction(user_input, result)
            
            return {
                "result": result,
                "decision": decision,
                "actions_taken": actions_taken,
                "similar_experiences_used": len(similar_experiences),
                "metadata": {
                    "processing_time": "TODO",
                    "confidence": decision.get("confidence", 0.0)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error processing task: {e}")
            return {"error": str(e)}

    def start_api_server(self):
        """Start the API server"""
        self.logger.info("Starting API server...")
        self.api_server.start()

if __name__ == "__main__":
    # Initialize and run the system
    ai_system = ChloeAI()
    
    # For development, start API server
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        ai_system.start_api_server()
    else:
        print("Chloe AI system initialized. Use 'python main.py api' to start the server.")