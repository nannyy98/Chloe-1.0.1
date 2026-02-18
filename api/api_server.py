"""
API Server - FastAPI interface for Chloe AI
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn

from core.decision_engine import DecisionEngine
from utils.config import Config
from utils.logger import setup_logger

# Pydantic models for API
class TaskRequest(BaseModel):
    user_input: str
    context: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None

class TaskResponse(BaseModel):
    result: Dict[str, Any]
    decision: Dict[str, Any]
    metadata: Dict[str, Any]

class MemoryQuery(BaseModel):
    query: str
    memory_type: str = "knowledge"  # knowledge, experience, interaction
    limit: int = 5

class LearningRequest(BaseModel):
    action: str  # "get_insights", "adapt_strategy", "get_metrics"
    parameters: Optional[Dict[str, Any]] = None

app = FastAPI(
    title="Chloe AI API",
    description="Adaptive Cognitive System API",
    version="0.1.0"
)

class APIServer:
    """FastAPI server for Chloe AI"""
    
    def __init__(self, decision_engine: DecisionEngine, config: Config):
        self.decision_engine = decision_engine
        self.config = config
        self.logger = setup_logger("api_server")
        self.app = app
        
        # Store reference to main components
        self.reasoning_core = decision_engine.reasoning_core
        self.memory_system = decision_engine.memory_system
        self.tool_manager = decision_engine.tool_manager
        self.learning_engine = decision_engine.learning_engine
        
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/")
        async def root():
            return {
                "message": "Chloe AI - Adaptive Cognitive System",
                "status": "running",
                "version": "0.1.0"
            }
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "components": {
                    "reasoning_core": "available",
                    "decision_engine": "available",
                    "memory_system": "available",
                    "tool_manager": "available",
                    "learning_engine": "available"
                }
            }
        
        @self.app.post("/process", response_model=TaskResponse)
        async def process_task(request: TaskRequest):
            """Process a user task through the cognitive system"""
            try:
                result = await self.decision_engine.make_decision(
                    request.user_input, 
                    request.context
                )
                
                # Execute the decision
                if result["action"] == "reason":
                    execution_result = await self.reasoning_core.process(
                        request.user_input, 
                        request.context
                    )
                elif result["action"] == "tool":
                    execution_result = await self.tool_manager.execute_tool(
                        result["tool_name"],
                        result.get("tool_params", {})
                    )
                elif result["action"] == "learn":
                    execution_result = await self.learning_engine.process_experience(
                        request.user_input
                    )
                else:
                    execution_result = {"message": "Unknown action type"}
                
                # Record experience
                await self.learning_engine.record_experience(
                    request.user_input, result, execution_result
                )
                
                # Store in memory
                if self.memory_system:
                    await self.memory_system.store_interaction(
                        request.user_input, 
                        execution_result, 
                        request.context, 
                        request.session_id
                    )
                
                return TaskResponse(
                    result=execution_result,
                    decision=result,
                    metadata={
                        "processing_time": "TODO",
                        "confidence": result.get("confidence", 0.0),
                        "session_id": request.session_id
                    }
                )
                
            except Exception as e:
                self.logger.error(f"Error processing task: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/react_process")
        async def react_process_task(request: TaskRequest):
            """Process a user task specifically using the ReAct loop"""
            try:
                # Use the reasoning core directly for ReAct processing
                execution_result = await self.reasoning_core.process(
                    request.user_input, 
                    request.context
                )
                
                # Create a decision indicating reasoning was used
                decision = {
                    "action": "reason",
                    "confidence": execution_result.get("confidence", 0.8),
                    "method": "ReAct_loop"
                }
                
                # Record experience
                await self.learning_engine.record_experience(
                    request.user_input, decision, execution_result
                )
                
                # Store in memory
                if self.memory_system:
                    await self.memory_system.store_interaction(
                        request.user_input, 
                        execution_result, 
                        request.context, 
                        request.session_id
                    )
                
                return TaskResponse(
                    result=execution_result,
                    decision=decision,
                    metadata={
                        "processing_time": "TODO",
                        "confidence": decision.get("confidence", 0.0),
                        "session_id": request.session_id
                    }
                )
                
            except Exception as e:
                self.logger.error(f"Error in ReAct processing task: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/tools")
        async def list_tools():
            """List available tools"""
            return {
                "available_tools": self.tool_manager.list_available_tools(),
                "status": "success"
            }
        
        @self.app.post("/tools/execute")
        async def execute_tool(tool_name: str, params: Dict[str, Any]):
            """Execute a specific tool"""
            try:
                result = await self.tool_manager.execute_tool(tool_name, params)
                return {"result": result, "status": "success"}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.post("/memory/store")
        async def store_memory(category: str, key: str, value: str, source: str = "api"):
            """Store information in long-term memory"""
            try:
                if self.memory_system:
                    await self.memory_system.store_knowledge(category, key, value, source)
                    return {"status": "success", "message": "Knowledge stored"}
                else:
                    raise HTTPException(status_code=500, detail="Memory system not available")
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/memory/search")
        async def search_memory(query: MemoryQuery):
            """Search memory (knowledge, experience, or interactions)"""
            try:
                if not self.memory_system:
                    raise HTTPException(status_code=500, detail="Memory system not available")
                
                if query.memory_type == "knowledge":
                    results = await self.memory_system.search_knowledge(query.query, query.limit)
                elif query.memory_type == "experience":
                    results = await self.memory_system.get_similar_experiences(query.query, query.limit)
                elif query.memory_type == "interaction":
                    results = await self.memory_system.get_recent_interactions(query.limit)
                else:
                    raise HTTPException(status_code=400, detail="Invalid memory type")
                
                return {"results": results, "status": "success"}
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/memory/stats")
        async def get_memory_stats():
            """Get memory system statistics"""
            try:
                if self.memory_system:
                    stats = await self.memory_system.get_memory_stats()
                    return {"stats": stats, "status": "success"}
                else:
                    raise HTTPException(status_code=500, detail="Memory system not available")
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/learning/process")
        async def process_learning(request: LearningRequest):
            """Process learning requests"""
            try:
                if request.action == "get_insights":
                    insights = await self.learning_engine.process_experience("system request")
                    return {"insights": insights, "status": "success"}
                elif request.action == "adapt_strategy":
                    adaptation = await self.learning_engine.adapt_strategy()
                    return {"adaptation": adaptation, "status": "success"}
                elif request.action == "get_metrics":
                    metrics = self.learning_engine.get_learning_metrics()
                    return {"metrics": metrics, "status": "success"}
                elif request.action == "get_state":
                    state = await self.learning_engine.get_current_state()
                    return {"state": state, "status": "success"}
                else:
                    raise HTTPException(status_code=400, detail="Unknown learning action")
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/system/status")
        async def system_status():
            """Get comprehensive system status"""
            try:
                status = {
                    "api": "running",
                    "reasoning_core": "available",
                    "decision_engine": "available",
                    "memory_system": "available" if self.memory_system else "unavailable",
                    "tool_manager": "available",
                    "learning_engine": "available"
                }
                
                # Add memory stats if available
                if self.memory_system:
                    try:
                        memory_stats = await self.memory_system.get_memory_stats()
                        status["memory_stats"] = memory_stats
                    except Exception:
                        status["memory_stats"] = "error"
                
                # Add learning metrics
                try:
                    learning_metrics = self.learning_engine.get_learning_metrics()
                    status["learning_metrics"] = learning_metrics
                except Exception:
                    status["learning_metrics"] = "error"
                
                return status
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    
    def start(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the API server"""
        api_config = self.config.get_api_config()
        host = api_config.get("host", host)
        port = api_config.get("port", port)
        
        self.logger.info(f"Starting API server on {host}:{port}")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )

# Example usage and testing
if __name__ == "__main__":
    # This would be run from main.py
    print("API server module ready")
    print("Available endpoints:")
    print("- POST /process - Process user tasks")
    print("- GET /tools - List available tools") 
    print("- POST /tools/execute - Execute specific tools")
    print("- POST /memory/store - Store knowledge")
    print("- POST /memory/search - Search memory")
    print("- GET /memory/stats - Memory statistics")
    print("- POST /learning/process - Learning operations")
    print("- GET /system/status - System status")