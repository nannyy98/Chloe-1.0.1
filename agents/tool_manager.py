"""
Tool Manager - Orchestrates all tool agents
Implements basic tools: python_repl, web_search, file_ops
"""

import asyncio
import subprocess
import tempfile
import os
import importlib
from typing import Dict, List, Any, Optional
from pathlib import Path
import requests
from duckduckgo_search import DDGS

from utils.config import Config
from utils.logger import setup_logger

class ToolManager:
    """Manages and executes various tool agents"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger("tool_manager")
        self.tools = {}
        self._load_tools()
        
    def _load_tools(self):
        """Load available tools from agents directory"""
        # Define basic tools as methods in this class
        basic_tools = {
            "python_repl": self._execute_python_repl,
            "web_search": self._execute_web_search,
            "file_ops": self._execute_file_operations
        }
        
        # Add basic tools
        for tool_name, tool_func in basic_tools.items():
            self.tools[tool_name] = tool_func
            self.logger.info(f"Loaded basic tool: {tool_name}")
        
        # Load agent-based tools
        agents_dir = Path(__file__).parent.parent / "agents"
        agent_tools = {
            "code_runner": "code_agent.CodeAgent",
            "web_agent": "web_agent.WebAgent", 
            "file_agent": "file_agent.FileAgent",
            "data_analysis_agent": "data_agent.DataAnalysisAgent"
        }
        
        for tool_name, module_path in agent_tools.items():
            try:
                module_name, class_name = module_path.split('.')
                module = importlib.import_module(f"agents.{module_name}")
                tool_class = getattr(module, class_name)
                self.tools[tool_name] = tool_class(self.config)
                self.logger.info(f"Loaded tool: {tool_name}")
            except Exception as e:
                self.logger.warning(f"Failed to load tool {tool_name}: {e}")
                # Create dummy tool for now
                self.tools[tool_name] = DummyTool(tool_name)
    
    def list_available_tools(self) -> List[str]:
        """List all available tools"""
        return list(self.tools.keys())
    
    async def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific tool with parameters"""
        if tool_name not in self.tools:
            return {"error": f"Tool '{tool_name}' not found", "available_tools": self.list_available_tools()}
        
        try:
            tool = self.tools[tool_name]
            
            # Check if tool is a function (basic tool) or an object (agent tool)
            if callable(tool):
                result = await tool(params)
            else:
                result = await tool.execute(params)
            
            self.logger.info(f"Executed tool {tool_name} successfully")
            return {
                "tool": tool_name,
                "result": result,
                "status": "success"
            }
        except Exception as e:
            self.logger.error(f"Error executing tool {tool_name}: {e}")
            return {
                "tool": tool_name,
                "error": str(e),
                "status": "error"
            }
    
    async def execute_multiple_tools(self, tool_sequence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute a sequence of tools"""
        results = []
        for tool_spec in tool_sequence:
            result = await self.execute_tool(
                tool_spec["tool_name"],
                tool_spec.get("params", {})
            )
            results.append(result)
        return results
    
    async def _execute_python_repl(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Python code in a safe environment"""
        code = params.get("code", "")
        if not code:
            return {"error": "No code provided"}
        
        try:
            # Create a temporary file to execute the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Execute the code in a subprocess to isolate it
            result = subprocess.run([
                'python', temp_file
            ], capture_output=True, text=True, timeout=30)
            
            # Clean up the temporary file
            os.unlink(temp_file)
            
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "success": result.returncode == 0
            }
        except subprocess.TimeoutExpired:
            return {"error": "Code execution timed out"}
        except Exception as e:
            return {"error": f"Python REPL execution failed: {str(e)}"}
    
    async def _execute_web_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform web search using DuckDuckGo"""
        query = params.get("query", "")
        if not query:
            return {"error": "No search query provided"}
        
        try:
            ddgs = DDGS()
            results = ddgs.text(query, max_results=params.get("max_results", 5))
            
            return {
                "query": query,
                "results": results,
                "count": len(results)
            }
        except Exception as e:
            return {"error": f"Web search failed: {str(e)}"}
    
    async def _execute_file_operations(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle file operations like read, write, list"""
        operation = params.get("operation", "read")  # read, write, list, delete
        filepath = params.get("filepath", "")
        content = params.get("content", "")
        
        try:
            if operation == "read":
                if not filepath:
                    return {"error": "No filepath provided for read operation"}
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                return {"content": content, "operation": "read", "filepath": filepath}
            
            elif operation == "write":
                if not filepath or not content:
                    return {"error": "Filepath and content required for write operation"}
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                return {"status": "success", "operation": "write", "filepath": filepath}
            
            elif operation == "list":
                directory = filepath if filepath else "."
                if not os.path.isdir(directory):
                    return {"error": f"Directory does not exist: {directory}"}
                files = os.listdir(directory)
                return {"files": files, "directory": directory, "operation": "list"}
            
            elif operation == "delete":
                if not filepath:
                    return {"error": "No filepath provided for delete operation"}
                os.remove(filepath)
                return {"status": "deleted", "operation": "delete", "filepath": filepath}
            
            else:
                return {"error": f"Unsupported operation: {operation}"}
        
        except FileNotFoundError:
            return {"error": f"File not found: {filepath}"}
        except PermissionError:
            return {"error": f"Permission denied: {filepath}"}
        except Exception as e:
            return {"error": f"File operation failed: {str(e)}"}

class DummyTool:
    """Placeholder tool for demonstration"""
    
    def __init__(self, name: str):
        self.name = name
        
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute dummy tool"""
        return {
            "message": f"Dummy tool {self.name} executed",
            "params": params,
            "result": f"Simulated result for {self.name}"
        }