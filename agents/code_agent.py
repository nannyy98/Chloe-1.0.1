"""
Code Agent - Executes code-related tasks
"""

import asyncio
import subprocess
import tempfile
import os
from typing import Dict, Any

from utils.config import Config
from utils.logger import setup_logger

class CodeAgent:
    """Agent for executing code and programming tasks"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger("code_agent")
        
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code-related task"""
        task_type = params.get("task_type", "run_code")
        
        if task_type == "run_code":
            return await self._run_code(params)
        elif task_type == "debug_code":
            return await self._debug_code(params)
        elif task_type == "analyze_code":
            return await self._analyze_code(params)
        else:
            return {"error": f"Unknown task type: {task_type}"}
    
    async def _run_code(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run Python code"""
        code = params.get("code", "")
        if not code:
            return {"error": "No code provided"}
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Execute code
            result = subprocess.run(
                ['python3', temp_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Clean up
            os.unlink(temp_file)
            
            return {
                "output": result.stdout,
                "error": result.stderr,
                "return_code": result.returncode,
                "success": result.returncode == 0
            }
            
        except subprocess.TimeoutExpired:
            return {"error": "Code execution timed out"}
        except Exception as e:
            return {"error": f"Execution failed: {str(e)}"}
    
    async def _debug_code(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Debug code by analyzing errors"""
        code = params.get("code", "")
        error = params.get("error", "")
        
        debug_prompt = f"""
        Debug this Python code:
        
        Code:
        {code}
        
        Error:
        {error}
        
        Provide:
        1. Root cause analysis
        2. Specific fix suggestions
        3. Corrected code
        """
        
        # This would integrate with LLM for actual debugging
        return {
            "analysis": "Debug analysis would be generated here",
            "suggestions": ["Fix syntax error", "Check variable scope"],
            "corrected_code": "# Corrected code would be here"
        }
    
    async def _analyze_code(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze code quality and complexity"""
        code = params.get("code", "")
        
        # Simple analysis
        lines = code.count('\n') + 1
        functions = code.count('def ')
        classes = code.count('class ')
        
        return {
            "metrics": {
                "lines_of_code": lines,
                "function_count": functions,
                "class_count": classes,
                "complexity_estimate": "low" if lines < 50 else "medium" if lines < 100 else "high"
            },
            "recommendations": ["Consider refactoring" if lines > 100 else "Code structure looks good"]
        }