"""
File Agent - Handles file system operations
"""

import asyncio
import os
import json
from pathlib import Path
from typing import Dict, Any

from utils.config import Config
from utils.logger import setup_logger

class FileAgent:
    """Agent for file system operations"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger("file_agent")
        self.workspace = Path(config.get("workspace", "."))
        
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute file-related task"""
        task_type = params.get("task_type", "read")
        
        if task_type == "read":
            return await self._read_file(params)
        elif task_type == "write":
            return await self._write_file(params)
        elif task_type == "list":
            return await self._list_files(params)
        elif task_type == "search":
            return await self._search_files(params)
        else:
            return {"error": f"Unknown task type: {task_type}"}
    
    async def _read_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Read file content"""
        file_path = params.get("path", "")
        if not file_path:
            return {"error": "No file path provided"}
        
        try:
            full_path = self.workspace / file_path
            if not full_path.exists():
                return {"error": f"File not found: {file_path}"}
            
            with open(full_path, 'r') as f:
                content = f.read()
            
            return {
                "path": str(full_path),
                "content": content,
                "size": len(content),
                "lines": content.count('\n') + 1
            }
            
        except Exception as e:
            return {"error": f"Failed to read file: {str(e)}"}
    
    async def _write_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Write content to file"""
        file_path = params.get("path", "")
        content = params.get("content", "")
        
        if not file_path:
            return {"error": "No file path provided"}
        
        try:
            full_path = self.workspace / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(full_path, 'w') as f:
                f.write(content)
            
            return {
                "path": str(full_path),
                "status": "success",
                "size": len(content)
            }
            
        except Exception as e:
            return {"error": f"Failed to write file: {str(e)}"}
    
    async def _list_files(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List files in directory"""
        directory = params.get("path", ".")
        pattern = params.get("pattern", "*")
        
        try:
            full_path = self.workspace / directory
            if not full_path.exists():
                return {"error": f"Directory not found: {directory}"}
            
            files = list(full_path.glob(pattern))
            file_info = []
            
            for file_path in files:
                if file_path.is_file():
                    stat = file_path.stat()
                    file_info.append({
                        "name": file_path.name,
                        "path": str(file_path.relative_to(self.workspace)),
                        "size": stat.st_size,
                        "modified": stat.st_mtime
                    })
            
            return {
                "directory": str(full_path),
                "files": file_info,
                "count": len(file_info)
            }
            
        except Exception as e:
            return {"error": f"Failed to list files: {str(e)}"}
    
    async def _search_files(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Search for files containing specific content"""
        search_term = params.get("term", "")
        directory = params.get("path", ".")
        
        if not search_term:
            return {"error": "No search term provided"}
        
        try:
            full_path = self.workspace / directory
            matches = []
            
            for file_path in full_path.rglob("*"):
                if file_path.is_file() and file_path.suffix in ['.txt', '.py', '.md', '.json']:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if search_term.lower() in content.lower():
                                matches.append({
                                    "file": str(file_path.relative_to(self.workspace)),
                                    "size": len(content)
                                })
                    except Exception:
                        continue  # Skip files that can't be read
            
            return {
                "search_term": search_term,
                "matches": matches,
                "match_count": len(matches)
            }
            
        except Exception as e:
            return {"error": f"Search failed: {str(e)}"}