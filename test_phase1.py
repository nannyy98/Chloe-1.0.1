#!/usr/bin/env python3
"""
Phase 1 Tests - Testing the implemented features for Phase 1
"""
import asyncio
import unittest
import tempfile
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
import sys
sys.path.insert(0, str(project_root))

from utils.config import Config
from core.reasoning_core import ReasoningCore
from agents.tool_manager import ToolManager
from memory.memory_system import MemorySystem
from utils.logger import setup_logger


class TestPhase1(unittest.TestCase):
    """Test suite for Phase 1 implementation"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = Config()
        self.logger = setup_logger("phase1_test")
        
    def test_config_loading(self):
        """Test that configuration loads correctly with Pydantic"""
        self.assertIsNotNone(self.config)
        self.assertEqual(self.config.llm_model, "qwen2.5:7b")
        self.assertEqual(self.config.api_host, "0.0.0.0")
        
    def test_reasoning_core_initialization(self):
        """Test that reasoning core initializes correctly"""
        reasoning_core = ReasoningCore(self.config)
        self.assertIsNotNone(reasoning_core)
        self.assertEqual(reasoning_core.max_steps, 20)
        
    def test_tool_manager_basic_tools(self):
        """Test that basic tools are available in tool manager"""
        tool_manager = ToolManager(self.config)
        available_tools = tool_manager.list_available_tools()
        
        # Check that basic tools are present
        self.assertIn("python_repl", available_tools)
        self.assertIn("web_search", available_tools)
        self.assertIn("file_ops", available_tools)
        
    async def test_python_repl_tool(self):
        """Test python_repl tool execution"""
        tool_manager = ToolManager(self.config)
        
        # Test simple Python code execution
        params = {
            "code": "print('Hello, World!')"
        }
        result = await tool_manager.execute_tool("python_repl", params)
        
        self.assertEqual(result["status"], "success")
        self.assertIn("Hello, World!", result["result"]["stdout"])
        
    async def test_file_operations_tool(self):
        """Test file_ops tool operations"""
        tool_manager = ToolManager(self.config)
        
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
            tmp_file.write("test content")
            tmp_path = tmp_file.name
        
        try:
            # Test read operation
            read_params = {
                "operation": "read",
                "filepath": tmp_path
            }
            read_result = await tool_manager.execute_tool("file_ops", read_params)
            
            self.assertEqual(read_result["status"], "success")
            self.assertEqual(read_result["result"]["content"], "test content")
            
            # Test write operation
            write_params = {
                "operation": "write",
                "filepath": tmp_path,
                "content": "updated content"
            }
            write_result = await tool_manager.execute_tool("file_ops", write_params)
            
            self.assertEqual(write_result["status"], "success")
            
            # Verify the write worked
            verify_read_result = await tool_manager.execute_tool("file_ops", read_params)
            self.assertEqual(verify_read_result["result"]["content"], "updated content")
            
        finally:
            # Clean up
            os.unlink(tmp_path)
            
    async def test_memory_system_deque(self):
        """Test that memory system uses deque for short-term memory"""
        memory_system = MemorySystem(self.config)
        
        # Check that short-term memory is a deque with max length 20
        self.assertTrue(hasattr(memory_system, 'short_term_memory'))
        self.assertEqual(memory_system.short_term_memory.maxlen, 20)
        
        # Test storing and retrieving interactions
        await memory_system.store_interaction("test input", {"response": "test output"})
        
        recent = await memory_system.get_recent_interactions(5)
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]["input"], "test input")
        
    async def test_react_loop_basic(self):
        """Test basic ReAct loop functionality"""
        reasoning_core = ReasoningCore(self.config)
        
        # Test a simple reasoning task
        result = await reasoning_core.process("What is 2+2?")
        
        # Should have a result with trace
        self.assertIn("trace", result)
        self.assertIn("final_answer", result)
        self.assertGreaterEqual(len(result["trace"]), 1)


def run_async_test(coro):
    """Helper to run async tests"""
    try:
        loop = asyncio.get_running_loop()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


class AsyncTestCase(unittest.TestCase):
    """Base class for async tests"""
    
    def setUp(self):
        self.config = Config()
    
    def async_run(self, coro):
        """Run an async coroutine in the test"""
        return run_async_test(coro)


class TestAsyncPhase1(AsyncTestCase):
    """Async tests for Phase 1"""
    
    def test_python_repl_tool(self):
        """Test python_repl tool execution"""
        async def test():
            tool_manager = ToolManager(self.config)
            
            # Test simple Python code execution
            params = {
                "code": "print('Hello, World!')"
            }
            result = await tool_manager.execute_tool("python_repl", params)
            
            self.assertEqual(result["status"], "success")
            self.assertIn("Hello, World!", result["result"]["stdout"])
            
        self.async_run(test())
        
    def test_file_operations_tool(self):
        """Test file_ops tool operations"""
        async def test():
            tool_manager = ToolManager(self.config)
            
            # Create a temporary file for testing
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
                tmp_file.write("test content")
                tmp_path = tmp_file.name
            
            try:
                # Test read operation
                read_params = {
                    "operation": "read",
                    "filepath": tmp_path
                }
                read_result = await tool_manager.execute_tool("file_ops", read_params)
                
                self.assertEqual(read_result["status"], "success")
                self.assertEqual(read_result["result"]["content"], "test content")
                
            finally:
                # Clean up
                os.unlink(tmp_path)
                
        self.async_run(test())
        
    def test_memory_system_deque(self):
        """Test that memory system uses deque for short-term memory"""
        async def test():
            memory_system = MemorySystem(self.config)
            
            # Check that short-term memory is a deque with max length 20
            self.assertTrue(hasattr(memory_system, 'short_term_memory'))
            self.assertEqual(memory_system.short_term_memory.maxlen, 20)
            
            # Test storing and retrieving interactions
            await memory_system.store_interaction("test input", {"response": "test output"})
            
            recent = await memory_system.get_recent_interactions(5)
            self.assertEqual(len(recent), 1)
            self.assertEqual(recent[0]["input"], "test input")
            
        self.async_run(test())
        
    def test_react_loop_basic(self):
        """Test basic ReAct loop functionality"""
        async def test():
            reasoning_core = ReasoningCore(self.config)
            
            # Test a simple reasoning task
            result = await reasoning_core.process("What is 2+2?")
            
            # Should have a result with trace
            self.assertIn("trace", result)
            self.assertIn("final_answer", result)
            self.assertGreaterEqual(len(result["trace"]), 1)
            
        self.async_run(test())


if __name__ == "__main__":
    print("Running Phase 1 Tests...")
    unittest.main(verbosity=2)