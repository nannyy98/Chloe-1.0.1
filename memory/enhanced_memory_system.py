"""
Enhanced Memory System for Chloe AI
Advanced memory management with short-term, long-term, and experience storage
"""

import asyncio
import json
import sqlite3
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import deque
import pickle
from pathlib import Path
import hashlib
from chromadb.utils import embedding_functions
import chromadb
from utils.config import Config
from utils.logger import setup_logger

class EnhancedMemorySystem:
    """Enhanced memory system with advanced capabilities"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger("enhanced_memory_system")
        self.data_path = Path(config.get("data_path", "./data"))
        self.logs_path = Path(config.get("logs_path", "./logs"))
        
        # Create directories
        self.data_path.mkdir(exist_ok=True)
        self.logs_path.mkdir(exist_ok=True)
        
        # Short-term memory using deque with max size
        self.short_term_memory = deque(maxlen=config.get("memory.short_term_limit", 100))
        
        # Context window management
        self.context_window = deque(maxlen=config.get("performance.context_window_max", 30))
        self.compression_threshold = config.get("performance.compression_threshold", 10)
        
        # Initialize databases
        self._initialize_sqlite_db()
        self._initialize_vector_db()
        
        # Memory statistics
        self.stats = {
            'short_term_count': 0,
            'knowledge_base_count': 0,
            'experience_count': 0,
            'compressions_performed': 0
        }
        
        self.logger.info("Enhanced Memory System initialized")
    
    def _initialize_sqlite_db(self):
        """Initialize SQLite database for structured memory storage"""
        db_path = self.data_path / "memory.db"
        self.db_conn = sqlite3.connect(db_path, check_same_thread=False)
        
        cursor = self.db_conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_base (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE,
                content TEXT,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task TEXT,
                result TEXT,
                outcome REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                context TEXT,
                metadata TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_input TEXT,
                ai_response TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                session_id TEXT,
                metadata TEXT
            )
        """)
        
        self.db_conn.commit()
    
    def _initialize_vector_db(self):
        """Initialize ChromaDB for semantic search"""
        try:
            self.chroma_client = chromadb.PersistentClient(path=str(self.data_path / "chroma_enhanced"))
            
            # Create collections
            self.collections = {
                "knowledge": self.chroma_client.get_or_create_collection(
                    name="knowledge",
                    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
                ),
                "experiences": self.chroma_client.get_or_create_collection(
                    name="experiences",
                    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
                ),
                "context": self.chroma_client.get_or_create_collection(
                    name="context",
                    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
                )
            }
        except Exception as e:
            self.logger.error(f"Failed to initialize vector DB: {e}")
            self.chroma_client = None
    
    async def store_interaction(self, user_input: str, ai_response: Dict[str, Any], session_id: str = None) -> bool:
        """Store interaction in memory"""
        try:
            # Add to short-term memory
            interaction = {
                "user_input": user_input,
                "ai_response": ai_response,
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id
            }
            
            self.short_term_memory.append(interaction)
            self.stats['short_term_count'] += 1
            
            # Add to context window
            self.context_window.append(interaction)
            
            # Compress if needed
            if len(self.context_window) >= self.compression_threshold:
                await self._compress_context()
            
            # Store in database
            cursor = self.db_conn.cursor()
            cursor.execute("""
                INSERT INTO interactions (user_input, ai_response, session_id, metadata)
                VALUES (?, ?, ?, ?)
            """, (
                user_input,
                json.dumps(ai_response),
                session_id,
                json.dumps({"source": "interaction"})
            ))
            
            self.db_conn.commit()
            
            # Store in vector database
            if self.chroma_client:
                self.collections["context"].add(
                    documents=[f"User: {user_input} AI: {json.dumps(ai_response)}"],
                    metadatas=[{"type": "interaction", "session_id": session_id, "timestamp": datetime.now().isoformat()}],
                    ids=[f"interaction_{hashlib.md5((user_input + str(datetime.now())).encode()).hexdigest()}"]
                )
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to store interaction: {e}")
            return False
    
    async def _compress_context(self):
        """Compress context window using LLM summary"""
        try:
            if len(self.context_window) < self.compression_threshold:
                return
            
            # Get items to compress
            items_to_compress = list(self.context_window)[:self.compression_threshold//2]
            
            # Summarize using an LLM (using a placeholder for now)
            summary = await self._summarize_context(items_to_compress)
            
            # Replace compressed items with summary
            for _ in range(len(items_to_compress)):
                if self.context_window:
                    self.context_window.popleft()
            
            # Add summary to context
            self.context_window.appendleft({
                "type": "summary",
                "content": summary,
                "timestamp": datetime.now().isoformat(),
                "original_items_count": len(items_to_compress)
            })
            
            self.stats['compressions_performed'] += 1
            self.logger.info(f"Context compressed: {len(items_to_compress)} items -> 1 summary")
            
        except Exception as e:
            self.logger.error(f"Failed to compress context: {e}")
    
    async def _summarize_context(self, items: List[Dict]) -> str:
        """Summarize context items using LLM"""
        # Placeholder implementation - in real system, this would call an LLM
        # For now, just return a simple concatenation
        summary_parts = []
        for item in items:
            if isinstance(item, dict):
                user_input = item.get('user_input', '')
                ai_response = str(item.get('ai_response', ''))
                summary_parts.append(f"Q: {user_input[:100]}... A: {ai_response[:100]}...")
            else:
                summary_parts.append(str(item)[:200])
        
        return "Summary: " + " | ".join(summary_parts)
    
    async def retrieve_similar_interactions(self, query: str, limit: int = 5) -> List[Dict]:
        """Retrieve similar interactions using semantic search"""
        try:
            if self.chroma_client:
                results = self.collections["context"].query(
                    query_texts=[query],
                    n_results=limit
                )
                
                # Format results
                formatted_results = []
                for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                    formatted_results.append({
                        "content": doc,
                        "metadata": meta
                    })
                
                return formatted_results
            else:
                # Fallback to SQLite
                cursor = self.db_conn.cursor()
                cursor.execute("""
                    SELECT user_input, ai_response, timestamp 
                    FROM interactions 
                    WHERE user_input LIKE ? OR ai_response LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (f"%{query}%", f"%{query}%", limit))
                
                rows = cursor.fetchall()
                return [{"user_input": row[0], "ai_response": json.loads(row[1]), "timestamp": row[2]} for row in rows]
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve similar interactions: {e}")
            return []
    
    async def store_knowledge(self, key: str, content: str, metadata: Dict = None) -> bool:
        """Store knowledge in long-term memory"""
        try:
            # Store in database
            cursor = self.db_conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO knowledge_base (key, content, metadata)
                VALUES (?, ?, ?)
            """, (key, content, json.dumps(metadata or {})))
            
            self.db_conn.commit()
            self.stats['knowledge_base_count'] += 1
            
            # Store in vector database
            if self.chroma_client:
                self.collections["knowledge"].add(
                    documents=[content],
                    metadatas=[metadata or {"key": key}],
                    ids=[f"kb_{hashlib.md5(key.encode()).hexdigest()}"]
                )
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to store knowledge: {e}")
            return False
    
    async def retrieve_knowledge(self, query: str, limit: int = 5) -> List[Dict]:
        """Retrieve knowledge using semantic search"""
        try:
            if self.chroma_client:
                results = self.collections["knowledge"].query(
                    query_texts=[query],
                    n_results=limit
                )
                
                # Format results
                formatted_results = []
                for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                    formatted_results.append({
                        "content": doc,
                        "metadata": meta
                    })
                
                return formatted_results
            else:
                # Fallback to SQLite
                cursor = self.db_conn.cursor()
                cursor.execute("""
                    SELECT key, content, metadata 
                    FROM knowledge_base 
                    WHERE key LIKE ? OR content LIKE ?
                    ORDER BY updated_at DESC
                    LIMIT ?
                """, (f"%{query}%", f"%{query}%", limit))
                
                rows = cursor.fetchall()
                return [{"key": row[0], "content": row[1], "metadata": json.loads(row[2])} for row in rows]
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve knowledge: {e}")
            return []
    
    async def store_experience(self, task: str, result: Dict[str, Any], outcome: float, context: Dict = None) -> bool:
        """Store experience for learning"""
        try:
            # Store in database
            cursor = self.db_conn.cursor()
            cursor.execute("""
                INSERT INTO experiences (task, result, outcome, context)
                VALUES (?, ?, ?, ?)
            """, (task, json.dumps(result), outcome, json.dumps(context or {})))
            
            self.db_conn.commit()
            self.stats['experience_count'] += 1
            
            # Store in vector database
            if self.chroma_client:
                self.collections["experiences"].add(
                    documents=[f"Task: {task} Result: {json.dumps(result)}"],
                    metadatas=[{"outcome": outcome, "task": task}],
                    ids=[f"exp_{hashlib.md5((task + str(outcome)).encode()).hexdigest()}"]
                )
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to store experience: {e}")
            return False
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        # Update stats
        self.stats['short_term_count'] = len(self.short_term_memory)
        self.stats['context_windows_used'] = len(self.context_window)
        
        if self.chroma_client:
            try:
                kb_count = self.collections["knowledge"].count()
                exp_count = self.collections["experiences"].count()
                ctx_count = self.collections["context"].count()
                
                self.stats['knowledge_base_size'] = kb_count
                self.stats['experience_store_size'] = exp_count
            except:
                # Fallback if vector DB is not available
                self.stats['knowledge_base_size'] = self.stats['knowledge_base_count']
                self.stats['experience_store_size'] = self.stats['experience_count']
        else:
            self.stats['knowledge_base_size'] = self.stats['knowledge_base_count']
            self.stats['experience_store_size'] = self.stats['experience_count']
        
        # Calculate recent performance
        recent_performance = await self._calculate_recent_performance()
        
        return {
            "short_term_interactions": self.stats['short_term_count'],
            "knowledge_base": self.stats['knowledge_base_size'],
            "learning_experiences": self.stats['experience_count'],
            "recent_performance": recent_performance,
            "vector_collections": {
                "knowledge": self.stats['knowledge_base_size'],
                "experiences": self.stats['experience_store_size'],
                "context": self.stats['context_windows_used']
            },
            "memory_management": {
                "compressions_performed": self.stats['compressions_performed'],
                "current_context_size": len(self.context_window),
                "max_context_size": self.context_window.maxlen
            },
            "cache_status": "available"
        }
    
    async def _calculate_recent_performance(self) -> Dict[str, float]:
        """Calculate recent performance metrics"""
        try:
            cursor = self.db_conn.cursor()
            # Get experiences from the last week
            cursor.execute("""
                SELECT outcome, timestamp 
                FROM experiences 
                WHERE timestamp > datetime('now', '-7 days')
                ORDER BY timestamp DESC
            """)
            
            rows = cursor.fetchall()
            
            if not rows:
                return {
                    "avg_success_rate": 0.0,
                    "avg_processing_time": 0.0,
                    "total_experiences": 0
                }
            
            outcomes = [row[0] for row in rows]
            avg_success_rate = sum(outcomes) / len(outcomes) if outcomes else 0.0
            
            return {
                "avg_success_rate": avg_success_rate,
                "avg_processing_time": 0.0,  # Placeholder
                "total_experiences": len(rows)
            }
        except Exception as e:
            self.logger.error(f"Failed to calculate recent performance: {e}")
            return {
                "avg_success_rate": 0.0,
                "avg_processing_time": 0.0,
                "total_experiences": 0
            }
    
    async def clear_memory(self, memory_type: str = "short_term") -> bool:
        """Clear specific type of memory"""
        try:
            if memory_type == "short_term":
                self.short_term_memory.clear()
                self.context_window.clear()
                self.stats['short_term_count'] = 0
            elif memory_type == "knowledge_base":
                if self.chroma_client:
                    self.chroma_client.delete_collection("knowledge")
                    self._initialize_vector_db()  # Re-create empty collection
                cursor = self.db_conn.cursor()
                cursor.execute("DELETE FROM knowledge_base")
                self.db_conn.commit()
                self.stats['knowledge_base_count'] = 0
            elif memory_type == "experiences":
                if self.chroma_client:
                    self.chroma_client.delete_collection("experiences")
                    self._initialize_vector_db()  # Re-create empty collection
                cursor = self.db_conn.cursor()
                cursor.execute("DELETE FROM experiences")
                self.db_conn.commit()
                self.stats['experience_count'] = 0
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear memory: {e}")
            return False
    
    def get_short_term_context(self, limit: int = 10) -> List[Dict]:
        """Get recent items from short-term memory"""
        return list(self.short_term_memory)[-limit:]
    
    def get_recent_interactions(self, limit: int = 10) -> List[Dict]:
        """Get recent interactions from short-term memory"""
        return list(self.short_term_memory)[-limit:]
    
    def get_context_window(self) -> List[Dict]:
        """Get current context window"""
        return list(self.context_window)
    
    async def close(self):
        """Close memory system resources"""
        if self.db_conn:
            self.db_conn.close()

# Test function
if __name__ == "__main__":
    from utils.config import Config
    import asyncio
    
    async def test_memory():
        config = Config()
        memory = EnhancedMemorySystem(config)
        
        print("✅ Enhanced Memory System initialized")
        
        # Test storing and retrieving
        await memory.store_interaction("Hello", {"response": "Hi there!"}, "test_session")
        await memory.store_knowledge("test_key", "This is test content")
        
        interactions = await memory.retrieve_similar_interactions("Hello")
        print(f"Found {len(interactions)} similar interactions")
        
        knowledge = await memory.retrieve_knowledge("test")
        print(f"Found {len(knowledge)} knowledge items")
        
        stats = await memory.get_memory_stats()
        print(f"Memory stats: {stats}")
        
        await memory.close()
    
    asyncio.run(test_memory())