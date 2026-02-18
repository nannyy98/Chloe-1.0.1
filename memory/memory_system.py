"""
Memory System - Handles all types of memory for Chloe AI
Implements short-term memory using Deque
"""

import asyncio
import json
import sqlite3
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
from collections import deque

import chromadb
from chromadb.config import Settings

from utils.config import Config
from utils.logger import setup_logger

class MemorySystem:
    """Manages short-term, long-term, and experience memory"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger("memory_system")
        # Initialize short-term memory with a deque with max length of 20 as specified
        self.short_term_memory = deque(maxlen=config.config.get("memory", {}).get("short_term_limit", 20))
        self._initialize_storage()
        
    def _initialize_storage(self):
        """Initialize different memory storage systems"""
        # SQLite for structured data
        self.db_path = Path("data/memory.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
        # ChromaDB for vector embeddings
        self.chroma_client = chromadb.PersistentClient(
            path="data/chroma_enhanced",
            settings=Settings(anonymized_telemetry=False)
        )
        self._init_collections()
        
        self.logger.info("Memory system initialized")
    
    def _init_database(self):
        """Initialize SQLite database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Short-term memory (recent interactions)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS short_term_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                user_input TEXT,
                ai_response TEXT,
                context TEXT,
                session_id TEXT
            )
        ''')
        
        # Experience memory (learning data)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experience_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task TEXT,
                decision TEXT,
                result TEXT,
                success_score REAL,
                timestamp TEXT,
                metadata TEXT
            )
        ''')
        
        # Long-term knowledge
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS long_term_knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT,
                key TEXT,
                value TEXT,
                source TEXT,
                confidence REAL,
                timestamp TEXT,
                expires_at TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _init_collections(self):
        """Initialize ChromaDB collections"""
        try:
            # Long-term knowledge collection
            self.knowledge_collection = self.chroma_client.get_or_create_collection(
                name="long_term_knowledge",
                metadata={"description": "Structured knowledge base"}
            )
            
            # Experience collection for learning
            self.experience_collection = self.chroma_client.get_or_create_collection(
                name="experience_memory",
                metadata={"description": "Learning experiences and outcomes"}
            )
            
            # Task-specific experience collection
            self.task_experience_collection = self.chroma_client.get_or_create_collection(
                name="task_experiences",
                metadata={"description": "Task-specific experiences with reflections"}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Chroma collections: {e}")
    
    async def store_interaction(self, user_input: str, ai_response: Dict, context: Dict = None, session_id: str = None):
        """Store interaction in short-term memory"""
        # Add to deque-based short-term memory
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "ai_response": ai_response,
            "context": context,
            "session_id": session_id or "default"
        }
        self.short_term_memory.append(interaction)
        
        # Also store in persistent storage
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO short_term_memory 
            (timestamp, user_input, ai_response, context, session_id)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            user_input,
            json.dumps(ai_response),
            json.dumps(context) if context else None,
            session_id or "default"
        ))
        
        conn.commit()
        conn.close()
        
        # Keep only recent interactions (last 100)
        self._cleanup_short_term_memory()
        
        self.logger.debug(f"Stored interaction: {user_input[:50]}...")
    
    async def get_recent_interactions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent interactions from short-term memory"""
        # Get from deque-based short-term memory first
        recent_from_deque = list(self.short_term_memory)[-limit:]
        
        # Also get from persistent storage
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, user_input, ai_response, context, session_id
            FROM short_term_memory
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "timestamp": row[0],
                "input": row[1],
                "response": json.loads(row[2]) if row[2] else {},
                "context": json.loads(row[3]) if row[3] else {},
                "session_id": row[4]
            })
        
        conn.close()
        
        # Combine results prioritizing recent ones from deque
        # If deque has more recent entries, use those
        deque_results = []
        for interaction in recent_from_deque:
            deque_results.append({
                "timestamp": interaction["timestamp"],
                "input": interaction["user_input"],
                "response": interaction["ai_response"],
                "context": interaction["context"],
                "session_id": interaction["session_id"]
            })
        
        # Return the combination, with deque results taking precedence
        # This ensures we get the most recent interactions
        return deque_results
    
    async def store_knowledge(self, category: str, key: str, value: str, source: str = "interaction", confidence: float = 1.0):
        """Store knowledge in long-term memory"""
        # Store in SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO long_term_knowledge 
            (category, key, value, source, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            category,
            key,
            value,
            source,
            confidence,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        # Store in ChromaDB for semantic search
        try:
            self.knowledge_collection.add(
                documents=[value],
                metadatas=[{
                    "category": category,
                    "key": key,
                    "source": source,
                    "confidence": confidence,
                    "timestamp": datetime.now().isoformat()
                }],
                ids=[f"knowledge_{datetime.now().timestamp()}"]
            )
        except Exception as e:
            self.logger.warning(f"Failed to store in ChromaDB: {e}")
    
    async def search_knowledge(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search long-term knowledge using semantic search"""
        try:
            results = self.knowledge_collection.query(
                query_texts=[query],
                n_results=limit
            )
            
            # Format results
            knowledge_results = []
            for i, doc in enumerate(results['documents'][0]):
                knowledge_results.append({
                    "content": doc,
                    "metadata": results['metadatas'][0][i],
                    "similarity": results['distances'][0][i] if 'distances' in results else None
                })
            
            return knowledge_results
            
        except Exception as e:
            self.logger.error(f"Knowledge search failed: {e}")
            return []
    
    async def store_experience(self, task: str, decision: Dict, result: Dict, success_score: float, metadata: Dict = None):
        """Store learning experience with enhanced structure"""
        # Store in SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO experience_memory 
            (task, decision, result, success_score, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            task,
            json.dumps(decision),
            json.dumps(result),
            success_score,
            datetime.now().isoformat(),
            json.dumps(metadata) if metadata else None
        ))
        
        conn.commit()
        conn.close()
        
        # Store in ChromaDB for pattern recognition
        try:
            experience_text = f"Task: {task}\nDecision: {json.dumps(decision)}\nResult: {json.dumps(result)}"
            self.experience_collection.add(
                documents=[experience_text],
                metadatas=[{
                    "task": task,
                    "success_score": success_score,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": metadata or {}
                }],
                ids=[f"experience_{datetime.now().timestamp()}"]
            )
        except Exception as e:
            self.logger.warning(f"Failed to store experience in ChromaDB: {e}")
    
    async def store_task_experience(self, task: str, actions: List[Dict], result: Dict, reflection: str, success_score: float, metadata: Dict = None):
        """Store comprehensive task experience with reflection"""
        timestamp = datetime.now().isoformat()
        experience_id = f"task_exp_{datetime.now().timestamp()}"
        
        # Create comprehensive experience document
        experience_text = f"""
        Task: {task}
        Actions Taken: {json.dumps(actions, indent=2)}
        Result: {json.dumps(result, indent=2)}
        Reflection: {reflection}
        Success Score: {success_score}
        Timestamp: {timestamp}
        """
        
        # Store in ChromaDB
        try:
            self.task_experience_collection.add(
                documents=[experience_text.strip()],
                metadatas=[{
                    "task": task,
                    "actions_count": len(actions),
                    "success_score": success_score,
                    "has_reflection": bool(reflection),
                    "timestamp": timestamp,
                    "metadata": metadata or {}
                }],
                ids=[experience_id]
            )
            self.logger.info(f"Stored task experience: {experience_id}")
        except Exception as e:
            self.logger.error(f"Failed to store task experience: {e}")
            
        # Also store in JSONL as fallback
        await self._store_experience_jsonl({
            "id": experience_id,
            "task": task,
            "actions": actions,
            "result": result,
            "reflection": reflection,
            "success_score": success_score,
            "timestamp": timestamp,
            "metadata": metadata or {}
        })
    
    async def _store_experience_jsonl(self, experience_data: Dict):
        """Store experience in JSONL format as fallback"""
        try:
            jsonl_path = Path("data/experiences.jsonl")
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(jsonl_path, "a") as f:
                f.write(json.dumps(experience_data, ensure_ascii=False) + "\n")
            self.logger.debug(f"Stored experience in JSONL: {experience_data['id']}")
        except Exception as e:
            self.logger.warning(f"Failed to store experience in JSONL: {e}")
    
    async def get_similar_experiences(self, task_description: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Find similar past experiences"""
        try:
            results = self.experience_collection.query(
                query_texts=[task_description],
                n_results=limit
            )
            
            experience_results = []
            for i, doc in enumerate(results['documents'][0]):
                experience_results.append({
                    "content": doc,
                    "metadata": results['metadatas'][0][i],
                    "similarity": results['distances'][0][i] if 'distances' in results else None
                })
            
            return experience_results
            
        except Exception as e:
            self.logger.error(f"Experience search failed: {e}")
            return []
    
    async def get_similar_task_experiences(self, task_description: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find similar task experiences with reflections"""
        try:
            # Query the task experience collection
            results = self.task_experience_collection.query(
                query_texts=[task_description],
                n_results=limit,
                where={"has_reflection": True}  # Only get experiences with reflections
            )
            
            task_experiences = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    task_experiences.append({
                        "content": doc,
                        "metadata": results['metadatas'][0][i],
                        "similarity": results['distances'][0][i] if 'distances' in results else None,
                        "id": results['ids'][0][i]
                    })
            
            return task_experiences
            
        except Exception as e:
            self.logger.error(f"Task experience search failed: {e}")
            # Fallback to JSONL search
            return await self._search_experiences_jsonl(task_description, limit)
    
    async def _search_experiences_jsonl(self, task_description: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search experiences in JSONL as fallback"""
        try:
            jsonl_path = Path("data/experiences.jsonl")
            if not jsonl_path.exists():
                return []
            
            experiences = []
            with open(jsonl_path, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            exp = json.loads(line)
                            # Simple text matching for now
                            if task_description.lower() in exp.get("task", "").lower() or \
                               any(task_description.lower() in str(action).lower() for action in exp.get("actions", [])):
                                experiences.append({
                                    "content": f"Task: {exp['task']}\nReflection: {exp.get('reflection', '')}\nSuccess: {exp.get('success_score', 0)}",
                                    "metadata": {"task": exp["task"], "success_score": exp.get("success_score", 0)},
                                    "similarity": 0.8,  # Approximate similarity
                                    "id": exp["id"]
                                })
                        except json.JSONDecodeError:
                            continue
            
            # Sort by similarity and return top results
            return sorted(experiences, key=lambda x: x["similarity"], reverse=True)[:limit]
            
        except Exception as e:
            self.logger.warning(f"JSONL experience search failed: {e}")
            return []
    
    def _cleanup_short_term_memory(self):
        """Remove old interactions to maintain size"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Keep only last 100 interactions
        cursor.execute('''
            DELETE FROM short_term_memory 
            WHERE id NOT IN (
                SELECT id FROM short_term_memory 
                ORDER BY timestamp DESC 
                LIMIT 100
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get counts
        cursor.execute("SELECT COUNT(*) FROM short_term_memory")
        short_term_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM long_term_knowledge")
        long_term_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM experience_memory")
        experience_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "short_term_interactions": short_term_count,
            "long_term_knowledge": long_term_count,
            "learning_experiences": experience_count,
            "collections": {
                "knowledge": self.knowledge_collection.count() if hasattr(self, 'knowledge_collection') else 0,
                "experiences": self.experience_collection.count() if hasattr(self, 'experience_collection') else 0
            }
        }
    
    async def clear_memory(self, memory_type: str = "all"):
        """Clear specific type of memory"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if memory_type in ["short_term", "all"]:
            cursor.execute("DELETE FROM short_term_memory")
        
        if memory_type in ["long_term", "all"]:
            cursor.execute("DELETE FROM long_term_knowledge")
            
        if memory_type in ["experience", "all"]:
            cursor.execute("DELETE FROM experience_memory")
        
        conn.commit()
        conn.close()
        
        # Clear ChromaDB collections
        if memory_type in ["long_term", "all"] and hasattr(self, 'knowledge_collection'):
            self.chroma_client.delete_collection("long_term_knowledge")
            self._init_collections()
            
        if memory_type in ["experience", "all"] and hasattr(self, 'experience_collection'):
            self.chroma_client.delete_collection("experience_memory")
            self._init_collections()
        
        self.logger.info(f"Cleared {memory_type} memory")

# Example usage
if __name__ == "__main__":
    # Test the memory system
    config = Config()
    memory = MemorySystem(config)
    
    # Test storing and retrieving
    asyncio.run(memory.store_interaction("Hello", {"response": "Hi there!"}))
    recent = asyncio.run(memory.get_recent_interactions(5))
    print(f"Recent interactions: {len(recent)}")