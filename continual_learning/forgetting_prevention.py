"""
Catastrophic Forgetting Prevention - Techniques to prevent loss of prior knowledge
"""
import asyncio
import numpy as np
import pickle
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import OrderedDict, deque

from utils.config import Config
from utils.logger import setup_logger
from memory.memory_system import MemorySystem
from learning.learning_engine import LearningEngine


class ForgettingPrevention:
    """Prevents catastrophic forgetting in neural networks and knowledge systems"""
    
    def __init__(self, config: Config, memory_system: MemorySystem):
        self.config = config
        self.memory_system = memory_system
        self.logger = setup_logger("forgetting_prevention")
        
        # Elastic Weight Consolidation (EWC) parameters
        self.fisher_information = {}
        self.optimal_params = {}
        self.importance_weight = config.get("continual_learning.importance_weight", 1000)
        
        # Replay buffer for experience replay
        self.replay_buffer = deque(maxlen=config.get("continual_learning.replay_buffer_size", 1000))
        
        # Knowledge distillation parameters
        self.previous_model_snapshots = []
        self.max_snapshots = config.get("continual_learning.max_snapshots", 5)
        
        # Regularization parameters
        self.regularization_strength = config.get("continual_learning.regularization_strength", 0.1)
        
        # Track important knowledge
        self.important_knowledge = {}
        self.knowledge_decay_rate = config.get("continual_learning.knowledge_decay", 0.95)
        
    async def update_important_weights(self, task_id: str, gradients: Dict[str, np.ndarray], 
                                     performance_before: float, performance_after: float):
        """Update important weights using Elastic Weight Consolidation (EWC)"""
        try:
            # Calculate Fisher Information Matrix approximation
            fisher_approx = {}
            for param_name, grad in gradients.items():
                fisher_approx[param_name] = grad ** 2  # Simplified Fisher approximation
            
            # Update Fisher information with exponential moving average
            for param_name, fisher_val in fisher_approx.items():
                if param_name in self.fisher_information:
                    self.fisher_information[param_name] = (
                        0.9 * self.fisher_information[param_name] + 
                        0.1 * fisher_val
                    )
                else:
                    self.fisher_information[param_name] = fisher_val
            
            # Store optimal parameters if performance improved
            if performance_after >= performance_before:
                # In a real implementation, this would store actual model parameters
                # For simulation, we'll store metadata
                self.optimal_params[task_id] = {
                    "timestamp": datetime.now().isoformat(),
                    "performance": performance_after,
                    "important_weights": list(gradients.keys())[:10]  # Top 10 for simulation
                }
            
            self.logger.info(f"Updated important weights for task {task_id}, params tracked: {len(self.fisher_information)}")
            
        except Exception as e:
            self.logger.error(f"Error updating important weights: {e}")
    
    def add_to_replay_buffer(self, experience: Dict[str, Any]):
        """Add experience to replay buffer for experience replay"""
        experience["timestamp"] = datetime.now().isoformat()
        experience["importance"] = experience.get("importance", 0.5)  # Default importance
        
        self.replay_buffer.append(experience)
        
        # Maintain buffer size
        if len(self.replay_buffer) > self.replay_buffer.maxlen:
            self.replay_buffer.popleft()
    
    async def sample_from_replay(self, batch_size: int = 32) -> List[Dict[str, Any]]:
        """Sample experiences from replay buffer"""
        if len(self.replay_buffer) == 0:
            return []
        
        # Convert to list and sort by importance (higher importance sampled more frequently)
        experiences = list(self.replay_buffer)
        experiences.sort(key=lambda x: x.get("importance", 0.5), reverse=True)
        
        # Sample based on importance
        sample_size = min(batch_size, len(experiences))
        return experiences[:sample_size]
    
    async def compute_regularization_loss(self, current_params: Dict[str, np.ndarray]) -> float:
        """Compute regularization loss to prevent forgetting"""
        if not self.optimal_params or not self.fisher_information:
            return 0.0
        
        total_reg_loss = 0.0
        
        # Calculate EWC regularization term
        for param_name, current_param in current_params.items():
            if param_name in self.fisher_information and param_name in self.optimal_params:
                optimal_param = self.optimal_params[list(self.optimal_params.keys())[0]].get("important_weights", [])
                fisher_val = self.fisher_information[param_name]
                
                # Simplified EWC calculation
                reg_term = self.importance_weight * fisher_val * ((current_param - optimal_param) ** 2).sum() if len(optimal_param) > 0 else 0
                total_reg_loss += reg_term
        
        return total_reg_loss * self.regularization_strength
    
    async def preserve_knowledge(self, task_name: str, knowledge_elements: List[Dict[str, Any]]):
        """Preserve important knowledge elements to prevent forgetting"""
        try:
            for knowledge in knowledge_elements:
                knowledge_id = knowledge.get("id", f"{task_name}_{datetime.now().timestamp()}")
                importance = knowledge.get("importance", 0.5)
                
                # Store in memory system with high importance
                await self.memory_system.store_experience(
                    task=task_name,
                    decision={"action": f"preserve_knowledge:{knowledge_id}"},
                    result=knowledge,
                    success_score=importance * 2.0,  # Boost importance for preservation
                    metadata={"experience_type": "preserved_knowledge", "importance": importance * 2.0}
                )
                
                # Track locally as well
                self.important_knowledge[knowledge_id] = {
                    "content": knowledge,
                    "importance": importance,
                    "last_accessed": datetime.now().isoformat(),
                    "access_count": 0
                }
            
            self.logger.info(f"Preserved {len(knowledge_elements)} knowledge elements for task {task_name}")
            
        except Exception as e:
            self.logger.error(f"Error preserving knowledge: {e}")
    
    async def retrieve_preserved_knowledge(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve preserved knowledge relevant to query"""
        try:
            # Search in memory system
            memory_results = await self.memory_system.get_similar_experiences(query, limit=top_k)
            
            # Also search in local important knowledge
            local_matches = []
            query_lower = query.lower()
            
            for knowledge_id, knowledge_data in self.important_knowledge.items():
                content_str = str(knowledge_data["content"]).lower()
                if query_lower in content_str:
                    knowledge_data["knowledge_id"] = knowledge_id
                    local_matches.append(knowledge_data)
            
            # Combine and sort by importance
            all_matches = memory_results + local_matches
            all_matches.sort(key=lambda x: x.get("importance", x.get("metadata", {}).get("importance", 0.5)), reverse=True)
            
            return all_matches[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error retrieving preserved knowledge: {e}")
            return []
    
    async def decay_knowledge_importance(self):
        """Apply decay to knowledge importance over time"""
        current_time = datetime.now()
        
        for knowledge_id, knowledge_data in list(self.important_knowledge.items()):
            # Calculate time since last access
            last_accessed = datetime.fromisoformat(knowledge_data["last_accessed"])
            days_since_access = (current_time - last_accessed).days
            
            # Apply decay
            decay_factor = self.knowledge_decay_rate ** days_since_access
            knowledge_data["importance"] *= decay_factor
            knowledge_data["last_accessed"] = current_time.isoformat()
            
            # Remove if importance drops too low
            if knowledge_data["importance"] < 0.1:
                del self.important_knowledge[knowledge_id]
    
    async def create_model_snapshot(self, model_state: Dict[str, Any], task_performance: Dict[str, float]):
        """Create a snapshot of the model state for knowledge distillation"""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "model_state": model_state,
            "task_performance": task_performance,
            "snapshot_id": f"snapshot_{datetime.now().timestamp()}"
        }
        
        self.previous_model_snapshots.append(snapshot)
        
        # Maintain maximum snapshots
        if len(self.previous_model_snapshots) > self.max_snapshots:
            self.previous_model_snapshots.pop(0)
        
        self.logger.info(f"Created model snapshot, total snapshots: {len(self.previous_model_snapshots)}")
    
    async def get_stable_knowledge(self) -> Dict[str, Any]:
        """Get stable knowledge that should not be forgotten"""
        stable_knowledge = {
            "important_weights": dict(list(self.fisher_information.items())[:20]),  # Top 20
            "preserved_experiences": list(self.important_knowledge.items())[:50],   # Top 50
            "model_snapshots": self.previous_model_snapshots,
            "replay_buffer_size": len(self.replay_buffer),
            "total_preserved_items": len(self.important_knowledge)
        }
        
        return stable_knowledge
    
    def save_state(self, filepath: str):
        """Save the current state of the forgetting prevention system"""
        state = {
            "fisher_information": {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                 for k, v in self.fisher_information.items()},
            "optimal_params": self.optimal_params,
            "replay_buffer": list(self.replay_buffer),
            "previous_model_snapshots": self.previous_model_snapshots,
            "important_knowledge": self.important_knowledge
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        self.logger.info(f"Saved forgetting prevention state to {filepath}")
    
    def load_state(self, filepath: str):
        """Load the state of the forgetting prevention system"""
        if not os.path.exists(filepath):
            self.logger.warning(f"State file does not exist: {filepath}")
            return
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.fisher_information = {k: np.array(v) if isinstance(v, list) else v 
                                 for k, v in state.get("fisher_information", {}).items()}
        self.optimal_params = state.get("optimal_params", {})
        self.replay_buffer = deque(state.get("replay_buffer", []), 
                                 maxlen=self.replay_buffer.maxlen)
        self.previous_model_snapshots = state.get("previous_model_snapshots", [])
        self.important_knowledge = state.get("important_knowledge", {})
        
        self.logger.info(f"Loaded forgetting prevention state from {filepath}")


# Example usage
if __name__ == "__main__":
    # This would be run with proper configuration
    pass