"""
Continual Learning System - Integrates all continual learning components
"""
import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

from utils.config import Config
from utils.logger import setup_logger
from memory.memory_system import MemorySystem
from learning.learning_engine import LearningEngine
from evaluation.evaluation_system import EvaluationSystem
from decision.adaptive_decision_maker import AdaptiveDecisionMaker
from continual_learning.forgetting_prevention import ForgettingPrevention
from continual_learning.automatic_retrainer import AutomaticRetrainer
from continual_learning.evolutionary_adaptation import EvolutionaryAdaptor


class ContinualLearningSystem:
    """Main system that integrates all continual learning components"""
    
    def __init__(self, config: Config, memory_system: MemorySystem, 
                 learning_engine: LearningEngine, evaluation_system: EvaluationSystem,
                 decision_maker: AdaptiveDecisionMaker):
        self.config = config
        self.memory_system = memory_system
        self.learning_engine = learning_engine
        self.evaluation_system = evaluation_system
        self.decision_maker = decision_maker
        self.logger = setup_logger("continual_learning_system")
        
        # Initialize component systems
        self.forgetting_prevention = ForgettingPrevention(config, memory_system)
        self.automatic_retrainer = AutomaticRetrainer(config, memory_system, learning_engine)
        self.evolutionary_adaptor = EvolutionaryAdaptor(config, learning_engine.strategy_ranker, evaluation_system)
        
        # Continual learning parameters
        self.enabled = config.get("continual_learning.enabled", True)
        self.performance_monitoring = config.get("continual_learning.performance_monitoring", True)
        
        # Learning state tracking
        self.learning_cycles = 0
        self.experience_count = 0
        self.retraining_count = 0
        self.evolution_count = 0
        
        # Task categorization
        self.task_categories = {}
        self.category_performance = {}
        
    async def process_experience(self, task: str, result: Dict, performance_score: float, 
                               task_category: str = "general") -> Dict[str, Any]:
        """Process a new experience and apply continual learning techniques"""
        if not self.enabled:
            return {"status": "disabled", "message": "Continual learning disabled"}
        
        try:
            self.experience_count += 1
            
            # Categorize the task
            category = await self._categorize_task(task)
            
            # Preserve important knowledge from this experience
            await self.forgetting_prevention.preserve_knowledge(
                task_name=task,
                knowledge_elements=[{
                    "id": f"exp_{self.experience_count}",
                    "task": task,
                    "result": result,
                    "category": category,
                    "performance_score": performance_score,
                    "timestamp": datetime.now().isoformat()
                }]
            )
            
            # Add to replay buffer for experience replay
            experience = {
                "task": task,
                "result": result,
                "performance_score": performance_score,
                "category": category,
                "importance": performance_score  # Higher performance = higher importance
            }
            self.forgetting_prevention.add_to_replay_buffer(experience)
            
            # Monitor performance for this category
            if self.performance_monitoring:
                await self.automatic_retrainer.monitor_performance(category, performance_score)
            
            # Update task category tracking
            if category not in self.task_categories:
                self.task_categories[category] = 0
                self.category_performance[category] = []
            
            self.task_categories[category] += 1
            self.category_performance[category].append(performance_score)
            
            # Keep only recent performance data
            max_performance_records = 50
            if len(self.category_performance[category]) > max_performance_records:
                self.category_performance[category] = self.category_performance[category][-max_performance_records:]
            
            # Add task to evolution pool for strategy evolution
            await self.evolutionary_adaptor.add_tasks_for_evolution([task])
            
            # Log the experience processing
            self.logger.info(f"Processed experience #{self.experience_count} for category '{category}', score: {performance_score:.3f}")
            
            return {
                "status": "success",
                "experience_id": self.experience_count,
                "category": category,
                "performance_score": performance_score,
                "continual_learning_applied": True
            }
            
        except Exception as e:
            self.logger.error(f"Error processing experience: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def _categorize_task(self, task: str) -> str:
        """Categorize a task based on its content"""
        task_lower = task.lower()
        
        # Define category keywords
        categories = {
            "mathematical": ["calculate", "compute", "solve", "math", "equation", "formula", "number"],
            "analytical": ["analyze", "compare", "evaluate", "assess", "review", "study"],
            "creative": ["create", "design", "develop", "write", "compose", "generate"],
            "planning": ["plan", "schedule", "organize", "coordinate", "arrange"],
            "research": ["find", "search", "investigate", "lookup", "discover", "explore"],
            "technical": ["code", "program", "debug", "algorithm", "data", "system"]
        }
        
        # Find the best matching category
        for category, keywords in categories.items():
            if any(keyword in task_lower for keyword in keywords):
                return category
        
        # Default to general if no specific category matches
        return "general"
    
    async def trigger_retraining(self):
        """Manually trigger the retraining process"""
        if self.enabled:
            await self.automatic_retrainer.force_retraining()
            self.retraining_count += 1
            self.logger.info(f"Manual retraining triggered, total: {self.retraining_count}")
    
    async def trigger_evolution(self):
        """Manually trigger the evolutionary adaptation process"""
        if self.enabled:
            result = await self.evolutionary_adaptor.evolve_and_apply()
            if result["status"] == "completed":
                self.evolution_count += 1
                self.logger.info(f"Manual evolution completed, total: {self.evolution_count}")
            return result
    
    async def run_continual_learning_cycle(self):
        """Run a complete continual learning cycle"""
        try:
            self.learning_cycles += 1
            cycle_start = datetime.now()
            
            # Sample from replay buffer to prevent forgetting
            replay_samples = await self.forgetting_prevention.sample_from_replay(batch_size=10)
            
            # Apply experience replay
            if replay_samples:
                for sample in replay_samples:
                    # Re-process important past experiences
                    task = sample.get("task", "")
                    result = sample.get("result", {})
                    score = sample.get("performance_score", 0.5)
                    category = sample.get("category", "general")
                    
                    # Re-integrate the experience
                    await self.process_experience(task, result, score, category)
            
            # Decay knowledge importance over time
            await self.forgetting_prevention.decay_knowledge_importance()
            
            # Check if evolution should be triggered based on accumulated tasks
            if len(self.evolutionary_adaptor.task_pool) >= 10:  # Arbitrary threshold
                await self.trigger_evolution()
            
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            
            self.logger.info(f"Continual learning cycle {self.learning_cycles} completed in {cycle_duration:.2f}s")
            
            return {
                "cycle": self.learning_cycles,
                "duration": cycle_duration,
                "replay_samples_processed": len(replay_samples),
                "total_experiences": self.experience_count,
                "retraining_count": self.retraining_count,
                "evolution_count": self.evolution_count
            }
            
        except Exception as e:
            self.logger.error(f"Error in continual learning cycle: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def get_continual_learning_status(self) -> Dict[str, Any]:
        """Get comprehensive status of continual learning system"""
        return {
            "enabled": self.enabled,
            "learning_cycles": self.learning_cycles,
            "total_experiences": self.experience_count,
            "retraining_count": self.retraining_count,
            "evolution_count": self.evolution_count,
            "task_categories": self.task_categories,
            "category_performance": {
                cat: {
                    "count": count,
                    "avg_performance": sum(self.category_performance[cat]) / len(self.category_performance[cat]) if self.category_performance[cat] else 0.0,
                    "recent_performance": self.category_performance[cat][-5:] if self.category_performance[cat] else []
                }
                for cat, count in self.task_categories.items()
            },
            "forgetting_prevention_status": await self.forgetting_prevention.get_stable_knowledge(),
            "retraining_status": await self.automatic_retrainer.get_retraining_status(),
            "evolution_status": self.evolutionary_adaptor.get_evolution_status(),
            "timestamp": datetime.now().isoformat()
        }
    
    async def save_state(self, filepath: str):
        """Save the state of the continual learning system"""
        state = {
            "learning_cycles": self.learning_cycles,
            "experience_count": self.experience_count,
            "retraining_count": self.retraining_count,
            "evolution_count": self.evolution_count,
            "task_categories": self.task_categories,
            "category_performance": self.category_performance,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        # Save component states
        self.forgetting_prevention.save_state(f"{filepath}.fp_state")
        
        self.logger.info(f"Saved continual learning system state to {filepath}")
    
    async def load_state(self, filepath: str):
        """Load the state of the continual learning system"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.learning_cycles = state.get("learning_cycles", 0)
            self.experience_count = state.get("experience_count", 0)
            self.retraining_count = state.get("retraining_count", 0)
            self.evolution_count = state.get("evolution_count", 0)
            self.task_categories = state.get("task_categories", {})
            self.category_performance = state.get("category_performance", {})
            
            # Load component states
            self.forgetting_prevention.load_state(f"{filepath}.fp_state")
            
            self.logger.info(f"Loaded continual learning system state from {filepath}")
            
        except FileNotFoundError:
            self.logger.warning(f"State file not found: {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading state: {e}")
    
    async def reset_system(self):
        """Reset the continual learning system to initial state"""
        self.learning_cycles = 0
        self.experience_count = 0
        self.retraining_count = 0
        self.evolution_count = 0
        self.task_categories = {}
        self.category_performance = {}
        
        # Reset component systems
        await self.forgetting_prevention.__init__(self.config, self.memory_system)
        await self.automatic_retrainer.__init__(self.config, self.memory_system, self.learning_engine)
        await self.evolutionary_adaptor.__init__(self.config, self.learning_engine.strategy_ranker, self.evaluation_system)
        
        self.logger.info("Reset continual learning system to initial state")


# Example usage
if __name__ == "__main__":
    # This would be run with proper configuration
    pass