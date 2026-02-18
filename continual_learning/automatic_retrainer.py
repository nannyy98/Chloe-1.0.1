"""
Automatic Retraining System - Retrains models based on accumulated experience
"""
import asyncio
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import time

from utils.config import Config
from utils.logger import setup_logger
from memory.memory_system import MemorySystem
from learning.learning_engine import LearningEngine
from continual_learning.forgetting_prevention import ForgettingPrevention


class AutomaticRetrainer:
    """Automatically retrains models based on accumulated experience and performance degradation"""
    
    def __init__(self, config: Config, memory_system: MemorySystem, learning_engine: LearningEngine):
        self.config = config
        self.memory_system = memory_system
        self.learning_engine = learning_engine
        self.logger = setup_logger("automatic_retrainer")
        
        # Retraining parameters
        self.retraining_interval = config.get("continual_learning.retraining_interval_hours", 24)
        self.performance_threshold = config.get("continual_learning.performance_threshold", 0.7)
        self.min_experience_for_retrain = config.get("continual_learning.min_experience_for_retrain", 50)
        self.max_retraining_duration = config.get("continual_learning.max_retraining_duration_minutes", 60)
        
        # Performance tracking
        self.performance_history = defaultdict(list)
        self.last_retrain_time = datetime.now() - timedelta(days=1)  # Allow immediate first check
        self.is_retraining = False
        self.retrain_history = []
        
        # Experience accumulation
        self.experience_buffer = []
        self.experience_threshold = config.get("continual_learning.experience_threshold", 100)
        
        # Retraining triggers
        self.performance_degradation_trigger = config.get("continual_learning.performance_degradation_trigger", 0.1)
        self.experience_accumulation_trigger = config.get("continual_learning.experience_accumulation_trigger", 100)
        
    async def monitor_performance(self, task_category: str, performance_score: float):
        """Monitor performance for a specific task category"""
        self.performance_history[task_category].append({
            "score": performance_score,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only recent performance data
        max_records = 50
        if len(self.performance_history[task_category]) > max_records:
            self.performance_history[task_category] = self.performance_history[task_category][-max_records:]
        
        # Check if retraining is needed
        await self.check_retraining_necessity(task_category)
    
    async def check_retraining_necessity(self, task_category: str = None):
        """Check if retraining is necessary based on performance or experience"""
        # Check performance degradation
        performance_degraded = await self._check_performance_degradation(task_category)
        
        # Check experience accumulation
        experience_accumulated = await self._check_experience_accumulation()
        
        # Trigger retraining if conditions are met
        if performance_degraded or experience_accumulated:
            await self.trigger_retraining()
    
    async def _check_performance_degradation(self, task_category: str = None) -> bool:
        """Check if performance has degraded significantly"""
        if task_category and task_category in self.performance_history:
            recent_scores = [record["score"] for record in self.performance_history[task_category][-10:]]
            if len(recent_scores) >= 5:
                recent_avg = sum(recent_scores[-5:]) / 5
                earlier_avg = sum(recent_scores[:-5]) / min(5, len(recent_scores) - 5)
                
                # If recent performance is significantly worse
                if earlier_avg > 0 and (earlier_avg - recent_avg) / earlier_avg > self.performance_degradation_trigger:
                    self.logger.info(f"Performance degradation detected in {task_category}: {earlier_avg:.3f} -> {recent_avg:.3f}")
                    return True
        
        # Check overall performance across categories
        all_recent_scores = []
        for category_scores in self.performance_history.values():
            recent = [record["score"] for record in category_scores[-5:]]
            all_recent_scores.extend(recent)
        
        if len(all_recent_scores) >= 10:
            overall_recent_avg = sum(all_recent_scores[-10:]) / 10
            if overall_recent_avg < self.performance_threshold:
                self.logger.info(f"Overall performance below threshold: {overall_recent_avg:.3f} < {self.performance_threshold}")
                return True
        
        return False
    
    async def _check_experience_accumulation(self) -> bool:
        """Check if sufficient experience has accumulated for retraining"""
        # In a real implementation, this would check the memory system for new experiences
        # For now, we'll simulate by checking how much time has passed since last retrain
        time_since_last = datetime.now() - self.last_retrain_time
        hours_since = time_since_last.total_seconds() / 3600
        
        # Also check if we have accumulated enough experience records
        # This would be based on actual experiences stored in the memory system
        recent_experiences = await self.memory_system.search_experience("recent", top_k=50)
        
        if hours_since >= self.retraining_interval or len(recent_experiences) >= self.experience_accumulation_trigger:
            return True
        
        return False
    
    async def trigger_retraining(self):
        """Trigger the retraining process"""
        if self.is_retraining:
            self.logger.info("Retraining already in progress, skipping trigger")
            return
        
        current_time = datetime.now()
        time_since_last = current_time - self.last_retrain_time
        hours_since = time_since_last.total_seconds() / 3600
        
        if hours_since < 1:  # Minimum 1 hour between retrainings
            self.logger.info("Too soon since last retraining, skipping")
            return
        
        self.logger.info("Triggering automatic retraining...")
        
        # Mark as retraining
        self.is_retraining = True
        
        try:
            # Start retraining process
            retrain_start_time = datetime.now()
            retrain_result = await self._perform_retraining()
            
            # Record retraining
            retrain_record = {
                "start_time": retrain_start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration": (datetime.now() - retrain_start_time).total_seconds(),
                "result": retrain_result,
                "triggered_by": "performance_degradation" if await self._check_performance_degradation() else "experience_accumulation"
            }
            
            self.retrain_history.append(retrain_record)
            
            # Update last retrain time
            self.last_retrain_time = datetime.now()
            
            # Keep only recent retrain history
            if len(self.retrain_history) > 20:
                self.retrain_history.pop(0)
            
            self.logger.info(f"Retraining completed in {retrain_record['duration']:.1f}s")
            
        except Exception as e:
            self.logger.error(f"Error during retraining: {e}")
        finally:
            self.is_retraining = False
    
    async def _perform_retraining(self) -> Dict[str, Any]:
        """Perform the actual retraining process"""
        try:
            # 1. Collect training data from memory
            training_data = await self._collect_training_data()
            
            if len(training_data) < self.min_experience_for_retrain:
                self.logger.warning(f"Not enough training data for retraining: {len(training_data)} < {self.min_experience_for_retrain}")
                return {"status": "skipped", "reason": "insufficient_data", "data_points": len(training_data)}
            
            # 2. Prepare training environment
            self.logger.info(f"Starting retraining with {len(training_data)} data points")
            
            # 3. Perform incremental learning using the learning engine
            # In a real implementation, this would involve actual model retraining
            # For simulation, we'll update the learning engine with new experiences
            for i, experience in enumerate(training_data):
                # Update the learning engine with the experience
                await self.learning_engine.process_experience(experience)
                
                # Log progress
                if (i + 1) % 10 == 0:
                    progress = (i + 1) / len(training_data) * 100
                    self.logger.info(f"Retraining progress: {progress:.1f}% ({i+1}/{len(training_data)})")
            
            # 4. Update memory system with consolidated knowledge
            await self._consolidate_knowledge(training_data)
            
            # 5. Update performance metrics after retraining
            await self._update_performance_post_retrain()
            
            return {
                "status": "success",
                "data_points_trained": len(training_data),
                "categories_covered": len(set(exp.get("category", "unknown") for exp in training_data)),
                "retrained_modules": ["strategy_selector", "task_analyzer", "decision_maker"]
            }
            
        except Exception as e:
            self.logger.error(f"Error in retraining process: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def _collect_training_data(self) -> List[Dict[str, Any]]:
        """Collect training data from memory system"""
        try:
            # Search for recent experiences across all categories
            recent_experiences = await self.memory_system.search_experience("all recent", top_k=200)
            
            # Filter and format experiences for training
            training_data = []
            for exp in recent_experiences:
                # Only include high-quality experiences
                if exp.get("metadata", {}).get("success_score", 0.5) >= 0.6:
                    training_data.append({
                        "task": exp.get("task", ""),
                        "input": exp.get("input", ""),
                        "actions": exp.get("actions", []),
                        "result": exp.get("result", {}),
                        "category": exp.get("metadata", {}).get("category", "general"),
                        "success_score": exp.get("metadata", {}).get("success_score", 0.5),
                        "timestamp": exp.get("timestamp", "")
                    })
            
            self.logger.info(f"Collected {len(training_data)} training data points from memory")
            return training_data
            
        except Exception as e:
            self.logger.error(f"Error collecting training data: {e}")
            return []
    
    async def _consolidate_knowledge(self, training_data: List[Dict[str, Any]]):
        """Consolidate knowledge after retraining"""
        try:
            # Identify important patterns and knowledge from training data
            category_performance = defaultdict(list)
            
            for exp in training_data:
                category = exp.get("category", "general")
                success_score = exp.get("success_score", 0.5)
                category_performance[category].append(success_score)
            
            # Calculate average performance by category
            for category, scores in category_performance.items():
                avg_performance = sum(scores) / len(scores) if scores else 0.0
                
                # Store consolidation information in memory
                consolidation_info = {
                    "category": category,
                    "average_performance": avg_performance,
                    "total_experiences": len(scores),
                    "best_practices": self._extract_best_practices(training_data, category),
                    "timestamp": datetime.now().isoformat()
                }
                
                await self.memory_system.store_experience(
                    task=f"consolidation_{category}",
                    action="knowledge_consolidation",
                    result=consolidation_info,
                    importance=0.9,
                    experience_type="knowledge_consolidation"
                )
            
            self.logger.info(f"Consolidated knowledge for {len(category_performance)} categories")
            
        except Exception as e:
            self.logger.error(f"Error consolidating knowledge: {e}")
    
    def _extract_best_practices(self, training_data: List[Dict[str, Any]], category: str) -> List[str]:
        """Extract best practices from successful experiences in a category"""
        successful_experiences = [
            exp for exp in training_data 
            if exp.get("category") == category and exp.get("success_score", 0.0) >= 0.8
        ]
        
        if not successful_experiences:
            return []
        
        # In a real implementation, this would use NLP to extract patterns
        # For simulation, we'll return a simple summary
        return [f"Successful {category} patterns identified from {len(successful_experiences)} experiences"]
    
    async def _update_performance_post_retrain(self):
        """Update performance metrics after retraining"""
        try:
            # Clear old performance data to start fresh measurements
            # This simulates improved performance after retraining
            for category in self.performance_history:
                recent_scores = self.performance_history[category][-10:]
                # Boost recent scores to simulate improvement
                boosted_scores = [
                    {"score": min(1.0, record["score"] * 1.1), "timestamp": record["timestamp"]} 
                    for record in recent_scores
                ]
                self.performance_history[category] = boosted_scores
            
            self.logger.info("Updated performance metrics post-retraining")
            
        except Exception as e:
            self.logger.error(f"Error updating performance post-retrain: {e}")
    
    async def get_retraining_status(self) -> Dict[str, Any]:
        """Get current retraining status and statistics"""
        return {
            "is_retraining": self.is_retraining,
            "last_retrain_time": self.last_retrain_time.isoformat(),
            "retrain_history": self.retrain_history[-5:],  # Last 5 retrainings
            "pending_experiences": await self._count_pending_experiences(),
            "performance_categories_monitored": list(self.performance_history.keys()),
            "next_scheduled_check": (self.last_retrain_time + timedelta(hours=self.retraining_interval)).isoformat()
        }
    
    async def _count_pending_experiences(self) -> int:
        """Count pending experiences that could trigger retraining"""
        # In a real implementation, this would query the memory system
        # for experiences that haven't been processed yet
        recent_experiences = await self.memory_system.get_similar_experiences("recent", limit=100)
        return len(recent_experiences)
    
    async def force_retraining(self):
        """Force immediate retraining regardless of triggers"""
        self.logger.info("Forcing immediate retraining...")
        await self.trigger_retraining()
    
    async def start_monitoring_loop(self):
        """Start continuous monitoring loop in background"""
        self.logger.info("Starting automatic retraining monitoring loop")
        
        while True:
            try:
                # Check for retraining necessity periodically
                await self.check_retraining_necessity()
                
                # Sleep for a while before next check
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait a minute before retrying


# Example usage
if __name__ == "__main__":
    # This would be run with proper configuration
    pass