"""
GAIA Benchmark Integration - Integration with GAIA benchmark for evaluation
"""
import asyncio
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import aiohttp
import pandas as pd

from utils.config import Config
from utils.logger import setup_logger
from decision.adaptive_decision_maker import AdaptiveDecisionMaker
from rewards.reward_system import RewardCalculator


class GAIAEvaluator:
    """Integrates with GAIA benchmark for evaluation"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger("gaia_evaluator")
        self.decision_maker = AdaptiveDecisionMaker(config)
        self.reward_calculator = RewardCalculator(config)
        
        # GAIA benchmark configuration
        self.gaia_dataset_url = config.get("benchmark.gaia_dataset_url", 
                                         "https://github.com/gaia-benchmark/GAIA/raw/main/")
        self.results_directory = config.get("benchmark.results_directory", "./benchmark_results")
        
        # Ensure results directory exists
        os.makedirs(self.results_directory, exist_ok=True)
        
        # Track benchmark performance
        self.benchmark_results = []
        self.performance_history = []
        
        # Continuous learning components
        self.learning_from_benchmarks = True
        self.performance_trends = []
        self.improvement_strategies = []
    
    async def load_gaia_tasks(self, subset: str = "validation") -> List[Dict]:
        """Load GAIA benchmark tasks"""
        try:
            # For now, we'll create sample tasks based on GAIA characteristics
            # In a real implementation, this would download from the actual GAIA dataset
            sample_tasks = [
                {
                    "task_id": "gaia_001",
                    "question": "Calculate the compound interest for $1000 invested at 5% annual rate for 3 years",
                    "reference_answer": "1157.63",
                    "category": "numerical_calculation",
                    "difficulty": "easy"
                },
                {
                    "task_id": "gaia_002", 
                    "question": "Analyze the pros and cons of renewable energy adoption in developing countries",
                    "reference_answer": "Comprehensive analysis with economic, social, and environmental considerations",
                    "category": "analytical_reasoning",
                    "difficulty": "medium"
                },
                {
                    "task_id": "gaia_003",
                    "question": "Plan a 5-day itinerary for a business trip to Tokyo including meetings and networking events",
                    "reference_answer": "Detailed schedule with time allocations and venue recommendations",
                    "category": "planning",
                    "difficulty": "hard"
                },
                {
                    "task_id": "gaia_004",
                    "question": "Compare the theoretical foundations of quantum computing versus classical computing",
                    "reference_answer": "Technical comparison covering computational models, algorithms, and applications",
                    "category": "technical_analysis",
                    "difficulty": "hard"
                },
                {
                    "task_id": "gaia_005",
                    "question": "Summarize the key findings from the latest IPCC climate report",
                    "reference_answer": "Summary covering main conclusions about climate change impacts and mitigation strategies",
                    "category": "information_synthesis",
                    "difficulty": "medium"
                }
            ]
            
            self.logger.info(f"Loaded {len(sample_tasks)} sample GAIA tasks for {subset} subset")
            return sample_tasks
            
        except Exception as e:
            self.logger.error(f"Error loading GAIA tasks: {e}")
            return self._get_fallback_tasks()
    
    async def run_gaia_evaluation(self, num_tasks: int = None, subset: str = "validation") -> Dict[str, Any]:
        """Run GAIA benchmark evaluation"""
        try:
            tasks = await self.load_gaia_tasks(subset)
            
            if num_tasks:
                tasks = tasks[:num_tasks]
            
            results = []
            
            for i, task in enumerate(tasks):
                self.logger.info(f"Processing GAIA task {i+1}/{len(tasks)}: {task['task_id']}")
                
                # Make adaptive decision about approach
                decision = await self.decision_maker.make_decision(
                    task=task["question"], 
                    context={"task_category": task["category"], "difficulty": task["difficulty"]}
                )
                
                # Execute the decision
                execution_result = await self.decision_maker.execute_decision(decision)
                
                # Calculate reward
                reward = await self.reward_calculator.calculate_reward(
                    task=task["question"],
                    result=execution_result["execution_result"],
                    expected_outcome=task["reference_answer"],
                    execution_time=execution_result.get("execution_time"),
                    strategy_used=decision["decision"]["selected_strategy"]
                )
                
                # Evaluate against reference answer
                evaluation = await self._evaluate_against_reference(
                    task["question"], 
                    execution_result["execution_result"], 
                    task["reference_answer"]
                )
                
                # Record result
                result_record = {
                    "task_id": task["task_id"],
                    "task": task["question"],
                    "category": task["category"],
                    "difficulty": task["difficulty"],
                    "reference_answer": task["reference_answer"],
                    "agent_response": execution_result["execution_result"],
                    "decision": decision["decision"],
                    "reward": reward,
                    "evaluation": evaluation,
                    "execution_time": execution_result.get("execution_time"),
                    "timestamp": datetime.now().isoformat()
                }
                
                results.append(result_record)
                
                # Learn from the result
                await self.decision_maker.learn_from_decision(
                    task=task["question"],
                    decision=decision["decision"],
                    execution_result=execution_result["execution_result"],
                    reward=reward["reward_score"]
                )
                
                self.logger.info(f"Completed task {task['task_id']}, Score: {evaluation['score']:.3f}")
            
            # Aggregate results
            aggregate_results = await self._aggregate_results(results)
            
            # Store results
            self.benchmark_results.extend(results)
            self.performance_history.append({
                "evaluation_run": datetime.now().isoformat(),
                "results_count": len(results),
                "aggregate_results": aggregate_results
            })
            
            # Save results to file
            await self._save_results(results, aggregate_results)
            
            return {
                "individual_results": results,
                "aggregate_results": aggregate_results,
                "total_tasks": len(results),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error running GAIA evaluation: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def _evaluate_against_reference(self, task: str, result: Dict, reference: str) -> Dict[str, Any]:
        """Evaluate agent response against reference answer"""
        try:
            evaluation_prompt = f"""
            Evaluate the agent's response against the reference answer for this task:
            
            Task: {task}
            Reference Answer: {reference}
            Agent Response: {result.get('result', result.get('final_answer', str(result)))}
            
            Provide evaluation in JSON format:
            {{
                "score": 0.0-1.0,
                "correctness": 0.0-1.0,
                "completeness": 0.0-1.0,
                "relevance": 0.0-1.0,
                "feedback": "Brief feedback on the response quality"
            }}
            """
            
            from agents.ollama_agent import OllamaAgent
            ollama_agent = OllamaAgent(self.config)
            
            evaluation_result = await ollama_agent.execute({
                "prompt": evaluation_prompt,
                "temperature": 0.2,
                "max_tokens": 500
            })
            
            if evaluation_result["status"] == "success":
                try:
                    return json.loads(evaluation_result["result"])
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            self.logger.warning(f"Evaluation against reference failed: {e}")
        
        # Fallback evaluation
        result_str = str(result.get("result", result.get("final_answer", str(result))))
        reference_str = str(reference)
        
        # Simple similarity check
        if result_str and reference_str:
            # Calculate rough similarity based on common words
            result_words = set(result_str.lower().split())
            reference_words = set(reference_str.lower().split())
            intersection = result_words.intersection(reference_words)
            union = result_words.union(reference_words)
            
            jaccard_similarity = len(intersection) / len(union) if union else 0.0
        else:
            jaccard_similarity = 0.0
        
        return {
            "score": jaccard_similarity,
            "correctness": jaccard_similarity,
            "completeness": jaccard_similarity,
            "relevance": jaccard_similarity,
            "feedback": "Fallback evaluation based on text similarity"
        }
    
    async def _aggregate_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate evaluation results"""
        if not results:
            return {
                "total_tasks": 0,
                "average_score": 0.0,
                "success_rate": 0.0,
                "average_reward": 0.0,
                "category_performance": {},
                "difficulty_performance": {},
                "execution_time_stats": {}
            }
        
        # Calculate overall metrics
        total_tasks = len(results)
        average_score = sum(r["evaluation"]["score"] for r in results) / len(results)
        success_rate = sum(1 for r in results if r["evaluation"]["score"] >= 0.7) / len(results)
        average_reward = sum(r["reward"]["reward_score"] for r in results) / len(results)
        
        # Calculate category-wise performance
        category_performance = {}
        difficulty_performance = {}
        
        for result in results:
            category = result["category"]
            difficulty = result["difficulty"]
            score = result["evaluation"]["score"]
            
            # Category performance
            if category not in category_performance:
                category_performance[category] = {"scores": [], "count": 0, "average": 0.0}
            category_performance[category]["scores"].append(score)
            category_performance[category]["count"] += 1
            
            # Difficulty performance
            if difficulty not in difficulty_performance:
                difficulty_performance[difficulty] = {"scores": [], "count": 0, "average": 0.0}
            difficulty_performance[difficulty]["scores"].append(score)
            difficulty_performance[difficulty]["count"] += 1
        
        # Calculate averages
        for cat_data in category_performance.values():
            cat_data["average"] = sum(cat_data["scores"]) / len(cat_data["scores"])
            del cat_data["scores"]  # Remove raw scores to keep result clean
        
        for diff_data in difficulty_performance.values():
            diff_data["average"] = sum(diff_data["scores"]) / len(diff_data["scores"])
            del diff_data["scores"]  # Remove raw scores to keep result clean
        
        # Execution time statistics
        execution_times = [r["execution_time"] for r in results if r["execution_time"] is not None]
        if execution_times:
            execution_time_stats = {
                "average": sum(execution_times) / len(execution_times),
                "min": min(execution_times),
                "max": max(execution_times),
                "std_dev": (sum((x - sum(execution_times)/len(execution_times))**2 for x in execution_times) / len(execution_times))**0.5 if execution_times else 0
            }
        else:
            execution_time_stats = {"average": 0, "min": 0, "max": 0, "std_dev": 0}
        
        return {
            "total_tasks": total_tasks,
            "average_score": average_score,
            "success_rate": success_rate,
            "average_reward": average_reward,
            "category_performance": category_performance,
            "difficulty_performance": difficulty_performance,
            "execution_time_stats": execution_time_stats,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _save_results(self, individual_results: List[Dict], aggregate_results: Dict[str, Any]):
        """Save benchmark results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save individual results
        individual_filename = os.path.join(self.results_directory, f"gaia_individual_{timestamp}.json")
        with open(individual_filename, 'w', encoding='utf-8') as f:
            json.dump(individual_results, f, indent=2, ensure_ascii=False)
        
        # Save aggregate results
        aggregate_filename = os.path.join(self.results_directory, f"gaia_aggregate_{timestamp}.json")
        with open(aggregate_filename, 'w', encoding='utf-8') as f:
            json.dump(aggregate_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved benchmark results to {self.results_directory}/")
    
    def _get_fallback_tasks(self) -> List[Dict]:
        """Get fallback tasks when GAIA loading fails"""
        return [
            {
                "task_id": "fallback_001",
                "question": "Calculate 2+2",
                "reference_answer": "4",
                "category": "basic_math",
                "difficulty": "easy"
            }
        ]
    
    def get_benchmark_performance(self) -> Dict[str, Any]:
        """Get benchmark performance metrics"""
        if not self.performance_history:
            return {
                "total_evaluations": 0,
                "latest_results": None,
                "performance_trend": "insufficient_data",
                "best_performance": 0.0
            }
        
        latest_run = self.performance_history[-1]["aggregate_results"]
        best_performance = max(
            (run["aggregate_results"]["average_score"] for run in self.performance_history),
            default=0.0
        )
        
        # Calculate trend if we have multiple runs
        if len(self.performance_history) >= 2:
            previous_score = self.performance_history[-2]["aggregate_results"]["average_score"]
            current_score = latest_run["average_score"]
            
            if current_score > previous_score:
                trend = "improving"
            elif current_score < previous_score:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "initial"
        
        return {
            "total_evaluations": len(self.performance_history),
            "latest_results": latest_run,
            "performance_trend": trend,
            "best_performance": best_performance,
            "all_evaluations": len(self.benchmark_results)
        }
    
    async def run_continuous_evaluation(self, interval_minutes: int = 60, num_tasks: int = 5):
        """Run continuous evaluation at specified intervals"""
        self.logger.info(f"Starting continuous GAIA evaluation (interval: {interval_minutes} minutes, tasks: {num_tasks})")
        
        while True:
            try:
                self.logger.info("Starting scheduled GAIA evaluation run...")
                result = await self.run_gaia_evaluation(num_tasks=num_tasks)
                
                if result.get("status") == "failed":
                    self.logger.error(f"Continuous evaluation failed: {result.get('error')}")
                else:
                    perf = self.get_benchmark_performance()
                    self.logger.info(f"Continuous evaluation completed. Avg score: {perf['latest_results']['average_score']:.3f}")
                    
                    # Apply continuous learning from results
                    if self.learning_from_benchmarks:
                        await self._apply_learning_from_results(result)
                
                # Wait for the specified interval
                await asyncio.sleep(interval_minutes * 60)
                
            except asyncio.CancelledError:
                self.logger.info("Continuous evaluation cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in continuous evaluation: {e}")
                await asyncio.sleep(interval_minutes * 60)  # Continue despite error

    async def _apply_learning_from_results(self, evaluation_results: Dict):
        """Apply learning from benchmark results to improve performance"""
        try:
            # Analyze performance trends
            current_results = evaluation_results.get("individual_results", [])
            if not current_results:
                return
            
            # Calculate performance metrics
            avg_score = sum(r["evaluation"]["score"] for r in current_results) / len(current_results)
            
            # Update performance trends
            self.performance_trends.append({
                "timestamp": datetime.now().isoformat(),
                "average_score": avg_score,
                "total_tasks": len(current_results),
                "success_rate": sum(1 for r in current_results if r["evaluation"]["score"] >= 0.7) / len(current_results)
            })
            
            # Keep only recent trends
            if len(self.performance_trends) > 100:
                self.performance_trends.pop(0)
            
            # Identify improvement strategies based on results
            improvement_opportunities = await self._identify_improvement_opportunities(current_results)
            
            # Apply improvements to the system
            await self._apply_improvements(improvement_opportunities)
            
            self.logger.info(f"Applied learning from {len(current_results)} benchmark results. Avg score: {avg_score:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error applying learning from results: {e}")

    async def _identify_improvement_opportunities(self, results: List[Dict]) -> List[Dict]:
        """Identify improvement opportunities from benchmark results"""
        try:
            # Analyze results by category
            category_analysis = {}
            for result in results:
                category = result["category"]
                score = result["evaluation"]["score"]
                
                if category not in category_analysis:
                    category_analysis[category] = {"scores": [], "count": 0, "avg_score": 0.0}
                category_analysis[category]["scores"].append(score)
                category_analysis[category]["count"] += 1
            
            # Calculate averages
            for cat_data in category_analysis.values():
                cat_data["avg_score"] = sum(cat_data["scores"]) / len(cat_data["scores"])
                del cat_data["scores"]  # Clean up
            
            # Identify low-performing categories
            low_performers = [
                cat for cat, data in category_analysis.items() 
                if data["avg_score"] < 0.7  # Below threshold
            ]
            
            # Analyze strategy effectiveness
            strategy_analysis = {}
            for result in results:
                strategy = result["decision"]["selected_strategy"]
                score = result["evaluation"]["score"]
                
                if strategy not in strategy_analysis:
                    strategy_analysis[strategy] = {"scores": [], "count": 0, "avg_score": 0.0}
                strategy_analysis[strategy]["scores"].append(score)
                strategy_analysis[strategy]["count"] += 1
            
            # Calculate strategy averages
            for strat_data in strategy_analysis.values():
                strat_data["avg_score"] = sum(strat_data["scores"]) / len(strat_data["scores"])
                del strat_data["scores"]  # Clean up
            
            # Identify underperforming strategies
            weak_strategies = [
                strat for strat, data in strategy_analysis.items()
                if data["count"] > 2 and data["avg_score"] < 0.6  # At least 3 uses and poor performance
            ]
            
            opportunities = [
                {
                    "type": "category_performance",
                    "target": low_performers,
                    "analysis": category_analysis,
                    "priority": "high" if low_performers else "low"
                },
                {
                    "type": "strategy_optimization",
                    "target": weak_strategies,
                    "analysis": strategy_analysis,
                    "priority": "medium" if weak_strategies else "low"
                }
            ]
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error identifying improvement opportunities: {e}")
            return []

    async def _apply_improvements(self, opportunities: List[Dict]):
        """Apply identified improvements to the system"""
        try:
            for opportunity in opportunities:
                opp_type = opportunity["type"]
                targets = opportunity["target"]
                priority = opportunity["priority"]
                
                if opp_type == "category_performance" and priority == "high":
                    # Adjust decision-making for low-performing categories
                    for category in targets:
                        await self._adjust_category_strategy(category)
                        
                elif opp_type == "strategy_optimization" and priority == "medium":
                    # Reduce usage of weak strategies
                    for strategy in targets:
                        await self._adjust_strategy_weights(strategy, adjustment=-0.1)
            
            self.logger.info(f"Applied improvements based on {len(opportunities)} opportunities")
            
        except Exception as e:
            self.logger.error(f"Error applying improvements: {e}")

    async def _adjust_category_strategy(self, category: str):
        """Adjust strategy selection for a specific category"""
        try:
            # This would involve fine-tuning the decision maker for specific categories
            # For now, we'll log the adjustment
            self.logger.info(f"Adjusting strategy selection for category: {category}")
            
            # In a real implementation, this might involve:
            # - Updating policy weights for category-specific decisions
            # - Adjusting reward calculations for category
            # - Modifying strategy rankings for category
            
        except Exception as e:
            self.logger.error(f"Error adjusting category strategy: {e}")

    async def _adjust_strategy_weights(self, strategy: str, adjustment: float):
        """Adjust weights for a specific strategy"""
        try:
            # Update policy optimizer weights
            if hasattr(self.decision_maker.policy_optimizer, 'policy_weights'):
                current_weight = self.decision_maker.policy_optimizer.policy_weights.get(strategy, 1.0)
                new_weight = max(0.1, current_weight + adjustment)  # Ensure minimum weight
                self.decision_maker.policy_optimizer.policy_weights[strategy] = new_weight
                
                self.logger.info(f"Adjusted weight for strategy {strategy}: {current_weight:.2f} -> {new_weight:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error adjusting strategy weights: {e}")

    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights about learning from benchmark results"""
        return {
            "performance_trends": self.performance_trends[-10:],  # Last 10 trends
            "improvement_opportunities": len(self.performance_trends),
            "applied_improvements": len(self.improvement_strategies),
            "learning_enabled": self.learning_from_benchmarks,
            "total_learning_cycles": len(self.performance_trends)
        }

# Example usage
if __name__ == "__main__":
    # This would be run with proper configuration
    pass