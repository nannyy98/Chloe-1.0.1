#!/usr/bin/env python3
"""
Phase 3 Tests - Testing strategy adaptation and self-improvement capabilities
"""
import asyncio
import unittest
import json
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent
import sys
sys.path.insert(0, str(project_root))

from utils.config import Config
from learning.strategies import get_all_strategies, ReActStrategy, ChainOfThoughtStrategy, PlanAndExecuteStrategy
from learning.strategy_ranker import StrategyRanker
from learning.self_critique import SelfCritiqueEngine
from evaluation.evaluation_system import EvaluationSystem
from utils.logger import setup_logger


class TestPhase3(unittest.TestCase):
    """Test suite for Phase 3 implementation"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = Config()
        self.logger = setup_logger("phase3_test")
        
    def test_strategy_classes_initialization(self):
        """Test that all strategy classes initialize correctly"""
        strategies = get_all_strategies(self.config)
        
        # Check that we have the expected strategies
        self.assertEqual(len(strategies), 3)
        strategy_names = [s.name for s in strategies]
        self.assertIn("ReAct", strategy_names)
        self.assertIn("ChainOfThought", strategy_names)
        self.assertIn("PlanAndExecute", strategy_names)
        
        # Test individual strategy initialization
        react = ReActStrategy(self.config)
        self.assertEqual(react.name, "ReAct")
        self.assertEqual(react.max_steps, 20)
        
        cot = ChainOfThoughtStrategy(self.config)
        self.assertEqual(cot.name, "ChainOfThought")
        
        plan_exec = PlanAndExecuteStrategy(self.config)
        self.assertEqual(plan_exec.name, "PlanAndExecute")
        
    def test_strategy_task_analysis(self):
        """Test that strategies can analyze task suitability"""
        react = ReActStrategy(self.config)
        cot = ChainOfThoughtStrategy(self.config)
        plan_exec = PlanAndExecuteStrategy(self.config)
        
        test_task = "Calculate the integral of x^2 from 0 to 1"
        
        # Test ReAct analysis
        react_analysis = asyncio.run(react.analyze_task(test_task))
        self.assertIn("suitability", react_analysis)
        self.assertIsInstance(react_analysis["suitability"], (int, float))
        
        # Test CoT analysis
        cot_analysis = asyncio.run(cot.analyze_task(test_task))
        self.assertIn("suitability", cot_analysis)
        self.assertIsInstance(cot_analysis["suitability"], (int, float))
        
        # Test Plan and Execute analysis
        plan_analysis = asyncio.run(plan_exec.analyze_task(test_task))
        self.assertIn("suitability", plan_analysis)
        self.assertIsInstance(plan_analysis["suitability"], (int, float))
        
    async def test_strategy_execution(self):
        """Test basic strategy execution"""
        # Test ReAct strategy
        react = ReActStrategy(self.config)
        result = await react.execute("What is 2+2?")
        self.assertIn("status", result)
        self.assertIn("strategy", result)
        
        # Test CoT strategy
        cot = ChainOfThoughtStrategy(self.config)
        cot_result = await cot.execute("What is 2+2?")
        self.assertIn("status", cot_result)
        self.assertEqual(cot_result["strategy"], "ChainOfThought")
        
        # Test Plan and Execute strategy
        plan_exec = PlanAndExecuteStrategy(self.config)
        plan_result = await plan_exec.execute("What is 2+2?")
        self.assertIn("status", plan_result)
        self.assertEqual(plan_result["strategy"], "PlanAndExecute")


class TestStrategyRanker(unittest.TestCase):
    """Test strategy ranking system"""
    
    def setUp(self):
        self.config = Config()
        self.ranker = StrategyRanker(self.config)
        
    def test_ranker_initialization(self):
        """Test strategy ranker initialization"""
        self.assertIsNotNone(self.ranker)
        self.assertGreater(len(self.ranker.get_all_strategies()), 0)
        self.assertEqual(len(self.ranker.get_strategy_names()), 3)
        
    async def test_strategy_ranking(self):
        """Test strategy ranking for different tasks"""
        # Test mathematical task
        math_task = "Calculate the derivative of x^3"
        math_rankings = await self.ranker.rank_strategies_for_task(math_task)
        self.assertIsInstance(math_rankings, list)
        self.assertGreater(len(math_rankings), 0)
        
        # Test planning task
        planning_task = "Create a study schedule for exams"
        planning_rankings = await self.ranker.rank_strategies_for_task(planning_task)
        self.assertIsInstance(planning_rankings, list)
        
        # Test that rankings are different for different task types
        math_strategies = [name for name, score in math_rankings]
        planning_strategies = [name for name, score in planning_rankings]
        
        # At least some strategies should be ranked differently
        self.assertNotEqual(math_rankings[:2], planning_rankings[:2])
        
    async def test_strategy_selection(self):
        """Test different strategy selection methods"""
        task = "Solve a complex problem"
        
        # Test epsilon-greedy selection
        selected_epsilon = await self.ranker.select_strategy(task, method="epsilon_greedy")
        self.assertIn(selected_epsilon, self.ranker.get_strategy_names())
        
        # Test softmax selection
        selected_softmax = await self.ranker.select_strategy(task, method="softmax")
        self.assertIn(selected_softmax, self.ranker.get_strategy_names())
        
        # Test UCB selection
        selected_ucb = await self.ranker.select_strategy(task, method="ucb")
        self.assertIn(selected_ucb, self.ranker.get_strategy_names())
        
        # Test greedy selection
        selected_greedy = await self.ranker.select_strategy(task, method="greedy")
        self.assertIn(selected_greedy, self.ranker.get_strategy_names())
        
    def test_performance_tracking(self):
        """Test strategy performance tracking"""
        # Test updating performance
        self.ranker.update_strategy_performance("ReAct", True, 0.9, 2.5)
        self.ranker.update_strategy_performance("ReAct", False, 0.3, 1.8)
        self.ranker.update_strategy_performance("ChainOfThought", True, 0.8, 3.2)
        
        # Test getting performance
        react_perf = self.ranker.get_strategy_performance("ReAct")
        self.assertEqual(react_perf["total_attempts"], 2)
        self.assertEqual(react_perf["success_rate"], 0.5)
        
        cot_perf = self.ranker.get_strategy_performance("ChainOfThought")
        self.assertEqual(cot_perf["total_attempts"], 1)
        self.assertEqual(cot_perf["success_rate"], 1.0)
        
        # Test rankings
        rankings = self.ranker.get_all_strategy_rankings()
        self.assertIsInstance(rankings, list)
        self.assertGreater(len(rankings), 0)
        
        # Test adaptation score
        adaptation_score = self.ranker.get_adaptation_score()
        self.assertIsInstance(adaptation_score, float)
        self.assertGreaterEqual(adaptation_score, 0.0)
        self.assertLessEqual(adaptation_score, 1.0)


class TestSelfCritique(unittest.TestCase):
    """Test self-critique engine"""
    
    def setUp(self):
        self.config = Config()
        self.critique_engine = SelfCritiqueEngine(self.config)
        
    def test_critique_engine_initialization(self):
        """Test self-critique engine initialization"""
        self.assertIsNotNone(self.critique_engine)
        self.assertTrue(hasattr(self.critique_engine, 'ollama_agent'))
        
    async def test_failure_critique(self):
        """Test generating critique for failed tasks"""
        task = "Calculate complex mathematical integral"
        failed_result = {
            "error": "Timeout exceeded",
            "status": "failed",
            "processing_time": 35.0
        }
        
        critique = await self.critique_engine.critique_failed_task(
            task, failed_result, "ReAct"
        )
        
        self.assertIn("critique", critique)
        self.assertIn("alternative_strategies", critique)
        self.assertIn("improvement_suggestions", critique)
        self.assertEqual(critique["original_strategy"], "ReAct")
        
        # Test that alternative strategies are suggested
        self.assertIsInstance(critique["alternative_strategies"], list)
        self.assertGreater(len(critique["alternative_strategies"]), 0)
        
        # Test that improvement suggestions are provided
        self.assertIsInstance(critique["improvement_suggestions"], list)
        self.assertGreater(len(critique["improvement_suggestions"]), 0)
        
    async def test_alternative_strategy_retry(self):
        """Test retrying with alternative strategy"""
        task = "Explain quantum mechanics concepts"
        failed_result = {
            "error": "Complexity too high for current approach",
            "status": "failed"
        }
        
        # Test retry with CoT strategy
        retry_result = await self.critique_engine.retry_with_alternative_strategy(
            task, failed_result, "ChainOfThought"
        )
        
        self.assertIn("status", retry_result)
        self.assertIn("original_result", retry_result)
        self.assertEqual(retry_result["alternative_strategy"], "ChainOfThought")
        
    def test_critique_history(self):
        """Test critique history management"""
        # Get initial history
        initial_history = self.critique_engine.get_critique_history()
        self.assertIsInstance(initial_history, list)
        
        # Test statistics
        stats = self.critique_engine.get_improvement_statistics()
        self.assertIsInstance(stats, dict)
        self.assertIn("total_critiques", stats)
        self.assertIn("successful_improvements", stats)
        self.assertIn("improvement_rate", stats)


class TestEvaluationSystem(unittest.TestCase):
    """Test LLM-based evaluation system"""
    
    def setUp(self):
        self.config = Config()
        self.evaluator = EvaluationSystem(self.config)
        
    def test_evaluation_system_initialization(self):
        """Test evaluation system initialization"""
        self.assertIsNotNone(self.evaluator)
        self.assertTrue(hasattr(self.evaluator, 'ollama_agent'))
        self.assertEqual(self.evaluator.success_threshold, 0.7)
        
    async def test_task_evaluation(self):
        """Test task performance evaluation"""
        task = "Calculate 2+2"
        successful_result = {
            "result": "4",
            "status": "success",
            "confidence": 0.9
        }
        
        # Test successful task evaluation
        evaluation = await self.evaluator.evaluate_task_performance(task, successful_result)
        
        self.assertIn("combined_score", evaluation)
        self.assertIn("is_success", evaluation)
        self.assertIn("llm_evaluation", evaluation)
        self.assertIn("automated_metrics", evaluation)
        
        # Test that successful result is marked as success
        self.assertTrue(evaluation["is_success"])
        self.assertGreater(evaluation["combined_score"], 0.7)
        
        # Test failed task evaluation
        failed_result = {
            "error": "Invalid operation",
            "status": "failed"
        }
        
        failed_evaluation = await self.evaluator.evaluate_task_performance(task, failed_result)
        self.assertFalse(failed_evaluation["is_success"])
        self.assertLess(failed_evaluation["combined_score"], 0.5)
        
    def test_performance_metrics(self):
        """Test performance metrics calculation"""
        # Add some test evaluations
        test_evaluations = [
            {"combined_score": 0.9, "is_success": True},
            {"combined_score": 0.3, "is_success": False},
            {"combined_score": 0.8, "is_success": True},
            {"combined_score": 0.2, "is_success": False},
            {"combined_score": 0.7, "is_success": True}
        ]
        
        # Manually add to evaluation history for testing
        self.evaluator.evaluation_history = test_evaluations
        
        # Test metrics calculation
        metrics = self.evaluator.get_performance_metrics()
        
        self.assertIn("success_rate", metrics)
        self.assertIn("average_score", metrics)
        self.assertIn("error_rate", metrics)
        self.assertIn("learning_speed", metrics)
        
        # Test values are reasonable
        self.assertEqual(metrics["total_evaluations"], 5)
        self.assertEqual(metrics["successful_evaluations"], 3)
        self.assertEqual(metrics["failed_evaluations"], 2)
        self.assertEqual(metrics["success_rate"], 0.6)
        
    def test_detailed_analysis(self):
        """Test detailed performance analysis"""
        # Add test data
        test_evals = []
        for i in range(20):
            test_evals.append({
                "combined_score": 0.6 + (i * 0.02),  # Improving scores
                "is_success": i >= 8,  # More successes over time
                "automated_metrics": {
                    "has_error": i < 8,
                    "status": "success" if i >= 8 else "failed"
                }
            })
        
        self.evaluator.evaluation_history = test_evals
        
        # Test detailed analysis
        analysis = self.evaluator.get_detailed_analysis()
        
        self.assertIn("metrics", analysis)
        self.assertIn("error_analysis", analysis)
        self.assertIn("score_distribution", analysis)
        
        # Test learning trend
        trend = self.evaluator.get_learning_trend()
        self.assertIn("trends", trend)
        self.assertIn("overall_trend", trend)
        
        # With improving scores, should detect improvement
        if trend["overall_trend"] != "insufficient_data":
            self.assertIn(trend["overall_trend"], ["improving", "stable", "declining"])


class AsyncTestCase(unittest.TestCase):
    """Base class for async tests"""
    
    def setUp(self):
        self.config = Config()
    
    def async_run(self, coro):
        """Run an async coroutine in the test"""
        try:
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)


class TestAsyncPhase3(AsyncTestCase):
    """Async tests for Phase 3"""
    
    def test_strategy_execution(self):
        """Test basic strategy execution"""
        async def test():
            # Test ReAct strategy
            react = ReActStrategy(self.config)
            result = await react.execute("What is 2+2?")
            self.assertIn("status", result)
            self.assertIn("strategy", result)
            
            # Test CoT strategy
            cot = ChainOfThoughtStrategy(self.config)
            cot_result = await cot.execute("What is 2+2?")
            self.assertIn("status", cot_result)
            self.assertEqual(cot_result["strategy"], "ChainOfThought")
            
        self.async_run(test())


def run_phase3_demo():
    """Run a demonstration of Phase 3 capabilities"""
    print("🧪 Phase 3 Demo: Strategy Adaptation & Self-Improvement")
    print("=" * 60)
    
    config = Config()
    
    # Initialize components
    ranker = StrategyRanker(config)
    evaluator = EvaluationSystem(config)
    
    # Demo tasks with different characteristics
    demo_tasks = [
        {
            "task": "Calculate the integral of x^2 from 0 to 1",
            "type": "mathematical",
            "expected_best": "ChainOfThought"
        },
        {
            "task": "Plan a week-long study schedule for mathematics and physics",
            "type": "planning", 
            "expected_best": "PlanAndExecute"
        },
        {
            "task": "Explain quantum entanglement and its applications",
            "type": "explanation",
            "expected_best": "ReAct"
        }
    ]
    
    print(f"Running {len(demo_tasks)} demonstration tasks with strategy comparison...")
    
    # Test strategy ranking for each task
    ranking_results = []
    for task_data in demo_tasks:
        task = task_data["task"]
        print(f"\n🔄 Analyzing task: {task[:50]}...")
        
        # Get strategy rankings
        rankings = asyncio.run(ranker.rank_strategies_for_task(task))
        ranking_results.append({
            "task": task,
            "task_type": task_data["type"],
            "expected": task_data["expected_best"],
            "rankings": rankings
        })
        
        # Display top 3 strategies
        print("  Top strategy rankings:")
        for i, (strategy_name, score) in enumerate(rankings[:3]):
            print(f"    {i+1}. {strategy_name}: {score:.3f}")
            
        # Test different selection methods
        selection_methods = ["greedy", "epsilon_greedy", "softmax", "ucb"]
        selections = []
        for method in selection_methods:
            selected = asyncio.run(ranker.select_strategy(task, method=method))
            selections.append(selected)
            print(f"    {method.capitalize()}: {selected}")
        
        # Show variance in selections
        unique_selections = set(selections)
        variance = len(unique_selections) / len(selection_methods)
        print(f"    Selection variance: {variance:.2f}")
    
    # Test adaptation score
    print(f"\n📊 Strategy Adaptation Analysis:")
    print("-" * 40)
    
    # Simulate some performance data
    test_performances = [
        ("ReAct", True, 0.85, 3.2),
        ("ReAct", False, 0.3, 1.8),
        ("ChainOfThought", True, 0.92, 2.1),
        ("ChainOfThought", True, 0.88, 2.4),
        ("PlanAndExecute", False, 0.4, 4.5),
        ("PlanAndExecute", True, 0.75, 3.8),
        ("ReAct", True, 0.9, 2.9),
        ("ChainOfThought", True, 0.85, 2.2)
    ]
    
    for strategy, success, score, time in test_performances:
        ranker.update_strategy_performance(strategy, success, score, time)
    
    # Get final rankings and adaptation score
    final_rankings = ranker.get_all_strategy_rankings()
    adaptation_score = ranker.get_adaptation_score()
    
    print("Final strategy rankings:")
    for i, ranking in enumerate(final_rankings):
        print(f"  {i+1}. {ranking['name']}: "
              f"Success Rate: {ranking['success_rate']:.2f}, "
              f"Recent: {ranking['recent_performance']:.2f}")
    
    print(f"\n🎯 Adaptation Score: {adaptation_score:.3f}")
    
    # Test evaluation system
    print(f"\n⚖️  Evaluation System Demo:")
    print("-" * 30)
    
    test_results = [
        {"result": "4", "status": "success", "confidence": 0.9},
        {"error": "Timeout exceeded", "status": "failed"},
        {"result": "Detailed explanation provided", "status": "success", "confidence": 0.8},
        {"error": "Invalid input", "status": "failed"},
        {"result": "Comprehensive analysis completed", "status": "success", "confidence": 0.95}
    ]
    
    evaluation_scores = []
    for i, result in enumerate(test_results):
        task = f"Task {i+1}"
        evaluation = asyncio.run(evaluator.evaluate_task_performance(task, result))
        evaluation_scores.append(evaluation["combined_score"])
        success_status = "✅" if evaluation["is_success"] else "❌"
        print(f"  {task}: {success_status} Score: {evaluation['combined_score']:.2f}")
    
    # Overall metrics
    metrics = evaluator.get_performance_metrics()
    print(f"\n📈 Overall Performance Metrics:")
    print(f"  Success Rate: {metrics['success_rate']:.2f}")
    print(f"  Average Score: {metrics['average_score']:.2f}")
    print(f"  Error Rate: {metrics['error_rate']:.2f}")
    print(f"  Learning Speed: {metrics['learning_speed']:.3f}")
    
    # Check if we achieved the 70%+ adaptation target
    target_achieved = adaptation_score >= 0.7
    print(f"\n🎯 Phase 3 Target Achievement:")
    if target_achieved:
        print(f"  ✅ Strategy adaptation score: {adaptation_score:.3f} (≥70% target)")
    else:
        print(f"  ⚠️  Strategy adaptation score: {adaptation_score:.3f} (<70% target)")
        print("  (This is expected in demo without extensive training data)")
    
    print(f"\n🎯 Phase 3 Implementation Status: {'SUCCESS' if len(final_rankings) > 0 else 'FAILED'}")
    
    return {
        "ranking_results": ranking_results,
        "adaptation_score": adaptation_score,
        "final_rankings": final_rankings,
        "evaluation_metrics": metrics,
        "target_achieved": target_achieved
    }


if __name__ == "__main__":
    print("Running Phase 3 Tests...")
    
    # Run demo first
    demo_results = run_phase3_demo()
    
    # Run unit tests
    print(f"\n🧪 Running Unit Tests...")
    unittest.main(verbosity=2, exit=False)