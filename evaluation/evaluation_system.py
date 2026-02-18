"""
Evaluation System - LLM-based evaluation and metrics tracking
"""
import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict

from utils.config import Config
from utils.logger import setup_logger
from agents.ollama_agent import OllamaAgent


class EvaluationSystem:
    """Manages LLM-based evaluation and performance metrics"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger("evaluation_system")
        self.ollama_agent = OllamaAgent(config)
        self.evaluation_history = []
        self.metrics_history = []
        self.success_threshold = config.get("learning.success_threshold", 0.7)
        
    async def evaluate_task_performance(self, task: str, result: Dict, 
                                      expected_outcome: str = None) -> Dict[str, Any]:
        """Evaluate task performance using LLM judge"""
        try:
            # Generate LLM-based evaluation
            llm_evaluation = await self._llm_judge_evaluation(task, result, expected_outcome)
            
            # Calculate automated metrics
            automated_metrics = self._calculate_automated_metrics(task, result)
            
            # Combine evaluations
            combined_score = self._combine_evaluations(llm_evaluation, automated_metrics)
            
            # Determine success
            is_success = combined_score >= self.success_threshold
            
            evaluation = {
                "task": task,
                "result": result,
                "llm_evaluation": llm_evaluation,
                "automated_metrics": automated_metrics,
                "combined_score": combined_score,
                "is_success": is_success,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store evaluation
            self.evaluation_history.append(evaluation)
            
            # Keep only recent evaluations
            if len(self.evaluation_history) > 100:
                self.evaluation_history.pop(0)
            
            self.logger.info(f"Task evaluation: Score={combined_score:.2f}, Success={is_success}")
            
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Error in task evaluation: {e}")
            return self._generate_fallback_evaluation(task, result)
    
    async def _llm_judge_evaluation(self, task: str, result: Dict, 
                                  expected_outcome: str = None) -> Dict[str, Any]:
        """Get LLM-based evaluation of task performance"""
        result_str = json.dumps(result, indent=2)
        expected_str = expected_outcome or "Appropriate completion of the requested task"
        
        prompt = f"""
        You are an expert AI evaluator. Assess the quality of this task completion.
        
        Task: {task}
        Result: {result_str}
        Expected Outcome: {expected_str}
        
        Evaluate on these criteria (0.0-1.0 scale):
        1. Correctness: Is the answer factually correct?
        2. Completeness: Does it fully address the task?
        3. Clarity: Is the response clear and well-structured?
        4. Relevance: Does it stay on topic and relevant?
        5. Quality: Overall quality of the response
        
        Provide your evaluation in JSON format:
        {{
            "criteria_scores": {{
                "correctness": 0.0-1.0,
                "completeness": 0.0-1.0,
                "clarity": 0.0-1.0,
                "relevance": 0.0-1.0,
                "quality": 0.0-1.0
            }},
            "overall_score": 0.0-1.0,
            "strengths": ["strength1", "strength2"],
            "weaknesses": ["weakness1", "weakness2"],
            "detailed_feedback": "Comprehensive feedback on performance",
            "improvement_areas": ["area1", "area2"]
        }}
        """
        
        try:
            eval_result = await self.ollama_agent.execute({
                "prompt": prompt,
                "temperature": 0.3,
                "max_tokens": 1000
            })
            
            if eval_result["status"] == "success":
                return json.loads(eval_result["result"])
            else:
                return self._generate_fallback_llm_evaluation(result)
                
        except Exception as e:
            self.logger.warning(f"LLM evaluation failed: {e}")
            return self._generate_fallback_llm_evaluation(result)
    
    def _generate_fallback_llm_evaluation(self, result: Dict) -> Dict[str, Any]:
        """Generate fallback LLM evaluation when LLM fails"""
        # Simple heuristic-based evaluation
        has_error = "error" in str(result).lower()
        has_result = "result" in result or "output" in result
        status_success = result.get("status") == "success"
        
        if has_error:
            overall_score = 0.2
        elif status_success and has_result:
            overall_score = 0.8
        elif has_result:
            overall_score = 0.6
        else:
            overall_score = 0.3
        
        return {
            "criteria_scores": {
                "correctness": overall_score,
                "completeness": overall_score,
                "clarity": 0.7,
                "relevance": 0.8,
                "quality": overall_score
            },
            "overall_score": overall_score,
            "strengths": ["Basic execution" if not has_error else "Error handling"],
            "weaknesses": ["Error occurred" if has_error else "Limited analysis"],
            "detailed_feedback": "Fallback evaluation based on result structure",
            "improvement_areas": ["Error reduction" if has_error else "Enhanced analysis"]
        }
    
    def _calculate_automated_metrics(self, task: str, result: Dict) -> Dict[str, Any]:
        """Calculate automated performance metrics"""
        # Extract key metrics
        has_error = "error" in str(result).lower()
        status = result.get("status", "unknown")
        result_content = str(result.get("result", "") or result.get("output", ""))
        result_length = len(result_content)
        
        # Success indicators
        success_indicators = [
            status == "success",
            not has_error,
            result_length > 10,  # Basic completeness check
            "final_answer" in result or "result" in result
        ]
        
        automated_score = sum(success_indicators) / len(success_indicators)
        
        return {
            "automated_score": automated_score,
            "has_error": has_error,
            "status": status,
            "result_length": result_length,
            "success_indicators_count": sum(success_indicators),
            "total_indicators": len(success_indicators)
        }
    
    def _combine_evaluations(self, llm_evaluation: Dict, automated_metrics: Dict) -> float:
        """Combine LLM and automated evaluations"""
        llm_score = llm_evaluation.get("overall_score", 0.5)
        automated_score = automated_metrics.get("automated_score", 0.5)
        
        # Weight more heavily towards LLM evaluation when available
        if llm_score != 0.5:  # 0.5 indicates fallback
            combined = (0.7 * llm_score) + (0.3 * automated_score)
        else:
            combined = automated_score
            
        return min(1.0, max(0.0, combined))
    
    def _generate_fallback_evaluation(self, task: str, result: Dict) -> Dict[str, Any]:
        """Generate fallback evaluation when everything fails"""
        automated_metrics = self._calculate_automated_metrics(task, result)
        
        return {
            "task": task,
            "result": result,
            "llm_evaluation": self._generate_fallback_llm_evaluation(result),
            "automated_metrics": automated_metrics,
            "combined_score": automated_metrics["automated_score"],
            "is_success": automated_metrics["automated_score"] >= self.success_threshold,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_performance_metrics(self, window: int = 50) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        if not self.evaluation_history:
            return self._get_empty_metrics()
        
        # Get recent evaluations
        recent_evals = self.evaluation_history[-window:]
        
        # Calculate metrics
        total_evaluations = len(recent_evals)
        successful_evaluations = sum(1 for eval in recent_evals if eval["is_success"])
        success_rate = successful_evaluations / total_evaluations if total_evaluations > 0 else 0.0
        
        # Score statistics
        scores = [eval["combined_score"] for eval in recent_evals]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0
        min_score = min(scores) if scores else 0.0
        
        # Error analysis
        errors = [eval for eval in recent_evals if eval["automated_metrics"]["has_error"]]
        error_rate = len(errors) / total_evaluations if total_evaluations > 0 else 0.0
        
        # Learning speed (improvement over time)
        learning_speed = self._calculate_learning_speed(recent_evals)
        
        metrics = {
            "total_evaluations": total_evaluations,
            "success_rate": success_rate,
            "successful_evaluations": successful_evaluations,
            "failed_evaluations": total_evaluations - successful_evaluations,
            "average_score": avg_score,
            "max_score": max_score,
            "min_score": min_score,
            "error_rate": error_rate,
            "learning_speed": learning_speed,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store metrics
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > 100:
            self.metrics_history.pop(0)
        
        return metrics
    
    def _get_empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure"""
        return {
            "total_evaluations": 0,
            "success_rate": 0.0,
            "successful_evaluations": 0,
            "failed_evaluations": 0,
            "average_score": 0.0,
            "max_score": 0.0,
            "min_score": 0.0,
            "error_rate": 0.0,
            "learning_speed": 0.0,
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_learning_speed(self, evaluations: List[Dict]) -> float:
        """Calculate learning speed based on performance improvement"""
        if len(evaluations) < 10:
            return 0.0
        
        # Split into two halves and compare average scores
        mid_point = len(evaluations) // 2
        first_half = evaluations[:mid_point]
        second_half = evaluations[mid_point:]
        
        if not first_half or not second_half:
            return 0.0
        
        first_avg = sum(eval["combined_score"] for eval in first_half) / len(first_half)
        second_avg = sum(eval["combined_score"] for eval in second_half) / len(second_half)
        
        # Calculate improvement rate per evaluation
        improvement = second_avg - first_avg
        evaluations_span = len(evaluations) / 2
        
        learning_speed = improvement / evaluations_span if evaluations_span > 0 else 0.0
        
        return learning_speed
    
    def get_detailed_analysis(self, window: int = 50) -> Dict[str, Any]:
        """Get detailed performance analysis"""
        metrics = self.get_performance_metrics(window)
        
        # Get recent evaluations for detailed analysis
        recent_evals = self.evaluation_history[-window:] if self.evaluation_history else []
        
        # Error pattern analysis
        error_types = defaultdict(int)
        for eval in recent_evals:
            if eval["automated_metrics"]["has_error"]:
                error_msg = str(eval["result"].get("error", "Unknown error"))
                if "timeout" in error_msg.lower():
                    error_types["timeout"] += 1
                elif "tool" in error_msg.lower():
                    error_types["tool_error"] += 1
                elif "memory" in error_msg.lower():
                    error_types["memory_error"] += 1
                else:
                    error_types["other_error"] += 1
        
        # Score distribution
        score_ranges = {
            "excellent": len([e for e in recent_evals if e["combined_score"] >= 0.9]),
            "good": len([e for e in recent_evals if 0.7 <= e["combined_score"] < 0.9]),
            "fair": len([e for e in recent_evals if 0.5 <= e["combined_score"] < 0.7]),
            "poor": len([e for e in recent_evals if e["combined_score"] < 0.5])
        }
        
        return {
            "metrics": metrics,
            "error_analysis": dict(error_types),
            "score_distribution": score_ranges,
            "total_history": len(self.evaluation_history),
            "recent_evaluations_count": len(recent_evals)
        }
    
    def get_learning_trend(self, windows: List[int] = [10, 25, 50]) -> Dict[str, Any]:
        """Analyze learning trends across different time windows"""
        trends = {}
        
        for window in windows:
            if len(self.evaluation_history) >= window:
                metrics = self.get_performance_metrics(window)
                trends[f"{window}_evaluations"] = {
                    "success_rate": metrics["success_rate"],
                    "average_score": metrics["average_score"],
                    "learning_speed": metrics["learning_speed"]
                }
        
        # Determine overall trend
        if len(trends) >= 2:
            windows_sorted = sorted([int(k.split('_')[0]) for k in trends.keys()], reverse=True)
            if len(windows_sorted) >= 2:
                recent_window = f"{windows_sorted[0]}_evaluations"
                older_window = f"{windows_sorted[1]}_evaluations"
                
                recent_success = trends[recent_window]["success_rate"]
                older_success = trends[older_window]["success_rate"]
                
                if recent_success > older_success + 0.1:
                    overall_trend = "improving"
                elif recent_success < older_success - 0.1:
                    overall_trend = "declining"
                else:
                    overall_trend = "stable"
            else:
                overall_trend = "insufficient_data"
        else:
            overall_trend = "insufficient_data"
        
        return {
            "trends": trends,
            "overall_trend": overall_trend,
            "total_evaluations": len(self.evaluation_history)
        }
    
    def export_evaluation_data(self) -> Dict[str, Any]:
        """Export all evaluation data for analysis"""
        return {
            "evaluation_history": self.evaluation_history,
            "metrics_history": self.metrics_history,
            "current_metrics": self.get_performance_metrics(),
            "detailed_analysis": self.get_detailed_analysis(),
            "learning_trend": self.get_learning_trend(),
            "export_timestamp": datetime.now().isoformat()
        }


# Example usage
if __name__ == "__main__":
    # Test the evaluation system
    config = Config()
    evaluator = EvaluationSystem(config)
    
    # Test evaluation
    async def test_evaluation():
        task = "Calculate 2+2"
        result = {"result": "4", "status": "success"}
        
        evaluation = await evaluator.evaluate_task_performance(task, result)
        print(f"Evaluation result: {evaluation['combined_score']:.2f}")
        print(f"Success: {evaluation['is_success']}")
        
        metrics = evaluator.get_performance_metrics()
        print(f"Success rate: {metrics['success_rate']:.2f}")
    
    # Run test
    asyncio.run(test_evaluation())