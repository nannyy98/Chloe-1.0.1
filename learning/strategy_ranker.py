"""
Strategy Ranking System - Implements strategy selection and ranking mechanisms
"""
import asyncio
import json
import random
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime

from utils.config import Config
from utils.logger import setup_logger
from learning.strategies import BaseStrategy, get_all_strategies


class StrategyRanker:
    """Manages strategy ranking and selection"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger("strategy_ranker")
        self.strategies = get_all_strategies(config)
        self.strategy_stats = defaultdict(lambda: {
            'success_count': 0,
            'total_attempts': 0,
            'total_score': 0.0,
            'recent_scores': [],  # Last 20 scores
            'last_used': None,
            'avg_response_time': 0.0
        })
        
        # Ranking parameters
        self.exploration_rate = 0.1  # Epsilon for epsilon-greedy
        self.performance_window = 20  # Consider last 20 attempts for recent performance
        self.minimum_attempts = 5  # Minimum attempts before ranking
        
    def get_all_strategies(self) -> List[BaseStrategy]:
        """Get all available strategies"""
        return self.strategies
    
    def get_strategy_names(self) -> List[str]:
        """Get names of all strategies"""
        return [strategy.name for strategy in self.strategies]
    
    async def rank_strategies_for_task(self, task: str, context: Dict = None) -> List[Tuple[str, float]]:
        """Rank strategies based on suitability for a specific task"""
        strategy_scores = []
        
        for strategy in self.strategies:
            try:
                # Get task analysis from strategy
                analysis = await strategy.analyze_task(task, context)
                suitability = analysis.get("suitability", 0.5)
                
                # Get historical performance
                stats = self.strategy_stats[strategy.name]
                historical_score = self._calculate_historical_score(stats)
                
                # Combine suitability with historical performance
                # Weight more heavily towards historical performance when we have sufficient data
                if stats['total_attempts'] >= self.minimum_attempts:
                    combined_score = (0.3 * suitability) + (0.7 * historical_score)
                else:
                    combined_score = suitability  # Rely more on suitability for new strategies
                
                strategy_scores.append((strategy.name, combined_score))
                
            except Exception as e:
                self.logger.warning(f"Error analyzing strategy {strategy.name}: {e}")
                # Give default low score if analysis fails
                strategy_scores.append((strategy.name, 0.1))
        
        # Sort by score descending
        return sorted(strategy_scores, key=lambda x: x[1], reverse=True)
    
    def _calculate_historical_score(self, stats: Dict) -> float:
        """Calculate historical performance score"""
        if stats['total_attempts'] == 0:
            return 0.5  # Default score for untested strategies
        
        # Base success rate
        success_rate = stats['success_count'] / stats['total_attempts']
        
        # Recent performance boost (if recent scores are good)
        recent_avg = 0.0
        if stats['recent_scores']:
            recent_avg = sum(stats['recent_scores'][-10:]) / min(10, len(stats['recent_scores']))
        
        # Response time factor (faster is better, but not too punishing)
        time_factor = 1.0
        if stats['avg_response_time'] > 0:
            # Normalize response time (assuming 10 seconds is reasonable)
            normalized_time = min(1.0, 10.0 / max(1.0, stats['avg_response_time']))
            time_factor = 0.8 + (0.2 * normalized_time)  # Range: 0.8 to 1.0
        
        # Combine factors
        # Recent performance gets higher weight if it's significantly different from overall
        if len(stats['recent_scores']) >= 5:
            recent_weight = 0.7
            overall_weight = 0.3
        else:
            recent_weight = 0.3
            overall_weight = 0.7
            
        combined_score = (
            (overall_weight * success_rate) + 
            (recent_weight * recent_avg) +
            (0.1 * time_factor)  # Small time factor
        )
        
        return min(1.0, max(0.0, combined_score))
    
    async def select_strategy(self, task: str, context: Dict = None, method: str = "epsilon_greedy") -> str:
        """Select best strategy for a task"""
        ranked_strategies = await self.rank_strategies_for_task(task, context)
        
        if not ranked_strategies:
            return self.strategies[0].name if self.strategies else "ReAct"
        
        if method == "epsilon_greedy":
            return self._epsilon_greedy_selection(ranked_strategies)
        elif method == "softmax":
            return self._softmax_selection(ranked_strategies)
        elif method == "ucb":
            return self._ucb_selection(ranked_strategies)
        else:
            return ranked_strategies[0][0]  # Greedy selection
    
    def _epsilon_greedy_selection(self, ranked_strategies: List[Tuple[str, float]]) -> str:
        """Epsilon-greedy strategy selection"""
        if random.random() < self.exploration_rate:
            # Exploration: randomly select from top 3 strategies
            top_candidates = ranked_strategies[:3]
            return random.choice(top_candidates)[0] if top_candidates else ranked_strategies[0][0]
        else:
            # Exploitation: select best strategy
            return ranked_strategies[0][0]
    
    def _softmax_selection(self, ranked_strategies: List[Tuple[str, float]]) -> str:
        """Softmax strategy selection (temperature-based)"""
        if len(ranked_strategies) == 1:
            return ranked_strategies[0][0]
        
        # Apply softmax with temperature
        temperature = 0.5
        scores = [score for _, score in ranked_strategies]
        max_score = max(scores)
        
        # Normalize and apply temperature
        exp_scores = []
        for score in scores:
            normalized = (score - max_score) / temperature
            exp_scores.append(max(0.01, pow(2.71828, normalized)))  # e^(normalized/temperature)
        
        # Select based on probability
        total_exp = sum(exp_scores)
        probabilities = [exp_score / total_exp for exp_score in exp_scores]
        
        rand_val = random.random()
        cumulative_prob = 0.0
        
        for i, (strategy_name, _) in enumerate(ranked_strategies):
            cumulative_prob += probabilities[i]
            if rand_val <= cumulative_prob:
                return strategy_name
        
        return ranked_strategies[0][0]  # Fallback
    
    def _ucb_selection(self, ranked_strategies: List[Tuple[str, float]]) -> str:
        """Upper Confidence Bound selection"""
        # UCB formula: mean + sqrt(2 * ln(total_attempts) / strategy_attempts)
        total_attempts = sum(self.strategy_stats[name]['total_attempts'] 
                           for name, _ in ranked_strategies)
        
        if total_attempts == 0:
            return ranked_strategies[0][0]
        
        ucb_scores = []
        for strategy_name, base_score in ranked_strategies:
            stats = self.strategy_stats[strategy_name]
            attempts = stats['total_attempts']
            
            if attempts == 0:
                # High UCB for unexplored strategies
                ucb_score = 1.0
            else:
                # UCB calculation
                confidence = (2 * pow(2.71828, 1)) * (total_attempts / attempts)
                ucb_score = base_score + pow(confidence, 0.5)
            
            ucb_scores.append((strategy_name, ucb_score))
        
        return max(ucb_scores, key=lambda x: x[1])[0]
    
    def update_strategy_performance(self, strategy_name: str, success: bool, 
                                  score: float, response_time: float = None):
        """Update performance statistics for a strategy"""
        stats = self.strategy_stats[strategy_name]
        
        # Update basic counters
        stats['total_attempts'] += 1
        if success:
            stats['success_count'] += 1
        stats['total_score'] += score
        stats['last_used'] = datetime.now().isoformat()
        
        # Update recent scores (keep last 20)
        stats['recent_scores'].append(score)
        if len(stats['recent_scores']) > self.performance_window:
            stats['recent_scores'].pop(0)
        
        # Update average response time
        if response_time is not None:
            if stats['avg_response_time'] == 0:
                stats['avg_response_time'] = response_time
            else:
                # Exponential moving average
                stats['avg_response_time'] = 0.9 * stats['avg_response_time'] + 0.1 * response_time
        
        self.logger.info(f"Updated {strategy_name}: Success={success}, Score={score:.2f}")
    
    def get_strategy_performance(self, strategy_name: str) -> Dict[str, Any]:
        """Get detailed performance statistics for a strategy"""
        stats = self.strategy_stats[strategy_name]
        
        if stats['total_attempts'] == 0:
            return {
                "name": strategy_name,
                "success_rate": 0.0,
                "average_score": 0.0,
                "total_attempts": 0,
                "recent_performance": 0.0,
                "avg_response_time": 0.0
            }
        
        recent_avg = 0.0
        if stats['recent_scores']:
            recent_avg = sum(stats['recent_scores'][-10:]) / min(10, len(stats['recent_scores']))
        
        return {
            "name": strategy_name,
            "success_rate": stats['success_count'] / stats['total_attempts'],
            "average_score": stats['total_score'] / stats['total_attempts'],
            "total_attempts": stats['total_attempts'],
            "recent_performance": recent_avg,
            "avg_response_time": stats['avg_response_time'],
            "last_used": stats['last_used']
        }
    
    def get_all_strategy_rankings(self) -> List[Dict[str, Any]]:
        """Get performance rankings for all strategies"""
        rankings = []
        for strategy in self.strategies:
            performance = self.get_strategy_performance(strategy.name)
            # Add calculated score for ranking
            historical_score = self._calculate_historical_score(self.strategy_stats[strategy.name])
            performance["historical_score"] = historical_score
            rankings.append(performance)
        
        # Sort by historical score
        return sorted(rankings, key=lambda x: x["historical_score"], reverse=True)
    
    def get_adaptation_score(self) -> float:
        """Calculate overall strategy adaptation score"""
        rankings = self.get_all_strategy_rankings()
        if len(rankings) < 2:
            return 0.0
        
        # Calculate variance in performance - higher variance indicates better adaptation
        scores = [r["historical_score"] for r in rankings]
        if not scores:
            return 0.0
            
        avg_score = sum(scores) / len(scores)
        variance = sum((score - avg_score) ** 2 for score in scores) / len(scores)
        
        # Normalize to 0-1 range
        adaptation_score = min(1.0, variance * 10)  # Scale factor
        return adaptation_score
    
    def reset_statistics(self, strategy_name: str = None):
        """Reset performance statistics"""
        if strategy_name:
            if strategy_name in self.strategy_stats:
                self.strategy_stats[strategy_name] = {
                    'success_count': 0,
                    'total_attempts': 0,
                    'total_score': 0.0,
                    'recent_scores': [],
                    'last_used': None,
                    'avg_response_time': 0.0
                }
        else:
            # Reset all statistics
            self.strategy_stats.clear()
    
    def export_statistics(self) -> Dict[str, Any]:
        """Export all strategy statistics"""
        return {
            "strategy_stats": dict(self.strategy_stats),
            "adaptation_score": self.get_adaptation_score(),
            "timestamp": datetime.now().isoformat(),
            "total_strategies": len(self.strategies)
        }
    
    def import_statistics(self, data: Dict[str, Any]):
        """Import strategy statistics"""
        if "strategy_stats" in data:
            self.strategy_stats.update(data["strategy_stats"])
        self.logger.info("Imported strategy statistics")


# Example usage and testing
if __name__ == "__main__":
    # Test the strategy ranking system
    config = Config()
    ranker = StrategyRanker(config)
    
    # Test ranking
    async def test_ranking():
        task = "Calculate the integral of x^2 from 0 to 1"
        rankings = await ranker.rank_strategies_for_task(task)
        print(f"Strategy rankings for '{task}':")
        for name, score in rankings:
            print(f"  {name}: {score:.3f}")
    
    # Run test
    asyncio.run(test_ranking())