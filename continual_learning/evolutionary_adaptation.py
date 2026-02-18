"""
Evolutionary Adaptation System - Uses genetic algorithms for strategy evolution
"""
import asyncio
import random
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from copy import deepcopy

from utils.config import Config
from utils.logger import setup_logger
from learning.strategies import BaseStrategy
from learning.strategy_ranker import StrategyRanker
from evaluation.evaluation_system import EvaluationSystem


class GeneticAlgorithm:
    """Genetic Algorithm implementation for evolving strategies and parameters"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger("genetic_algorithm")
        
        # GA parameters
        self.population_size = config.get("evolution.population_size", 20)
        self.mutation_rate = config.get("evolution.mutation_rate", 0.1)
        self.crossover_rate = config.get("evolution.crossover_rate", 0.8)
        self.elitism_rate = config.get("evolution.elitism_rate", 0.2)
        self.generations = config.get("evolution.generations", 50)
        
        # Evolution parameters
        self.param_bounds = config.get("evolution.param_bounds", {
            "learning_rate": (0.001, 0.1),
            "exploration_rate": (0.01, 0.5),
            "discount_factor": (0.7, 0.99),
            "temperature": (0.1, 2.0)
        })
        
        # Population tracking
        self.current_population = []
        self.fitness_history = []
        self.best_individual_history = []
    
    def create_random_individual(self) -> Dict[str, Any]:
        """Create a random individual with parameters to evolve"""
        individual = {}
        for param, (min_val, max_val) in self.param_bounds.items():
            individual[param] = random.uniform(min_val, max_val)
        
        # Add strategy composition
        individual["strategy_weights"] = {
            "ReAct": random.random(),
            "ChainOfThought": random.random(),
            "PlanAndExecute": random.random()
        }
        
        # Normalize strategy weights
        total_weight = sum(individual["strategy_weights"].values())
        if total_weight > 0:
            for key in individual["strategy_weights"]:
                individual["strategy_weights"][key] /= total_weight
        
        individual["id"] = f"ind_{datetime.now().timestamp()}_{random.randint(1000, 9999)}"
        individual["generation"] = 0
        
        return individual
    
    def initialize_population(self) -> List[Dict[str, Any]]:
        """Initialize the population with random individuals"""
        population = []
        for _ in range(self.population_size):
            individual = self.create_random_individual()
            population.append(individual)
        
        self.current_population = population
        self.logger.info(f"Initialized population of size {self.population_size}")
        return population
    
    def mutate_individual(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate an individual by randomly changing some parameters"""
        mutated = deepcopy(individual)
        
        for param, (min_val, max_val) in self.param_bounds.items():
            if random.random() < self.mutation_rate:
                # Gaussian mutation
                current_val = mutated[param]
                mutation_strength = (max_val - min_val) * 0.1
                mutated_val = current_val + random.gauss(0, mutation_strength)
                
                # Clamp to bounds
                mutated[param] = max(min_val, min(max_val, mutated_val))
        
        # Mutate strategy weights
        if random.random() < self.mutation_rate:
            # Add small random changes to strategy weights
            for key in mutated["strategy_weights"]:
                if random.random() < 0.3:  # Only mutate some weights
                    current_weight = mutated["strategy_weights"][key]
                    mutation = random.gauss(0, 0.1)
                    mutated["strategy_weights"][key] = max(0.0, current_weight + mutation)
            
            # Renormalize weights
            total_weight = sum(mutated["strategy_weights"].values())
            if total_weight > 0:
                for key in mutated["strategy_weights"]:
                    mutated["strategy_weights"][key] /= total_weight
        
        return mutated
    
    def crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> tuple:
        """Perform crossover between two parents"""
        child1 = deepcopy(parent1)
        child2 = deepcopy(parent2)
        
        for param in self.param_bounds.keys():
            if random.random() < self.crossover_rate:
                # Blend crossover
                alpha = random.random()
                child1[param] = alpha * parent1[param] + (1 - alpha) * parent2[param]
                child2[param] = alpha * parent2[param] + (1 - alpha) * parent1[param]
        
        # Crossover strategy weights
        if random.random() < self.crossover_rate:
            alpha = random.random()
            for key in child1["strategy_weights"]:
                child1["strategy_weights"][key] = (
                    alpha * parent1["strategy_weights"][key] + 
                    (1 - alpha) * parent2["strategy_weights"][key]
                )
                child2["strategy_weights"][key] = (
                    alpha * parent2["strategy_weights"][key] + 
                    (1 - alpha) * parent1["strategy_weights"][key]
                )
        
        # Renormalize weights
        for child in [child1, child2]:
            total_weight = sum(child["strategy_weights"].values())
            if total_weight > 0:
                for key in child["strategy_weights"]:
                    child["strategy_weights"][key] /= total_weight
        
        return child1, child2
    
    async def evaluate_fitness(self, individual: Dict[str, Any], tasks: List[str]) -> float:
        """Evaluate fitness of an individual on given tasks"""
        try:
            # In a real implementation, this would test the individual's parameters
            # against actual tasks and return performance score
            # For simulation, we'll calculate a synthetic fitness based on parameter values
            # and add some randomness
            
            # Calculate fitness based on parameter combinations
            fitness_score = 0.0
            
            # Encourage balanced parameters
            param_values = [individual[param] for param in self.param_bounds.keys()]
            param_variance = np.var(param_values) if param_values else 0
            fitness_score -= param_variance * 0.1  # Penalize extreme variance
            
            # Encourage diverse strategy weights
            strategy_weights = list(individual["strategy_weights"].values())
            strategy_variance = np.var(strategy_weights) if strategy_weights else 0
            fitness_score += strategy_variance * 0.2  # Reward diversity
            
            # Add task-specific performance simulation
            task_performance = []
            for task in tasks:
                # Simulate performance based on task characteristics and individual parameters
                task_score = await self._simulate_task_performance(task, individual)
                task_performance.append(task_score)
            
            avg_task_performance = sum(task_performance) / len(task_performance) if task_performance else 0.5
            fitness_score += avg_task_performance
            
            # Ensure fitness is within reasonable bounds
            fitness_score = max(0.0, min(1.0, fitness_score))
            
            return fitness_score
            
        except Exception as e:
            self.logger.error(f"Error evaluating fitness: {e}")
            return 0.1  # Low fitness for failed evaluation
    
    async def _simulate_task_performance(self, task: str, individual: Dict[str, Any]) -> float:
        """Simulate task performance based on individual parameters"""
        # This would be replaced with actual task execution in a real implementation
        # For simulation, we'll return a score based on how well parameters match task requirements
        
        task_lower = task.lower()
        
        # Adjust score based on task type and parameters
        if "calculate" in task_lower or "math" in task_lower:
            # Mathematical tasks might benefit from higher precision parameters
            precision_score = min(1.0, individual["learning_rate"] * 10)
            return precision_score * 0.7 + random.random() * 0.3
        elif "analyze" in task_lower or "compare" in task_lower:
            # Analytical tasks might benefit from more exploration
            exploration_score = individual["exploration_rate"]
            return exploration_score * 0.6 + random.random() * 0.4
        else:
            # General tasks get baseline score
            return 0.5 + random.random() * 0.3
    
    def select_parents(self, population: List[Dict[str, Any]], fitness_scores: List[float]) -> tuple:
        """Select parents for reproduction using tournament selection"""
        tournament_size = 3
        parent1 = self._tournament_selection(population, fitness_scores, tournament_size)
        parent2 = self._tournament_selection(population, fitness_scores, tournament_size)
        return parent1, parent2
    
    def _tournament_selection(self, population: List[Dict[str, Any]], 
                            fitness_scores: List[float], tournament_size: int) -> Dict[str, Any]:
        """Select an individual using tournament selection"""
        tournament_indices = random.sample(range(len(population)), 
                                        min(tournament_size, len(population)))
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
        return population[winner_index]
    
    async def evolve_generation(self, tasks: List[str]) -> Dict[str, Any]:
        """Evolve one generation of the population"""
        # Evaluate fitness for all individuals
        fitness_scores = []
        for individual in self.current_population:
            fitness = await self.evaluate_fitness(individual, tasks)
            fitness_scores.append(fitness)
        
        # Sort population by fitness (descending)
        sorted_pop = [x for _, x in sorted(zip(fitness_scores, self.current_population), 
                                         key=lambda pair: pair[0], reverse=True)]
        sorted_fitness = sorted(fitness_scores, reverse=True)
        
        # Track best individual
        best_individual = deepcopy(sorted_pop[0])
        best_fitness = sorted_fitness[0]
        
        self.best_individual_history.append({
            "individual": best_individual,
            "fitness": best_fitness,
            "timestamp": datetime.now().isoformat()
        })
        
        # Calculate and store fitness statistics
        self.fitness_history.append({
            "generation": len(self.fitness_history),
            "best_fitness": best_fitness,
            "avg_fitness": sum(sorted_fitness) / len(sorted_fitness),
            "worst_fitness": sorted_fitness[-1],
            "timestamp": datetime.now().isoformat()
        })
        
        # Create new population
        new_population = []
        
        # Elitism: keep best individuals
        elite_count = int(self.elitism_rate * self.population_size)
        new_population.extend(deepcopy(sorted_pop[:elite_count]))
        
        # Fill rest of population through selection, crossover, and mutation
        while len(new_population) < self.population_size:
            parent1, parent2 = self.select_parents(sorted_pop, sorted_fitness)
            
            if random.random() < self.crossover_rate:
                child1, child2 = self.crossover(parent1, parent2)
            else:
                child1, child2 = deepcopy(parent1), deepcopy(parent2)
            
            child1 = self.mutate_individual(child1)
            child2 = self.mutate_individual(child2)
            
            # Update generation counter
            child1["generation"] = len(self.fitness_history)
            child2["generation"] = len(self.fitness_history)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        self.current_population = new_population[:self.population_size]
        
        return {
            "best_fitness": best_fitness,
            "avg_fitness": sum(sorted_fitness) / len(sorted_fitness),
            "population_size": len(self.current_population)
        }


class EvolutionaryAdaptor:
    """Main evolutionary adaptation system that manages the genetic algorithm"""
    
    def __init__(self, config: Config, strategy_ranker: StrategyRanker, evaluation_system: EvaluationSystem):
        self.config = config
        self.strategy_ranker = strategy_ranker
        self.evaluation_system = evaluation_system
        self.logger = setup_logger("evolutionary_adaptor")
        
        # Initialize genetic algorithm
        self.ga = GeneticAlgorithm(config)
        
        # Evolution tracking
        self.evolution_history = []
        self.best_solution = None
        self.best_fitness = 0.0
        
        # Task sampling for evolution
        self.task_pool = []
        self.min_tasks_for_evolution = config.get("evolution.min_tasks_for_evolution", 5)
    
    async def add_tasks_for_evolution(self, tasks: List[str]):
        """Add tasks to the evolution task pool"""
        self.task_pool.extend(tasks)
        
        # Keep only recent tasks
        max_tasks = 50
        if len(self.task_pool) > max_tasks:
            self.task_pool = self.task_pool[-max_tasks:]
    
    async def evolve_strategies(self, generations: int = None) -> Dict[str, Any]:
        """Evolve strategies using genetic algorithm"""
        if len(self.task_pool) < self.min_tasks_for_evolution:
            self.logger.warning(f"Not enough tasks for evolution: {len(self.task_pool)} < {self.min_tasks_for_evolution}")
            return {"status": "skipped", "reason": "insufficient_tasks", "task_count": len(self.task_pool)}
        
        generations = generations or self.ga.generations
        
        # Initialize population if needed
        if not self.ga.current_population:
            self.ga.initialize_population()
        
        self.logger.info(f"Starting evolution for {generations} generations with {len(self.task_pool)} tasks")
        
        evolution_start_time = datetime.now()
        
        for gen in range(generations):
            # Evolve one generation
            gen_stats = await self.ga.evolve_generation(self.task_pool)
            
            # Log progress
            if gen % 10 == 0 or gen == generations - 1:
                self.logger.info(f"Generation {gen + 1}/{generations}: Best fitness = {gen_stats['best_fitness']:.4f}")
            
            # Update best solution if improved
            if gen_stats["best_fitness"] > self.best_fitness:
                self.best_fitness = gen_stats["best_fitness"]
                self.best_solution = deepcopy(self.ga.best_individual_history[-1]["individual"])
        
        evolution_duration = (datetime.now() - evolution_start_time).total_seconds()
        
        # Record evolution result
        evolution_result = {
            "start_time": evolution_start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": evolution_duration,
            "generations_completed": generations,
            "best_fitness": self.best_fitness,
            "best_solution": self.best_solution,
            "fitness_history": self.ga.fitness_history[-10:],  # Last 10 generations
            "status": "completed"
        }
        
        self.evolution_history.append(evolution_result)
        
        # Keep only recent evolution history
        if len(self.evolution_history) > 10:
            self.evolution_history.pop(0)
        
        self.logger.info(f"Evolution completed in {evolution_duration:.2f}s, best fitness: {self.best_fitness:.4f}")
        
        return evolution_result
    
    async def apply_evolved_solution(self, evolved_params: Dict[str, Any]):
        """Apply the evolved parameters to the system"""
        try:
            # Update strategy ranker with evolved parameters
            if "exploration_rate" in evolved_params:
                self.strategy_ranker.exploration_rate = evolved_params["exploration_rate"]
            
            if "learning_rate" in evolved_params:
                # In a real implementation, this would update learning rates in various components
                pass
            
            # Update strategy weights if present
            if "strategy_weights" in evolved_params:
                for strategy_name, weight in evolved_params["strategy_weights"].items():
                    if hasattr(self.strategy_ranker, 'strategy_stats'):
                        # Update the strategy performance stats with evolved weights
                        if strategy_name in self.strategy_ranker.strategy_stats:
                            # This is a simplified approach - in reality, weights would affect selection probability
                            pass
            
            self.logger.info("Applied evolved solution to system components")
            
        except Exception as e:
            self.logger.error(f"Error applying evolved solution: {e}")
    
    async def evolve_and_apply(self) -> Dict[str, Any]:
        """Evolve strategies and apply the best solution to the system"""
        # Run evolution
        evolution_result = await self.evolve_strategies()
        
        if evolution_result["status"] == "completed" and self.best_solution:
            # Apply the best evolved solution
            await self.apply_evolved_solution(self.best_solution)
            
            # Update the result with application status
            evolution_result["solution_applied"] = True
            evolution_result["applied_solution"] = self.best_solution
        else:
            evolution_result["solution_applied"] = False
        
        return evolution_result
    
    def get_evolution_status(self) -> Dict[str, Any]:
        """Get current evolution status and statistics"""
        return {
            "active_population_size": len(self.ga.current_population) if hasattr(self.ga, 'current_population') else 0,
            "evolution_generations": len(self.ga.fitness_history),
            "best_fitness_ever": self.best_fitness,
            "evolution_history": self.evolution_history[-3:],  # Last 3 evolution runs
            "task_pool_size": len(self.task_pool),
            "parameters_evolved": list(self.ga.param_bounds.keys()) if hasattr(self.ga, 'param_bounds') else [],
            "current_best_solution": self.best_solution
        }
    
    async def reset_evolution(self):
        """Reset the evolution process"""
        self.ga.current_population = []
        self.ga.fitness_history = []
        self.ga.best_individual_history = []
        self.best_solution = None
        self.best_fitness = 0.0
        self.evolution_history = []
        
        self.logger.info("Reset evolution process")


# Example usage
if __name__ == "__main__":
    # This would be run with proper configuration
    pass