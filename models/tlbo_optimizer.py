import numpy as np
from typing import Dict, List, Callable, Tuple, Any
import random


class TLBOOptimizer:
    """
    Teaching-Learning-Based Optimization (TLBO) for hyperparameter tuning.
    
    Based on: Rao et al. (2011) - Teaching-Learning-Based Optimization
    
    Used in: Yilma et al. (2026) - Sequential metaheuristic optimization
    of stacked-LSTM hyperparameters
    """
    
    def __init__(
        self,
        search_space: Dict[str, Tuple[Any, Any]],
        population_size: int = 20,
        n_iterations: int = 30,
        verbose: bool = True
    ):
        self.search_space = search_space
        self.population_size = population_size
        self.n_iterations = n_iterations
        self.verbose = verbose
        
        self.param_names = list(search_space.keys())
        self.population = []
        self.best_solution = None
        self.best_score = float('inf')
        
    def _initialize_population(self) -> List[Dict]:
        """Initialize random population within search space."""
        population = []
        
        for _ in range(self.population_size):
            individual = {}
            for param, (low, high) in self.search_space.items():
                if isinstance(low, int) and isinstance(high, int):
                    individual[param] = random.randint(low, high)
                elif isinstance(low, float):
                    individual[param] = random.uniform(low, high)
                else:
                    individual[param] = random.choice(low)
            population.append(individual)
        
        return population
    
    def _evaluate(self, individual: Dict) -> float:
        """Evaluate objective function. To be implemented by user."""
        raise NotImplementedError("Subclass must implement _evaluate method")
    
    def _teaching_phase(
        self,
        population: List[Dict],
        teacher: Dict,
        tf: int = 2
    ) -> List[Dict]:
        """
        Teaching phase: learners learn from the teacher.
        
        Args:
            population: Current population
            teacher: Best individual (teacher)
            tf: Teaching factor (1 or 2)
        """
        new_population = []
        mean = {}
        
        for param in self.param_names:
            values = [ind[param] for ind in population]
            if all(isinstance(v, (int, float)) for v in values):
                mean[param] = sum(values) / len(values)
            else:
                mean[param] = values[0]
        
        for learner in population:
            new_learner = learner.copy()
            for param in self.param_names:
                if isinstance(learner[param], (int, float)):
                    r = random.random()
                    new_val = learner[param] + r * (
                        teacher[param] - tf * mean[param]
                    )
                    low, high = self.search_space[param]
                    new_learner[param] = self._clip(new_val, low, high)
            new_population.append(new_learner)
        
        return new_population
    
    def _learning_phase(
        self,
        population: List[Dict]
    ) -> List[Dict]:
        """
        Learning phase: learners learn from each other.
        """
        new_population = []
        
        for i, learner in enumerate(population):
            new_learner = learner.copy()
            
            partner_idx = random.randint(0, len(population) - 1)
            while partner_idx == i:
                partner_idx = random.randint(0, len(population) - 1)
            
            partner = population[partner_idx]
            
            learner_score = self._evaluate(learner)
            partner_score = self._evaluate(partner)
            
            for param in self.param_names:
                if isinstance(learner[param], (int, float)):
                    if learner_score < partner_score:
                        diff = learner[param] - partner[param]
                    else:
                        diff = partner[param] - learner[param]
                    
                    r = random.random()
                    new_val = learner[param] + r * diff
                    low, high = self.search_space[param]
                    new_learner[param] = self._clip(new_val, low, high)
            
            new_population.append(new_learner)
        
        return new_population
    
    def _clip(self, value: float, low: Any, high: Any) -> Any:
        """Clip value to search space bounds."""
        if isinstance(low, int) and isinstance(high, int):
            return int(np.clip(value, low, high))
        return np.clip(value, low, high)
    
    def optimize(
        self,
        objective_fn: Callable[[Dict], float]
    ) -> Tuple[Dict, float]:
        """
        Run TLBO optimization.
        
        Args:
            objective_fn: Function that takes params dict and returns score
            
        Returns:
            Tuple of (best_params, best_score)
        """
        self._evaluate = objective_fn
        
        self.population = self._initialize_population()
        
        scores = [self._evaluate(ind) for ind in self.population]
        best_idx = np.argmin(scores)
        self.best_solution = self.population[best_idx].copy()
        self.best_score = scores[best_idx]
        
        for iteration in range(self.n_iterations):
            teacher = self.best_solution.copy()
            tf = random.choice([1, 2])
            
            self.population = self._teaching_phase(
                self.population, teacher, tf
            )
            
            self.population = self._learning_phase(self.population)
            
            scores = [self._evaluate(ind) for ind in self.population]
            best_idx = np.argmin(scores)
            
            if scores[best_idx] < self.best_score:
                self.best_score = scores[best_idx]
                self.best_solution = self.population[best_idx].copy()
            
            if self.verbose and iteration % 5 == 0:
                print(f"TLBO Iter {iteration + 1}/{self.n_iterations}, "
                      f"Best Score: {self.best_score:.4f}")
        
        return self.best_solution, self.best_score


def create_search_space(
    hidden_size: Tuple[int, int] = (32, 128),
    num_layers: Tuple[int, int] = (1, 4),
    dropout: Tuple[float, float] = (0.1, 0.5),
    learning_rate: Tuple[float, float] = (1e-4, 1e-2),
    batch_size: Tuple[int, int] = (16, 64),
    sequence_length: Tuple[int, int] = (20, 50)
) -> Dict:
    """Create default search space for LSTM hyperparameters."""
    return {
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'dropout': dropout,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'sequence_length': sequence_length
    }