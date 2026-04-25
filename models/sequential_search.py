import numpy as np
from typing import Dict, List, Callable, Tuple, Optional
import copy

from .tlbo_optimizer import TLBOOptimizer, create_search_space
from .pso_optimizer import PSOOptimizer


class SequentialMetaheuristicOptimizer:
    """
    Sequential Metaheuristic Optimization for LSTM hyperparameters.
    
    Implements the two-phase approach from:
    Yilma et al. (2026) - "Remaining useful life prediction using sequential 
    metaheuristic optimization of stacked-LSTM hyperparameters"
    
    Phase 1: TLBO for broad exploration of hyperparameter space
    Phase 2: PSO for refined local search around best region
    """
    
    def __init__(
        self,
        search_space: Dict[str, Tuple],
        tlbo_population: int = 20,
        tlbo_iterations: int = 25,
        pso_particles: int = 20,
        pso_iterations: int = 20,
        verbose: bool = True
    ):
        self.search_space = search_space
        self.tlbo_population = tlbo_population
        self.tlbo_iterations = tlbo_iterations
        self.pso_particles = pso_particles
        self.pso_iterations = pso_iterations
        self.verbose = verbose
        
        self.tlbo_optimizer = TLBOOptimizer(
            search_space=search_space,
            population_size=tlbo_population,
            n_iterations=tlbo_iterations,
            verbose=False
        )
        
        self.pso_optimizer = PSOOptimizer(
            search_space=search_space,
            n_particles=pso_particles,
            n_iterations=pso_iterations,
            verbose=False
        )
        
        self.best_params = None
        self.best_score = float('inf')
        self.optimization_history = []
    
    def optimize(
        self,
        objective_fn: Callable[[Dict], float],
        use_refined_search_space: bool = True
    ) -> Tuple[Dict, float]:
        """
        Run sequential optimization: TLBO -> PSO.
        
        Args:
            objective_fn: Function that takes params dict and returns score
            use_refined_search_space: If True, refine search space around TLBO's best
            
        Returns:
            Tuple of (best_params, best_score)
        """
        if self.verbose:
            print("=" * 60)
            print("PHASE 1: TLBO Optimization (Exploration)")
            print("=" * 60)
        
        tlbo_params, tlbo_score = self.tlbo_optimizer.optimize(objective_fn)
        
        self.optimization_history.append({
            'phase': 'TLBO',
            'params': tlbo_params,
            'score': tlbo_score
        })
        
        if self.verbose:
            print(f"TLBO Best Score: {tlbo_score:.4f}")
            print(f"TLBO Best Params: {tlbo_params}")
            print()
            print("=" * 60)
            print("PHASE 2: PSO Optimization (Refinement)")
            print("=" * 60)
        
        if use_refined_search_space:
            refined_space = self._create_refined_search_space(tlbo_params)
            
            if self.verbose:
                print("Using refined search space around TLBO solution")
            
            pso_optimizer = PSOOptimizer(
                search_space=refined_space,
                n_particles=self.pso_particles,
                n_iterations=self.pso_iterations,
                verbose=False
            )
            
            pso_params, pso_score = pso_optimizer.optimize(
                objective_fn,
                initial_positions=[tlbo_params]
            )
            
            self.best_params = pso_params
            self.best_score = pso_score
        else:
            self.best_params = tlbo_params
            self.best_score = tlbo_score
        
        self.optimization_history.append({
            'phase': 'PSO',
            'params': self.best_params,
            'score': self.best_score
        })
        
        if self.verbose:
            print(f"PSO Best Score: {self.best_score:.4f}")
            print(f"Final Best Params: {self.best_params}")
            print()
            print("=" * 60)
            print("OPTIMIZATION COMPLETE")
            print("=" * 60)
            print(f"Improvement: {tlbo_score - self.best_score:.4f}")
        
        return self.best_params, self.best_score
    
    def _create_refined_search_space(
        self,
        center_params: Dict,
        refinement_factor: float = 0.3
    ) -> Dict[str, Tuple]:
        """
        Create a refined search space centered around the given parameters.
        
        Args:
            center_params: Parameters to center around
            refinement_factor: Fraction of original range to use
            
        Returns:
            Refined search space
        """
        refined_space = {}
        
        for param, (low, high) in self.search_space.items():
            center = center_params.get(param, (low + high) / 2)
            
            if isinstance(low, (int, float)) and isinstance(high, (int, float)):
                range_size = high - low
                refine_range = range_size * refinement_factor
                
                new_low = max(low, center - refine_range)
                new_high = min(high, center + refine_range)
                
                if isinstance(low, int):
                    refined_space[param] = (int(new_low), int(new_high))
                else:
                    refined_space[param] = (new_low, new_high)
            else:
                refined_space[param] = (low, high)
        
        return refined_space
    
    def get_optimization_summary(self) -> Dict:
        """Get summary of optimization process."""
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'tlbo_score': self.optimization_history[0]['score'] if self.optimization_history else None,
            'improvement': self.optimization_history[0]['score'] - self.best_score if len(self.optimization_history) > 1 else 0,
            'history': self.optimization_history
        }


def create_lstm_search_space() -> Dict[str, Tuple]:
    """Create default search space for LSTM hyperparameters."""
    return create_search_space(
        hidden_size=(32, 128),
        num_layers=(1, 4),
        dropout=(0.1, 0.5),
        learning_rate=(1e-4, 1e-2),
        batch_size=(16, 64),
        sequence_length=(20, 50)
    )