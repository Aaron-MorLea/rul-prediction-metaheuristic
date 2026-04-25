import numpy as np
from typing import Dict, List, Callable, Tuple, Any
import random


class PSOOptimizer:
    """
    Particle Swarm Optimization (PSO) for hyperparameter tuning.
    
    Based on: Eberhart & Kennedy (1995)
    
    Used in: Yilma et al. (2026) - Second phase of sequential optimization
    """
    
    def __init__(
        self,
        search_space: Dict[str, Tuple],
        n_particles: int = 20,
        n_iterations: int = 30,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        verbose: bool = True
    ):
        self.search_space = search_space
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.verbose = verbose
        
        self.param_names = list(search_space.keys())
        self.particles = []
        self.velocities = []
        self.best_positions = []
        self.best_scores = []
        self.global_best = None
        self.global_best_score = float('inf')
        
    def _initialize_particles(self) -> Tuple[List[Dict], List[Dict]]:
        """Initialize particles and velocities."""
        particles = []
        velocities = []
        
        for _ in range(self.n_particles):
            particle = {}
            velocity = {}
            for param, (low, high) in self.search_space.items():
                if isinstance(low, int) and isinstance(high, int):
                    particle[param] = random.randint(low, high)
                    velocity[param] = random.uniform(-(high - low) / 10, (high - low) / 10)
                elif isinstance(low, float):
                    particle[param] = random.uniform(low, high)
                    velocity[param] = random.uniform(-(high - low) / 10, (high - low) / 10)
                else:
                    particle[param] = random.choice(low)
                    velocity[param] = 0
            particles.append(particle)
            velocities.append(velocity)
        
        return particles, velocities
    
    def _clip(self, value: float, low: Any, high: Any) -> Any:
        """Clip value to search space bounds."""
        if isinstance(low, int) and isinstance(high, int):
            return int(np.clip(value, low, high))
        return np.clip(value, low, high)
    
    def _evaluate(self, particle: Dict) -> float:
        """Evaluate objective function."""
        raise NotImplementedError("Subclass must implement _evaluate method")
    
    def _get_value(self, particle: Dict, param: str) -> float:
        """Get numeric value from particle."""
        val = particle[param]
        if isinstance(self.search_space[param][0], (list, tuple)):
            return float(val)
        return val
    
    def _set_value(self, particle: Dict, param: str, value: float) -> Dict:
        """Set value in particle."""
        low, high = self.search_space[param]
        if isinstance(low, int) and isinstance(high, int):
            particle[param] = int(value)
        else:
            particle[param] = value
        return particle
    
    def optimize(
        self,
        objective_fn: Callable[[Dict], float],
        initial_positions: List[Dict] = None
    ) -> Tuple[Dict, float]:
        """
        Run PSO optimization.
        
        Args:
            objective_fn: Function that takes params dict and returns score
            initial_positions: Optional warm-start from previous optimizer
            
        Returns:
            Tuple of (best_params, best_score)
        """
        self._evaluate = objective_fn
        
        if initial_positions:
            self.particles = initial_positions[:self.n_particles]
            while len(self.particles) < self.n_particles:
                self.particles.append(self._random_particle())
        else:
            self.particles, self.velocities = self._initialize_particles()
        
        self.best_positions = [p.copy() for p in self.particles]
        self.best_scores = [self._evaluate(p) for p in self.particles]
        
        best_idx = np.argmin(self.best_scores)
        self.global_best = self.best_positions[best_idx].copy()
        self.global_best_score = self.best_scores[best_idx]
        
        for iteration in range(self.n_iterations):
            for i in range(self.n_particles):
                for param in self.param_names:
                    if param not in self.velocities:
                        self.velocities[i][param] = 0
                    
                    r1, r2 = random.random(), random.random()
                    
                    current_val = self._get_value(self.particles[i], param)
                    best_val = self._get_value(self.best_positions[i], param)
                    global_val = self._get_value(self.global_best, param)
                    
                    v_new = (self.w * self.velocities[i][param] +
                             self.c1 * r1 * (best_val - current_val) +
                             self.c2 * r2 * (global_val - current_val))
                    
                    self.velocities[i][param] = v_new
                    
                    low, high = self.search_space[param]
                    new_val = current_val + v_new
                    self.particles[i] = self._set_value(
                        self.particles[i], param, 
                        self._clip(new_val, low, high)
                    )
                
                score = self._evaluate(self.particles[i])
                
                if score < self.best_scores[i]:
                    self.best_scores[i] = score
                    self.best_positions[i] = self.particles[i].copy()
                    
                    if score < self.global_best_score:
                        self.global_best_score = score
                        self.global_best = self.particles[i].copy()
            
            if self.verbose and iteration % 5 == 0:
                print(f"PSO Iter {iteration + 1}/{self.n_iterations}, "
                      f"Best Score: {self.global_best_score:.4f}")
        
        return self.global_best, self.global_best_score
    
    def _random_particle(self) -> Dict:
        """Generate a random particle."""
        particle = {}
        for param, (low, high) in self.search_space.items():
            if isinstance(low, int) and isinstance(high, int):
                particle[param] = random.randint(low, high)
            elif isinstance(low, float):
                particle[param] = random.uniform(low, high)
            else:
                particle[param] = random.choice(low)
        return particle