import numpy as np
import torch
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.lstm_model import build_model, train_model, evaluate_model
from models.sequential_search import SequentialMetaheuristicOptimizer, create_lstm_search_space
from models.fuzzy_integration import create_fuzzy_integrator
from pipelines.feature_engineering import prepare_data


class TrainingPipeline:
    """
    End-to-end training pipeline for RUL prediction.
    
    Orchestrates: Data preparation -> Metaheuristic optimization -> Model training -> Evaluation
    """
    
    def __init__(
        self,
        data_dir: str = 'data/raw',
        subset: str = 'FD001',
        max_rul: int = 125,
        sequence_length: int = 30,
        device: str = 'cpu',
        output_dir: str = 'models/checkpoints'
    ):
        self.data_dir = data_dir
        self.subset = subset
        self.max_rul = max_rul
        self.sequence_length = sequence_length
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.feature_cols = None
        
        self.best_model = None
        self.best_params = None
        self.best_score = float('inf')
        
    def load_data(self, val_split: float = 0.2):
        """Load and split data."""
        print("Loading data...")
        X_train, y_train, X_test, y_test, feature_cols = prepare_data(
            self.data_dir, self.subset, self.max_rul, self.sequence_length
        )
        
        n_train = int(len(X_train) * (1 - val_split))
        self.X_train = X_train[:n_train]
        self.y_train = y_train[:n_train]
        self.X_val = X_train[n_train:]
        self.y_val = y_train[n_train:]
        
        self.X_test = X_test
        self.y_test = y_test
        self.feature_cols = feature_cols
        
        print(f"Train: {len(self.X_train)}, Val: {len(self.X_val)}, Test: {len(self.X_test)}")
        
    def _objective_function(self, params: Dict) -> float:
        """Objective function for hyperparameter optimization."""
        
        model = build_model(
            input_size=self.X_train.shape[2],
            hidden_size=params.get('hidden_size', 64),
            num_layers=params.get('num_layers', 2),
            dropout=params.get('dropout', 0.2),
            device=self.device
        )
        
        seq_length = int(params.get('sequence_length', self.sequence_length))
        
        if seq_length != self.sequence_length:
            X_train_seq = self.X_train[:, -seq_length:, :]
            X_val_seq = self.X_val[:, -seq_length:, :]
        else:
            X_train_seq = self.X_train
            X_val_seq = self.X_val
        
        history = train_model(
            model=model,
            X_train=X_train_seq,
            y_train=self.y_train,
            X_val=X_val_seq,
            y_val=self.y_val,
            epochs=10,
            batch_size=int(params.get('batch_size', 32)),
            learning_rate=params.get('learning_rate', 0.001),
            device=self.device,
            early_stopping_patience=5
        )
        
        return history['best_val_loss']
    
    def run_hyperparameter_optimization(
        self,
        use_sequential: bool = True,
        quick_test: bool = True
    ) -> Dict:
        """Run hyperparameter optimization using TLBO + PSO."""
        
        if quick_test:
            search_space = {
                'hidden_size': (32, 64),
                'num_layers': (1, 2),
                'dropout': (0.1, 0.3),
                'learning_rate': (1e-3, 1e-2),
                'batch_size': (16, 32),
                'sequence_length': (20, 30)
            }
        else:
            search_space = create_lstm_search_space()
        
        if use_sequential:
            print("\n" + "="*60)
            print("Running Sequential Metaheuristic Optimization")
            print("Phase 1: TLBO -> Phase 2: PSO")
            print("="*60 + "\n")
            
            optimizer = SequentialMetaheuristicOptimizer(
                search_space=search_space,
                tlbo_population=10 if quick_test else 20,
                tlbo_iterations=5 if quick_test else 25,
                pso_particles=10 if quick_test else 20,
                pso_iterations=5 if quick_test else 20,
                verbose=True
            )
            
            best_params, best_score = optimizer.optimize(
                self._objective_function,
                use_refined_search_space=True
            )
            
            summary = optimizer.get_optimization_summary()
        else:
            from models.tlbo_optimizer import TLBOOptimizer
            
            optimizer = TLBOOptimizer(
                search_space=search_space,
                population_size=10 if quick_test else 20,
                n_iterations=5 if quick_test else 30,
                verbose=True
            )
            
            best_params, best_score = optimizer.optimize(self._objective_function)
        
        self.best_params = best_params
        self.best_score = best_score
        
        return best_params
    
    def train_final_model(self, params: Optional[Dict] = None) -> Dict:
        """Train final model with best parameters."""
        
        if params is None:
            params = self.best_params
        
        print("\n" + "="*60)
        print("Training Final Model")
        print("="*60)
        
        model = build_model(
            input_size=self.X_train.shape[2],
            hidden_size=params.get('hidden_size', 64),
            num_layers=params.get('num_layers', 2),
            dropout=params.get('dropout', 0.2),
            device=self.device
        )
        
        seq_length = int(params.get('sequence_length', self.sequence_length))
        
        X_train_final = self.X_train[:, -seq_length:, :] if seq_length != self.sequence_length else self.X_train
        X_val_final = self.X_val[:, -seq_length:, :] if seq_length != self.sequence_length else self.X_val
        
        history = train_model(
            model=model,
            X_train=X_train_final,
            y_train=self.y_train,
            X_val=X_val_final,
            y_val=self.y_val,
            epochs=50,
            batch_size=params.get('batch_size', 32),
            learning_rate=params.get('learning_rate', 0.001),
            device=self.device,
            early_stopping_patience=10
        )
        
        self.best_model = model
        
        model_path = self.output_dir / f'rul_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'
        model.save(str(model_path))
        print(f"Model saved to {model_path}")
        
        return {
            'model_path': str(model_path),
            'history': history,
            'best_params': params
        }
    
    def evaluate(self) -> Dict:
        """Evaluate model on test set."""
        
        if self.best_model is None:
            raise ValueError("Model not trained. Run train_final_model first.")
        
        print("\n" + "="*60)
        print("Evaluating Model")
        print("="*60)
        
        metrics = evaluate_model(
            self.best_model,
            self.X_test,
            self.y_test,
            device=self.device
        )
        
        fuzzy = create_fuzzy_integrator()
        predictions = metrics['predictions'].flatten()
        
        fuzzy_results = fuzzy.batch_classify(predictions)
        
        risk_distribution = {}
        for result in fuzzy_results:
            label = result['risk_label']
            risk_distribution[label] = risk_distribution.get(label, 0) + 1
        
        print(f"\nTest RMSE: {metrics['rmse']:.2f}")
        print(f"Test MAE: {metrics['mae']:.2f}")
        print(f"Risk Distribution: {risk_distribution}")
        
        return {
            'metrics': metrics,
            'fuzzy_results': fuzzy_results,
            'risk_distribution': risk_distribution
        }
    
    def save_results(self, results: Dict):
        """Save training results to JSON."""
        
        results_path = self.output_dir / 'training_results.json'
        
        results_to_save = {
            'timestamp': datetime.now().isoformat(),
            'best_params': self.best_params,
            'best_score': self.best_score,
            'test_rmse': results.get('metrics', {}).get('rmse'),
            'test_mae': results.get('metrics', {}).get('mae'),
            'risk_distribution': results.get('risk_distribution', {})
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"Results saved to {results_path}")
    
    def run(self, optimize_hyperparams: bool = True, quick_test: bool = True):
        """Run complete training pipeline."""
        
        self.load_data()
        
        if optimize_hyperparams:
            self.run_hyperparameter_optimization(quick_test=quick_test)
        
        training_results = self.train_final_model()
        
        eval_results = self.evaluate()
        
        self.save_results(eval_results)
        
        return {
            'best_params': self.best_params,
            'training': training_results,
            'evaluation': eval_results
        }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/raw')
    parser.add_argument('--subset', default='FD001')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    parser.add_argument('--no-optimize', action='store_true', help='Skip hyperparameter optimization')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    pipeline = TrainingPipeline(
        data_dir=args.data_dir,
        subset=args.subset,
        device=device
    )
    
    results = pipeline.run(
        optimize_hyperparams=not args.no_optimize,
        quick_test=args.quick
    )
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)