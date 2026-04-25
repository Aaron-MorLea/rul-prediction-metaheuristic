"""
ML Pipeline Testing Suite
"""

import pytest
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm_model import StackedLSTMRegressor, create_sequences, build_model
from models.tlbo_optimizer import TLBOOptimizer, create_search_space
from models.pso_optimizer import PSOOptimizer
from models.fuzzy_integration import Type2FuzzyIntegrator


class TestStackedLSTM:
    """Tests for LSTM model."""
    
    def test_model_creation(self):
        model = build_model(input_size=21, hidden_size=32, num_layers=2)
        assert model is not None
        assert model.hidden_size == 32
        assert model.num_layers == 2
    
    def test_forward_pass(self):
        model = build_model(input_size=21, hidden_size=32, num_layers=1)
        x = torch.randn(8, 30, 21)
        output = model(x)
        assert output.shape == (8, 1)
    
    def test_sequence_creation(self):
        data = np.random.randn(100, 21)
        labels = np.random.randint(0, 125, 100)
        X, y = create_sequences(data, labels, sequence_length=20)
        assert X.shape == (80, 20, 21)
        assert y.shape == (80,)


class TestTLBO:
    """Tests for TLBO optimizer."""
    
    def test_search_space_creation(self):
        space = create_search_space()
        assert 'hidden_size' in space
        assert 'num_layers' in space
    
    def test_optimization(self):
        def objective(params):
            return params['hidden_size'] + params['num_layers']
        
        space = {'hidden_size': (32, 64), 'num_layers': (1, 3)}
        optimizer = TLBOOptimizer(space, population_size=5, n_iterations=3)
        best_params, score = optimizer.optimize(objective)
        
        assert best_params is not None
        assert 32 <= best_params['hidden_size'] <= 64
        assert 1 <= best_params['num_layers'] <= 3


class TestPSO:
    """Tests for PSO optimizer."""
    
    def test_optimization(self):
        def objective(params):
            return params['hidden_size'] / 100
        
        space = {'hidden_size': (32, 128)}
        optimizer = PSOOptimizer(space, n_particles=5, n_iterations=3)
        best_params, score = optimizer.optimize(objective)
        
        assert best_params is not None
        assert 32 <= best_params['hidden_size'] <= 128


class TestFuzzyIntegration:
    """Tests for Type-2 Fuzzy integration."""
    
    def test_fuzzy_system_creation(self):
        fuzzy = Type2FuzzyIntegrator()
        assert fuzzy is not None
    
    def test_risk_classification(self):
        fuzzy = Type2FuzzyIntegrator()
        result = fuzzy.classify_risk(rul_prediction=50, uncertainty=0.2)
        
        assert 'risk_level' in result
        assert 'recommendation' in result
        assert result['rul_prediction'] == 50
    
    def test_batch_classification(self):
        fuzzy = Type2FuzzyIntegrator()
        predictions = np.array([30, 80, 150, 250])
        results = fuzzy.batch_classify(predictions)
        
        assert len(results) == 4
        assert all('risk_level' in r for r in results)


class TestPipeline:
    """Integration tests."""
    
    def test_full_prediction_flow(self):
        model = build_model(input_size=21, hidden_size=32, num_layers=1)
        
        x = torch.randn(4, 30, 21)
        output = model(x)
        
        fuzzy = Type2FuzzyIntegrator()
        rul = output[0].item()
        result = fuzzy.classify_risk(rul, 0.2)
        
        assert 'risk_label' in result
        assert 'recommendation' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])