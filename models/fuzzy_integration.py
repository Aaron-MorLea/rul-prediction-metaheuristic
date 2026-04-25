import numpy as np
from typing import Dict, Tuple, List
import skfuzzy as fuzz
from skfuzzy import control as fuzz_ctrl


class Type2FuzzyIntegrator:
    """
    Type-2 Fuzzy System for RUL Risk Classification.
    
    Inspired by:
    Melin et al. (2024) - "A New Hybrid Approach for Clustering, 
    Classification, and Prediction Combining General Type-2 Fuzzy 
    Systems and Neural Networks"
    
    This module integrates LSTM predictions with fuzzy logic for
    risk classification and maintenance recommendation.
    """
    
    def __init__(self, use_general_type2: bool = False):
        self.use_general_type2 = use_general_type2
        self.rul_system = None
        self._build_fuzzy_system()
    
    def _build_fuzzy_system(self):
        """Build Type-2 Fuzzy Inference System for RUL classification."""
        
        rul_input = fuzz_ctrl.Antecedent(
            np.arange(0, 301, 1), 'rul_cycles'
        )
        
        uncertainty_input = fuzz_ctrl.Antecedent(
            np.arange(0, 1.01, 0.01), 'model_uncertainty'
        )
        
        risk_level = fuzz_ctrl.Consequent(
            np.arange(0, 101, 1), 'risk_level'
        )
        
        maintenance_action = fuzz_ctrl.Consequent(
            np.arange(0, 101, 1), 'maintenance_action'
        )
        
        rul_input['low'] = fuzz.trimf(rul_input.universe, [0, 0, 100])
        rul_input['medium'] = fuzz.trimf(rul_input.universe, [50, 125, 200])
        rul_input['high'] = fuzz.trimf(rul_input.universe, [150, 300, 300])
        
        uncertainty_input['low'] = fuzz.trimf(uncertainty_input.universe, [0, 0, 0.3])
        uncertainty_input['medium'] = fuzz.trimf(uncertainty_input.universe, [0.2, 0.5, 0.8])
        uncertainty_input['high'] = fuzz.trimf(uncertainty_input.universe, [0.7, 1.0, 1.0])
        
        risk_level['critical'] = fuzz.trimf(risk_level.universe, [0, 0, 25])
        risk_level['high'] = fuzz.trimf(risk_level.universe, [15, 40, 60])
        risk_level['medium'] = fuzz.trimf(risk_level.universe, [40, 60, 80])
        risk_level['low'] = fuzz.trimf(risk_level.universe, [70, 100, 100])
        
        maintenance_action['immediate'] = fuzz.trimf(maintenance_action.universe, [0, 0, 20])
        maintenance_action['schedule_soon'] = fuzz.trimf(maintenance_action.universe, [10, 35, 55])
        maintenance_action['plan_next'] = fuzz.trimf(maintenance_action.universe, [45, 70, 90])
        maintenance_action['monitor'] = fuzz.trimf(maintenance_action.universe, [80, 100, 100])
        
        rules = [
            fuzz_ctrl.Rule(
                rul_input['low'] & uncertainty_input['low'],
                [risk_level['critical'], maintenance_action['immediate']]
            ),
            fuzz_ctrl.Rule(
                rul_input['low'] & uncertainty_input['medium'],
                [risk_level['critical'], maintenance_action['immediate']]
            ),
            fuzz_ctrl.Rule(
                rul_input['low'] & uncertainty_input['high'],
                [risk_level['high'], maintenance_action['schedule_soon']]
            ),
            fuzz_ctrl.Rule(
                rul_input['medium'] & uncertainty_input['low'],
                [risk_level['medium'], maintenance_action['plan_next']]
            ),
            fuzz_ctrl.Rule(
                rul_input['medium'] & uncertainty_input['medium'],
                [risk_level['medium'], maintenance_action['plan_next']]
            ),
            fuzz_ctrl.Rule(
                rul_input['medium'] & uncertainty_input['high'],
                [risk_level['high'], maintenance_action['schedule_soon']]
            ),
            fuzz_ctrl.Rule(
                rul_input['high'] & uncertainty_input['low'],
                [risk_level['low'], maintenance_action['monitor']]
            ),
            fuzz_ctrl.Rule(
                rul_input['high'] & uncertainty_input['medium'],
                [risk_level['medium'], maintenance_action['plan_next']]
            ),
            fuzz_ctrl.Rule(
                rul_input['high'] & uncertainty_input['high'],
                [risk_level['medium'], maintenance_action['plan_next']]
            ),
        ]
        
        self.rul_system = fuzz_ctrl.ControlSystem(rules)
        self.rul_simulator = fuzz_ctrl.ControlSystemSimulation(self.rul_system)
    
    def classify_risk(
        self,
        rul_prediction: float,
        uncertainty: float = 0.2
    ) -> Dict:
        """
        Classify risk level and maintenance action based on RUL prediction.
        
        Args:
            rul_prediction: Predicted Remaining Useful Life in cycles
            uncertainty: Model uncertainty (0-1)
            
        Returns:
            Dict with risk_level, maintenance_action, and recommendations
        """
        try:
            self.rul_simulator.input['rul_cycles'] = min(300, max(0, rul_prediction))
            self.rul_simulator.input['model_uncertainty'] = min(1.0, max(0.0, uncertainty))
            
            self.rul_simulator.compute()
            
            risk = self.rul_simulator.output['risk_level']
            action = self.rul_simulator.output['maintenance_action']
            
            if risk < 25:
                risk_label = 'CRITICAL'
                action_label = 'IMMEDIATE'
                recommendation = 'Stop operation and perform maintenance NOW'
            elif risk < 45:
                risk_label = 'HIGH'
                action_label = 'SCHEDULE_SOON'
                recommendation = 'Schedule maintenance within 24-48 hours'
            elif risk < 70:
                risk_label = 'MEDIUM'
                action_label = 'PLAN_NEXT'
                recommendation = 'Plan maintenance for next scheduled window'
            else:
                risk_label = 'LOW'
                action_label = 'MONITOR'
                recommendation = 'Continue normal operation, monitor closely'
            
            return {
                'rul_prediction': rul_prediction,
                'uncertainty': uncertainty,
                'risk_level': risk,
                'risk_label': risk_label,
                'maintenance_action': action,
                'action_label': action_label,
                'recommendation': recommendation
            }
            
        except Exception as e:
            return self._fallback_classification(rul_prediction, uncertainty)
    
    def _fallback_classification(
        self,
        rul_prediction: float,
        uncertainty: float
    ) -> Dict:
        """Fallback classification without fuzzy inference."""
        if rul_prediction < 50:
            return {
                'rul_prediction': rul_prediction,
                'uncertainty': uncertainty,
                'risk_level': 90,
                'risk_label': 'CRITICAL',
                'maintenance_action': 10,
                'action_label': 'IMMEDIATE',
                'recommendation': 'Stop operation and perform maintenance NOW'
            }
        elif rul_prediction < 100:
            return {
                'rul_prediction': rul_prediction,
                'uncertainty': uncertainty,
                'risk_level': 60,
                'risk_label': 'HIGH',
                'maintenance_action': 40,
                'action_label': 'SCHEDULE_SOON',
                'recommendation': 'Schedule maintenance within 24-48 hours'
            }
        elif rul_prediction < 180:
            return {
                'rul_prediction': rul_prediction,
                'uncertainty': uncertainty,
                'risk_level': 40,
                'risk_label': 'MEDIUM',
                'maintenance_action': 65,
                'action_label': 'PLAN_NEXT',
                'recommendation': 'Plan maintenance for next scheduled window'
            }
        else:
            return {
                'rul_prediction': rul_prediction,
                'uncertainty': uncertainty,
                'risk_level': 15,
                'risk_label': 'LOW',
                'maintenance_label': 'MONITOR',
                'recommendation': 'Continue normal operation, monitor closely'
            }
    
    def batch_classify(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray = None
    ) -> List[Dict]:
        """Classify risk for multiple predictions."""
        if uncertainties is None:
            uncertainties = np.full(len(predictions), 0.2)
        
        results = []
        for rul, unc in zip(predictions, uncertainties):
            results.append(self.classify_risk(rul, unc))
        
        return results


def create_fuzzy_integrator(use_general_type2: bool = False) -> Type2FuzzyIntegrator:
    """Create and return a Type-2 Fuzzy Integrator."""
    return Type2FuzzyIntegrator(use_general_type2=use_general_type2)