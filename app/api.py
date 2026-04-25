from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
import numpy as np
import joblib
import os

from models.lstm_model import StackedLSTMRegressor
from models.fuzzy_integration import create_fuzzy_integrator


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, fuzzy_integrator
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "..", "models", "checkpoints", "rul_model.pt")
    model_path = os.path.normpath(model_path)
    device = "cpu"
    
    fuzzy_integrator = create_fuzzy_integrator()
    
    if os.path.exists(model_path):
        try:
            model = StackedLSTMRegressor.load(model_path, device=device)
            print(f"Modelo cargado desde: {model_path}")
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            model = None
    else:
        print(f"Modelo no encontrado en: {model_path}")
        model = None
    
    yield
    
    # Cleanup
    model = None


app = FastAPI(
    title="RUL Prediction API",
    description="Remaining Useful Life Prediction with LSTM + Fuzzy Integration",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RULPredictionRequest(BaseModel):
    unit_number: int
    sensor_data: List[List[float]]
    sequence_length: int = 30


class RULPredictionResponse(BaseModel):
    unit_number: int
    predicted_rul: float
    risk_level: str
    maintenance_action: str
    recommendation: str
    confidence: float


class BatchPredictionRequest(BaseModel):
    predictions: List[dict]


model = None
fuzzy_integrator = None


@app.get("/")
async def root():
    return {
        "message": "RUL Prediction API",
        "version": "1.0.0",
        "endpoints": [
            "/predict_rul",
            "/predict_batch",
            "/health",
            "/metrics"
        ]
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict_rul", response_model=RULPredictionResponse)
async def predict_rul(request: RULPredictionRequest):
    if model is None:
        raise HTTPException(status_code=400, detail="Model not loaded")
    
    try:
        X = np.array(request.sensor_data, dtype=np.float32)
        
        if X.shape[0] < request.sequence_length:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least {request.sequence_length} time steps"
            )
        
        X = X[-request.sequence_length:].reshape(1, request.sequence_length, -1)
        
        prediction = model.predict(X)[0][0]
        
        uncertainty = 0.2
        
        fuzzy_result = fuzzy_integrator.classify_risk(prediction, uncertainty)
        
        return RULPredictionResponse(
            unit_number=request.unit_number,
            predicted_rul=float(prediction),
            risk_level=fuzzy_result['risk_label'],
            maintenance_action=fuzzy_result['action_label'],
            recommendation=fuzzy_result['recommendation'],
            confidence=1.0 - uncertainty
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch")
async def predict_batch(request: BatchPredictionRequest):
    if model is None:
        raise HTTPException(status_code=400, detail="Model not loaded")
    
    results = []
    
    for item in request.predictions:
        try:
            X = np.array(item['sensor_data'], dtype=np.float32)
            seq_len = item.get('sequence_length', 30)
            
            X = X[-seq_len:].reshape(1, seq_len, -1)
            
            prediction = model.predict(X)[0][0]
            
            fuzzy_result = fuzzy_integrator.classify_risk(prediction, 0.2)
            
            results.append({
                'unit_number': item.get('unit_number', 0),
                'predicted_rul': float(prediction),
                'risk_level': fuzzy_result['risk_label'],
                'maintenance_action': fuzzy_result['action_label'],
                'recommendation': fuzzy_result['recommendation']
            })
        
        except Exception as e:
            results.append({
                'unit_number': item.get('unit_number', 0),
                'error': str(e)
            })
    
    return {"results": results}


@app.get("/metrics")
async def metrics():
    return {
        "model_type": "StackedLSTM",
        "optimization": "TLBO + PSO",
        "fuzzy_integration": "Type-2 Fuzzy",
        "max_rul_cap": 125,
        "dataset": "NASA C-MAPSS"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)