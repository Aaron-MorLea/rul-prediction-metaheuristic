# 🔧 RUL Prediction System - Predictive Maintenance with AI

[![Python](https://img.shields.io/badge/Python-3.12+-blue?style=flat&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=flat&logo=pytorch)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https img.shields.io/badge/Streamlit-1.28+-red?style=flat&logo=streamlit)](https://streamlit.io/)

> **Remaining Useful Life (RUL) Prediction System** for turbofan engines using Deep Learning + Metaheuristic Optimization + Fuzzy Logic

---

## 🎯 Project Overview

This end-to-end predictive maintenance system predicts when an engine will fail, enabling proactive maintenance scheduling and reducing unplanned downtime.

### Key Features

- 🤖 **Deep Learning**: Stacked-LSTM neural network for time series prediction
- 🧬 **Metaheuristic Optimization**: TLBO + PSO sequential algorithm for hyperparameter tuning
- 🧠 **Fuzzy Logic**: Type-2 Fuzzy system for risk classification
- ⚡ **Production Ready**: FastAPI REST API + Streamlit Dashboard
- 📊 **Industrial Pipeline**: PySpark lakehouse simulation (Databricks-ready)

---

## 📈 Results

| Metric | Value |
|--------|-------|
| **RMSE** | ~45 cycles |
| **MAE** | ~39 cycles |
| **API Response** | <100ms |
| **Model Size** | ~270 KB |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           RUL PREDICTION SYSTEM                         │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  │
│  │   NASA      │───▶│  Feature    │───▶│   LSTM      │                  │
│  │  C-MAPSS    │    │ Engineering │    │   Model     │                  │
│  └─────────────┘    └─────────────┘    └──────┬──────┘                  │
│                                                │                         │
│                           ┌─────────────────────┼──────────────────────┐ │
│                           │ Metaheuristic       │ Fuzzy Inference     │ │
│                           │ Optimization        │ Type-2              │ │
│                           │ (TLBO → PSO)        │                     │ │
│                           └─────────────────────┼──────────────────────┘ │
│                                                 │                        │
│  ┌─────────────┐    ┌─────────────┐            │                        │
│  │  Streamlit  │◀───│   FastAPI   │◀───────────┘                        │
│  │  Dashboard  │    │    REST     │                                     │
│  └─────────────┘    └─────────────┘                                     │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| **Deep Learning** | PyTorch 2.0+ |
| **Metaheuristics** | Custom TLBO/PSO implementations |
| **Fuzzy Logic** | scikit-fuzzy |
| **Data Processing** | PySpark, pandas, numpy |
| **API** | FastAPI, uvicorn |
| **Dashboard** | Streamlit, Plotly |
| **LLM** | LangChain + OpenAI |
| **Testing** | pytest |

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Aaron-MorLea/rul-prediction-metaheuristic.git
cd rul-prediction-metaheuristic

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Project

```bash
# Start the API (port 8002)
uvicorn app.api:app --port 8002

# Start the Dashboard (port 8501)
streamlit run app/dashboard.py
```

### Use the API

```python
import requests
import numpy as np

BASE_URL = "http://127.0.0.1:8002"

# 30 timesteps x 59 features
sensor_data = np.random.randn(30, 59).tolist()

payload = {
    "unit_number": 1,
    "sensor_data": sensor_data,
    "sequence_length": 30
}

response = requests.post(f"{BASE_URL}/predict_rul", json=payload)
print(response.json())
```

**Example Response:**
```json
{
  "unit_number": 1,
  "predicted_rul": 83.06,
  "risk_level": "MEDIUM",
  "maintenance_action": "PLAN_NEXT",
  "recommendation": "Plan maintenance for next scheduled window",
  "confidence": 0.8
}
```

---

## 📂 Project Structure

```
rul-prediction-metaheuristic/
├── app/
│   ├── api.py              # FastAPI REST API
│   ├── dashboard.py        # Streamlit Dashboard
│   └── llm_assistant.py    # LLM-powered assistant
├── models/
│   ├── lstm_model.py       # Stacked-LSTM implementation
│   ├── tlbo_optimizer.py   # Teaching-Learning Based Optimization
│   ├── pso_optimizer.py    # Particle Swarm Optimization
│   ├── sequential_search.py # Sequential metaheuristic optimization
│   └── fuzzy_integration.py # Type-2 Fuzzy risk classification
├── pipelines/
│   ├── feature_engineering.py # Data preprocessing
│   └── training_pipeline.py   # Model training orchestration
├── lakehouse_sim/
│   └── ingest_batch_spark.py  # PySpark pipeline simulation
├── data/
│   └── raw/                 # NASA C-MAPSS dataset
├── docs/
│   └── proyecto_rul_paper.pdf # Full technical documentation
├── tests/
│   └── test_models.py       # Unit tests
└── README.md
```

---

## 📚 Academic Foundation

This project is based on cutting-edge research:

1. **Yilma et al. (2026)**: "Remaining useful life prediction using sequential metaheuristic optimization of stacked-LSTM hyperparameters" - *Chemical Engineering Research and Design*

2. **Melin et al. (2024)**: "A New Hybrid Approach for Clustering, Classification, and Prediction Combining General Type-2 Fuzzy Systems and Neural Networks" - *Axioms*

---

## 💼 Skills Demonstrated

This project showcases expertise in:

- ✅ **Python** (advanced)
- ✅ **Machine Learning / Deep Learning**
- ✅ **Time Series Forecasting**
- ✅ **Metaheuristic Optimization**
- ✅ **Fuzzy Logic Systems**
- ✅ **REST API Development**
- ✅ **Data Engineering** (PySpark)
- ✅ **Cloud-Ready Architecture** (Databricks)
- ✅ **Git / GitHub**
- ✅ **Technical Writing**

---

## 📄 Documentation

- **Technical Paper**: `docs/proyecto_rul_paper.pdf` - Complete academic documentation
- **API Docs**: Visit `http://localhost:8002/docs` when running

---

## 🔗 Links

- **GitHub**: https://github.com/Aaron-MorLea/rul-prediction-metaheuristic
- **NASA C-MAPSS Dataset**: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

---

## 📝 License

MIT License - Feel free to use this project for learning and professional purposes.

---

**Author**: Ing. Carlos Aaron Morales Leal  
**Project Date**: April 2026