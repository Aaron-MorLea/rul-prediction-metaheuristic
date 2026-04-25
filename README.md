# RUL Prediction with Metaheuristic Optimization

End-to-end predictive maintenance project for turbofan engines using stacked-LSTM with sequential metaheuristic hyperparameter optimization, Type-2 Fuzzy integration, and industrial-grade data pipelines.

## Overview

This project implements a **Remaining Useful Life (RUL)** prediction system for turbofan engines using:

- **Stacked-LSTM** model (inspired by Yilma et al., 2026)
- **Sequential Metaheuristic Optimization**: TLBO → PSO for hyperparameter tuning
- **Type-2 Fuzzy Integration**: Post-processing inspired by Melin et al. (2024)
- **PySpark Lakehouse**: Industrial data pipeline simulation
- **FastAPI + Streamlit**: Production-ready API and dashboard
- **LLM Assistant**: Natural language explanations for RUL predictions

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Data Ingestion │ ──▶ │  Feature Pipeline │ ──▶ │  Model Training │
│  (PySpark)      │     │  (Windows/Norm)   │     │  (LSTM+TLBO+PSO)│
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
┌─────────────────┐     ┌──────────────────┐             │
│  Dashboard      │ ◀── │  Fuzzy Inference │ ◀───────────┘
│  (Streamlit)    │     │  (Type-2)        │
└─────────────────┘     └──────────────────┘
```

## Methodology

Based on:
- **Yilma et al. (2026)**: "Remaining useful life prediction using sequential metaheuristic optimization of stacked-LSTM hyperparameters" - Chemical Engineering Research and Design
- **Melin et al. (2024)**: "A New Hybrid Approach for Clustering, Classification, and Prediction Combining General Type-2 Fuzzy Systems and Neural Networks" - Axioms

### Key Innovations
1. **Sequential TLBO → PSO** optimization for hyperparameter tuning
2. **Type-2 Fuzzy** integration for risk classification
3. **Industrial pipeline** design ready for Databricks migration

## Tech Stack

| Layer | Technology |
|-------|------------|
| Deep Learning | PyTorch |
| Metaheuristics | Custom TLBO/PSO implementations |
| Fuzzy Logic | scikit-fuzzy |
| Data Processing | PySpark (simulated lakehouse) |
| API | FastAPI |
| Dashboard | Streamlit |
| LLM Integration | LangChain + OpenAI (configurable) |
| Testing | pytest |

## Quick Start

### 1. Setup

```bash
# Clone and enter directory
cd rul-prediction-metaheuristic

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data

```bash
# Download NASA C-MAPSS dataset
# https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
# Place in data/raw/
```

### 3. Run Pipeline

```bash
# Feature engineering
python pipelines/feature_engineering.py

# Train model with metaheuristic optimization
python pipelines/training_pipeline.py

# Start API
uvicorn app.api:app --reload

# Start dashboard
streamlit run app/dashboard.py
```

## Project Structure

```
rul-prediction-metaheuristic/
├── data/
│   ├── raw/                  # C-MAPSS original (NASA)
│   ├── processed/            # Features, ventanas, splits
│   └── external/             # Simulación de datos de planta
├── models/
│   ├── lstm_model.py         # Stacked-LSTM
│   ├── tlbo_optimizer.py     # TLBO
│   ├── pso_optimizer.py      # PSO
│   ├── sequential_search.py  # Sequential optimization
│   └── fuzzy_integration.py  # Type-2 Fuzzy
├── pipelines/
│   ├── feature_engineering.py
│   ├── training_pipeline.py
│   └── scoring_pipeline.py
├── lakehouse_sim/
│   ├── ingest_batch_spark.py
│   └── delta_tables_demo.md
├── app/
│   ├── api.py                # FastAPI
│   ├── dashboard.py          # Streamlit
│   └── llm_assistant.py      # LLM-powered assistant
├── notebooks/
│   ├── 00_eda_cmapss.ipynb
│   ├── 01_baseline_lstm.ipynb
│   ├── 02_metaheuristic_tuning.ipynb
│   └── 03_fuzzy_integration_and_results.ipynb
├── tests/
│   ├── test_models.py
│   ├── test_pipelines.py
│   └── test_app.py
└── docs/
    ├── architecture.md
    └── databricks_mapping.md
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict_rul` | POST | Predict RUL for engine unit |
| `/predict_batch` | POST | Batch prediction |
| `/health` | GET | Health check |
| `/metrics` | GET | Model metrics |
| `/explanation` | POST | LLM-powered explanation |

## Results

Based on experiments with C-MAPSS dataset:

- **Binary Classification Accuracy**: ~98.9%
- **Multi-class Precision**: Up to 100%
- **RMSE (Regression)**: ~12-15 cycles

See `notebooks/` for detailed experiments.

## Databricks Migration

This project is designed to run locally but can be migrated to Databricks:

1. **Data Lake**: Replace `data/raw/` with ADLS/S3 paths
2. **Jobs**: Convert `pipelines/` to Databricks jobs
3. **MLflow**: Add model tracking (future work)
4. **Delta Tables**: Already simulated in `lakehouse_sim/`

See `docs/databricks_mapping.md` for detailed migration guide.

## License

MIT License

## References

- Yilma, A. A., Yang, C. L., & Woldegiorgis, B. H. (2026). Remaining useful life prediction using sequential metaheuristic optimization of stacked-LSTM hyperparameters. *Chemical Engineering Research and Design*, 228, 323-335.

- Ramírez, M., Melin, P., & Castillo, O. (2024). A New Hybrid Approach for Clustering, Classification, and Prediction Combining General Type-2 Fuzzy Systems and Neural Networks. *Axioms*, 13(6), 368.

- Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage propagation modeling for aircraft engine run-to-failure simulation. *Annual conference of the prognostics and health management society*.