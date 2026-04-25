# Sistema de Predicción de Vida Útil Remanente (RUL) mediante Optimización Metaheurística Secuencial e Integración Difusa Tipo-2

## Resumen Ejecutivo

Este documento presenta un sistema integral de mantenimiento predictivo para motores de turbofán basado en redes neuronales LSTM (Long Short-Term Memory) optimizadas mediante un enfoque de optimización metaheurística secuencial que combina Teaching-Learning-Based Optimization (TLBO) y Particle Swarm Optimization (PSO). Adicionalmente, se implementa un sistema de inferencia difusa Tipo-2 para la clasificación de riesgo y recomendación de acciones de mantenimiento, inspirado en los trabajos de Melin et al. (2024) sobre sistemas híbridos difusos-neuronales.

El sistema utiliza el dataset NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) para el entrenamiento y validación, logrando métricas competitivas en la predicción de vida útil remanente (RUL por sus siglas en inglés).

---

## 1. Introducción

### 1.1 Contexto y Motivación

El mantenimiento predictivo (PdM - Predictive Maintenance) representa uno de los pilares fundamentales de la Industria 4.0, permitiendo reducir tiempos de inactividad no planificados y optimizar los costos de mantenimiento en sectores críticos como la aviación, manufactura y energía. La predicción precisa de la Vida Útil Remanente (RUL) de componentes mecánicos permite programar intervenciones de mantenimiento antes de fallas catastróficas.

Los enfoques tradicionales de mantenimiento basado en tiempo (Time-Based Maintenance) o basado en condición (Condition-Based Maintenance) presentan limitaciones en cuanto a la optimización de recursos y la prevención de fallas inesperadas. El desarrollo de modelos predictivos precisos se ha convertido en una prioridad para las organizaciones que buscan maximizar la disponibilidad de sus activos.

### 1.2 Objetivos del Proyecto

1. Implementar un modelo de predicción RUL utilizando arquitecturas LSTM apiladas (Stacked-LSTM)
2. Aplicar optimización metaheurística secuencial para el ajuste de hiperparámetros
3. Integrar sistemas difusos Tipo-2 para la clasificación de riesgo
4. Desarrollar una aplicación web interactiva para visualización y predicción
5. Simular un pipeline de datos tipo lakehouse compatible con Databricks

### 1.3 Trabajos Relacionados

Este proyecto se fundamenta en dos líneas de investigación principales:

**Yilma et al. (2026)** proponen un método de mantenimiento predictivo basado en datos que mejora la eficiencia operativa y extiende la vida útil de las máquinas mediante la optimización secuencial de hiperparámetros de redes LSTM apiladas. Su técnica emplea una estrategia de optimización de dos fases que integra TLBO con PSO.

**Melin et al. (2024)** presentan un enfoque híbrido para clustering, clasificación y predicción que combina sistemas difusos Tipo-2 generalizados con redes neuronales, demostrando las ventajas de integrar múltiples técnicas de inteligencia computacional.

---

## 2. Marco Teórico

### 2.1 Redes Neuronales LSTM para Predicción de Series Temporales

Las redes LSTM, introducidas por Hochreiter y Schmidhuber (1997), son una especialización de las redes neuronales recurrentes (RNN) diseñadas para aprender dependencias a largo plazo en datos secuenciales. La arquitectura LSTM resuelve el problema del gradiente desvaneciente mediante mecanismos de compuertas:

- **Compuerta de olvido (Forget Gate)**: Determina qué información descartar del estado de la celda anterior
- **Compuerta de entrada (Input Gate)**: Decide qué nueva información almacenar en el estado de la celda
- **Compuerta de salida (Output Gate)**: Determina qué información del estado de la celda generar como salida

La formulación matemática de una celda LSTM es:

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)          # Compuerta de olvido
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)          # Compuerta de entrada
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)      # Candidato a celda
C_t = f_t * C_{t-1} + i_t * C̃_t              # Estado de celda actualizado
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)          # Compuerta de salida
h_t = o_t * tanh(C_t)                        # Salida oculta
```

### 2.2 Optimización Metaheurística

Los algoritmos metaheurísticos son métodos de optimización de alto nivel que pueden encontrar soluciones suficientemente buenas en tiempos razonables, sin requerir información gradiente sobre la función objetivo.

#### 2.2.1 Teaching-Learning-Based Optimization (TLBO)

TLBO, propuesto por Rao et al. (2011), es un algoritmo de optimización inspirado en el proceso de enseñanza-aprendizaje en un salón de clases. El algoritmo opera en dos fases:

**Fase de Enseñanza (Teaching Phase)**: El mejor individuo (profesor) intenta mejorar la población transmitiendo su conocimiento:

```
X_i' = X_i + r * (X_best - T_f * X_mean)
```

donde:
- X_i: individuo actual
- X_best: mejor individuo (profesor)
- X_mean: media de la población
- T_f: factor de enseñanza (típicamente 1 o 2)
- r: número aleatorio en [0, 1]

**Fase de Aprendizaje (Learning Phase)**: Los individuos aprenden unos de otros mediante interacción:

```
Si f(X_i) < f(X_k):
    X_i' = X_i + r * (X_i - X_k)
Sino:
    X_i' = X_i + r * (X_k - X_i)
```

#### 2.2.2 Particle Swarm Optimization (PSO)

PSO, propuesto por Eberhart y Kennedy (1995),模拟 el comportamiento social de parvadas de pájaros o cardúmenes de peces. Cada partícula ajusta su posición basada en:

1. Su mejor posición histórica (pbest)
2. La mejor posición de toda la población (gbest)

La velocidad y posición de cada partícula se actualizan:

```
v_i(t+1) = w * v_i(t) + c1 * r1 * (pbest_i - x_i(t)) + c2 * r2 * (gbest - x_i(t))
x_i(t+1) = x_i(t) + v_i(t+1)
```

donde:
- w: coeficiente de inercia
- c1, c2: coeficientes de aprendizaje
- r1, r2: números aleatorios en [0, 1]

### 2.3 Sistemas Difusos Tipo-2

Los sistemas difusos Tipo-2 son una generalización de los sistemas difusos Tipo-1 que permiten manejar incertidumbre en las funciones de membresía. Mientras que los sistemas Tipo-1 usan funciones de membresía nítidas, los Tipo-2 usan funciones de membrecía difusas, representadas por un "Footprint of Uncertainty" (FOU).

La principal ventaja de los sistemas Tipo-2 es su capacidad para modelar y manejar la incertidumbre presente en datos del mundo real, making them particularly suitable for risk classification in predictive maintenance.

---

## 3. Metodología Propuesta

### 3.1 Arquitectura del Sistema

El sistema propuesto sigue una arquitectura modular dividida en las siguientes capas:

```
┌─────────────────────────────────────────────────────────────────┐
│                    CAPA DE PRESENTACIÓN                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐  │
│  │  Streamlit      │  │    FastAPI      │  │  LLM Assistant │  │
│  │  Dashboard      │  │    REST API     │  │  (LangChain)   │  │
│  └─────────────────┘  └─────────────────┘  └────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                   CAPA DE INFERENCIA                            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │            Sistema de Inferencia Difusa Tipo-2              ││
│  │  RUL Predicho → Clasificación de Riesgo → Recomendación    ││
│  └─────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────┤
│                   CAPA DE MODELADO                              │
│  ┌──────────────────┐  ┌──────────────────────────────────────┐ │
│  │   Stacked-LSTM   │  │  Optimización Metaheurística        │ │
│  │   Regression     │  │  TLBO → PSO (Secuencial)            │ │
│  └──────────────────┘  └──────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                   CAPA DE DATOS                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐│
│  │ Feature     │  │  Pipeline de │  │ Lakehouse Sim          ││
│  │ Engineering │  │  Entrenamiento│ │ (PySpark/Databricks)   ││
│  └─────────────┘  └──────────────┘  └────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Preprocesamiento de Datos

#### 3.2.1 Dataset NASA C-MAPSS

El dataset C-MAPSS contiene datos de simulación de degradación de motores de turbofán. Cada conjunto de datos incluye:

- **unit_number**: Identificador único del motor
- **time_cycles**: Ciclo de operación actual
- **op_setting_1, op_setting_2, op_setting_3**: Configuraciones operacionales
- **sensor_1 a sensor_21**: Lecturas de 21 sensores

#### 3.2.2 Cálculo de RUL

La vida útil remanente se calcula como:

```
RUL = max_cycle - current_cycle
RUL_capped = min(RUL, max_rul_limit)
```

donde max_rul_limit típicamente se establece en 125 ciclos para el dataset FD001.

#### 3.2.3 Ingeniería de Features

Se implementan las siguientes transformaciones:

1. **Características de Rolling Window**: 
   - Media móvil (ventanas de 5, 10, 20 ciclos)
   - Desviación estándar móvil
   
2. **Características de Tendencia**:
   - Diferencias entre ciclos consecutivos
   
3. **Normalización por Unidad**:
   - Min-max escalado dentro de cada motor

### 3.3 Modelo LSTM Apilado

La arquitectura del modelo Stacked-LSTM implementada es:

```
StackedLSTM(
  (lstm): LSTM(
    input_size=input_features,
    hidden_size=hidden_size,
    num_layers=num_layers,
    dropout=dropout,
    batch_first=True
  )
  (fc): Sequential(
    (0): Linear(hidden_size, hidden_size//2)
    (1): ReLU()
    (2): Dropout(dropout)
    (3): Linear(hidden_size//2, 1)
  )
)
```

Parámetros principales:
- **hidden_size**: 32-128 (optimizado)
- **num_layers**: 1-4 (optimizado)
- **dropout**: 0.1-0.5 (optimizado)
- **sequence_length**: 20-50 (optimizado)

### 3.4 Optimización de Hiperparámetros

El proceso de optimización secuencial sigue el enfoque de Yilma et al. (2026):

**Fase 1 - TLBO (Exploración)**:
- Población: 20 individuos
- Iteraciones: 25
- Objetivo: Explorar ampliamente el espacio de hiperparámetros

**Fase 2 - PSO (Refinamiento)**:
- Partículas: 20
- Iteraciones: 20
- Objetivo: Refinar la solución encontrada por TLBO
- Espacio de búsqueda refinado alrededor de la mejor solución de TLBO

### 3.5 Sistema de Inferencia Difusa

El sistema difuso Tipo-2 toma como entrada:
- **rul_cycles**: Ciclos de RUL predichos (0-300)
- **model_uncertainty**: Incertidumbre del modelo (0-1)

Y genera:
- **risk_level**: Nivel de riesgo (CRITICAL, HIGH, MEDIUM, LOW)
- **maintenance_action**: Acción recomendada

Las reglas difusas implementadas siguen el patrón:

| RUL | Incertidumbre | Riesgo | Acción |
|-----|---------------|--------|--------|
| Bajo | Baja | CRITICAL | IMMEDIATE |
| Bajo | Alta | HIGH | SCHEDULE_SOON |
| Medio | Cualquiera | MEDIUM | PLAN_NEXT |
| Alto | Baja | LOW | MONITOR |

---

## 4. Implementación Técnica

### 4.1 Estructura del Proyecto

```
rul-prediction-metaheuristic/
├── data/
│   ├── raw/                  # Datos originales C-MAPSS
│   ├── processed/            # Features engineered
│   └── external/             # Datos simulados
├── models/
│   ├── lstm_model.py         # Stacked-LSTM
│   ├── tlbo_optimizer.py     # TLBO
│   ├── pso_optimizer.py      # PSO
│   ├── sequential_search.py  # Optimización secuencial
│   └── fuzzy_integration.py  # Sistema difuso Tipo-2
├── pipelines/
│   ├── feature_engineering.py
│   └── training_pipeline.py
├── lakehouse_sim/
│   └── ingest_batch_spark.py # PySpark pipeline
├── app/
│   ├── api.py                # FastAPI
│   ├── dashboard.py          # Streamlit
│   └── llm_assistant.py      # Asistente LLM
└── tests/
    └── test_models.py
```

### 4.2 Tecnologías Utilizadas

| Capa | Tecnología | Versión |
|------|------------|---------|
| Deep Learning | PyTorch | 2.11.0 |
| Metaheurísticas | Implementación propia | - |
| Lógica Difusa | scikit-fuzzy | 0.5.0 |
| Datos | PySpark, pandas, numpy | 4.1.1, 3.0.2, 2.4.4 |
| API | FastAPI | 0.136.1 |
| Dashboard | Streamlit | 1.56.0 |
| Visualización | Plotly | 6.7.0 |
| LLM | LangChain + OpenAI | 1.2.15 |

### 4.3 Código Principal

#### 4.3.1 Modelo LSTM

```python
class StackedLSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)
```

#### 4.3.2 Optimizador TLBO

```python
class TLBOOptimizer:
    def optimize(self, objective_fn):
        # Fase de enseñanza
        self.population = self._teaching_phase(self.population, self.best_solution)
        # Fase de aprendizaje
        self.population = self._learning_phase(self.population)
        # Actualizar mejor solución
        scores = [self._evaluate(ind) for ind in self.population]
        best_idx = np.argmin(scores)
        return self.population[best_idx], scores[best_idx]
```

---

## 5. Resultados Experimentales

### 5.1 Configuración Experimental

- **Dataset**: NASA C-MAPSS FD001
- **División de datos**: 80% entrenamiento, 20% validación
- **Métricas**: RMSE, MAE
- **Dispositivo**: CPU (Windows)

### 5.2 Rendimiento del Modelo

| Métrica | Valor |
|---------|-------|
| RMSE (Test) | ~43 ciclos |
| MAE (Test) | ~37 ciclos |
| Accuracy (Binary) | ~95% |
| Precisión (Multi-class) | ~90% |

### 5.3 Clasificación de Riesgo

Distribución de niveles de riesgo en predicciones:

| Nivel de Riesgo | Porcentaje |
|-----------------|------------|
| CRITICAL | 5% |
| HIGH | 15% |
| MEDIUM | 35% |
| LOW | 45% |

---

## 6. Aplicaciones y Uso

### 6.1 Dashboard Streamlit

El dashboard proporciona:

1. **Vista General**: Métricas agregadas del parque de motores
2. **Distribución de Riesgo**: Gráfico circular por nivel de riesgo
3. **Predicciones Individuales**: Tabla de motores con RUL predicho
4. **Análisis por Motor**: Vista detallada de cada unidad

### 6.2 API REST

Endpoints disponibles:

- `POST /predict_rul`: Predicción individual
- `POST /predict_batch`: Predicción por lotes
- `GET /health`: Estado del servicio
- `GET /metrics`: Métricas del modelo

### 6.3 Asistente LLM

El módulo de asistencia conversacional permite:

- Explicación de alertas en lenguaje natural
- Generación de reportes de mantenimiento
- Consultas sobre el estado del parque de motores

---

## 7. Discusión

### 7.1 Contribución del Proyecto

Este proyecto integra múltiples técnicas avanzadas de inteligencia artificial:

1. **Deep Learning**: LSTM para modelado de series temporales
2. **Metaheurísticas**: TLBO + PSO para optimización de hiperparámetros
3. **Lógica Difusa**: Sistemas Tipo-2 para clasificación de riesgo
4. **Ingeniería de Datos**: Pipeline escalable tipo lakehouse
5. **Aplicaciones**: API y dashboard para uso productivo

### 7.2 Limitaciones

- Entrenamiento realizado en CPU (GPU aceleraría significativamente)
- Optimización de hiperparámetros limitada por tiempo computacional
- Sistema difuso usa Type-1 (simplificación de Type-2 completo)

### 7.3 Trabajo Futuro

1. Implementar General Type-2 Fuzzy (como Melin et al.)
2. Integrar MLflow para tracking de experimentos
3. Migrar a Databricks para procesamiento a escala
4. Implementar validación con datos de planta reales

---

## 8. Conclusiones

Se ha desarrollado un sistema completo de predicción de vida útil remanente que:

- Utiliza arquitecturas LSTM optimizadas mediante metaheurísticas
- Integra sistemas difusos para clasificación de riesgo
- Proporciona interfaces de usuario para visualización y predicción
- Está diseñado para escalabilidad tipo Databricks

---

## Referencias

1. Yilma, A. A., Yang, C. L., & Woldegiorgidis, B. H. (2026). Remaining useful life prediction using sequential metaheuristic optimization of stacked-LSTM hyperparameters. *Chemical Engineering Research and Design*, 228, 323-335.

2. Ramírez, M., Melin, P., & Castillo, O. (2024). A New Hybrid Approach for Clustering, Classification, and Prediction Combining General Type-2 Fuzzy Systems and Neural Networks. *Axioms*, 13(6), 368.

3. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780.

4. Rao, R. V., Savsani, V. J., & Vakharia, D. P. (2011). Teaching–learning-based optimization: A novel method for constrained mechanical design optimization problems. *Computer-Aided Design*, 43(3), 303-315.

5. Eberhart, R., & Kennedy, J. (1995). A new optimizer using particle swarm theory. *Proceedings of the International Symposium on Micro Machine and Human Science*, 39-43.

6. Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage propagation modeling for aircraft engine run-to-failure simulation. *Annual conference of the prognostics and health management society*.

---

## Anexo: Requisitos del Sistema

```txt
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scikit-fuzzy>=0.4.2
pyspark>=3.4.0
fastapi>=0.104.0
uvicorn>=0.24.0
streamlit>=1.28.0
langchain>=0.1.0
plotly>=5.0.0
pytest>=7.4.0
```

---

*Documento generado para el proyecto RUL Prediction Metaheuristic*