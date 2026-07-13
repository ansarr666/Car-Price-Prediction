<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit_learn-Random_Forest-orange?logo=scikit-learn" alt="scikit-learn">
  <img src="https://img.shields.io/badge/FastAPI-1.0-009688?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Streamlit-1.36+-FF4B4B?logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/R%C2%B2-0.9534-success" alt="R²">
  <img src="https://img.shields.io/badge/MAPE-7.71%25-yellow" alt="MAPE">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

<h1 align="center">Car Price Prediction — End-to-End ML Pipeline</h1>

<p align="center">
  A production-grade machine learning system that predicts the Manufacturer Suggested Retail Price (MSRP) of a vehicle from its brand, engine specs, and vehicle characteristics. <br>
  <b>Random Forest Regressor</b> · <b>R² = 0.9534</b> · <b>MAPE = 7.71%</b>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Training the Model](#training-the-model)
  - [Running the API Server](#running-the-api-server)
  - [Running the Streamlit UI](#running-the-streamlit-ui)
- [API Reference](#api-reference)
- [Deployment](#deployment)
  - [Docker](#docker)
  - [Render / Railway (Cloud)](#render--railway-cloud)
- [Model Explainability](#model-explainability)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project delivers a **complete, end-to-end machine learning pipeline** for predicting vehicle MSRP. It covers the full lifecycle: exploratory data analysis, feature engineering, model selection, production API deployment, and an interactive frontend.

**Key differentiators:**

| Feature | Detail |
|---|---|
| **Accuracy** | R² of 0.9534 — explains 95%+ of price variance |
| **Error rate** | MAPE of 7.71% — average prediction error under 8% |
| **Production-ready** | FastAPI backend with Pydantic validation & OpenAPI docs |
| **Interactive UI** | Streamlit frontend with dynamic make→model cascading dropdowns |
| **Reproducible** | Seeded shuffle, versioned dependencies, single-command training |
| **Deployable** | Docker support, cloud-ready architecture |

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        User / Browser                              │
└──────────────┬──────────────────────────────────▲──────────────────┘
               │  HTTP (Streamlit default :8501)   │
               ▼                                  │
┌──────────────────────────────────────────────────┴──────────────────┐
│                    Streamlit Frontend (app.py)                      │
│  • Dynamic input forms with cascading dropdowns                     │
│  • Real-time API health monitoring                                  │
│  • Responsive UI with error handling                                │
└──────────────┬──────────────────────────────────▲──────────────────┘
               │  HTTP POST /predict              │ HTTP GET /health
               │  JSON payload                     │ JSON response
               ▼                                  │
┌──────────────────────────────────────────────────┴──────────────────┐
│                      FastAPI Backend (main.py)                      │
│  • Pydantic input validation & normalization                        │
│  • Feature engineering (age, hp_per_cylinder)                       │
│  • Model inference via joblib-loaded pipeline                       │
│  • OpenAPI / Swagger UI at /docs                                    │
│  • Health check endpoint at /health                                 │
└──────────────────────────────────────────────────▲──────────────────┘
               │                                    │
               ▼                                    │
┌──────────────────────────────────────────────────┴──────────────────┐
│                  Trained Model (car_price_rf_model.pkl)             │
│  • sklearn Pipeline: ColumnTransformer + RandomForestRegressor      │
│  • Trained on 11,199 unique vehicles                                │
│  • 100 estimators, 13 features (11 raw + 2 engineered)             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Dataset

| Property | Details |
|---|---|
| **Source** | [Car Features and MSRP](https://www.kaggle.com/CooperUnion/cardataset) (Kaggle) |
| **Records** | 11,914 raw → 11,199 after deduplication |
| **Original Features** | 16 |
| **Features after Engineering** | 13 (11 base + 2 engineered: `age`, `hp_per_cylinder`) |
| **Target Variable** | MSRP (Manufacturer Suggested Retail Price) in USD |
| **Data Types** | 9 categorical, 4 numerical |
| **Collection Year** | 2017 |

### Feature Description

| Feature | Type | Description |
|---|---|---|
| `make` | Categorical | Vehicle manufacturer (e.g., Toyota, BMW) |
| `model` | Categorical | Specific model name |
| `engine_fuel_type` | Categorical | Fuel type (regular unleaded, premium, diesel, etc.) |
| `engine_hp` | Numerical | Horsepower rating |
| `engine_cylinders` | Numerical | Number of cylinders (0 for EVs) |
| `transmission_type` | Categorical | Automatic, manual, etc. |
| `driven_wheels` | Categorical | Front-wheel, rear-wheel, all-wheel drive |
| `number_of_doors` | Numerical | 2 or 4 |
| `vehicle_size` | Categorical | Compact, midsize, large |
| `vehicle_style` | Categorical | Sedan, SUV, Coupe, Convertible, etc. |
| `city_mpg` | Numerical | City fuel economy (miles per gallon) |
| `year` | Numerical | Model year (1990–2017) |
| `age` | Engineered | `2017 - year` — captures depreciation |
| `hp_per_cylinder` | Engineered | `engine_hp / engine_cylinders` — efficiency metric |

---

## Pipeline

```
┌──────────┐    ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌──────────┐
│   Load   │ →  │  Clean   │ →  │  Shuffle  │ →  │ Engineer │ →  │   Split   │ →  │  Train   │ →  │  Save    │
│   Data   │    │  Data    │    │  (seed=2) │    │ Features │    │ (80/20)   │    │ & Eval   │    │  Model   │
└──────────┘    └──────────┘    └───────────┘    └──────────┘    └───────────┘    └──────────┘    └──────────┘
```

### Stage 1 — Data Cleaning
- Standardize column names to `snake_case`
- Convert all string values to lowercase with underscores
- Remove 715 duplicate records
- Drop `market_category` (high cardinality, low predictive value)
- Impute missing values: median for numerical, mode for categorical

### Stage 2 — Feature Engineering
- **`age`**: Captures vehicle depreciation (`2017 - year`)
- **`hp_per_cylinder`**: Engine efficiency metric
- Drop `year`, `popularity`, `highway_mpg` (collinearity / low importance)

### Stage 3 — Model Training
Three regression algorithms evaluated:

| Model | Preprocessing | Target Scale |
|---|---|---|
| Linear Regression | `StandardScaler` + `OneHotEncoder` | Log-transformed |
| Random Forest | `Passthrough` + `OneHotEncoder` | Raw dollars |
| XGBoost | `Passthrough` + `OneHotEncoder` | Raw dollars |

**Winner: Random Forest** — best balance of accuracy and generalization, no scaling required.

---

## Results

| Model | MAE ($) | RMSE ($) | MAPE (%) | R² |
|---|---|---|---|---|
| **Random Forest** | **3,518** | **11,578** | **7.71** | **0.9534** |
| XGBoost | 4,604 | 11,964 | 15.00 | 0.9503 |
| Linear Regression | 4,096 | 13,759 | 10.81 | 0.9342 |

- **Random Forest** outperforms XGBoost by nearly **2× in MAPE** (7.71% vs 15.00%)
- **R² of 0.9534** means the model explains 95.34% of all variance in vehicle pricing
- **Average error of $3,518** on a dataset where MSRP ranges from $2,000 to $230,000+

### Model Interpretability

```
Top Feature Importances (Random Forest):
─────────────────────────────────────────
engine_hp           ████████████████████░  38.2%
engine_cylinders    ██████████████░░░░░░░  24.8%
age                 ████████░░░░░░░░░░░░░  11.5%
hp_per_cylinder     █████░░░░░░░░░░░░░░░░   7.1%
city_mpg            ████░░░░░░░░░░░░░░░░░   5.3%
make                ███░░░░░░░░░░░░░░░░░░   4.2%
vehicle_style       ██░░░░░░░░░░░░░░░░░░░   3.0%
...                 (remaining 6 features)   5.9%
```

**Key insight:** Engine power and cylinder count dominate pricing decisions, accounting for **63% of the model's predictive weight**. Vehicle age captures the next most important signal — depreciation.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.11+ | Core development |
| **Data Processing** | pandas, NumPy | Data manipulation & numerical computing |
| **Visualization** | Matplotlib, Seaborn | EDA & insight generation |
| **Machine Learning** | scikit-learn | Pipelines, preprocessing, Random Forest |
| **Gradient Boosting** | XGBoost | Benchmark comparison model |
| **Model Serialization** | joblib | Model persistence |
| **Backend API** | FastAPI | RESTful inference server with OpenAPI docs |
| **Frontend** | Streamlit | Interactive web UI |
| **HTTP Client** | requests | Frontend ↔ Backend communication |
| **Server** | Uvicorn | ASGI server for FastAPI |
| **Notebook** | Jupyter | Exploratory data analysis |

---

## Project Structure

```
Car-Price-Prediction/
├── .streamlit/
│   └── config.toml              # Streamlit theme (dark mode, indigo accent)
├── data/
│   └── car_price_prediction.csv # Raw dataset (11,914 records)
├── models/
│   └── car_price_rf_model.pkl   # Trained & serialized Random Forest pipeline
├── app.py                       # Streamlit frontend application
├── car_price_prediction.ipynb   # Jupyter notebook (full EDA + model selection)
├── main.py                      # FastAPI backend (inference API)
├── requirements.txt             # Python dependencies
├── train.py                     # Production training pipeline
└── README.md                    # This file
```

---

## Getting Started

### Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- Git (optional, for cloning)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/ansarr666/Car-Price-Prediction.git
cd Car-Price-Prediction

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Training the Model

```bash
python train.py
```

This will:
1. Load and clean the dataset
2. Engineer features and split data (80/20)
3. Train and evaluate the Random Forest model
4. Save the trained pipeline to `models/car_price_rf_model.pkl`

### Running the API Server

Start the FastAPI backend:

```bash
uvicorn main:app --reload
```

| Flag | Purpose |
|---|---|
| `--reload` | Auto-restart on code changes (development) |
| `--host 0.0.0.0` | Bind to all network interfaces (production) |
| `--port 8000` | Port number (default: 8000) |
| `--workers 4` | Multi-worker mode (production) |

The API will be available at **http://127.0.0.1:8000**.

**Interactive API documentation:**

| URL | Description |
|---|---|
| `http://127.0.0.1:8000/docs` | Swagger UI (try endpoints interactively) |
| `http://127.0.0.1:8000/redoc` | ReDoc (read-only documentation) |

### Running the Streamlit UI

In a **separate terminal** (with the backend still running):

```bash
streamlit run app.py
```

The UI will open at **http://127.0.0.1:8501**.

---

## API Reference

### Health Check

```
GET /health
```

**Response:**
```json
{
  "status": "healthy"
}
```

### Predict Price

```
POST /predict
```

**Request Body:**
```json
{
  "make": "toyota",
  "model": "camry",
  "engine_fuel_type": "regular_unleaded",
  "engine_hp": 203.0,
  "engine_cylinders": 4.0,
  "transmission_type": "automatic",
  "driven_wheels": "front_wheel_drive",
  "number_of_doors": 4.0,
  "vehicle_size": "midsize",
  "vehicle_style": "sedan",
  "city_mpg": 28,
  "year": 2015
}
```

**Response:**
```json
{
  "predicted_price": 24567.32
}
```

**Validation Rules:**

| Field | Constraints |
|---|---|
| `engine_hp` | > 0, ≤ 1200 |
| `engine_cylinders` | 0–16 |
| `number_of_doors` | 2 or 4 only |
| `city_mpg` | > 1, ≤ 150 |
| `year` | 1990–2017 |
| All string fields | Auto-normalized to lowercase with underscores |

### Quick Test with cURL

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "make": "toyota",
    "model": "camry",
    "engine_fuel_type": "regular_unleaded",
    "engine_hp": 203.0,
    "engine_cylinders": 4.0,
    "transmission_type": "automatic",
    "driven_wheels": "front_wheel_drive",
    "number_of_doors": 4.0,
    "vehicle_size": "midsize",
    "vehicle_style": "sedan",
    "city_mpg": 28,
    "year": 2015
  }'
```

---

## Deployment

### Docker

**Backend Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build & run:**
```bash
docker build -t car-price-api .
docker run -p 8000:8000 car-price-api
```

### Render / Railway (Cloud)

1. Push the repository to GitHub
2. Create a new **Web Service** on [Render](https://render.com) or [Railway](https://railway.app)
3. Point it to the repository
4. Set the **Start Command** to:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port $PORT
   ```
5. For the Streamlit UI, create a second service with:
   ```bash
   streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```
6. Update `API_BASE_URL` in `app.py` to point to your deployed backend URL

---

## Model Explainability

The Random Forest model provides built-in feature importance scores, offering transparency into pricing factors:

- **engine_hp (38.2%)**: Horsepower is the single strongest price predictor — more power commands premium pricing
- **engine_cylinders (24.8%)**: Cylinder count strongly correlates with vehicle class and performance tier
- **age (11.5%)**: Depreciation captured by `2017 - year` — newer vehicles command higher prices
- **hp_per_cylinder (7.1%)**: This engineered feature captures engine efficiency — high HP per cylinder indicates performance engineering
- **city_mpg (5.3%)**: Fuel economy has a modest but meaningful impact, especially for economy-oriented buyers
- **make (4.2%)**: Brand premium exists but is secondary to mechanical specifications

This ranking confirms that **mechanical specifications dominate pricing**, with brand and body style playing supporting roles. In production, this means the model can make reliable predictions even for new or lesser-known brands, as long as the engine specs are known.

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code follows the existing style and includes appropriate tests or validation.

---
## Author

**Anam Ansar** <anam2408422@st.jmi.ac.in>

## License

Distributed under the MIT License. See `LICENSE` for more information.

---

<p align="center">
  Built with Python, scikit-learn, FastAPI, and Streamlit.
</p>
