# Stellar Analytics — Backend

Stellar Analytics Backend is a high-performance RESTful API and machine-learning inference service designed for exoplanet analysis. Powered by **FastAPI** and **scikit-learn**, the backend ingests transit observation metrics and host stellar parameters, validates the incoming payload, handles missing feature imputation, and leverages pre-trained Random Forest ensemble models to predict exoplanet disposition classification and estimate planetary radius.

The backend is deployed on **Render** and serves predictions to the separately hosted [Stellar Analytics Frontend](https://steller-frontend.vercel.app/) deployed on Vercel.

---

## Overview

The Stellar Analytics backend functions as the computational and predictive engine of the platform:

1. **Ingests Exoplanet Observation Data**: Receives 23 astronomical parameters encompassing Kepler Object of Interest (KOI) transit signatures and stellar host properties.
2. **Validates & Preprocesses Payload**: Validates data types through Pydantic schemas, ensures all-null payloads are rejected, orders features into the exact matrix required by the models, and imputes missing attributes with median values.
3. **Generates Dual ML Predictions**:
   - **Classification**: Determines whether the candidate is a genuine exoplanet (**Confirmed**) or a spurious detection (**False Positive**) alongside a confidence probability.
   - **Regression**: Estimates the planetary radius ($R_p$, measured in Earth radii $R_\oplus$) using an inverse log-space transformation ($\text{expm1}$).
4. **Returns Structured Responses**: Serializes results into lightweight, formatted JSON consumed directly by the frontend web application.

---

## Architecture

The end-to-end operational flow between user interactions, the API server, and machine-learning pipelines is structured as follows:

```text
Stellar Analytics Frontend (Vercel)
         │
         ▼  [HTTPS POST /predict with JSON payload]
FastAPI Application (Render Web Service)
         │
         ├─► [CORS Middleware: Cross-origin validation]
         │
         ├─► [Pydantic Input Validation (StellarInput model)]
         │         │
         │         ├─ All fields null? ──► [HTTP 400 Bad Request]
         │         └─ Invalid types?    ──► [HTTP 422 Unprocessable Entity]
         │
         ▼  [Convert input dictionary to Pandas DataFrame]
Feature Alignment & Preprocessing
         │
         └─► Reindex columns to match exact training feature order (feature_names_in_)
         │
         ▼
Scikit-Learn Inference Pipelines
         │
         ├──► [Classification Pipeline (classification_model.pkl)]
         │         │
         │         ├─ SimpleImputer(strategy='median')
         │         └─ RandomForestClassifier (300 trees, class_weight='balanced')
         │                  │
         │                  └─► class_pred (0 / 1) & class probability
         │
         └──► [Regression Pipeline (regression_model.pkl)]
                   │
                   ├─ SimpleImputer(strategy='median')
                   └─ RandomForestRegressor (400 trees)
                            │
                            └─► log-scale prediction ──► np.expm1() ──► planetary radius
         │
         ▼
JSON Response Serialization
         │
         └─► {"predicted_planet_radius", "habitability_class", "habitability_probability"}
         │
         ▼
Stellar Analytics Frontend (Interactive 3D Visualizations & Reports)
```

---

## Features

- **RESTful API Service**: Built on top of FastAPI, offering asynchronous support, low latency, and automatic OpenAPI generation.
- **Dual Machine Learning Predictions**: Delivers both planetary classification and continuous radius estimation in a single unified API call.
- **Exoplanet Disposition Classification**: Classifies candidate transits into `Confirmed` or `False Positive` with a calibrated classification probability score.
- **Planetary Radius Regression**: Estimates physical planetary radius in Earth radii ($R_\oplus$) with exponential re-scaling from logarithmic training targets.
- **Automated Missing Data Handling**: Embedded `SimpleImputer` steps inside both model pipelines fill unprovided or missing observational attributes with training medians.
- **Input Validation & Guards**: Enforced via Pydantic model schemas; guards against entirely empty payloads with explicit HTTP 400 feedback.
- **CORS Enabled**: Fully configured Cross-Origin Resource Sharing middleware enabling seamless communication with the Vercel frontend.
- **Interactive Documentation**: Out-of-the-box Swagger UI (`/docs`) and ReDoc (`/redoc`) explorers for interactive testing.
- **Zero-Database Runtime**: High-throughput deployment utilizing serialized Joblib model artifacts loaded once into memory on startup.

---

## API

### Base URLs

- **Production API Base**: `https://steller-backend.onrender.com`
- **Local Development**: `http://localhost:10000`

---

### Available Endpoints

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `GET` | `/` | Service health check and status confirmation | No |
| `POST` | `/predict` | Primary machine-learning prediction endpoint | No |
| `GET` | `/docs` | Interactive Swagger UI documentation | No |
| `GET` | `/redoc` | ReDoc alternative interactive documentation | No |
| `GET` | `/openapi.json` | Raw OpenAPI 3.0 specification schema | No |

---

### 1. Root / Health Check Endpoint

Confirms the API is active and ready to accept requests.

- **URL**: `GET /`
- **Headers**: `Accept: application/json`
- **Response**:

```json
{
  "message": "Stellar Prediction API is running"
}
```

---

### 2. Prediction Endpoint

Generates exoplanet disposition classification and estimated planetary radius from candidate transit and stellar features.

- **URL**: `POST /predict`
- **Content-Type**: `application/json`
- **Headers**: `Content-Type: application/json`

---

## Prediction Request

The `/predict` endpoint accepts a JSON object containing up to **23 input parameters**. All fields are defined as optional floating-point values; missing or unprovided parameters are automatically imputed with median values learned during model training. At least one feature must be provided.

### Parameter Reference Table

| Parameter | Type | Description | Unit / Format |
|---|---|---|---|
| `koi_period` | `float` (Optional) | Orbital period of the candidate exoplanet | Days |
| `koi_duration` | `float` (Optional) | Total transit duration from first to fourth contact | Hours |
| `koi_depth` | `float` (Optional) | Flux loss at minimum transit depth | Parts per million (ppm) |
| `koi_impact` | `float` (Optional) | Sky-projected distance between stellar and planetary disk centers | Dimensionless |
| `koi_model_snr` | `float` (Optional) | Transit model Signal-to-Noise Ratio (SNR) | Dimensionless ratio |
| `koi_num_transits` | `float` (Optional) | Total number of transits observed over the Kepler baseline | Count |
| `koi_ror` | `float` (Optional) | Planet-to-star radius ratio ($R_p / R_\star$) | Dimensionless ratio |
| `st_teff` | `float` (Optional) | Photospheric effective temperature of the host star | Kelvin (K) |
| `st_logg` | `float` (Optional) | Host star surface gravity in base-10 logarithmic scale | $\log_{10}(\text{cm/s}^2)$ |
| `st_met` | `float` (Optional) | Host star metallicity abundance relative to solar ([Fe/H]) | Dex |
| `st_mass` | `float` (Optional) | Host star mass | Solar masses ($M_\odot$) |
| `st_radius` | `float` (Optional) | Host star radius | Solar radii ($R_\odot$) |
| `st_dens` | `float` (Optional) | Host star mean bulk density | $\text{g/cm}^3$ |
| `teff_err1` | `float` (Optional) | Upper uncertainty in stellar effective temperature | +Kelvin (+K) |
| `teff_err2` | `float` (Optional) | Lower uncertainty in stellar effective temperature | -Kelvin (-K) |
| `logg_err1` | `float` (Optional) | Upper uncertainty in stellar surface gravity | +Dex |
| `logg_err2` | `float` (Optional) | Lower uncertainty in stellar surface gravity | -Dex |
| `feh_err1` | `float` (Optional) | Upper uncertainty in stellar metallicity ([Fe/H]) | +Dex |
| `feh_err2` | `float` (Optional) | Lower uncertainty in stellar metallicity ([Fe/H]) | -Dex |
| `mass_err1` | `float` (Optional) | Upper uncertainty in stellar mass | +$M_\odot$ |
| `mass_err2` | `float` (Optional) | Lower uncertainty in stellar mass | -$M_\odot$ |
| `radius_err1` | `float` (Optional) | Upper uncertainty in stellar radius | +$R_\odot$ |
| `radius_err2` | `float` (Optional) | Lower uncertainty in stellar radius | -$R_\odot$ |

---

### Example Request Body

```json
{
  "koi_period": 9.488036,
  "koi_duration": 2.9575,
  "koi_depth": 615.8,
  "koi_impact": 0.146,
  "koi_model_snr": 35.8,
  "koi_num_transits": 142.0,
  "koi_ror": 0.022344,
  "st_teff": 5762.0,
  "st_logg": 4.426,
  "st_met": 0.14,
  "st_mass": 0.985,
  "st_radius": 0.989,
  "st_dens": 1.469,
  "teff_err1": 123.0,
  "teff_err2": -123.0,
  "logg_err1": 0.068,
  "logg_err2": -0.243,
  "feh_err1": 0.15,
  "feh_err2": -0.15,
  "mass_err1": 0.1315,
  "mass_err2": -0.08685,
  "radius_err1": 0.465,
  "radius_err2": -0.114
}
```

---

## Prediction Response

### Example Success Response (`200 OK`)

```json
{
  "predicted_planet_radius": 2.2600,
  "habitability_class": "Confirmed",
  "habitability_probability": 0.9125
}
```

### Response Field Descriptions

| Field | Type | Description |
|---|---|---|
| `predicted_planet_radius` | `float` | Estimated planetary radius expressed in Earth radii ($R_\oplus$), rounded to 4 decimal places. Derived from the Random Forest Regressor predicting $\ln(1 + R_p)$ and inverted using $\text{expm1}(x) = e^x - 1$. |
| `habitability_class` | `string` | Categorical disposition outcome. Returns `"Confirmed"` if candidate classification evaluates to 1, or `"False Positive"` if evaluated as 0. |
| `habitability_probability` | `float` | Estimated confidence probability (between `0.0000` and `1.0000`) for the `"Confirmed"` exoplanet class, rounded to 4 decimal places. |

---

## Machine Learning Pipeline

The project implements a complete machine learning lifecycle documented in `source.ipynb`:

```text
Raw Kepler KOI Data (supernova_dataset.csv)
                    │
                    ├──► Exploratory Data Analysis & Skewness Analysis
                    │
                    ├──► Filtering:
                    │      - Classification: Subset CONFIRMED (1) vs FALSE POSITIVE (0)
                    │      - Regression: Subset CONFIRMED with non-null koi_prad
                    │
                    ├──► Target Transformation (Regression):
                    │      - y_reg = np.log1p(koi_prad)
                    │
                    ├──► Train/Test Stratified Splits (80% Train, 20% Test)
                    │
                    ├──► Model Pipelines:
                    │      - SimpleImputer(strategy='median')
                    │      - Classification: RandomForestClassifier(300 estimators, balanced)
                    │      - Regression: RandomForestRegressor(400 estimators)
                    │
                    ├──► Cross-Validation & Permutation Feature Importance
                    │
                    └──► Serialization:
                           - classification_model.pkl
                           - regression_model.pkl
```

### 1. Data Ingestion & Sanitization
The raw data is loaded and filtered to remove non-evaluable candidate records. Identification columns (`kepid`) and target columns (`koi_disposition`, `koi_prad`) are decoupled from the 23 feature predictors.

### 2. Target Transformation for Regression
The raw planetary radius column (`koi_prad`) possesses heavy positive skewness (~52.12). To stabilize variance and penalize relative rather than absolute errors across vast exoplanet scales, targets are transformed into log space:
$$\tilde{y} = \ln(1 + R_p) = \text{log1p}(R_p)$$

During inference in `app.py`, predictions are inverted back to original Earth radii scale via:
$$R_p = e^{\tilde{y}} - 1 = \text{expm1}(\tilde{y})$$

### 3. Pipeline Design & Imputation
Both models are packaged as self-contained scikit-learn `Pipeline` objects. The first step of each pipeline is `SimpleImputer(strategy='median')`. This guarantees that partial or sparse client inputs are filled with the baseline median values observed during training.

---

## Models

The backend utilizes two pre-trained model artifacts serialized with Joblib:

### 1. `classification_model.pkl`

- **Purpose**: Evaluates candidate validity by distinguishing authentic exoplanetary transit signals from false positive astrophysical contaminants.
- **Pipeline Architecture**:
  1. `SimpleImputer(strategy='median')`
  2. `RandomForestClassifier(n_estimators=300, max_depth=None, class_weight='balanced', random_state=42)`
- **How Loaded**: Loaded once at startup via `joblib.load("classification_model.pkl")`.
- **Verified Metrics (from `source.ipynb`)**:
  - **Test Accuracy**: `0.93` (93%)
  - **Test ROC-AUC**: `0.9814`
  - **Macro Avg Precision / Recall / F1**: `0.93 / 0.92 / 0.92`
  - **5-Fold Cross-Validation ROC-AUC**: `[0.9591, 0.9866, 0.9625, 0.9955, 0.9656]` (Mean: `0.9739`)
  - **Top Predictive Features**: `koi_ror` (radius ratio), `koi_model_snr`, `koi_period`, `koi_duration`, `koi_impact`, `koi_depth`, `feh_err2`.

### 2. `regression_model.pkl`

- **Purpose**: Estimates planetary radius ($R_p$ in Earth radii $R_\oplus$) for confirmed exoplanet systems.
- **Pipeline Architecture**:
  1. `SimpleImputer(strategy='median')`
  2. `RandomForestRegressor(n_estimators=400, random_state=42)`
- **How Loaded**: Loaded once at startup via `joblib.load("regression_model.pkl")`.
- **Target Space**: Trained on log-transformed radius `np.log1p(y)`.
- **Verified Metrics (from `source.ipynb`)**:
  - **Test RMSE (original scale)**: `1.0054` Earth radii
  - **Test MAE (original scale)**: `0.2212` Earth radii
  - **5-Fold Cross-Validation RMSE (log space)**: `[0.1171, 0.0838, 0.0812, 0.0585, 0.1315]` (Mean: `0.0944`)
  - **Top Predictive Features**: `koi_ror` (71.18% importance), `st_dens` (8.97%), `radius_err2` (4.83%), `st_mass` (3.77%), `st_radius` (2.35%).

---

## Dataset

### `supernova_dataset.csv`

Despite its naming convention within the repository, the dataset consists of archival observations from the **NASA Kepler Space Telescope** (Kepler Objects of Interest / KOI Cumulative Catalog).

- **Total Entries**: 9,564 observation rows across 26 features.
- **Key Columns**:
  - `kepid`: Unique Kepler star catalog identifier.
  - `koi_disposition`: Ground-truth disposition category (`FALSE POSITIVE`: 4,839, `CONFIRMED`: 2,746, `CANDIDATE`: 1,979).
  - `koi_prad`: Physical planetary radius ($R_\oplus$).
  - 23 astronomical predictors: 7 transit characteristics (`koi_*`), 6 host star parameters (`st_*`), and 10 observational uncertainty margins (`*_err1`, `*_err2`).
- **Runtime Usage**: **The CSV file is not loaded or referenced during backend runtime.** It is retained in the repository for training reproducibility and offline exploratory data analysis in `source.ipynb`. The API exclusively queries the serialized `.pkl` models for fast and low-memory inference.

---

## Technology Stack

All technologies and libraries are verified directly from `requirements.txt`, `runtime.txt`, and codebase imports:

| Category | Component | Version / Detail |
|---|---|---|
| **Runtime** | Python | `3.10.13` (declared in `runtime.txt`) |
| **Framework** | FastAPI | High-performance modern web framework |
| **ASGI Server** | Uvicorn | Lightning-fast ASGI server implementation |
| **Data Validation** | Pydantic | Schema definition, coercion, and validation |
| **Machine Learning** | scikit-learn | Version `1.6.1` (pinned in `requirements.txt`) |
| **Ensemble Library** | XGBoost | Gradient boosted tree support |
| **Data Manipulation** | Pandas & NumPy | In-memory tabular and numerical array operations |
| **Model Serialization** | Joblib | Efficient persistence for Python objects and arrays |
| **Multipart Parsing** | python-multipart | Form and multipart request streaming |
| **Hosting Platform** | Render | Cloud containerized web service hosting |

---

## Project Structure

The repository maintains a clean, flat architecture optimized for continuous deployment on Render:

```text
Steller_backend/
├── app.py                      # FastAPI application entry point, CORS, schema & routes
├── classification_model.pkl    # Serialized Random Forest classification pipeline
├── regression_model.pkl        # Serialized Random Forest regression pipeline
├── requirements.txt            # Locked Python dependency manifest
├── runtime.txt                 # Pinned Python version (python-3.10.13)
├── source.ipynb                # End-to-end ML notebook (EDA, training, validation, export)
├── supernova_dataset.csv       # Archival Kepler KOI dataset for training reproducibility
└── README.md                   # Complete repository documentation
```

---

## Local Development

Follow these steps to clone, configure, and run the backend locally:

### 1. Prerequisites

- Python `3.10.x` (matching `runtime.txt: python-3.10.13`).
- Git CLI.

> **Note on Scikit-Learn Compatibility**: The pre-trained model artifacts (`classification_model.pkl` and `regression_model.pkl`) were trained and serialized using `scikit-learn==1.6.1` under Python 3.10. Using a virtual environment with `pip install -r requirements.txt` ensures identical library versions and prevents unpickling `InconsistentVersionWarning` or attribute discrepancies.

### 2. Clone Repository

```bash
git clone https://github.com/kunj-16/Steller_backend.git
cd Steller_backend
```

### 3. Create Virtual Environment

- **Windows (PowerShell / Command Prompt)**:
  ```powershell
  python -m venv venv
  venv\Scripts\activate
  ```

- **Linux / macOS**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Launch Application

You can launch the server either directly through Python or with Uvicorn:

- **Via Python script** (uses the internal configuration in `app.py`):
  ```bash
  python app.py
  ```

- **Via Uvicorn directly** (with hot reload enabled for development):
  ```bash
  uvicorn app:app --host 0.0.0.0 --port 10000 --reload
  ```

The server will initialize on `http://localhost:10000` (or `http://127.0.0.1:10000`).

---

## API Documentation

FastAPI automatically generates interactive schema documentation accessible in any standard web browser:

- **Swagger UI**: Accessible at `http://localhost:10000/docs` (Production: `https://steller-backend.onrender.com/docs`)
  - Features an interactive **Try it out** button to execute live test requests against `/predict`.
- **ReDoc**: Accessible at `http://localhost:10000/redoc` (Production: `https://steller-backend.onrender.com/redoc`)
  - Provides a structured, three-panel reference view of the API schema.
- **OpenAPI JSON**: Available at `http://localhost:10000/openapi.json` for external tool integrations or Postman imports.

---

## CORS Configuration

Cross-Origin Resource Sharing is actively configured in `app.py` via FastAPI's `CORSMiddleware`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

- **Allowed Origins**: `["*"]` — Grants universal access across web origins, ensuring zero cross-origin rejection when called from the Vercel frontend (`https://steller-frontend.vercel.app/`) or localhost during development.
- **Allowed Methods**: `["*"]` — Permits all standard HTTP verbs (`GET`, `POST`, `OPTIONS`, etc.).
- **Allowed Headers**: `["*"]` — Allows all request headers including custom content headers.
- **Allow Credentials**: `True` — Enables transmission of credentials across origins.

---

## Deployment

The backend service is deployed on **Render** as an automated Web Service:

- **Deployment Provider**: Render Cloud Platform
- **Live Base URL**: `https://steller-backend.onrender.com/`
- **Prediction Endpoint**: `https://steller-backend.onrender.com/predict`
- **Runtime Specification**: `runtime.txt` pins the execution environment to `python-3.10.13`.
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python app.py` (or `uvicorn app:app --host 0.0.0.0 --port 10000`)
- **Port Binding**: Binds to host `0.0.0.0` and port `10000` as required by Render.

---

## Frontend Integration

The backend is connected directly to the [Stellar Analytics Frontend](https://github.com/kunj-16/Steller_Frontend):

- **Frontend Live URL**: [https://steller-frontend.vercel.app/](https://steller-frontend.vercel.app/)
- **Backend API URL**: [https://steller-backend.onrender.com/predict](https://steller-backend.onrender.com/predict)

### Integration Workflow

```text
User adjusts astronomical sliders / input fields in the Frontend UI
                              │
                              ▼
Frontend constructs JSON POST request and dispatches it to:
https://steller-backend.onrender.com/predict
                              │
                              ▼
Render Backend ingests payload, validates values, and executes inference
                              │
                              ▼
Backend responds with 200 OK JSON payload:
{
  "predicted_planet_radius": 2.2600,
  "habitability_class": "Confirmed",
  "habitability_probability": 0.9125
}
                              │
                              ▼
Frontend renders real-time 3D planetary rendering, metrics card, and classification badge
```

---

## Environment Variables

The backend application does **not** require any custom `.env` environment variables to function. Model paths (`classification_model.pkl` and `regression_model.pkl`), network host (`0.0.0.0`), and port (`10000`) are referenced relatively from the project root, enabling zero-configuration deployments.

---

## Error Handling

The backend implements structured error responses with appropriate HTTP status codes:

### 1. HTTP 400 Bad Request — Empty Payload
Triggered when all 23 input parameters evaluate to `null` or are omitted entirely:

```json
{
  "detail": "At least one feature must be provided for prediction."
}
```

### 2. HTTP 422 Unprocessable Entity — Schema Validation Error
Automatically raised by FastAPI if non-numeric values or incompatible data types are submitted:

```json
{
  "detail": [
    {
      "loc": ["body", "koi_period"],
      "msg": "Input should be a valid number, unable to parse string as an extractable float",
      "type": "float_parsing"
    }
  ]
}
```

### 3. HTTP 500 Internal Server Error — Prediction Exception
Raised if an unexpected error occurs during data manipulation or model inference:

```json
{
  "detail": "Prediction failed: <error_message>"
}
```

---

## Limitations

- **Median Imputation Sensitivity**: While the embedded median imputer prevents pipeline crashes when attributes are omitted, submitting inputs with an excessive number of missing values will bias predictions toward dataset medians.
- **Static Artifacts**: Pre-trained model `.pkl` files are static snapshots. Retraining requires re-running `source.ipynb` and re-committing new pickle files.
- **Render Free-Tier Spin-Up**: When hosted on Render's free tier, the service may spin down after periods of inactivity, causing an initial cold-start latency of approximately 30–50 seconds on the first request.

---

## Future Improvements

The following architectural and operational enhancements are planned as potential future upgrades:

- [ ] **Automated Test Suite**: Integrate `pytest` and `httpx.AsyncClient` test suites for continuous integration testing.
- [ ] **Dedicated Health Probes**: Add a dedicated `/healthz` endpoint verifying model availability and memory consumption.
- [ ] **API Rate Limiting**: Implement request throttling (e.g. `slowapi`) to safeguard against denial-of-service traffic.
- [ ] **Structured Logging**: Introduce structured JSON logging (e.g. `structlog`) and OpenTelemetry tracing for API observability.
- [ ] **Model Versioning**: Implement MLflow or DVC tracking for artifact versioning and automated retraining workflows.

---

## Author

**Kunj Bhasin**
- GitHub: [@kunj-16](https://github.com/kunj-16)
- Frontend Repository: [Steller_Frontend](https://github.com/kunj-16/Steller_Frontend)
- Backend Repository: [Steller_backend](https://github.com/kunj-16/Steller_backend)
