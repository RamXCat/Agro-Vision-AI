<h1 align="center">🌿 Agro-Vision AI</h1>

<p align="center">
  <b>Multimodal Agricultural Decision-Support &amp; Precision Farming Platform</b><br/>
  Empowering farmers with AI-driven plant disease diagnostics, crop recommendation, yield estimation, and wholesale price forecasting.
</p>

<p align="center">
  <a href="https://github.com/RamXCat/Agro-Vision-AI/stargazers"><img src="https://img.shields.io/github/stars/RamXCat/Agro-Vision-AI?style=for-the-badge&logo=github&color=gold" alt="Stars"/></a>
  <a href="https://github.com/RamXCat/Agro-Vision-AI/network/members"><img src="https://img.shields.io/github/forks/RamXCat/Agro-Vision-AI?style=for-the-badge&logo=github&color=blue" alt="Forks"/></a>
  <a href="https://github.com/RamXCat/Agro-Vision-AI/issues"><img src="https://img.shields.io/github/issues/RamXCat/Agro-Vision-AI?style=for-the-badge&logo=github&color=red" alt="Issues"/></a>
  <a href="https://github.com/RamXCat/Agro-Vision-AI/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="License"/></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-3.x-000000?style=flat-square&logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.17-FF6F00?style=flat-square&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-1.5-F7931E?style=flat-square&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/MobileNetV2-99.73%25_Val_Acc-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/UI-Mobile--First%20%7C%20Bilingual-informational?style=flat-square"/>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture & Tech Stack](#-architecture--tech-stack)
- [Model Performance & ML Pipeline](#-model-performance--ml-pipeline)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Datasets & Training](#-datasets--training)
- [Automated Testing](#-automated-testing)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌾 Overview

**Agro-Vision AI** is an intelligent, multimodal agricultural management platform tailored for farmers, agronomists, and agribusinesses. Built with a mobile-first philosophy to thrive in field conditions with spotty internet connectivity, the system combines **Computer Vision**, **Tabular Machine Learning**, and **Time-Series Forecasting** to replace guesswork with data-driven insights.

### Core Problems Addressed
- 🍃 **Late Disease Detection:** Eliminates delayed diagnoses with instant leaf image scans and actionable treatment guidance.
- 🌱 **Suboptimal Crop Selection:** Analyzes soil chemistry (NPK, pH) and microclimates to recommend high-compatibility crops.
- 📈 **Harvest Uncertainty:** Accurately forecasts expected yields (metric tons per hectare).
- 💰 **Market Price Fluctuations:** Predicts wholesale Mandi prices to help growers optimize sale timing.
- 🧠 **Fragmented Data:** Features an **Integrated Farm Advisor** synthesizing cross-module metrics into holistic profitability reports.

---

## ✨ Key Features

| Feature | Description | Architecture / Implementation |
| :--- | :--- | :--- |
| 🔬 **Leaf Disease Detection** | Real-time classification with confidence scoring & uncertainty guardrails | **MobileNetV2 Transfer Learning** (ImageNet base) + OpenCV segmentation fallback |
| 🌾 **Crop Recommendation** | Calibrated top-3 crop suggestions matching soil and weather parameters | Multi-class **RandomForest / XGBoost** classifiers |
| 📊 **Yield Prediction** | Quantitative yield estimation per hectare and total acreage output | Gradient Boosted / Linear regression pipelines with domain heuristics |
| 📉 **Mandi Price Forecast** | Predictive price trends across harvest cycles | Multi-step recursive **LSTM / ARIMA** time-series modeling |
| 🧠 **Cross-Module Advisor** | Unified revenue forecast, risk flags, and agronomic recommendations | End-to-end analytical pipeline cross-referencing all sub-models |
| 📱 **Field-Ready Mobile UI** | Responsive design tested down to 360px viewports with large tap targets | Bootstrap 5 + On-device HTML5 Canvas compression (~1024px) |
| 🌐 **Bilingual Support (i18n)** | Seamless real-time switching between English and Hindi (हिन्दी) | Client-side localized dictionary mapping |
| 🔐 **User Accounts & History** | Secure user registration, authentication, and past prediction history | Flask-Login / Werkzeug security with SQLite/MySQL schemas |

---

## 🛠 Architecture & Tech Stack

```text
       [ Mobile Browser / Client ]
                    │
                    ▼  (Canvas-compressed ~1024px images / JSON)
       [ Flask Application Gateway ]
                    │
  ┌─────────────────┼──────────────────┐
  ▼                 ▼                  ▼
[ Computer Vision ] [ Tabular ML ]     [ Time-Series ]
  • MobileNetV2       • Scikit-Learn     • LSTM
  • OpenCV fallback   • XGBoost          • Statsmodels
  └─────────────────┬──────────────────┘
                    │
                    ▼
         [ SQLite / MySQL DB ]
```

- **Backend:** Python 3.11+, Flask 3.0, SQLAlchemy, Flask-WTF, Werkzeug
- **Machine Learning & CV:** TensorFlow 2.17, Keras, Scikit-Learn, XGBoost, OpenCV, Pillow, Statsmodels
- **Frontend:** HTML5, CSS3, JavaScript (ES6+), Bootstrap 5, Jinja2
- **Testing & Quality:** Pytest, Latency profiling harnesses

---

## 📊 Model Performance & ML Pipeline

### Disease Detection — MobileNetV2 (Binary Classifier)

Trained on curated subsets of the **PlantVillage** dataset using a two-phase transfer learning regimen:
1. **Phase 1 (Feature Extraction):** MobileNetV2 base weights frozen; custom top layers (GlobalAveragePooling -> Dropout -> Dense) trained for 10 epochs.
2. **Phase 2 (Fine-Tuning):** Top 30 layers unfrozen and trained with a reduced learning rate ($10^{-5}$) to adapt leaf-specific patterns.

```text
Validation Accuracy : 99.73%
Validation Loss     : 0.0099
Input Resolution    : 224 x 224 px
Inference Latency   : < 150ms (CPU)
```

> **Uncertainty Guardrail:** Outputs with confidence $< 60\%$ are automatically labeled as *Uncertain* to avoid dangerous false positives in critical agricultural treatments.

---

## 📁 Project Structure

```text
Agro-Vision-AI/
├── app/
│   ├── __init__.py              # Application factory & model initialization
│   ├── extensions.py            # Database and security extensions
│   ├── models_db.py             # SQLAlchemy models (User, Scans, Queries)
│   ├── blueprints/              # Modular controller routes
│   │   ├── advisor.py           # Cross-module unified advisor
│   │   ├── auth.py              # User authentication & session management
│   │   ├── disease.py           # Leaf scan diagnostic endpoints
│   │   ├── history.py           # Farmer scan and query logs
│   │   ├── info.py              # Model transparency registry
│   │   ├── price.py             # Mandi price forecasting
│   │   ├── recommend.py         # Soil-based crop recommendation
│   │   └── yield_.py            # Yield prediction endpoint
│   ├── services/                # Business logic, inference & fallback engines
│   ├── templates/               # Jinja2 mobile-first templates
│   ├── static/                  # Responsive CSS, client-side JS & i18n JSONs
│   └── ml_artifacts/            # Serialized models (.keras, .joblib)
├── ml/
│   ├── data/                    # Raw & processed datasets (gitignored)
│   ├── train_disease.py         # Transfer learning training script
│   ├── plant_disease_model.keras# Trained disease classification model
│   └── classes.json             # Target class dictionary
├── tests/                       # Pytest test suite & latency benchmarks
├── docs/                        # Architecture diagrams and assets
├── config.py                    # Environment and app configuration
├── requirements.txt             # Project dependencies
└── .env.example                 # Environment variables template
```

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python 3.11** or **3.12**
- **Git**

### 2. Clone the Repository
```bash
git clone https://github.com/RamXCat/Agro-Vision-AI.git
cd Agro-Vision-AI
```

### 3. Create & Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Setup Environment Configuration
```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```
*(Optional: Configure database credentials and secrets inside `.env` if using MySQL; defaults to local SQLite).*

### 6. Launch the Application
```bash
flask run --host=0.0.0.0 --port=5000
```
Open **`http://localhost:5000`** in your browser.

---

## 📦 Datasets & Training

To train or fine-tune models locally, place datasets in `ml/data/raw/`:

| Dataset | Module | Format |
| :--- | :--- | :--- |
| **PlantVillage** | Disease Detection | Image folders organized by plant/disease class |
| **Crop_recommendation.csv** | Crop Recommendation | Soil nutrients (N, P, K), pH, climate & labels |
| **crop_yield.csv** | Yield Prediction | Historical records of soil, weather, acreage & yield |
| **mandi_prices.csv** | Price Forecasting | Historical wholesale mandi price time-series |

To run the disease detection training pipeline:
```bash
python ml/train_disease.py
```

---

## 🧪 Automated Testing

Agro-Vision AI includes a comprehensive test suite covering routing, service layer fallbacks, edge-case validation, and latency benchmarks.

```bash
# Run all tests
pytest -v tests/

# Benchmark inference latency
python tests/measure_latency.py
```

---

## 🤝 Contributing

Contributions make the open-source community thrive! Follow these steps to contribute:

1. **Fork** the Project
2. Create your Feature Branch (`git checkout -b feature/NewFeature`)
3. Commit your Changes (`git commit -m 'feat: Add NewFeature'`)
4. Push to the Branch (`git push origin feature/NewFeature`)
5. Open a **Pull Request**

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more details.

---

<p align="center">
  Developed with ❤️ for sustainable agriculture and smart farming.
</p>