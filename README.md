Website link : https://breastcancerwebsite-d64if64b4kt9jxwkaywb2z.streamlit.app/

# 🩺 Breast Cancer Diagnosis Predictor

A Streamlit web app that predicts whether a breast tumor is **benign** or **malignant** from cell nuclei measurements, using a linear-kernel Support Vector Machine (SVM) trained on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

**⚠️ Disclaimer:** This tool is for educational and portfolio purposes only. It is **not** a substitute for professional medical diagnosis.

---

## 🔍 Overview

The app takes 30 diagnostic features computed from a digitized image of a breast mass biopsy (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension — each reported as `mean`, `standard error`, and `worst` value) and classifies the sample as **Benign** or **Malignant**.

It supports three ways to provide input:

| Mode | Description |
|---|---|
| **Upload file** | Upload a CSV/Excel report (batch predictions for multiple patients) |
| **Sliders** | Adjust measurements with sliders, grouped by Mean / SE / Worst |
| **Number boxes** | Enter exact numeric values for each feature |

## ✨ Features

- **Single-patient prediction** — adjust measurements and get an instant Benign/Malignant result with a decision-score gauge
- **Batch prediction** — upload a CSV/Excel file of multiple patient reports and download the results
- **Downloadable sample template** — a ready-to-fill CSV so users know the expected input format
- **Decision confidence score** — the signed distance from the SVM's decision boundary, showing how confidently a case falls on either side
- **Dataset-driven defaults** — sliders and number boxes default to the dataset's mean values, with min/max bounds pulled from training data statistics

## 🧠 Model

- **Algorithm:** Support Vector Machine (linear kernel)
- **Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC) — 569 samples, 30 features
- **Preprocessing:** Features scaled with `StandardScaler` before training and inference
- **Test accuracy:** ≈ 96.5%

## 📁 Project Structure

```
.
├── app.py                        # Streamlit application
├── svm_model.pkl                 # Trained SVM classifier
├── scaler.pkl                    # Fitted StandardScaler
├── feature_stats.csv             # Per-feature min/max/mean used for input ranges & defaults
├── sample_report_template.csv    # Downloadable template for batch CSV uploads
├── requirements.txt              # Python dependencies
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/breast-cancer-diagnosis-predictor.git
cd breast-cancer-diagnosis-predictor

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the app

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

## 📊 Usage

### Single-patient prediction
1. Select **Sliders** or **Number boxes** in the sidebar.
2. Adjust the cell nuclei measurements (grouped under Mean / Standard Error / Worst).
3. Click **🔍 Predict Diagnosis** to view the result and decision score.

### Batch prediction
1. Select **Upload file** in the sidebar.
2. Download the sample template to see the required column format.
3. Upload a CSV or Excel file with the same columns.
4. Click **🔍 Analyze Report** to classify every row.
5. Download the results as a CSV.

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — web app framework
- [scikit-learn](https://scikit-learn.org/) — SVM model & StandardScaler
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data handling
- [joblib](https://joblib.readthedocs.io/) — model serialization

## 📈 Dataset

This project uses the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset, a widely used benchmark dataset in machine learning containing 569 instances of digitized biopsy features with 30 real-valued predictors.

## 👤 Author

**Bishal Tamang**
Data Science Portfolio Project

---

*Built as part of an ongoing machine learning portfolio exploring classification, EDA, and interactive model deployment.*

