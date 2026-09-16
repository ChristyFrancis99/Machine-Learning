# URL-Based Phishing Website Detection Using Machine Learning and Artificial Neural Networks

## Abstract

Phishing websites are designed to deceive users into revealing sensitive information such as passwords, financial details, and personal data. This project presents a machine-learning-based approach for detecting phishing websites from URL and webpage-related characteristics. The system uses preprocessing, exploratory analysis, feature selection, and an Artificial Neural Network (ANN) to classify websites as legitimate or phishing. The project also includes robustness evaluation, model validation, result visualization, and a URL screening interface.

## Problem Statement

Traditional phishing detection methods often depend on manually maintained blacklists or fixed rules. These approaches can struggle with newly created phishing websites and changing attack patterns. The objective of this project is to develop a data-driven system that can learn distinguishing characteristics of legitimate and phishing websites and use them for automated classification.

## Solution

The project follows an end-to-end machine learning pipeline:

1. Load and inspect the phishing URL dataset.
2. Clean and preprocess the data.
3. Perform exploratory data analysis to understand feature and class distributions.
4. Remove raw or non-predictive identifier fields.
5. Separate input features and the target class.
6. Scale numerical features using training data.
7. Select informative features using Mutual Information-based feature selection.
8. Train an Artificial Neural Network for binary classification.
9. Evaluate the model using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.
10. Perform additional robustness and ablation evaluation.
11. Generate model comparisons, figures, and HTML result reports.
12. Provide a URL screening utility for testing URL-structure-based risk indicators.

### Dataset

The project uses the PhiUSIIL Phishing URL Dataset. The dataset contains legitimate and phishing website samples with URL and webpage-related features.

## Result

The project produces:

- Trained machine-learning model artifacts in `models/`.
- Evaluation metrics and comparison results.
- Robustness and ablation analysis.
- Generated plots and figures in `results/`.
- An HTML-based project results dashboard.
- A URL prediction/screening page for testing a raw URL.

The repository keeps the implementation and evaluation workflow reproducible rather than presenting unsupported or fabricated performance values. Run the pipeline to generate the current results from the available project data and model configuration.

## Conclusion

This project demonstrates an end-to-end approach for phishing website detection using machine learning and an Artificial Neural Network. By combining preprocessing, informative feature selection, neural-network classification, and robustness evaluation, the system provides a structured approach to distinguishing phishing websites from legitimate websites. The modular pipeline also makes it possible to retrain, evaluate, and inspect the model as the dataset or experimental configuration changes.

## How to Install

### 1. Clone the repository

```powershell
git clone https://github.com/ChristyFrancis99/Machine-Learning.git
cd Machine-Learning
```

### 2. Create a Python virtual environment

Python 3.11 is recommended for the current project environment.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell activation is blocked, use Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

### 3. Install backend dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
```

If the required ML packages are not available in the environment, install them with:

```powershell
python -m pip install numpy pandas matplotlib seaborn scikit-learn xgboost tensorflow jupyter ipykernel
```

### 4. Install frontend dependencies

```powershell
cd frontend
npm install
cd ..
```

## How to Run the Full Project

The project has a backend API and a frontend interface. Run them in separate terminals.

### Terminal 1 — Start the Backend

From the project root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.app:app --reload --port 8000
```

The backend API runs at:

```text
http://127.0.0.1:8000
```

### Terminal 2 — Start the Frontend

From the project root:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL displayed by Vite in your browser.

The frontend calls the backend endpoint:

```text
http://127.0.0.1:8000/api/analyze
```

If the backend is hosted somewhere else, copy `frontend/.env.example` to `frontend/.env` and configure:

```text
VITE_API_URL=<your-backend-api-url>
```

## Run the Complete ML Pipeline

From the project root, activate the virtual environment and run the stages in order:

```powershell
python backend/pipeline/run_phishing_pipeline.py
python backend/pipeline/run_stage2_robustness.py
python backend/pipeline/sanity_check.py
python backend/pipeline/display_results.py
```

### Pipeline Scripts

| Script | Purpose |
|---|---|
| `backend/pipeline/run_phishing_pipeline.py` | Main training and evaluation pipeline |
| `backend/pipeline/run_stage2_robustness.py` | Robustness and additional validation |
| `backend/pipeline/sanity_check.py` | Model artifact and inference checks |
| `backend/pipeline/display_results.py` | Generates the project results report |
| `backend/pipeline/predict_url.py` | Screens a raw URL using visible URL-structure signals |

Generated models are stored in `models/`, while figures and reports are stored in `results/`.

## View Results

Run:

```powershell
python backend/pipeline/display_results.py
```

This generates and opens:

```text
results/project_results.html
```

The report contains the available model comparison, robustness summary, cross-dataset evaluation, ablation results, and generated figures.

## Test a URL

To screen a raw URL:

```powershell
python backend/pipeline/predict_url.py https://www.example.com
```

The tool generates:

```text
results/url_prediction.html
```

It displays an estimated phishing-risk percentage and explains visible URL signals. This URL screening score is based on URL-structure indicators and should not be interpreted as the trained ANN probability, because the ANN requires the engineered dataset features, including webpage-derived features.

## Optional Commands

Run only the URL-feature ablation:

```powershell
python backend/pipeline/run_stage2_robustness.py --part-c-only
```

Run the completed notebook:

```powershell
jupyter lab FA1_completed.ipynb
```

## Project Structure

```text
Machine-Learning/
├── backend/
│   ├── pipeline/
│   ├── app.py
│   └── requirements.txt
├── data/
├── models/
├── notebooks/
├── results/
├── uci_dataset/
├── frontend/
├── docs/
└── readme.md
```

## License

This repository is intended for academic and research purposes.
