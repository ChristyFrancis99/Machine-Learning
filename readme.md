**Run the full project**

Open a terminal in the project folder:

```powershell
cd "C:\Users\Christy\Desktop\Website\Machine-Learning"
```

Start the backend API in one terminal:

```powershell
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.app:app --reload --port 8000
```

Then start the frontend in a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

The frontend calls `http://127.0.0.1:8000/api/analyze` by default. Copy
`frontend/.env.example` to `frontend/.env` and change `VITE_API_URL` when the API is hosted elsewhere.

Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies. The current environment is missing `tensorflow` and `xgboost`.

```powershell
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib seaborn scikit-learn xgboost tensorflow jupyter ipykernel
```

Run the complete workflow in this order:

```powershell
python backend/pipeline/run_phishing_pipeline.py
python backend/pipeline/run_stage2_robustness.py
python backend/pipeline/sanity_check.py
python backend/pipeline/display_results.py
python backend/pipeline/predict_url.py https://www.example.com
```

The scripts are:

- Stage 1 training and evaluation: `backend/pipeline/run_phishing_pipeline.py`
- Stage 2 robustness validation: `backend/pipeline/run_stage2_robustness.py`
- Model artifact/inference check: `backend/pipeline/sanity_check.py`
- Results report generator: `backend/pipeline/display_results.py`
- Raw URL screening display: `backend/pipeline/predict_url.py`

Generated models are saved in `models`, and figures/results are saved in `results`.

`backend/pipeline/display_results.py` creates and opens `results/project_results.html`, which displays the model comparison, robustness summary, cross-dataset evaluation, ablation results, and generated figures.

`backend/pipeline/predict_url.py` creates and opens `results/url_prediction.html`. It accepts a raw URL, displays an estimated phishing-risk percentage, and explains the visible URL signals. This is a transparent URL-structure screening score, not the trained ANN probability: the ANN requires the original dataset's engineered feature values, including webpage-derived features.

Example:

```powershell
python backend/pipeline/predict_url.py https://www.example.com
```

For only the URL-feature ablation:

```powershell
python backend/pipeline/run_stage2_robustness.py --part-c-only
```

To run the completed notebook instead:

```powershell
jupyter lab FA1_completed.ipynb
```

If PowerShell activation is blocked, use Command Prompt:

```cmd
.venv\Scripts\activate.bat
```
