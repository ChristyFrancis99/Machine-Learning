"""HTTP API for the PhishLens frontend.

The API deliberately analyses URL text only.  It never resolves, downloads, or
opens a submitted URL.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .pipeline.predict_url import extract_features, heuristic_reasons, screening_probability

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"

app = FastAPI(title="PhishLens API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class AnalyzeRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)


def risk_category(probability: float) -> str:
    percentage = probability * 100
    return "High" if percentage >= 70 else "Medium" if percentage >= 35 else "Low"


def feature_interpretation(name: str, value: object) -> str:
    descriptions = {
        "IsHTTPS": "Encrypted scheme" if value else "Unencrypted scheme",
        "IsDomainIP": "IP host detected" if value else "Named domain",
        "HasObfuscation": "Encoded text found" if value else "Not detected",
        "URLLength": "Long" if isinstance(value, int) and value > 75 else "Normal range",
        "NoOfSubDomain": "Elevated" if isinstance(value, int) and value >= 2 else "Typical",
    }
    return descriptions.get(name, "Extracted URL value")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/analyze")
def analyze(payload: AnalyzeRequest) -> dict[str, object]:
    try:
        normalized_url, parsed, features, _ = extract_features(payload.url)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    probability = screening_probability(features)
    phishing_reasons, legitimate_signals = heuristic_reasons(parsed, features)
    signals = []
    for reason in phishing_reasons:
        signals.append({"name": "Risk signal", "description": reason, "status": "risk"})
    for reason in legitimate_signals:
        signals.append({"name": "Legitimate signal", "description": reason, "status": "safe"})
    if not signals:
        signals.append({"name": "URL structure", "description": "No decisive URL-level signal was found.", "status": "neutral"})

    return {
        "normalizedUrl": normalized_url,
        "phishing": round(probability * 100, 2),
        "legitimate": round((1 - probability) * 100, 2),
        "classification": "LIKELY PHISHING" if probability >= 0.5 else "LIKELY LEGITIMATE",
        "risk": risk_category(probability),
        "signals": signals,
        "features": [
            {"feature": name, "value": round(value, 6) if isinstance(value, float) else value, "interpretation": feature_interpretation(name, value)}
            for name, value in features.items()
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": "URL-structure screening",
    }


def read_csv(name: str) -> list[dict[str, str]]:
    path = RESULTS_DIR / name
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"Research artifact is unavailable: {name}")
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


@app.get("/api/research")
def research() -> dict[str, object]:
    """Serve saved evaluation artifacts so the dashboard has one data source."""
    return {
        "models": read_csv("model_comparison_table6.csv"),
        "featureRanking": read_csv("mutual_information_ranking.csv"),
        "crossValidation": read_csv("kfold_cross_validation_results.csv"),
        "generalization": read_csv("cross_dataset_generalization.csv"),
        "urlOnlyAblation": read_csv("url_only_ablation_results.csv"),
    }
