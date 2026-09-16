# PhishGuard Pro / PhishLens

PhishLens is the research-console frontend for the **URL-Based Phishing Website Detection Using Machine Learning and Artificial Neural Networks** project.

The application provides five research views:

1. **URL Analyzer** — validates a URL and performs a deterministic URL-structure screening estimate without opening or fetching the website.
2. **Model Performance** — presents the embedded research evaluation snapshot currently shipped with the frontend.
3. **Robustness Validation** — presents the saved robustness/validation view used by the dashboard.
4. **Feature Analysis** — presents the embedded feature-ranking snapshot.
5. **About Project** — documents the dataset, preprocessing pipeline, feature selection, and ANN architecture.

## Important research limitation

The browser-based URL analyzer is **not the trained ANN inference pipeline**. The original ANN expects the project's model-compatible engineered features, scalers, and feature selector. A raw URL entered into this frontend is therefore handled as an explainable **URL-structure screening estimate**.

The analyzer:

- never opens or crawls the submitted URL;
- never executes page content;
- treats the URL as untrusted text;
- checks HTTPS usage, IP-hosting, URL length, subdomains, encoded characters, digit ratio, query parameters, and suspicious wording;
- shows the signals used to produce the deterministic estimate.

Do not interpret the displayed screening percentage as a calibrated ANN probability.

## Research data status

The current GitHub repository contains the React/TanStack frontend, but the original Python `results/*.csv` artifacts and trained model files are **not present in this repository**. The dashboard therefore uses an explicitly labelled embedded research snapshot for the performance and feature views.

When the original result artifacts are added, the frontend should be changed to load those files dynamically rather than duplicating their values in TypeScript.

## Technology

- React 19
- TypeScript
- Vite
- TanStack Router / Start
- Tailwind CSS
- Lucide React
- Bun lockfile

## Local development

```bash
bun install
bun run dev
```

Open the local Vite URL shown in the terminal.

## Production build

```bash
bun run build
bun run preview
```

## Linting

```bash
bun run lint
```

## Repository structure

```text
src/
  components/
    PhishLensDashboard.tsx
    ui/
  lib/
    phishlens.ts
  routes/
  main.tsx
public/
package.json
bun.lock
vite.config.ts
tsconfig.json
```

## Research pipeline documented by the project

```text
Dataset
  -> Cleaning
  -> Identifier / leakage-prone column removal
  -> Stratified train/test split
  -> Scaling
  -> Mutual Information feature selection
  -> ANN training
  -> Baseline comparison
  -> Cross-validation
  -> Cross-dataset evaluation
```

The intended ANN architecture is:

```text
Input (selected features)
    -> Dense(128, ReLU)
    -> Dropout(0.30)
    -> Dense(64, ReLU)
    -> Dropout(0.30)
    -> Dense(32, ReLU)
    -> Dense(1, Sigmoid)
```

## Label convention

The research pipeline uses:

- `0 = Phishing`
- `1 = Legitimate`

## Safety

This project is for research and educational use. A URL classification result is not a guarantee of safety. Never open a suspicious URL simply because a screening result appears legitimate.
