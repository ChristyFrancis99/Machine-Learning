# PhishLens frontend dashboard

## Build
- Replace the placeholder screen with a complete five-view cybersecurity research dashboard.
- Add a fixed desktop sidebar, mobile drawer navigation, and compact top status bar with theme control.
- Use the dark charcoal/navy visual system, semantic teal/green/amber/red states, Space Grotesk headings, IBM Plex Sans body text, restrained borders, and subtle reveal motion.

## URL Analyzer
- Build URL entry, clear, example selection, empty, validation, staged loading, and result states.
- Treat every URL as text only; never navigate to or request it.
- Generate deterministic demo scores from HTTPS, IP host, URL length, subdomains, encoded characters, digits, query parameters, and suspicious terms.
- Show mock classification, probabilities, risk meter, timestamp, explanation rows, disclaimer, and a collapsible extracted-signal table.

## Research views
- Model Performance: render the supplied model metrics in comparison tables and four compact charts, highlighting XGBoost and the Proposed ANN.
- Robustness Validation: show cross-validation, cross-dataset warning, and URL-only ablation comparisons.
- Feature Analysis: show feature counts, mutual-information explanation, horizontal ranking chart, and table.
- About Project: show goal, datasets, pipeline, ANN architecture, technologies, limitations, and future integration roadmap.

## Technical details
- Keep all behavior frontend-only in React; no Python, backend, model, CSV, or network dependencies.
- Add small reusable dashboard/chart primitives and use existing design-system controls plus Lucide icons.
- Define all visual colors and fonts as semantic tokens in the global stylesheet.
- Add unique page metadata for the dashboard and remove template metadata.
- Verify desktop and mobile rendering, navigation, example analysis, invalid URL handling, and result display in the live preview.
