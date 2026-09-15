"""Generate a browser-readable report from the saved project results."""

from html import escape
from pathlib import Path
import webbrowser

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results"
REPORT_PATH = RESULTS_DIR / "project_results.html"


def load_csv(filename):
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    return pd.read_csv(path)


def format_percent(value):
    try:
        return f"{float(value) * 100:.3f}%"
    except (TypeError, ValueError):
        return str(value)


def metric_value(dataframe, column, row_index=0):
    if dataframe is None or column not in dataframe.columns or dataframe.empty:
        return "N/A"
    return format_percent(dataframe.iloc[row_index][column])


def dataframe_table(dataframe, percent_columns=()):
    if dataframe is None or dataframe.empty:
        return '<p class="muted">Run the corresponding pipeline stage to create this result.</p>'

    display_df = dataframe.copy()
    for column in percent_columns:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(format_percent)
    return display_df.to_html(index=False, classes="data-table", border=0, escape=True)


def figure_block(filename, title):
    if not (RESULTS_DIR / filename).exists():
        return ""
    return f'''
        <figure>
          <img src="{escape(filename)}" alt="{escape(title)}">
          <figcaption>{escape(title)}</figcaption>
        </figure>
    '''


def build_report():
    comparison = load_csv("model_comparison_table6.csv")
    ablation = load_csv("ablation_study_table7.csv")
    robustness = load_csv("stage2_consolidated_summary.csv")
    cross_dataset = load_csv("cross_dataset_generalization.csv")

    ann_row = None
    if comparison is not None and "Model" in comparison.columns:
        ann_matches = comparison[comparison["Model"].str.contains("Proposed ANN", na=False)]
        if not ann_matches.empty:
            ann_row = ann_matches.iloc[0]

    ann_accuracy = format_percent(ann_row["Accuracy"]) if ann_row is not None else "N/A"
    ann_f1 = format_percent(ann_row["F1-Score"]) if ann_row is not None else "N/A"
    ann_auc = f"{float(ann_row['ROC-AUC']):.6f}" if ann_row is not None else "N/A"

    comparison_table = dataframe_table(
        comparison,
        ("Accuracy", "Precision", "Recall", "F1-Score"),
    )
    ablation_table = dataframe_table(
        ablation,
        ("Accuracy", "Precision", "Recall", "F1-Score"),
    )
    robustness_table = dataframe_table(
        robustness,
        ("Accuracy", "F1_Score"),
    )
    cross_dataset_table = dataframe_table(
        cross_dataset,
        ("Accuracy", "Precision", "Recall", "F1"),
    )

    html = f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Phishing Detection Results</title>
  <style>
    :root {{
      --ink: #17212b;
      --muted: #65727e;
      --paper: #f5f7f8;
      --panel: #ffffff;
      --line: #dce3e8;
      --accent: #087f8c;
      --accent-soft: #dff4f1;
      --warning: #b45309;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; color: var(--ink); background: var(--paper); font-family: Georgia, 'Times New Roman', serif; }}
    main {{ width: min(1180px, calc(100% - 32px)); margin: 0 auto; padding: 44px 0 72px; }}
    header {{ border-bottom: 1px solid var(--line); padding-bottom: 28px; margin-bottom: 28px; }}
    .eyebrow {{ color: var(--accent); font: 700 12px/1.2 Arial, sans-serif; letter-spacing: 1.8px; text-transform: uppercase; }}
    h1 {{ max-width: 760px; margin: 10px 0; font-size: clamp(2.2rem, 5vw, 4.5rem); line-height: .98; font-weight: 500; }}
    h2 {{ margin: 42px 0 14px; font-size: 1.7rem; font-weight: 500; }}
    h3 {{ margin: 0 0 8px; font-size: 1.1rem; }}
    p {{ color: var(--muted); line-height: 1.6; }}
    .lede {{ max-width: 700px; font-size: 1.08rem; }}
    .metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 26px 0; }}
    .metric {{ background: var(--panel); border: 1px solid var(--line); border-top: 4px solid var(--accent); padding: 18px; }}
    .metric strong {{ display: block; margin-top: 8px; font: 700 1.7rem/1 Arial, sans-serif; }}
    .metric span {{ color: var(--muted); font: 700 11px/1.2 Arial, sans-serif; letter-spacing: 1px; text-transform: uppercase; }}
    .panel {{ background: var(--panel); border: 1px solid var(--line); padding: 18px; overflow-x: auto; }}
    .data-table {{ width: 100%; border-collapse: collapse; font: 13px/1.45 Arial, sans-serif; white-space: nowrap; }}
    .data-table th {{ background: var(--accent-soft); color: var(--ink); text-align: left; }}
    .data-table th, .data-table td {{ border-bottom: 1px solid var(--line); padding: 9px 10px; }}
    .data-table tr:last-child td {{ border-bottom: 0; }}
    .muted {{ color: var(--muted); font-style: italic; }}
    .figures {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 18px; }}
    figure {{ margin: 0; background: var(--panel); border: 1px solid var(--line); padding: 10px; }}
    figure img {{ display: block; width: 100%; height: auto; }}
    figcaption {{ padding: 10px 4px 2px; color: var(--muted); font: 12px Arial, sans-serif; }}
    .note {{ border-left: 4px solid var(--warning); padding: 3px 0 3px 14px; }}
    footer {{ margin-top: 44px; color: var(--muted); font: 12px Arial, sans-serif; }}
    @media (max-width: 720px) {{
      main {{ width: min(100% - 22px, 1180px); padding-top: 28px; }}
      .metrics, .figures {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <div class="eyebrow">Machine Learning Project · Results Report</div>
    <h1>Phishing detection, measured.</h1>
    <p class="lede">A browser-readable summary of the trained ANN, baseline comparisons, ablation study, and robustness experiments generated by this project.</p>
  </header>

  <section class="metrics" aria-label="Proposed ANN metrics">
    <div class="metric"><span>ANN accuracy</span><strong>{escape(ann_accuracy)}</strong></div>
    <div class="metric"><span>ANN F1-score</span><strong>{escape(ann_f1)}</strong></div>
    <div class="metric"><span>ANN ROC-AUC</span><strong>{escape(ann_auc)}</strong></div>
  </section>

  <h2>Model comparison</h2>
  <div class="panel">{comparison_table}</div>

  <h2>Robustness summary</h2>
  <div class="panel">{robustness_table}</div>

  <h2>Cross-dataset evaluation</h2>
  <p class="note">The external UCI result is shown separately because the feature definitions and data distribution differ from PhiUSIIL.</p>
  <div class="panel">{cross_dataset_table}</div>

  <h2>Feature-selection ablation</h2>
  <div class="panel">{ablation_table}</div>

  <h2>Generated figures</h2>
  <div class="figures">
    {figure_block("fig1_target_distribution.png", "Target class distribution")}
    {figure_block("fig2_https_usage.png", "HTTPS usage by class")}
    {figure_block("fig3_correlation_heatmap.png", "Feature correlation heatmap")}
    {figure_block("fig4_top20_mi_features.png", "Top features by mutual information")}
    {figure_block("fig5_ann_training_history.png", "ANN training and validation curves")}
    {figure_block("fig6_ann_confusion_matrix.png", "ANN confusion matrix")}
    {figure_block("fig7_model_comparison.png", "Model comparison")}
  </div>

  <footer>Generated from the CSV and PNG artifacts in the results folder.</footer>
</main>
</body>
</html>'''

    RESULTS_DIR.mkdir(exist_ok=True)
    REPORT_PATH.write_text(html, encoding="utf-8")
    return REPORT_PATH


if __name__ == "__main__":
    report_path = build_report()
    print(f"Results report created: {report_path}")
    webbrowser.open(report_path.as_uri())
