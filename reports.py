import os
from typing import Dict

import pandas as pd


def export_reports_csv(report_map: Dict[str, pd.DataFrame], report_name: str, base_path: str):
    os.makedirs(base_path, exist_ok=True)
    for suffix, report in report_map.items():
        if isinstance(report, pd.DataFrame) and not report.empty:
            report.to_csv(os.path.join(base_path, f"{report_name}_{suffix}.csv"), index=False)


def export_reports_html(report_map: Dict[str, pd.DataFrame], summary: pd.DataFrame, report_name: str, base_path: str):
    os.makedirs(base_path, exist_ok=True)
    sections = [
        "<h1>Preprocessing Report</h1>",
        "<h2>Summary</h2>",
        summary.to_html(index=False),
    ]
    for title, report in report_map.items():
        if isinstance(report, pd.DataFrame) and not report.empty:
            sections.append(f"<h2>{title}</h2>")
            sections.append(report.to_html(index=False))
    html_path = os.path.join(base_path, f"{report_name}.html")
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(sections))
