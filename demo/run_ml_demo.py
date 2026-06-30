from pathlib import Path
import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from preprocessing_pipeline import PreprocessingPipeline
from preprocessing_pipeline.runner import load_config_from_file


def main():
    demo_dir = Path(__file__).resolve().parent
    config = load_config_from_file(str(demo_dir / "demo_config.py"))

    pipeline = PreprocessingPipeline(config)
    artifacts = pipeline.run()

    transformed = artifacts.transformed_data.copy()
    target_column = config["target_column"]
    cohort_column = config["cohort_column"]
    reference_cohort = config["reference_cohort"]

    all_cohorts = sorted(transformed[cohort_column].astype(str).unique().tolist())
    evaluation_cohorts = [value for value in all_cohorts if value != reference_cohort]
    holdout_cohort = evaluation_cohorts[-1]

    feature_columns = [feature for feature in artifacts.selected_features if feature in transformed.columns]
    train_mask = transformed[cohort_column].astype(str) != holdout_cohort
    test_mask = transformed[cohort_column].astype(str) == holdout_cohort

    x_train = transformed.loc[train_mask, feature_columns]
    y_train = transformed.loc[train_mask, target_column]
    x_test = transformed.loc[test_mask, feature_columns]
    y_test = transformed.loc[test_mask, target_column]

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = model.predict(x_test)

    metrics_df = pd.DataFrame(
        [
            {
                "model_name": "logistic_regression",
                "train_cohorts": ",".join(sorted(transformed.loc[train_mask, cohort_column].astype(str).unique().tolist())),
                "holdout_cohort": holdout_cohort,
                "train_rows": int(train_mask.sum()),
                "test_rows": int(test_mask.sum()),
                "auc": float(roc_auc_score(y_test, probabilities)),
                "accuracy": float(accuracy_score(y_test, predictions)),
                "feature_count": len(feature_columns),
            }
        ]
    )

    coefficients_df = pd.DataFrame(
        {
            "feature_name": feature_columns,
            "coefficient": model.coef_[0],
        }
    ).sort_values("coefficient", ascending=False, key=lambda series: series.abs()).reset_index(drop=True)

    reports_dir = demo_dir / "demo_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "ml_demo_metrics.csv"
    coefficients_path = reports_dir / "ml_demo_coefficients.csv"
    metrics_df.to_csv(metrics_path, index=False)
    coefficients_df.to_csv(coefficients_path, index=False)

    print("ML demo completed.")
    print(f"Training cohorts: {sorted(transformed.loc[train_mask, cohort_column].astype(str).unique().tolist())}")
    print(f"Holdout cohort: {holdout_cohort}")
    print(f"Modeling features: {feature_columns}")
    print(metrics_df.to_string(index=False))
    print("\nTop coefficients:")
    print(coefficients_df.head().to_string(index=False))
    print(f"\nMetrics file: {metrics_path}")
    print(f"Coefficients file: {coefficients_path}")


if __name__ == "__main__":
    main()
