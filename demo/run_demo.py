from pathlib import Path
import sys

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

    reports_dir = demo_dir / "demo_reports"
    pipeline.get_report_csv(base_path=str(reports_dir))
    pipeline.get_report_html(base_path=str(reports_dir))

    print("Demo completed.")
    print(f"Selected features: {artifacts.selected_features}")
    if artifacts.transformed_data is not None:
        print(f"Output file: {config['path_df_output']}")
        print(f"Output shape: {artifacts.transformed_data.shape}")
        print(artifacts.transformed_data.head().to_string(index=False))


if __name__ == "__main__":
    main()
