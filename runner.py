import argparse
import importlib.util
from pathlib import Path

if __package__:
    from .library import PreprocessingPipeline
else:
    import sys

    PACKAGE_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = PACKAGE_DIR.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from preprocessing_pipeline.library import PreprocessingPipeline


def load_config_from_file(config_path: str):
    path = Path(config_path)
    spec = importlib.util.spec_from_file_location("pipeline_config", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load config from '{config_path}'.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "config"):
        return module.config
    if hasattr(module, "EXAMPLE_CONFIG"):
        return module.EXAMPLE_CONFIG
    raise ValueError(f"Config file '{config_path}' must expose 'config' or 'EXAMPLE_CONFIG'.")


def main():
    parser = argparse.ArgumentParser(description="Run the open-source preprocessing pipeline.")
    parser.add_argument("--config", required=True, help="Path to a Python config file.")
    parser.add_argument("--start", default="utils", help="Stage to start from.")
    parser.add_argument("--end", default="transform", help="Stage to finish at.")
    parser.add_argument("--reports-dir", default="", help="Optional folder to export CSV and HTML reports.")
    args = parser.parse_args()

    config = load_config_from_file(args.config)
    pipeline = PreprocessingPipeline(config)
    artifacts = pipeline.run(start=args.start, end=args.end)

    if args.reports_dir:
        pipeline.get_report_csv(base_path=args.reports_dir)
        pipeline.get_report_html(base_path=args.reports_dir)

    print("Pipeline finished.")
    print(f"Selected features: {len(artifacts.selected_features or artifacts.candidate_features)}")
    if artifacts.transformed_data is not None:
        print(f"Output shape: {artifacts.transformed_data.shape}")


if __name__ == "__main__":
    main()
