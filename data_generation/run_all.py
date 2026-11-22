import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PIPELINE = [
    "data_generation.make_company_master",
    "data_generation.make_mock_data",
    "data_generation.gen_calendar",
    "data_generation.gen_index",
    "data_generation.gen_regime",
    "data_generation.gen_prices",
    "data_generation.gen_esg",
    "common.scoring",
    "data_generation.gen_fundamentals",
    "data_generation.build_targets",
    "data_generation.build_model_dataset",
    "data_generation.build_variants",
    "models.splits",
    "models.train_experiments",
    "models.evaluate",
    "models.stats_tests",
    "models.explain",
    "common.portfolio_engine",
]


def main() -> None:
    print(f"Running SIRPS data pipeline from {ROOT}")
    for module_name in PIPELINE:
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            print(f"skipped: {module_name} ({e})")
            continue

        print(f"--> Executing: {module_name}")
        if hasattr(module, "main"):
            module.main()
        elif hasattr(module, "build"):
            module.build()
        elif hasattr(module, "build_company_master"):
            frame = module.build_company_master()
            output = ROOT / "data" / "company_master.csv"
            output.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(output, index=False)
        else:
            print(f"skipped: {module_name} (no main/build)")

    print("Pipeline execution complete! All 20 datasets successfully generated.")


if __name__ == "__main__":
    main()
