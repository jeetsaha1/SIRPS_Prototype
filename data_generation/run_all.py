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
    "data_generation.gen_prices",
    "data_generation.gen_esg",
    "data_generation.gen_fundamentals",
    "data_generation.gen_regime",
    "data_generation.build_targets",
    "data_generation.build_model_dataset",
    "data_generation.build_variants",
]


def main() -> None:
    print(f"Running SIRPS data pipeline from {ROOT}")
    for module_name in PIPELINE:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            print(f"skipped: {module_name}")
            continue

        if hasattr(module, "main"):
            module.main()
        elif hasattr(module, "build"):
            module.build()
        elif hasattr(module, "build_company_master"):
            frame = module.build_company_master()
            output = ROOT / "data" / "company_master.csv"
            frame.to_csv(output, index=False)
        else:
            print(f"skipped: {module_name}")


if __name__ == "__main__":
    main()
