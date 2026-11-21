from pathlib import Path
import importlib


GENERATORS = [
    "data_generation.make_company_master",
    "data_generation.make_mock_data",
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    print(f"Running SIRPS data pipeline from {root}")
    for module_name in GENERATORS:
        module = importlib.import_module(module_name)
        if hasattr(module, "main"):
            module.main()
        elif hasattr(module, "build"):
            module.build()
        elif hasattr(module, "build_company_master"):
            frame = module.build_company_master()
            output = root / "data" / "company_master.csv"
            frame.to_csv(output, index=False)
        else:
            print(f"skipped: {module_name}")


if __name__ == "__main__":
    main()
