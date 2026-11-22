from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def build():
    """Generates variant flag mappings for Experiments C (identity control) and P (placebo)."""
    dataset_path = ROOT / "data" / "model_dataset.csv"
    if dataset_path.exists():
        df = pd.read_csv(dataset_path)
        # Variants C and P use matched rows
        df["in_C"] = df["in_matched"]
        df["in_P"] = df["in_matched"]
        df.to_csv(dataset_path, index=False)


if __name__ == "__main__":
    build()
