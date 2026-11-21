import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.schemas import FILES


def main() -> None:
    for name, spec in FILES.items():
        print(f"{name}: {spec['folder']}/{spec['file']}")
        print("  " + ", ".join(spec["columns"]))
        if spec["dates"]:
            print("  date columns: " + ", ".join(spec["dates"]))


if __name__ == "__main__":
    main()
