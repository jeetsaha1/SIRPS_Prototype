from common.schemas import FILES


def main() -> None:
    for name, spec in FILES.items():
        print(f"{name}: {spec['folder']}/{spec['file']}")
        print("  " + ", ".join(spec["columns"]))
        if spec["dates"]:
            print("  date columns: " + ", ".join(spec["dates"]))


if __name__ == "__main__":
    main()
