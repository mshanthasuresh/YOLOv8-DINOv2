"""Download datasets and write the fixed split YAMLs.

    python -m tools.prepare_data            # VOC only
    python -m tools.prepare_data --visdrone # also the second-domain dataset
"""

from __future__ import annotations

import argparse

from nafuse import data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visdrone", action="store_true")
    parser.add_argument("--low-fraction", type=float, default=0.1)
    args = parser.parse_args()

    data.download_voc()
    for name, path in data.write_voc_yamls(args.low_fraction).items():
        print(f"{name}: {path}")
    if args.visdrone:
        data.download_visdrone()
        print(f"visdrone: {data.write_visdrone_yaml()}")


if __name__ == "__main__":
    main()
