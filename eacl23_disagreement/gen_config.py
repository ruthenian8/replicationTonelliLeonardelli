#!/usr/bin/env python3
"""Generate MaChAmp dataset configs."""

import argparse
import json
import os

TEMPLATE = {
    "MD": {
        "train_data_path": "",
        "dev_data_path": "",
        "test_data_path": "",
        "sent_idxs": [0],
        "tasks": {
            "offense": {"task_type": "classification", "column_idx": 1}
        },
    }
}


def add_aux_agr3(config):
    config["MD"]["tasks"]["agr3"] = {
        "task_type": "classification",
        "column_idx": 2,
    }
    return config


def add_aux_agr6(config):
    config["MD"]["tasks"]["agr6"] = {
        "task_type": "classification",
        "column_idx": 3,
    }
    return config


def main():
    parser = argparse.ArgumentParser(description="Generate MaChAmp config JSONs.")
    parser.add_argument("--split_dir", required=True, help="Directory with train/dev/test TSVs")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument(
        "--mode",
        choices=["single", "mtl3", "mtl6"],
        default="single",
        help="single-task or multi-task variants",
    )
    args = parser.parse_args()

    config = json.loads(json.dumps(TEMPLATE))
    config["MD"]["train_data_path"] = os.path.join(args.split_dir, "train.tsv")
    config["MD"]["dev_data_path"] = os.path.join(args.split_dir, "dev.tsv")
    config["MD"]["test_data_path"] = os.path.join(args.split_dir, "test.tsv")

    if args.mode == "mtl3":
        config = add_aux_agr3(config)
    if args.mode == "mtl6":
        config = add_aux_agr6(config)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf8") as handle:
        json.dump(config, handle, indent=2)


if __name__ == "__main__":
    main()
