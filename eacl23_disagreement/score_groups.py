#!/usr/bin/env python3
"""Compute F1 scores overall and by group for MaChAmp predictions."""

import argparse
import glob
import json
from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score


def read_split(tsv_path):
    rows = []
    with open(tsv_path, encoding="utf8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            rows.append(parts)
    return rows


def read_preds(pred_path):
    with open(pred_path, encoding="utf8") as handle:
        return [line.strip() for line in handle]


def f1s(y_true, y_pred):
    overall = f1_score(y_true, y_pred, average="micro", labels=["OFF", "NOT"])
    off_f1 = f1_score(y_true, y_pred, pos_label="OFF")
    not_f1 = f1_score(y_true, y_pred, pos_label="NOT")
    return overall, off_f1, not_f1


def main():
    parser = argparse.ArgumentParser(description="Score MaChAmp prediction files.")
    parser.add_argument("--test_tsv", required=True)
    parser.add_argument(
        "--pred_glob",
        required=True,
        help="Glob matching prediction files (one label per line)",
    )
    parser.add_argument(
        "--group_by",
        choices=["category", "subtype", "none"],
        default="none",
        help="Aggregate metrics per group",
    )
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    test_rows = read_split(args.test_tsv)
    gold = [row[1] for row in test_rows]
    categories = [row[4] if len(row) > 4 else "" for row in test_rows]
    subtypes = [row[5] if len(row) > 5 else "" for row in test_rows]

    groups = {"__ALL__": list(range(len(gold)))}
    if args.group_by == "category":
        groups = defaultdict(list)
        for idx, category in enumerate(categories):
            if category:
                groups[category].append(idx)
    elif args.group_by == "subtype":
        groups = defaultdict(list)
        for idx, subtype in enumerate(subtypes):
            if subtype:
                groups[subtype].append(idx)

    runs = []
    for pred_path in sorted(glob.glob(args.pred_glob)):
        preds = read_preds(pred_path)
        if len(preds) != len(gold):
            raise ValueError(
                f"Prediction length mismatch: {pred_path} has {len(preds)} items but gold has {len(gold)}"
            )
        per_group = {}
        for group_name, indices in groups.items():
            if not indices:
                continue
            gold_subset = [gold[i] for i in indices]
            pred_subset = [preds[i] for i in indices]
            overall, off_f1, not_f1 = f1s(gold_subset, pred_subset)
            per_group[group_name] = {
                "ALL": overall,
                "OFF": off_f1,
                "NOT": not_f1,
            }
        runs.append(per_group)

    if not runs:
        raise ValueError("No prediction files matched the provided glob.")

    aggregated = {}
    for group_name in runs[0]:
        for metric in ["ALL", "OFF", "NOT"]:
            values = [run[group_name][metric] for run in runs]
            aggregated.setdefault(group_name, {})[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }

    with open(args.out_json, "w", encoding="utf8") as handle:
        json.dump(aggregated, handle, indent=2)
    print("Wrote", args.out_json)


if __name__ == "__main__":
    main()
