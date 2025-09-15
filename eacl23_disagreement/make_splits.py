#!/usr/bin/env python3
"""Create training variants for disagreement experiments."""

import argparse
import os
import shutil


def read_tsv(path):
    with open(path, encoding="utf8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            yield parts


def write_tsv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf8") as handle:
        for row in rows:
            handle.write("\t".join(row) + "\n")


def _norm(value):
    return (value or "").strip().lower().replace(" ", "_")


def cat_is(row, key):
    category = _norm(row[4] if len(row) > 4 else "")
    return category == _norm(key)


def main():
    parser = argparse.ArgumentParser(description="Create MaChAmp split directories")
    parser.add_argument("--base_dir", default="data/md_agreement")
    parser.add_argument("--out_root", default="data/splits")
    args = parser.parse_args()

    train_rows = list(read_tsv(os.path.join(args.base_dir, "train.tsv")))

    is_App = lambda row: row[2] in {"A++", "A+"}
    is_A0 = lambda row: row[2] == "A0"

    variants = {
        "App": [row for row in train_rows if is_App(row)],
        "App_A0all": train_rows[:],
        "App_A0_SUBJ": [
            row
            for row in train_rows
            if is_App(row) or (is_A0(row) and cat_is(row, "Subjectivity"))
        ],
        "App_A0_MISS": [
            row
            for row in train_rows
            if is_App(row) or (is_A0(row) and cat_is(row, "Missing_Info"))
        ],
        "App_A0_AMB": [
            row
            for row in train_rows
            if is_App(row) or (is_A0(row) and cat_is(row, "Ambiguity"))
        ],
    }

    for name, rows in variants.items():
        out_dir = os.path.join(args.out_root, name)
        write_tsv(os.path.join(out_dir, "train.tsv"), rows)
        os.makedirs(out_dir, exist_ok=True)
        shutil.copy2(os.path.join(args.base_dir, "dev.tsv"), os.path.join(out_dir, "dev.tsv"))
        shutil.copy2(os.path.join(args.base_dir, "test.tsv"), os.path.join(out_dir, "test.tsv"))
        print(name, "train_size:", len(rows))


if __name__ == "__main__":
    main()
