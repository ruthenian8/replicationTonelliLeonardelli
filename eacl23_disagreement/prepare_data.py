#!/usr/bin/env python3
"""Normalize MD-Agreement data for MaChAmp."""

import argparse
import csv
import os
import re
from collections import Counter


def _sniff_sep(path):
    """Heuristically choose delimiter for CSV/TSV files."""
    with open(path, "r", encoding="utf8", newline="") as f:
        head = f.read(4096)
    if "\t" in head and head.count("\t") >= head.count(","):
        return "\t"
    return ","


def _norm_off(label):
    """Normalize offensive labels to OFF/NOT."""
    label = str(label).strip().lower()
    if label in {"off", "offensive", "1", "true", "toxic"}:
        return "OFF"
    if label in {"not", "non-offensive", "0", "false", "clean", "none"}:
        return "NOT"
    raise ValueError(f"Unrecognized label: {label}")


def _majority_and_agreement(row, annotator_cols):
    """Derive majority offensive label and agreement tier."""
    labels = [
        _norm_off(row[col])
        for col in annotator_cols
        if col in row and str(row[col]).strip() != ""
    ]
    if len(labels) != 5:
        raise ValueError("Expected 5 annotator labels to derive majority/agreement.")

    counts = Counter(labels)
    majority_label, majority_votes = counts.most_common(1)[0]
    if majority_votes == 5:
        agreement = "A++"
    elif majority_votes == 4:
        agreement = "A+"
    elif majority_votes == 3:
        agreement = "A0"
    else:
        raise ValueError(f"Unexpected vote distribution: {counts}")
    return majority_label, agreement


def _read_dict(path, sep=None):
    """Yield rows from CSV/TSV as dictionaries."""
    delimiter = sep or _sniff_sep(path)
    with open(path, encoding="utf8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter=delimiter)


ID_DIGITS = re.compile(r"^(\d+)")


def _id_base(identifier):
    if identifier is None:
        return ""
    identifier = str(identifier).strip()
    match = ID_DIGITS.match(identifier)
    return match.group(1) if match else identifier


def load_taxonomy(tax_path):
    """Load taxonomy annotations from Category_dataset.tsv."""
    if not tax_path or not os.path.exists(tax_path):
        return {}

    delimiter = _sniff_sep(tax_path)
    taxonomy = {}
    for row in _read_dict(tax_path, sep=delimiter):
        tweet_id = _id_base(row.get("ID"))
        taxonomy[tweet_id] = {
            "text": (row.get("Text") or "").strip(),
            "agr": (row.get("Agreement_level") or "").strip(),
            "primary_cat": (row.get("Primary_category") or "").strip(),
            "primary_subcat": (row.get("Primary_subcategry") or "").strip(),
            "secondary_cat": (row.get("Secondary_category") or "").strip(),
            "secondary_subcat": (row.get("Seconday_subcategory") or "").strip(),
        }
    return taxonomy


def process_split(
    in_path,
    out_path,
    taxonomy,
    id_col,
    text_col,
    gold_col=None,
    agr_col=None,
    ann_cols=None,
    prefer_taxonomy_agr=True,
):
    """Convert a raw MD-Agreement split into MaChAmp TSV format."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    delimiter = _sniff_sep(in_path)

    with open(in_path, encoding="utf8", newline="") as in_handle, open(
        out_path, "w", encoding="utf8"
    ) as out_handle:
        reader = csv.DictReader(in_handle, delimiter=delimiter)
        for row in reader:
            tweet_id = _id_base(row.get(id_col))
            text = (row.get(text_col) or "").replace("\n", " ").strip()

            if gold_col and row.get(gold_col, "") != "":
                offensive = _norm_off(row[gold_col])
            elif ann_cols:
                offensive, _ = _majority_and_agreement(row, ann_cols)
            else:
                raise ValueError("Need --gold_col or --ann_cols to derive OFF/NOT.")

            if prefer_taxonomy_agr and tweet_id in taxonomy and taxonomy[tweet_id]["agr"]:
                agreement = taxonomy[tweet_id]["agr"]
            elif agr_col and row.get(agr_col, "") != "":
                agreement = row[agr_col].strip()
            elif ann_cols:
                _, agreement = _majority_and_agreement(row, ann_cols)
            else:
                agreement = ""

            agr6 = f"{agreement}_{offensive}" if agreement else ""

            info = taxonomy.get(tweet_id, {})
            main_cat = info.get("primary_cat", "")
            subtype = info.get("primary_subcat", "")
            secondary_cat = info.get("secondary_cat", "")
            secondary_subtype = info.get("secondary_subcat", "")

            columns = [
                text,
                offensive,
                agreement,
                agr6,
                main_cat,
                subtype,
                secondary_cat,
                secondary_subtype,
            ]
            out_handle.write("\t".join(columns) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Normalize MD-Agreement splits for MaChAmp training."
    )
    parser.add_argument("--train_csv", required=True, help="MD-Agreement train split")
    parser.add_argument("--dev_csv", required=True, help="MD-Agreement dev split")
    parser.add_argument("--test_csv", required=True, help="MD-Agreement test split")
    parser.add_argument(
        "--taxonomy_path",
        required=True,
        help="Path to Category_dataset.tsv (taxonomy annotations)",
    )
    parser.add_argument("--id_col", default="tweet_id")
    parser.add_argument("--text_col", default="text")
    parser.add_argument(
        "--gold_col",
        default=None,
        help="Column containing gold OFF/NOT labels (optional if deriving from annotators)",
    )
    parser.add_argument(
        "--agr_col",
        default=None,
        help="Column containing agreement labels (A++/A+/A0), optional",
    )
    parser.add_argument(
        "--ann_cols",
        nargs="*",
        default=None,
        help="Names of five annotator columns to derive labels if needed",
    )
    parser.add_argument(
        "--prefer_taxonomy_agr",
        action="store_true",
        help="Use taxonomy agreement labels when available",
    )
    parser.add_argument("--outdir", default="data/md_agreement")
    args = parser.parse_args()

    taxonomy = load_taxonomy(args.taxonomy_path)
    os.makedirs(args.outdir, exist_ok=True)

    process_split(
        args.train_csv,
        os.path.join(args.outdir, "train.tsv"),
        taxonomy,
        args.id_col,
        args.text_col,
        args.gold_col,
        args.agr_col,
        args.ann_cols,
        args.prefer_taxonomy_agr,
    )
    process_split(
        args.dev_csv,
        os.path.join(args.outdir, "dev.tsv"),
        taxonomy,
        args.id_col,
        args.text_col,
        args.gold_col,
        args.agr_col,
        args.ann_cols,
        args.prefer_taxonomy_agr,
    )
    process_split(
        args.test_csv,
        os.path.join(args.outdir, "test.tsv"),
        taxonomy,
        args.id_col,
        args.text_col,
        args.gold_col,
        args.agr_col,
        args.ann_cols,
        args.prefer_taxonomy_agr,
    )


if __name__ == "__main__":
    main()
