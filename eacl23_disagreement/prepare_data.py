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


def _majority_and_agreement(labels):
    """Derive majority offensive label and agreement tier."""
    if len(labels) != 5:
        raise ValueError(
            f"Expected 5 annotator labels to derive majority/agreement, got {labels}"
        )

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

ANNOT_SPLIT = re.compile(r"[;,/\\|\s]+")

VALID_SPLITS = {"train", "dev", "test"}

SPLIT_ALIASES = {
    "training": "train",
    "trainset": "train",
    "development": "dev",
    "devset": "dev",
    "validation": "dev",
    "val": "dev",
    "testing": "test",
}


def _normalize_split_name(value):
    if value is None:
        return ""
    normalized = str(value).strip().lower()
    if not normalized:
        return ""
    mapped = SPLIT_ALIASES.get(normalized, normalized)
    return mapped if mapped in VALID_SPLITS else ""


def _split_identifier(identifier):
    """Return (numeric_id, split_name) parsed from raw identifier."""

    if identifier is None:
        return "", ""

    text = str(identifier).strip()
    if not text:
        return "", ""

    match = ID_DIGITS.match(text)
    base = match.group(1) if match else text

    split = ""
    for sep in ("_", "-"):
        if sep in text:
            candidate = text.rsplit(sep, 1)[1]
            split = _normalize_split_name(candidate)
            if split:
                break
            split = ""

    return base, split


def _id_base(identifier):
    base, _ = _split_identifier(identifier)
    return base


def _parse_annotation_sequence(value):
    """Parse comma/space separated annotator labels from a single field."""

    if value is None:
        return []

    text = str(value).strip()
    if not text:
        return []

    # Remove common brackets and quotes.
    text = text.strip("[](){}\"'")
    if not text:
        return []

    parts = [part for part in ANNOT_SPLIT.split(text) if part]
    if len(parts) <= 1:
        return []

    return [_norm_off(part) for part in parts]


def _collect_votes(row, annotator_cols=None, annotations_field=None):
    """Collect annotator votes either from dedicated columns or a single field."""

    if annotator_cols:
        votes = []
        for col in annotator_cols:
            if col not in row or row[col] is None or str(row[col]).strip() == "":
                raise ValueError(f"Missing annotator column '{col}' in row {row}")
            votes.append(_norm_off(row[col]))
        return votes

    if annotations_field:
        return _parse_annotation_sequence(row.get(annotations_field))

    return []


def load_taxonomy(tax_path):
    """Load taxonomy annotations from Category_dataset.tsv."""
    if not tax_path or not os.path.exists(tax_path):
        return {}

    delimiter = _sniff_sep(tax_path)
    taxonomy = {}
    for row in _read_dict(tax_path, sep=delimiter):
        base_id, split = _split_identifier(row.get("ID"))
        if not base_id:
            continue

        key = (base_id, split or None)
        taxonomy[key] = {
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
    ann_field=None,
    prefer_taxonomy_agr=True,
    expected_split=None,
):
    """Convert a raw MD-Agreement split into MaChAmp TSV format."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    delimiter = _sniff_sep(in_path)

    expected_split = _normalize_split_name(expected_split)

    def lookup_taxonomy(base_id, split_name):
        if not base_id:
            return {}

        key_specific = (base_id, split_name or None)
        if key_specific in taxonomy:
            return taxonomy[key_specific]

        if split_name:
            fallback_key = (base_id, None)
            return taxonomy.get(fallback_key, {})

        return {}

    with open(in_path, encoding="utf8", newline="") as in_handle, open(
        out_path, "w", encoding="utf8"
    ) as out_handle:
        reader = csv.DictReader(in_handle, delimiter=delimiter)
        for row in reader:
            tweet_id, row_split = _split_identifier(row.get(id_col))
            if expected_split:
                if row_split and row_split != expected_split:
                    continue
                row_split = row_split or expected_split

            text = (row.get(text_col) or "").replace("\n", " ").strip()

            votes = _collect_votes(row, ann_cols, ann_field)
            derived_offensive = None
            derived_agreement = None
            if votes:
                derived_offensive, derived_agreement = _majority_and_agreement(votes)

            offensive = None
            gold_agreement = None
            if gold_col and row.get(gold_col) not in (None, ""):
                gold_raw = str(row[gold_col]).strip()
                gold_votes = _parse_annotation_sequence(gold_raw)
                if gold_votes:
                    if len(gold_votes) != 5:
                        raise ValueError(
                            f"Expected 5 annotator labels in '{gold_col}', got {gold_votes}"
                        )
                    offensive, gold_agreement = _majority_and_agreement(gold_votes)
                    if not votes:
                        derived_offensive, derived_agreement = offensive, gold_agreement
                else:
                    offensive = _norm_off(gold_raw)

            if offensive is None and derived_offensive is not None:
                offensive = derived_offensive

            if offensive is None:
                raise ValueError("Need gold or annotator labels to derive OFF/NOT.")

            info = lookup_taxonomy(tweet_id, row_split)

            if prefer_taxonomy_agr and info.get("agr"):
                agreement = info["agr"]
            elif agr_col and row.get(agr_col) not in (None, ""):
                agreement = str(row[agr_col]).strip()
            elif gold_agreement is not None:
                agreement = gold_agreement
            elif derived_agreement is not None:
                agreement = derived_agreement
            else:
                agreement = ""

            agr6 = f"{agreement}_{offensive}" if agreement else ""

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
    parser.add_argument("--id_col", default="ID")
    parser.add_argument("--text_col", default="Text")
    parser.add_argument(
        "--gold_col",
        default="Offensive_binary_label",
        help="Column containing gold OFF/NOT labels (optional if deriving from annotators)",
    )
    parser.add_argument(
        "--agr_col",
        default="Agreement_level",
        help="Column containing agreement labels (A++/A+/A0), optional",
    )
    parser.add_argument(
        "--ann_cols",
        nargs="*",
        default=None,
        help="Names of five annotator columns to derive labels if needed",
    )
    parser.add_argument(
        "--ann_field",
        default="Individual_Annotations",
        help="Column containing comma-separated annotator labels (if not using --ann_cols)",
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
        args.ann_field,
        args.prefer_taxonomy_agr,
        expected_split="train",
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
        args.ann_field,
        args.prefer_taxonomy_agr,
        expected_split="dev",
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
        args.ann_field,
        args.prefer_taxonomy_agr,
        expected_split="test",
    )


if __name__ == "__main__":
    main()
