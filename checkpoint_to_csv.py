"""
SC4021 - Checkpoint JSON → CSV Converter
=========================================
Reads the checkpoint JSON saved by ir_label_enrichment.py and converts
it into a proper CSV, merging back with the original data.

Two modes:
  1. --merge (default): Join labels back onto the original CSV
  2. --labels-only: Output only the label columns (no original data)

Usage:
    python checkpoint_to_csv.py --checkpoint ir_enrich_checkpoint.json --original crawled_clean_with_predictions.csv --output enriched.csv
    python checkpoint_to_csv.py --checkpoint ir_enrich_checkpoint.json --labels-only --output labels_only.csv

Author: Zhang
Date: 2026-04
"""

import os
import csv
import json
import argparse
from collections import Counter
from typing import Dict, List, Tuple


# ============================================================================
# CLI
# ============================================================================
parser = argparse.ArgumentParser(
    description="Convert ir_enrich_checkpoint.json → CSV",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--checkpoint', type=str, default='ir_enrich_checkpoint.json',
                    help='Checkpoint JSON file from ir_label_enrichment.py')
parser.add_argument('--original', type=str, default='crawled_clean_with_predictions.csv',
                    help='Original input CSV (used in --merge mode)')
parser.add_argument('--output', type=str, default='crawled_enriched.csv',
                    help='Output CSV file')
parser.add_argument('--labels-only', action='store_true',
                    help='Output only label columns (no original CSV merge)')
parser.add_argument('--include-per-model', action='store_true',
                    help='Include per-model raw labels (_a / _b columns)')
parser.add_argument('--stats', action='store_true',
                    help='Print label distribution statistics after conversion')

args = parser.parse_args()


# ============================================================================
# Defaults (must match ir_label_enrichment.py)
# ============================================================================
VALID_LABELS = {
    "subjectivity": ["objective", "subjective"],
    "sarcasm": ["sarcastic", "not_sarcastic"],
    "education_level": [
        "psle", "nlevel", "olevel", "alevel", "jc", "poly", "ite",
        "uni", "masters", "general"
    ],
    "subject": [
        "english", "math", "physics", "chemistry", "biology",
        "history", "geography", "economics", "gp", "chinese",
        "literature", "computing", "art", "music", "poa", "none"
    ],
    "intent": [
        "advice_seeking", "advice_giving", "rant", "discussion",
        "resource_sharing", "experience_sharing", "question",
        "celebration", "commiseration", "comparison"
    ],
    "topic": [
        "exam_prep", "results", "school_choice", "study_tips",
        "mental_health", "career", "daily_life", "admissions",
        "academic_policy", "extracurricular", "other"
    ],
    "emotion": [
        "neutral", "anxiety", "joy", "sadness", "anger",
        "frustration", "hope", "fear", "disgust", "surprise"
    ],
    "specificity": ["high", "medium", "low"],
    "temporal_context": [
        "before_exam", "during_exam", "after_exam", "results_day",
        "enrollment_period", "semester", "holiday", "general"
    ],
}

DEFAULTS = {
    "subjectivity": "subjective",
    "sarcasm": "not_sarcastic",
    "education_level": "general",
    "subject": "none",
    "intent": "discussion",
    "topic": "other",
    "emotion": "neutral",
    "school_mentioned": "none",
    "specificity": "medium",
    "temporal_context": "general",
    "label_agreement": 0.0,
}

FINAL_LABEL_FIELDS = [
    "subjectivity", "sarcasm", "education_level", "subject",
    "intent", "topic", "emotion", "school_mentioned",
    "specificity", "temporal_context", "label_agreement",
]

PER_MODEL_FIELDS = []
for _f in list(VALID_LABELS.keys()) + ["school_mentioned"]:
    PER_MODEL_FIELDS.append(f"{_f}_a")
    PER_MODEL_FIELDS.append(f"{_f}_b")


# ============================================================================
# Load checkpoint
# ============================================================================
def load_checkpoint(path: str) -> Tuple[List[int], Dict[str, dict]]:
    """Load checkpoint JSON, return (processed_indices, results_dict)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed = data.get('processed_indices', [])
    results = data.get('results', {})
    timestamp = data.get('timestamp', 'unknown')

    print(f"Checkpoint loaded: {path}")
    print(f"  Timestamp:  {timestamp}")
    print(f"  Processed:  {len(processed)} rows")
    print(f"  Result keys: {len(results)}")

    return processed, results


# ============================================================================
# Load original CSV
# ============================================================================
def load_original_csv(path: str) -> Tuple[List[str], List[dict]]:
    """Load original CSV, return (fieldnames, rows)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Original CSV not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    print(f"Original CSV loaded: {path}")
    print(f"  Columns: {fieldnames}")
    print(f"  Rows:    {len(rows)}")

    return fieldnames, rows


# ============================================================================
# Build output
# ============================================================================
def build_label_row(result: dict, include_per_model: bool) -> dict:
    """Extract a clean label row from a checkpoint result entry."""
    row = {}
    for field in FINAL_LABEL_FIELDS:
        row[field] = result.get(field, DEFAULTS.get(field, ""))
    if include_per_model:
        for field in PER_MODEL_FIELDS:
            row[field] = result.get(field, "")
    return row


def build_default_label_row(include_per_model: bool) -> dict:
    """Build a row with all defaults (for unprocessed rows)."""
    row = {}
    for field in FINAL_LABEL_FIELDS:
        row[field] = DEFAULTS.get(field, "")
    if include_per_model:
        for field in PER_MODEL_FIELDS:
            row[field] = ""
    return row


# ============================================================================
# Statistics
# ============================================================================
def print_statistics(results: Dict[str, dict]):
    """Print label distribution statistics."""
    n = len(results)
    if n == 0:
        print("No results to analyze.")
        return

    print("\n" + "=" * 60)
    print(f"Label Distribution Statistics  (n={n})")
    print("=" * 60)

    # Agreement
    agreements = [
        v['label_agreement'] for v in results.values()
        if isinstance(v.get('label_agreement'), (int, float))
    ]
    if agreements:
        avg = sum(agreements) / len(agreements)
        full = sum(1 for a in agreements if a >= 0.99)
        print(f"\nInter-model agreement:")
        print(f"  Average:      {avg:.2%}")
        print(f"  Full (10/10): {full}/{len(agreements)} ({full/len(agreements):.1%})")

    # Per-field distribution
    for field in VALID_LABELS:
        dist = Counter(v.get(field) for v in results.values())
        print(f"\n--- {field} ---")
        for label, count in dist.most_common():
            bar = "█" * int(count / n * 40)
            print(f"  {label:25s} {count:6d}  ({count/n*100:5.1f}%)  {bar}")

    # School mentions
    all_schools = []
    for v in results.values():
        schools = v.get('school_mentioned', 'none')
        if schools and schools.lower() != 'none':
            for s in schools.split(','):
                s = s.strip()
                if s:
                    all_schools.append(s)
    if all_schools:
        school_dist = Counter(all_schools)
        print(f"\n--- school_mentioned (top 20) ---")
        for school, count in school_dist.most_common(20):
            print(f"  {school:25s} {count:6d}")
    else:
        print(f"\n--- school_mentioned ---")
        print("  (no schools extracted yet)")


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print("Checkpoint JSON → CSV Converter")
    print("=" * 60)

    # Load checkpoint
    processed_indices, results = load_checkpoint(args.checkpoint)

    # Determine new columns
    label_cols = list(FINAL_LABEL_FIELDS)
    if args.include_per_model:
        label_cols += PER_MODEL_FIELDS

    if args.labels_only:
        # ---- Labels-only mode ----
        print(f"\nMode: labels-only")
        fieldnames = ["row_index"] + label_cols

        with open(args.output, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for idx_str in sorted(results.keys(), key=lambda x: int(x)):
                row = {"row_index": idx_str}
                row.update(build_label_row(results[idx_str], args.include_per_model))
                writer.writerow(row)

        print(f"\nWrote {len(results)} rows to {args.output}")

    else:
        # ---- Merge mode ----
        print(f"\nMode: merge with original CSV")
        orig_fieldnames, orig_rows = load_original_csv(args.original)
        total = len(orig_rows)

        fieldnames = orig_fieldnames + label_cols

        processed_count = 0
        default_count = 0

        with open(args.output, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(total):
                row = dict(orig_rows[i])
                key = str(i)
                if key in results:
                    row.update(build_label_row(results[key], args.include_per_model))
                    processed_count += 1
                else:
                    row.update(build_default_label_row(args.include_per_model))
                    default_count += 1
                writer.writerow(row)

        print(f"\nWrote {total} rows to {args.output}")
        print(f"  With labels:    {processed_count}")
        print(f"  With defaults:  {default_count}")

    # Stats
    if args.stats:
        print_statistics(results)

    print("\nDone!")


if __name__ == '__main__':
    main()
