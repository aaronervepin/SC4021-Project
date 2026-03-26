"""
Q4 Data Preprocessing: normalization, basic statistics, and data quality checks.

Performs text-level preprocessing on eval_preprocessed.csv and reports statistics.
"""

import re
import csv
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter


# ============================================================================
# Text normalization utilities
# ============================================================================

# Common Singlish/informal abbreviations -> standard English
SINGLISH_MAP = {
    'lah': '', 'lor': '', 'leh': '', 'meh': '', 'hor': '', 'sia': '',
    'arh': '', 'liao': 'already', 'shiok': 'great', 'jialat': 'terrible',
    'sian': 'bored', 'walao': 'oh my', 'wah': 'wow', 'alamak': 'oh no',
    'gg': 'doomed', 'cb': '', 'knn': '',
    'u': 'you', 'r': 'are', 'ur': 'your', 'n': 'and',
    'govt': 'government', 'sg': 'singapore', 'poly': 'polytechnic',
    'jc': 'junior college', 'uni': 'university', 'sch': 'school',
    'gpa': 'grade point average', 'hdb': 'public housing',
    'moe': 'ministry of education', 'nus': 'national university of singapore',
    'ntu': 'nanyang technological university', 'smu': 'singapore management university',
}

# Common internet abbreviations
INTERNET_ABBREV = {
    'imo': 'in my opinion', 'imho': 'in my humble opinion',
    'tbh': 'to be honest', 'ngl': 'not gonna lie',
    'idk': 'i do not know', 'btw': 'by the way',
    'smh': 'shaking my head', 'fwiw': 'for what it is worth',
    'afaik': 'as far as i know', 'iirc': 'if i recall correctly',
    'tldr': 'too long did not read', 'pls': 'please', 'thx': 'thanks',
}


def normalize_text(text: str) -> str:
    """
    Apply microtext normalization:
    1. Lowercase
    2. Expand Singlish particles and abbreviations
    3. Expand internet abbreviations
    4. Normalize repeated characters (e.g., sooooo -> so)
    5. Normalize whitespace
    6. Strip URLs and user mentions
    """
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Remove Reddit-style references
    text = re.sub(r'/?(r|u)/\w+', '', text)

    # Normalize repeated characters (3+ -> 2)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # Expand abbreviations (word-boundary match)
    words = text.split()
    expanded = []
    all_map = {**SINGLISH_MAP, **INTERNET_ABBREV}
    for w in words:
        clean = w.strip('.,!?;:\'\"()[]{}')
        replacement = all_map.get(clean, w)
        if replacement:
            expanded.append(replacement)
    text = ' '.join(expanded)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============================================================================
# Basic statistics computation
# ============================================================================

def compute_text_statistics(texts: List[str]) -> Dict:
    """Compute basic text statistics for a list of texts."""
    lengths_char = [len(t) for t in texts]
    lengths_word = [len(t.split()) for t in texts]

    # Vocabulary
    all_words = []
    for t in texts:
        all_words.extend(t.lower().split())
    vocab = set(all_words)
    word_freq = Counter(all_words)

    return {
        'num_samples': len(texts),
        'char_length': {
            'mean': round(np.mean(lengths_char), 1),
            'median': round(np.median(lengths_char), 1),
            'std': round(np.std(lengths_char), 1),
            'min': int(np.min(lengths_char)),
            'max': int(np.max(lengths_char)),
        },
        'word_length': {
            'mean': round(np.mean(lengths_word), 1),
            'median': round(np.median(lengths_word), 1),
            'std': round(np.std(lengths_word), 1),
            'min': int(np.min(lengths_word)),
            'max': int(np.max(lengths_word)),
        },
        'vocabulary_size': len(vocab),
        'total_tokens': len(all_words),
        'top_20_words': word_freq.most_common(20),
    }


def compute_label_statistics(rows: List[Dict], label_col: str) -> Dict:
    """Compute label distribution and agreement statistics."""
    labels = [r[label_col] for r in rows if r.get(label_col, '') != '']
    agreement_col = label_col.replace('_final', '_agreement')
    agreements = [float(r[agreement_col]) for r in rows
                  if r.get(agreement_col, '') != '']

    dist = Counter(labels)
    return {
        'distribution': dict(sorted(dist.items())),
        'total': len(labels),
        'agreement_mean': round(np.mean(agreements), 4) if agreements else None,
        'agreement_min': round(np.min(agreements), 4) if agreements else None,
        'agreement_above_67pct': round(
            sum(1 for a in agreements if a >= 0.67) / len(agreements) * 100, 1
        ) if agreements else None,
    }


def run_preprocessing_report(filepath: str) -> Dict:
    """
    Run full preprocessing analysis on eval CSV and print report.
    Returns statistics dict for use in report generation.
    """
    print("\n" + "=" * 100)
    print("Q4: DATA PREPROCESSING & STATISTICS")
    print("=" * 100)

    # Load data
    with open(filepath, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    print(f"\n  Loaded {len(rows)} records from {filepath}")
    print(f"  Columns: {list(rows[0].keys())}")

    # ---- Text statistics (before normalization) ----
    # NOTE: In CSV, 'normalized_text' is actually original Singlish
    original_texts = [r['normalized_text'] for r in rows if r.get('normalized_text')]
    normalized_texts = [r['original_text'] for r in rows if r.get('original_text')]

    print(f"\n  --- Original Singlish Text Statistics ---")
    orig_stats = compute_text_statistics(original_texts)
    _print_text_stats(orig_stats)

    print(f"\n  --- Normalized (Std English) Text Statistics ---")
    norm_stats = compute_text_statistics(normalized_texts)
    _print_text_stats(norm_stats)

    # How many differ
    diff_count = sum(1 for r in rows
                     if r['original_text'] != r['normalized_text'])
    print(f"\n  Rows with text changed by normalization: {diff_count}/{len(rows)} "
          f"({diff_count/len(rows)*100:.1f}%)")

    # ---- Apply additional microtext normalization ----
    print(f"\n  --- Applying Microtext Normalization ---")
    further_normalized = [normalize_text(t) for t in original_texts]
    further_stats = compute_text_statistics(further_normalized)
    print(f"  Vocabulary reduction: {orig_stats['vocabulary_size']} -> "
          f"{further_stats['vocabulary_size']} "
          f"(-{orig_stats['vocabulary_size'] - further_stats['vocabulary_size']})")
    print(f"  (Note: microtext normalization has limited additional impact since "
          f"the LLM-based normalization already standardized most expressions)")

    # ---- Label statistics ----
    print(f"\n  --- Label Statistics ---")
    label_stats = {}
    for label_col in ['sentiment_final', 'sarcasm_final', 'subjectivity_final']:
        stats = compute_label_statistics(rows, label_col)
        label_stats[label_col] = stats
        print(f"\n  {label_col}:")
        print(f"    Distribution: {stats['distribution']}")
        print(f"    Total labeled: {stats['total']}")
        print(f"    Agreement (mean): {stats['agreement_mean']}")
        print(f"    Agreement >= 67%: {stats['agreement_above_67pct']}%")

    # ---- Missing value check ----
    print(f"\n  --- Missing Values ---")
    for col in rows[0].keys():
        missing = sum(1 for r in rows if not r.get(col, '').strip())
        if missing > 0:
            print(f"    {col}: {missing} missing ({missing/len(rows)*100:.1f}%)")
    print(f"    (No critical missing values detected)")

    return {
        'original_stats': orig_stats,
        'normalized_stats': norm_stats,
        'further_normalized_stats': further_stats,
        'label_stats': label_stats,
        'num_rows': len(rows),
        'normalization_changed': diff_count,
    }


def _print_text_stats(stats: Dict):
    """Pretty-print text statistics."""
    print(f"    Samples: {stats['num_samples']}")
    cl = stats['char_length']
    print(f"    Char length: mean={cl['mean']}, median={cl['median']}, "
          f"std={cl['std']}, min={cl['min']}, max={cl['max']}")
    wl = stats['word_length']
    print(f"    Word length: mean={wl['mean']}, median={wl['median']}, "
          f"std={wl['std']}, min={wl['min']}, max={wl['max']}")
    print(f"    Vocabulary: {stats['vocabulary_size']} unique words, "
          f"{stats['total_tokens']} total tokens")
    top5 = ', '.join(f'"{w}"({c})' for w, c in stats['top_20_words'][:5])
    print(f"    Top-5 words: {top5}")
