"""
SC4021 Information Retrieval - Preprocessed Data Classification & Comparison
=============================================================================
Based on classification.py, this script systematically evaluates the impact of:
1. Text representation: original_text vs normalized_text
2. Multiple label tasks: sentiment_final, sarcasm_final, subjectivity_final, emotion_final

It runs TF-IDF + multiple classifiers with 5-fold stratified CV for every
(text_col, label_col) combination and produces a consolidated comparison report.

Usage:
    # Run all experiments (default)
    python classification_preprocessed.py

    # Specify classifiers
    python classification_preprocessed.py --classifiers logistic svm nb

    # Custom TF-IDF parameters
    python classification_preprocessed.py --max_features 80000 --ngram_max 3

Author: Zhang
Date: 2026-03
"""

import os
import csv
import time
import argparse
import json
import numpy as np
from collections import Counter
from typing import List, Tuple, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

import sys
import builtins
import warnings
warnings.filterwarnings('ignore')

# Force all print() output to flush immediately (avoids buffering issues)
_original_print = builtins.print
def print(*a, **kw):
    kw.setdefault('flush', True)
    _original_print(*a, **kw)
builtins.print = print


# ============================================================================
# CLI Arguments
# ============================================================================
parser = argparse.ArgumentParser(
    description="Preprocessed Data Classification & Comparison for SC4021",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# Data
parser.add_argument('--input', type=str, default='eval_preprocessed.csv',
                    help='Input preprocessed CSV file')
parser.add_argument('--output_report', type=str, default='classification_comparison_report.json',
                    help='Path to save JSON comparison report')

# Experiment selection
parser.add_argument('--text_cols', type=str, nargs='+',
                    default=['original_text', 'normalized_text'],
                    help='Text columns to compare')
parser.add_argument('--label_cols', type=str, nargs='+',
                    default=['sentiment_final', 'sarcasm_final',
                             'subjectivity_final', 'emotion_final'],
                    help='Label columns to evaluate')

# TF-IDF parameters
parser.add_argument('--max_features', type=int, default=50000,
                    help='Max TF-IDF features')
parser.add_argument('--ngram_min', type=int, default=1,
                    help='Min n-gram range')
parser.add_argument('--ngram_max', type=int, default=2,
                    help='Max n-gram range')
parser.add_argument('--min_df', type=int, default=2,
                    help='Min document frequency for TF-IDF')
parser.add_argument('--max_df', type=float, default=0.95,
                    help='Max document frequency for TF-IDF')
parser.add_argument('--sublinear_tf', action='store_true', default=True,
                    help='Use sublinear TF scaling (log)')
parser.add_argument('--use_idf', action='store_true', default=True,
                    help='Use IDF weighting')

# Classifier selection
parser.add_argument('--classifiers', type=str, nargs='+',
                    default=['logistic', 'svm', 'nb', 'rf'],
                    choices=['logistic', 'svm', 'nb', 'rf', 'gb'],
                    help='Classifiers to evaluate')

# Cross-validation
parser.add_argument('--n_folds', type=int, default=5,
                    help='Number of CV folds')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed')

# Preprocessing
parser.add_argument('--lowercase', action='store_true', default=True,
                    help='Lowercase text before TF-IDF')
parser.add_argument('--strip_accents', type=str, default='unicode',
                    choices=['unicode', 'ascii', 'none'],
                    help='Accent stripping mode')

args = parser.parse_args()


# ============================================================================
# Label display name mappings (for readability)
# ============================================================================
LABEL_COL_DISPLAY = {
    'sentiment_final':    'Sentiment (3-class)',
    'sarcasm_final':      'Sarcasm (binary)',
    'subjectivity_final': 'Subjectivity (binary)',
    'emotion_final':      'Emotion (7-class)',
}

TEXT_COL_DISPLAY = {
    'original_text':   'Original (Singlish)',
    'normalized_text': 'Normalized (Std English)',
}

CLASSIFIER_NAMES = {
    'logistic': 'Logistic Regression',
    'svm':      'Linear SVM',
    'nb':       'Multinomial Naive Bayes',
    'rf':       'Random Forest',
    'gb':       'Gradient Boosting',
}


# ============================================================================
# Data Loading
# ============================================================================

def load_data(filepath: str, text_col: str, label_col: str
              ) -> Tuple[List[str], np.ndarray, List[str]]:
    """
    Load CSV and extract text + labels.
    Returns: (texts, labels_int, label_names)
    """
    texts = []
    raw_labels = []

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        available_cols = reader.fieldnames

        if text_col not in available_cols:
            raise ValueError(
                f"Text column '{text_col}' not found. "
                f"Available: {available_cols}"
            )
        if label_col not in available_cols:
            raise ValueError(
                f"Label column '{label_col}' not found. "
                f"Available: {available_cols}"
            )

        for row in reader:
            text = row[text_col]
            label = row[label_col]
            if text and label != '':
                texts.append(text.strip())
                raw_labels.append(str(label).strip())

    # Encode labels
    le = LabelEncoder()
    labels_int = le.fit_transform(raw_labels)
    label_names = list(le.classes_)

    return texts, labels_int, label_names


# ============================================================================
# Classifier Factory
# ============================================================================

def get_classifier(name: str, seed: int):
    """Return classifier instance by name"""
    classifiers = {
        'logistic': LogisticRegression(
            max_iter=1000, C=1.0, solver='lbfgs', random_state=seed
        ),
        'svm': LinearSVC(
            max_iter=2000, C=1.0, random_state=seed
        ),
        'nb': MultinomialNB(alpha=0.1),
        'rf': RandomForestClassifier(
            n_estimators=200, max_depth=None,
            random_state=seed, n_jobs=-1
        ),
        'gb': GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=seed
        ),
    }
    return classifiers[name]


# ============================================================================
# 5-Fold Cross Validation (single experiment)
# ============================================================================

def run_5fold(
    texts: List[str],
    labels: np.ndarray,
    label_names: List[str],
    classifier_name: str,
    verbose: bool = True
) -> Dict:
    """
    Run stratified 5-fold CV with TF-IDF + classifier.
    Returns detailed metrics dict.
    """
    clf_display = CLASSIFIER_NAMES.get(classifier_name, classifier_name)

    skf = StratifiedKFold(
        n_splits=args.n_folds, shuffle=True, random_state=args.seed
    )

    fold_results = []
    all_y_true = []
    all_y_pred = []
    total_train_time = 0
    total_predict_time = 0
    total_predict_samples = 0

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(texts, labels)):
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        y_train = labels[train_idx]
        y_test = labels[test_idx]

        # Build TF-IDF
        strip_val = None if args.strip_accents == 'none' else args.strip_accents
        vectorizer = TfidfVectorizer(
            max_features=args.max_features,
            ngram_range=(args.ngram_min, args.ngram_max),
            min_df=args.min_df,
            max_df=args.max_df,
            sublinear_tf=args.sublinear_tf,
            use_idf=args.use_idf,
            lowercase=args.lowercase,
            strip_accents=strip_val,
            token_pattern=r'(?u)\b\w+\b',
        )

        X_train = vectorizer.fit_transform(train_texts)
        X_test = vectorizer.transform(test_texts)

        # Train
        clf = get_classifier(classifier_name, args.seed)
        t0 = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - t0
        total_train_time += train_time

        # Predict
        t0 = time.time()
        y_pred = clf.predict(X_test)
        predict_time = time.time() - t0
        total_predict_time += predict_time
        total_predict_samples += len(test_idx)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='macro', zero_division=0
        )
        prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )

        fold_results.append({
            'fold': fold_idx + 1,
            'accuracy': round(acc, 4),
            'macro_precision': round(prec, 4),
            'macro_recall': round(rec, 4),
            'macro_f1': round(f1, 4),
            'weighted_precision': round(prec_w, 4),
            'weighted_recall': round(rec_w, 4),
            'weighted_f1': round(f1_w, 4),
            'train_time_sec': round(train_time, 3),
            'predict_time_sec': round(predict_time, 3),
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'vocab_size': len(vectorizer.vocabulary_),
        })

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        if verbose:
            print(f"    Fold {fold_idx+1}/{args.n_folds}: "
                  f"Acc={acc:.4f}  F1(macro)={f1:.4f}  "
                  f"F1(weighted)={f1_w:.4f}  "
                  f"[train={train_time:.2f}s, predict={predict_time:.3f}s]")

    # Aggregate metrics
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    overall_acc = accuracy_score(all_y_true, all_y_pred)
    overall_prec, overall_rec, overall_f1, _ = precision_recall_fscore_support(
        all_y_true, all_y_pred, average='macro', zero_division=0
    )
    overall_prec_w, overall_rec_w, overall_f1_w, _ = precision_recall_fscore_support(
        all_y_true, all_y_pred, average='weighted', zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred)

    # Per-class report
    report = classification_report(
        all_y_true, all_y_pred,
        target_names=label_names,
        zero_division=0,
        output_dict=True
    )

    # Speed metrics
    records_per_sec = (total_predict_samples / total_predict_time
                       if total_predict_time > 0 else 0)

    return {
        'classifier': clf_display,
        'classifier_key': classifier_name,
        'n_folds': args.n_folds,
        'fold_results': fold_results,
        'aggregate': {
            'accuracy': round(overall_acc, 4),
            'macro_precision': round(overall_prec, 4),
            'macro_recall': round(overall_rec, 4),
            'macro_f1': round(overall_f1, 4),
            'weighted_precision': round(overall_prec_w, 4),
            'weighted_recall': round(overall_rec_w, 4),
            'weighted_f1': round(overall_f1_w, 4),
        },
        'confusion_matrix': cm.tolist(),
        'per_class_report': {
            k: v for k, v in report.items()
            if k in label_names
        },
        'performance': {
            'total_train_time_sec': round(total_train_time, 3),
            'total_predict_time_sec': round(total_predict_time, 3),
            'records_per_second': round(records_per_sec, 1),
        }
    }


# ============================================================================
# Print Helpers
# ============================================================================

def print_experiment_result(result: Dict, label_names: List[str]):
    """Print detailed metrics for one experiment run"""
    agg = result['aggregate']
    perf = result['performance']
    cm = np.array(result['confusion_matrix'])

    print(f"    Accuracy:           {agg['accuracy']:.4f}")
    print(f"    Macro Precision:    {agg['macro_precision']:.4f}")
    print(f"    Macro Recall:       {agg['macro_recall']:.4f}")
    print(f"    Macro F1:           {agg['macro_f1']:.4f}")
    print(f"    Weighted F1:        {agg['weighted_f1']:.4f}")
    print(f"    Speed:              {perf['records_per_second']:.0f} records/sec")

    # Confusion matrix
    print(f"\n    Confusion Matrix:")
    header = "    " + " " * 14 + "  ".join(f"{n:>10}" for n in label_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>10}" for v in row)
        print(f"    {label_names[i]:>12}  {row_str}")

    # Per-class report
    per_class = result['per_class_report']
    print(f"\n    Per-Class Report:")
    print(f"    {'Class':>14}  {'Prec':>8}  {'Recall':>8}  {'F1':>8}  {'Support':>8}")
    for name in label_names:
        if name in per_class:
            r = per_class[name]
            print(f"    {name:>14}  {r['precision']:>8.4f}  "
                  f"{r['recall']:>8.4f}  {r['f1-score']:>8.4f}  "
                  f"{r['support']:>8.0f}")


def print_comparison_table(comparison_data: List[Dict]):
    """Print a comparison table across all experiments"""
    # Group by label_col for side-by-side text_col comparison
    from itertools import groupby

    print("\n" + "=" * 100)
    print("COMPARISON TABLE: original_text vs normalized_text")
    print("=" * 100)

    # Organize: {(label_col, classifier): {text_col: metrics}}
    grouped = {}
    for entry in comparison_data:
        key = (entry['label_col'], entry['classifier_key'])
        if key not in grouped:
            grouped[key] = {}
        grouped[key][entry['text_col']] = entry['aggregate']

    # Print per task
    seen_labels = []
    for (label_col, clf_key), text_results in sorted(grouped.items()):
        if label_col not in seen_labels:
            seen_labels.append(label_col)
            task_name = LABEL_COL_DISPLAY.get(label_col, label_col)
            print(f"\n  {'─'*90}")
            print(f"  Task: {task_name} ({label_col})")
            print(f"  {'─'*90}")
            print(f"  {'Classifier':<22} │ {'Text Type':<28} │ "
                  f"{'Acc':>7} {'M-Prec':>7} {'M-Rec':>7} {'M-F1':>7} {'W-F1':>7}")
            print(f"  {'─'*22}─┼─{'─'*28}─┼─"
                  f"{'─'*7}─{'─'*7}─{'─'*7}─{'─'*7}─{'─'*7}")

        clf_display = CLASSIFIER_NAMES.get(clf_key, clf_key)

        for text_col in args.text_cols:
            text_display = TEXT_COL_DISPLAY.get(text_col, text_col)
            if text_col in text_results:
                m = text_results[text_col]
                print(f"  {clf_display:<22} │ {text_display:<28} │ "
                      f"{m['accuracy']:>7.4f} {m['macro_precision']:>7.4f} "
                      f"{m['macro_recall']:>7.4f} {m['macro_f1']:>7.4f} "
                      f"{m['weighted_f1']:>7.4f}")


def print_delta_analysis(comparison_data: List[Dict]):
    """Print the improvement/regression from normalization for each task+classifier"""
    print("\n" + "=" * 100)
    print("DELTA ANALYSIS: Normalized vs Original (Macro-F1 difference)")
    print("=" * 100)

    # Organize by (label_col, classifier)
    grouped = {}
    for entry in comparison_data:
        key = (entry['label_col'], entry['classifier_key'])
        if key not in grouped:
            grouped[key] = {}
        grouped[key][entry['text_col']] = entry['aggregate']

    print(f"\n  {'Task':<28} {'Classifier':<22} "
          f"{'Orig F1':>9} {'Norm F1':>9} {'Delta':>9} {'Change':>9}")
    print(f"  {'─'*28} {'─'*22} {'─'*9} {'─'*9} {'─'*9} {'─'*9}")

    task_deltas = {}  # label_col -> list of deltas

    for (label_col, clf_key), text_results in sorted(grouped.items()):
        task_name = LABEL_COL_DISPLAY.get(label_col, label_col)
        clf_display = CLASSIFIER_NAMES.get(clf_key, clf_key)

        orig_f1 = text_results.get('original_text', {}).get('macro_f1', None)
        norm_f1 = text_results.get('normalized_text', {}).get('macro_f1', None)

        if orig_f1 is not None and norm_f1 is not None:
            delta = norm_f1 - orig_f1
            pct = (delta / orig_f1 * 100) if orig_f1 > 0 else 0
            sign = "+" if delta >= 0 else ""
            print(f"  {task_name:<28} {clf_display:<22} "
                  f"{orig_f1:>9.4f} {norm_f1:>9.4f} "
                  f"{sign}{delta:>8.4f} {sign}{pct:>7.1f}%")

            if label_col not in task_deltas:
                task_deltas[label_col] = []
            task_deltas[label_col].append(delta)

    # Average delta per task
    print(f"\n  {'─'*90}")
    print(f"  Average F1 change per task (Normalized - Original):")
    for label_col, deltas in task_deltas.items():
        task_name = LABEL_COL_DISPLAY.get(label_col, label_col)
        avg = np.mean(deltas)
        sign = "+" if avg >= 0 else ""
        print(f"    {task_name:<28}: {sign}{avg:.4f}")


def print_best_per_task(comparison_data: List[Dict]):
    """Print the best classifier + text combination per task"""
    print("\n" + "=" * 100)
    print("BEST CONFIGURATION PER TASK")
    print("=" * 100)

    # Group by label_col
    by_task = {}
    for entry in comparison_data:
        lc = entry['label_col']
        if lc not in by_task:
            by_task[lc] = []
        by_task[lc].append(entry)

    print(f"\n  {'Task':<28} {'Best Classifier':<22} {'Text':<28} "
          f"{'Macro-F1':>9} {'Accuracy':>9}")
    print(f"  {'─'*28} {'─'*22} {'─'*28} {'─'*9} {'─'*9}")

    for label_col in args.label_cols:
        if label_col not in by_task:
            continue
        task_name = LABEL_COL_DISPLAY.get(label_col, label_col)
        entries = by_task[label_col]
        best = max(entries, key=lambda e: e['aggregate']['macro_f1'])
        clf_display = CLASSIFIER_NAMES.get(best['classifier_key'], best['classifier_key'])
        text_display = TEXT_COL_DISPLAY.get(best['text_col'], best['text_col'])
        print(f"  {task_name:<28} {clf_display:<22} {text_display:<28} "
              f"{best['aggregate']['macro_f1']:>9.4f} "
              f"{best['aggregate']['accuracy']:>9.4f}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 100)
    print("SC4021 - Preprocessed Data Classification & Comparison")
    print("=" * 100)
    print(f"Input:        {args.input}")
    print(f"Text cols:    {args.text_cols}")
    print(f"Label cols:   {args.label_cols}")
    print(f"Folds:        {args.n_folds}")
    print(f"Classifiers:  {args.classifiers}")
    print(f"TF-IDF:       max_features={args.max_features}, "
          f"ngram=({args.ngram_min},{args.ngram_max}), "
          f"min_df={args.min_df}, max_df={args.max_df}")

    total_experiments = len(args.text_cols) * len(args.label_cols) * len(args.classifiers)
    print(f"\nTotal experiments: {total_experiments} "
          f"({len(args.text_cols)} text x {len(args.label_cols)} label x "
          f"{len(args.classifiers)} classifiers)")
    print()

    global_start_time = time.time()

    # Pre-load data for each (text_col, label_col) combination
    data_cache = {}
    for text_col in args.text_cols:
        for label_col in args.label_cols:
            key = (text_col, label_col)
            try:
                texts, labels_int, label_names = load_data(
                    args.input, text_col, label_col
                )
                labels = np.array(labels_int)
                dist = Counter(labels_int)
                data_cache[key] = {
                    'texts': texts,
                    'labels': labels,
                    'label_names': label_names,
                    'dist': dist,
                }
                print(f"  Loaded [{text_col}] x [{label_col}]: "
                      f"{len(texts)} samples, "
                      f"labels={label_names}")
            except Exception as e:
                print(f"  ERROR loading [{text_col}] x [{label_col}]: {e}")

    # ============================================================
    # Run all experiments
    # ============================================================
    comparison_data = []
    all_experiment_results = {}
    experiment_idx = 0

    for label_col in args.label_cols:
        task_name = LABEL_COL_DISPLAY.get(label_col, label_col)

        print(f"\n{'#'*100}")
        print(f"# TASK: {task_name}")
        print(f"{'#'*100}")

        for text_col in args.text_cols:
            text_display = TEXT_COL_DISPLAY.get(text_col, text_col)
            key = (text_col, label_col)

            if key not in data_cache:
                print(f"\n  [SKIP] No data for {text_col} x {label_col}")
                continue

            data = data_cache[key]
            texts = data['texts']
            labels = data['labels']
            label_names = data['label_names']
            dist = data['dist']

            print(f"\n  {'='*90}")
            print(f"  Text: {text_display} ({text_col})")
            print(f"  Samples: {len(texts)}, Labels: {label_names}")
            dist_str = ", ".join(f"{label_names[k]}={v}"
                                for k, v in sorted(dist.items()))
            print(f"  Distribution: {dist_str}")
            print(f"  {'='*90}")

            for clf_name in args.classifiers:
                experiment_idx += 1
                clf_display = CLASSIFIER_NAMES.get(clf_name, clf_name)

                elapsed = time.time() - global_start_time
                print(f"\n  [{experiment_idx}/{total_experiments}] "
                      f"{clf_display} | {text_display} | {task_name} "
                      f"(elapsed: {elapsed:.1f}s)")

                try:
                    result = run_5fold(
                        texts, labels, label_names, clf_name, verbose=True
                    )

                    # Print summary
                    print(f"\n    --- Aggregate ({args.n_folds}-Fold) ---")
                    print_experiment_result(result, label_names)

                    # Store
                    exp_key = f"{text_col}__{label_col}__{clf_name}"
                    all_experiment_results[exp_key] = result

                    comparison_entry = {
                        'text_col': text_col,
                        'label_col': label_col,
                        'classifier_key': clf_name,
                        'classifier': clf_display,
                        'label_names': label_names,
                        'aggregate': result['aggregate'],
                        'performance': result['performance'],
                    }
                    comparison_data.append(comparison_entry)

                except Exception as e:
                    import traceback
                    print(f"\n    [ERROR in {clf_display}] {type(e).__name__}: {e}")
                    traceback.print_exc()

    # ============================================================
    # Comparison outputs
    # ============================================================
    if len(comparison_data) > 0:
        print_comparison_table(comparison_data)
        print_delta_analysis(comparison_data)
        print_best_per_task(comparison_data)

    # ============================================================
    # Per-task summary tables
    # ============================================================
    print("\n" + "=" * 100)
    print("PER-TASK CLASSIFIER RANKING (by Macro-F1)")
    print("=" * 100)

    by_task = {}
    for entry in comparison_data:
        lc = entry['label_col']
        if lc not in by_task:
            by_task[lc] = []
        by_task[lc].append(entry)

    for label_col in args.label_cols:
        if label_col not in by_task:
            continue
        task_name = LABEL_COL_DISPLAY.get(label_col, label_col)
        entries = sorted(by_task[label_col],
                         key=lambda e: e['aggregate']['macro_f1'],
                         reverse=True)

        print(f"\n  Task: {task_name}")
        print(f"  {'Rank':<5} {'Classifier':<22} {'Text':<28} "
              f"{'Acc':>7} {'M-F1':>7} {'W-F1':>7} {'Speed':>10}")
        print(f"  {'─'*5} {'─'*22} {'─'*28} {'─'*7} {'─'*7} {'─'*7} {'─'*10}")

        for rank, entry in enumerate(entries, 1):
            clf_d = CLASSIFIER_NAMES.get(entry['classifier_key'], entry['classifier_key'])
            txt_d = TEXT_COL_DISPLAY.get(entry['text_col'], entry['text_col'])
            m = entry['aggregate']
            spd = entry['performance']['records_per_second']
            print(f"  {rank:<5} {clf_d:<22} {txt_d:<28} "
                  f"{m['accuracy']:>7.4f} {m['macro_f1']:>7.4f} "
                  f"{m['weighted_f1']:>7.4f} {spd:>7.0f}r/s")

    # ============================================================
    # Save Report
    # ============================================================
    report = {
        'config': {
            'input': args.input,
            'text_cols': args.text_cols,
            'label_cols': args.label_cols,
            'classifiers': args.classifiers,
            'n_folds': args.n_folds,
            'max_features': args.max_features,
            'ngram_range': [args.ngram_min, args.ngram_max],
            'min_df': args.min_df,
            'max_df': args.max_df,
            'seed': args.seed,
        },
        'data_summary': {
            f"{tc}__{lc}": {
                'num_samples': len(data_cache[(tc, lc)]['texts']),
                'label_names': data_cache[(tc, lc)]['label_names'],
                'label_distribution': {
                    data_cache[(tc, lc)]['label_names'][k]: v
                    for k, v in sorted(data_cache[(tc, lc)]['dist'].items())
                },
            }
            for tc in args.text_cols
            for lc in args.label_cols
            if (tc, lc) in data_cache
        },
        'experiments': {
            k: v for k, v in all_experiment_results.items()
        },
        'comparison': comparison_data,
        'best_per_task': {},
    }

    # Best per task
    for label_col in args.label_cols:
        if label_col not in by_task:
            continue
        entries = by_task[label_col]
        best = max(entries, key=lambda e: e['aggregate']['macro_f1'])
        report['best_per_task'][label_col] = {
            'task': LABEL_COL_DISPLAY.get(label_col, label_col),
            'best_classifier': best['classifier'],
            'best_text_col': best['text_col'],
            'macro_f1': best['aggregate']['macro_f1'],
            'accuracy': best['aggregate']['accuracy'],
        }

    total_elapsed = time.time() - global_start_time
    report_path = args.output_report
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n\nReport saved to: {report_path}")
    print(f"Total time: {total_elapsed:.1f}s")
    print("Done!")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[Interrupted by user]")
    except Exception as e:
        import traceback
        print(f"\n\n[ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
