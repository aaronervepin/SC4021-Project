"""
SC4021 Information Retrieval - 5-Fold TF-IDF Classification
============================================================
Features:
1. Configurable text column and label column (supports original/normalized/subtask)
2. TF-IDF + multiple classifiers (Logistic Regression, SVM, Naive Bayes, Random Forest)
3. Stratified 5-fold cross-validation
4. Per-fold and aggregate metrics: Precision, Recall, F1, Accuracy
5. Confusion matrix, per-class report, classification speed benchmarks
6. Optional n-gram range, max features, preprocessing toggles

Usage examples:
    # Original text + original label (3-class sentiment)
    python classification_5fold.py --input eval.csv --text_col text --label_col label

    # Preprocessed: normalized text + majority-voted sentiment
    python classification_5fold.py --input eval_preprocessed.csv \\
        --text_col normalized_text --label_col sentiment_final

    # Subtask: original text + sarcasm label (binary)
    python classification_5fold.py --input eval_preprocessed.csv \\
        --text_col original_text --label_col sarcasm_final

    # Subtask: normalized text + emotion label (multi-class)
    python classification_5fold.py --input eval_preprocessed.csv \\
        --text_col normalized_text --label_col emotion_final

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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CLI Arguments
# ============================================================================
parser = argparse.ArgumentParser(
    description="5-Fold TF-IDF Classification for SC4021 eval dataset",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# Data
parser.add_argument('--input', type=str, default='eval.csv',
                    help='Input CSV file')
parser.add_argument('--text_col', type=str, default='text',
                    help='Column name for text input')
parser.add_argument('--label_col', type=str, default='label',
                    help='Column name for label')
parser.add_argument('--output_report', type=str, default=None,
                    help='Path to save JSON report (default: auto-named)')

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
# Data Loading
# ============================================================================

def load_data(filepath: str, text_col: str, label_col: str
              ) -> Tuple[List[str], List[int], List[str]]:
    """
    Load CSV and extract text + labels.
    Returns: (texts, labels_int, label_names)
    """
    texts = []
    raw_labels = []

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        available_cols = reader.fieldnames
        print(f"Available columns: {available_cols}")

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

def get_classifier(name: str):
    """Return classifier instance by name"""
    classifiers = {
        'logistic': LogisticRegression(
            max_iter=1000, C=1.0, solver='lbfgs',
            random_state=args.seed
        ),
        'svm': LinearSVC(
            max_iter=2000, C=1.0, random_state=args.seed
        ),
        'nb': MultinomialNB(alpha=0.1),
        'rf': RandomForestClassifier(
            n_estimators=200, max_depth=None,
            random_state=args.seed, n_jobs=-1
        ),
        'gb': GradientBoostingClassifier(
            n_estimators=100, max_depth=5,
            random_state=args.seed
        ),
    }
    return classifiers[name]


CLASSIFIER_NAMES = {
    'logistic': 'Logistic Regression',
    'svm': 'Linear SVM',
    'nb': 'Multinomial Naive Bayes',
    'rf': 'Random Forest',
    'gb': 'Gradient Boosting',
}


# ============================================================================
# 5-Fold Cross Validation
# ============================================================================

def run_5fold(
    texts: List[str],
    labels: np.ndarray,
    label_names: List[str],
    classifier_name: str
) -> Dict:
    """
    Run stratified 5-fold CV with TF-IDF + classifier.
    Returns detailed metrics dict.
    """
    clf_display = CLASSIFIER_NAMES.get(classifier_name, classifier_name)
    print(f"\n{'='*60}")
    print(f"Classifier: {clf_display}")
    print(f"{'='*60}")

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
        clf = get_classifier(classifier_name)
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

        print(f"  Fold {fold_idx+1}/{args.n_folds}: "
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

    print(f"\n  --- Aggregate ({args.n_folds}-Fold) ---")
    print(f"  Accuracy:           {overall_acc:.4f}")
    print(f"  Macro Precision:    {overall_prec:.4f}")
    print(f"  Macro Recall:       {overall_rec:.4f}")
    print(f"  Macro F1:           {overall_f1:.4f}")
    print(f"  Weighted F1:        {overall_f1_w:.4f}")
    print(f"  Speed:              {records_per_sec:.0f} records/sec")
    print(f"\n  Confusion Matrix:")
    # Print header
    header = "  " + " " * 12 + "  ".join(f"{n:>8}" for n in label_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>8}" for v in row)
        print(f"  {label_names[i]:>10}  {row_str}")

    print(f"\n  Per-Class Report:")
    print(f"  {'Class':>12}  {'Prec':>8}  {'Recall':>8}  {'F1':>8}  {'Support':>8}")
    for name in label_names:
        if name in report:
            r = report[name]
            print(f"  {name:>12}  {r['precision']:>8.4f}  "
                  f"{r['recall']:>8.4f}  {r['f1-score']:>8.4f}  "
                  f"{r['support']:>8.0f}")

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
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("SC4021 - 5-Fold TF-IDF Classification")
    print("=" * 70)
    print(f"Input:       {args.input}")
    print(f"Text col:    {args.text_col}")
    print(f"Label col:   {args.label_col}")
    print(f"Folds:       {args.n_folds}")
    print(f"Classifiers: {args.classifiers}")
    print(f"TF-IDF:      max_features={args.max_features}, "
          f"ngram=({args.ngram_min},{args.ngram_max}), "
          f"min_df={args.min_df}, max_df={args.max_df}")
    print()

    # Load data
    texts, labels, label_names = load_data(
        args.input, args.text_col, args.label_col
    )
    print(f"\nLoaded: {len(texts)} samples")
    print(f"Labels: {label_names}")
    dist = Counter(labels)
    for lbl_idx, name in enumerate(label_names):
        print(f"  {name}: {dist.get(lbl_idx, 0)}")
    labels = np.array(labels)

    # Run each classifier
    all_results = {}
    for clf_name in args.classifiers:
        result = run_5fold(texts, labels, label_names, clf_name)
        all_results[clf_name] = result

    # ============================================================
    # Summary Table
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY - All Classifiers")
    print("=" * 70)
    print(f"{'Classifier':<25} {'Accuracy':>9} {'Macro-F1':>9} "
          f"{'W-F1':>9} {'Speed':>12}")
    print("-" * 70)
    best_f1 = -1
    best_clf = ""
    for clf_name in args.classifiers:
        r = all_results[clf_name]
        agg = r['aggregate']
        spd = r['performance']['records_per_second']
        print(f"{r['classifier']:<25} {agg['accuracy']:>9.4f} "
              f"{agg['macro_f1']:>9.4f} {agg['weighted_f1']:>9.4f} "
              f"{spd:>9.0f} rec/s")
        if agg['macro_f1'] > best_f1:
            best_f1 = agg['macro_f1']
            best_clf = r['classifier']

    print(f"\nBest: {best_clf} (Macro-F1 = {best_f1:.4f})")

    # ============================================================
    # Save Report
    # ============================================================
    report_path = args.output_report
    if report_path is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        report_path = f"classification_report_{base}_{args.text_col}_{args.label_col}.json"

    report = {
        'config': {
            'input': args.input,
            'text_col': args.text_col,
            'label_col': args.label_col,
            'n_folds': args.n_folds,
            'max_features': args.max_features,
            'ngram_range': [args.ngram_min, args.ngram_max],
            'min_df': args.min_df,
            'max_df': args.max_df,
            'seed': args.seed,
            'num_samples': len(texts),
            'label_names': label_names,
            'label_distribution': {
                label_names[k]: v for k, v in sorted(dist.items())
            },
        },
        'results': all_results,
        'best_classifier': best_clf,
        'best_macro_f1': best_f1,
    }

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved to: {report_path}")
    print("Done!")


if __name__ == '__main__':
    main()