"""
Cross-validation evaluation and metrics computation.
"""

import time
import numpy as np
from typing import List, Dict

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

from .config import CLASSIFIER_NAMES
from .models import get_classifier, build_tfidf_vectorizer


def run_5fold(
    texts: List[str],
    labels: np.ndarray,
    label_names: List[str],
    classifier_name: str,
    args,
    clf_instance=None,
    verbose: bool = True
) -> Dict:
    """
    Run stratified 5-fold CV with TF-IDF + classifier.
    If clf_instance is provided, use it directly instead of factory.
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

        vectorizer = build_tfidf_vectorizer(args)
        X_train = vectorizer.fit_transform(train_texts)
        X_test = vectorizer.transform(test_texts)

        if clf_instance is not None:
            from sklearn.base import clone
            clf = clone(clf_instance)
        else:
            clf = get_classifier(classifier_name, args.seed)

        t0 = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - t0
        total_train_time += train_time

        t0 = time.time()
        y_pred = clf.predict(X_test)
        predict_time = time.time() - t0
        total_predict_time += predict_time
        total_predict_samples += len(test_idx)

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

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    overall_acc = accuracy_score(all_y_true, all_y_pred)
    overall_prec, overall_rec, overall_f1, _ = precision_recall_fscore_support(
        all_y_true, all_y_pred, average='macro', zero_division=0
    )
    overall_prec_w, overall_rec_w, overall_f1_w, _ = precision_recall_fscore_support(
        all_y_true, all_y_pred, average='weighted', zero_division=0
    )

    cm = confusion_matrix(all_y_true, all_y_pred)

    report = classification_report(
        all_y_true, all_y_pred,
        target_names=label_names,
        zero_division=0,
        output_dict=True
    )

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
