"""
Q5 Innovations for enhancing sentiment classification.

Innovations implemented (with ablation study):
1. Hybrid classification: TF-IDF (subsymbolic) + VADER lexicon features (symbolic)
2. Enhanced classification: Sarcasm-aware sentiment correction
3. Ensemble classification: Stacked ensemble (LR + NB + RF -> LR)

Ablation study shows the incremental contribution of each innovation.
"""

import re
import time
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter

from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import clone

from .config import CLASSIFIER_NAMES
from .models import build_tfidf_vectorizer


# ============================================================================
# Feature Engineering: VADER lexicon + text statistics (Hybrid / Symbolic)
# ============================================================================

def _get_vader():
    """Lazy-load VADER to avoid import cost if not needed."""
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


def extract_hybrid_features(texts: List[str]) -> np.ndarray:
    """
    Extract symbolic/knowledge-based features:
    - VADER sentiment scores (neg, neu, pos, compound)  [lexicon-based]
    - Text statistics (length, word count, avg word length,
      exclamation/question marks, caps ratio, emoji count) [rule-based]
    Returns (n_samples, n_features) array.
    """
    sid = _get_vader()

    features = []
    for text in texts:
        # VADER scores (knowledge-based sentiment lexicon)
        vs = sid.polarity_scores(text)

        # Text statistics (rule-based features)
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        avg_word_len = np.mean([len(w) for w in words]) if words else 0

        excl_count = text.count('!')
        ques_count = text.count('?')
        caps_ratio = sum(1 for c in text if c.isupper()) / max(char_count, 1)

        # Simple emoji/emoticon pattern count
        emoji_pattern = re.compile(r'[:;][-(]?[)D(P/\\|]|[<>]3|[\U0001F600-\U0001F64F]', re.UNICODE)
        emoji_count = len(emoji_pattern.findall(text))

        # Negation word count (rule-based)
        negation_words = {'not', 'no', 'never', 'neither', 'nobody', 'nothing',
                          "n't", 'cant', 'cannot', 'wont', 'dont', 'doesnt',
                          'isnt', 'arent', 'wasnt', 'werent', 'havent', 'hasnt'}
        neg_count = sum(1 for w in words if w.lower().strip("'\".,!?") in negation_words)

        features.append([
            vs['neg'], vs['neu'], vs['pos'], vs['compound'],  # VADER (4)
            char_count, word_count, avg_word_len,               # text stats (3)
            excl_count, ques_count, caps_ratio,                 # punctuation/style (3)
            emoji_count, neg_count,                             # emoji + negation (2)
        ])

    return np.array(features, dtype=np.float64)


# ============================================================================
# Sarcasm-aware sentiment: use sarcasm signal to adjust predictions
# ============================================================================

def extract_sarcasm_features(texts: List[str]) -> np.ndarray:
    """
    Extract sarcasm-indicative features:
    - Contrast indicators (e.g., "but", "however", "although")
    - Hyperbole markers (e.g., "totally", "absolutely", "literally")
    - Quote marks (often used in sarcasm)
    - Mixed sentiment signal (positive VADER but negative keywords or vice versa)
    """
    sid = _get_vader()

    contrast_words = {'but', 'however', 'although', 'though', 'yet',
                      'actually', 'technically', 'supposedly', 'apparently'}
    hyperbole_words = {'totally', 'absolutely', 'literally', 'obviously',
                       'clearly', 'definitely', 'surely', 'completely',
                       'utterly', 'perfectly', 'amazing', 'incredible',
                       'brilliant', 'genius', 'wonderful'}
    # Singlish sarcasm markers
    sg_sarcasm = {'right', 'sure', 'wow', 'wah', 'yah', 'lor', 'meh', 'hor'}

    features = []
    for text in texts:
        words = [w.lower().strip("'\".,!?") for w in text.split()]
        vs = sid.polarity_scores(text)

        contrast_count = sum(1 for w in words if w in contrast_words)
        hyperbole_count = sum(1 for w in words if w in hyperbole_words)
        sg_sarcasm_count = sum(1 for w in words if w in sg_sarcasm)
        quote_count = text.count('"') + text.count("'") // 2
        ellipsis_count = text.count('...')

        # Mixed sentiment signal: high positive VADER but contains negative words
        # or high negative VADER but contains positive words
        sentiment_contrast = abs(vs['pos'] - vs['neg'])

        # /s sarcasm tag (Reddit convention)
        has_s_tag = 1.0 if '/s' in text.lower() else 0.0

        features.append([
            contrast_count,
            hyperbole_count,
            sg_sarcasm_count,
            quote_count,
            ellipsis_count,
            sentiment_contrast,
            has_s_tag,
        ])

    return np.array(features, dtype=np.float64)


# ============================================================================
# Core ablation experiment runner
# ============================================================================

def _run_cv_with_features(
    texts: List[str],
    labels: np.ndarray,
    label_names: List[str],
    args,
    use_hybrid: bool = False,
    use_sarcasm: bool = False,
    use_ensemble: bool = False,
    config_name: str = "Baseline",
) -> Dict:
    """
    Run 5-fold CV with optional feature augmentation and ensemble.
    """
    skf = StratifiedKFold(
        n_splits=args.n_folds, shuffle=True, random_state=args.seed
    )

    # Pre-compute extra features if needed (outside fold loop for consistency)
    hybrid_feats = extract_hybrid_features(texts) if use_hybrid else None
    sarcasm_feats = extract_sarcasm_features(texts) if use_sarcasm else None

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

        # TF-IDF base features
        vectorizer = build_tfidf_vectorizer(args)
        X_train = vectorizer.fit_transform(train_texts)
        X_test = vectorizer.transform(test_texts)

        # Augment with extra features
        if use_hybrid or use_sarcasm:
            extra_train_parts = []
            extra_test_parts = []

            if use_hybrid:
                scaler_h = MinMaxScaler()
                h_train = scaler_h.fit_transform(hybrid_feats[train_idx])
                h_test = scaler_h.transform(hybrid_feats[test_idx])
                extra_train_parts.append(csr_matrix(h_train))
                extra_test_parts.append(csr_matrix(h_test))

            if use_sarcasm:
                scaler_s = MinMaxScaler()
                s_train = scaler_s.fit_transform(sarcasm_feats[train_idx])
                s_test = scaler_s.transform(sarcasm_feats[test_idx])
                extra_train_parts.append(csr_matrix(s_train))
                extra_test_parts.append(csr_matrix(s_test))

            X_train = hstack([X_train] + extra_train_parts, format='csr')
            X_test = hstack([X_test] + extra_test_parts, format='csr')

        # Select classifier
        if use_ensemble:
            # Use non-negative features for NB compatibility
            # NB needs non-negative, so clip
            X_train_nn = X_train.copy()
            X_test_nn = X_test.copy()
            X_train_nn[X_train_nn < 0] = 0
            X_test_nn[X_test_nn < 0] = 0

            estimators = [
                ('lr', LogisticRegression(max_iter=1000, C=1.0,
                                          solver='lbfgs', random_state=args.seed)),
                ('nb', MultinomialNB(alpha=0.1)),
                ('rf', RandomForestClassifier(n_estimators=100, random_state=args.seed,
                                              n_jobs=-1)),
            ]
            clf = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(max_iter=500, random_state=args.seed),
                cv=3, n_jobs=-1,
            )
            # For stacking with NB, we need non-negative input
            X_train_use = X_train_nn
            X_test_use = X_test_nn
        else:
            clf = LinearSVC(max_iter=2000, C=1.0, random_state=args.seed)
            X_train_use = X_train
            X_test_use = X_test

        t0 = time.time()
        clf.fit(X_train_use, y_train)
        train_time = time.time() - t0
        total_train_time += train_time

        t0 = time.time()
        y_pred = clf.predict(X_test_use)
        predict_time = time.time() - t0
        total_predict_time += predict_time
        total_predict_samples += len(test_idx)

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='macro', zero_division=0
        )
        _, _, f1_w, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        print(f"    Fold {fold_idx+1}/{args.n_folds}: "
              f"Acc={acc:.4f}  F1(macro)={f1:.4f}  F1(w)={f1_w:.4f}  "
              f"[train={train_time:.2f}s]")

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    overall_acc = accuracy_score(all_y_true, all_y_pred)
    overall_prec, overall_rec, overall_f1, _ = precision_recall_fscore_support(
        all_y_true, all_y_pred, average='macro', zero_division=0
    )
    _, _, overall_f1_w, _ = precision_recall_fscore_support(
        all_y_true, all_y_pred, average='weighted', zero_division=0
    )

    cm = confusion_matrix(all_y_true, all_y_pred)
    report = classification_report(
        all_y_true, all_y_pred, target_names=label_names,
        zero_division=0, output_dict=True
    )

    records_per_sec = (total_predict_samples / total_predict_time
                       if total_predict_time > 0 else 0)

    return {
        'config_name': config_name,
        'aggregate': {
            'accuracy': round(overall_acc, 4),
            'macro_precision': round(overall_prec, 4),
            'macro_recall': round(overall_rec, 4),
            'macro_f1': round(overall_f1, 4),
            'weighted_f1': round(overall_f1_w, 4),
        },
        'confusion_matrix': cm.tolist(),
        'per_class_report': {
            k: v for k, v in report.items() if k in label_names
        },
        'performance': {
            'total_train_time_sec': round(total_train_time, 3),
            'total_predict_time_sec': round(total_predict_time, 3),
            'records_per_second': round(records_per_sec, 1),
        }
    }


# ============================================================================
# Full ablation study
# ============================================================================

def run_ablation_study(texts: List[str], labels: np.ndarray,
                       label_names: List[str], args) -> Dict:
    """
    Ablation study for Q5 innovations on sentiment classification.

    Configurations (incremental):
    A. Baseline: TF-IDF + Linear SVM
    B. + Hybrid features (VADER + text stats)         [Innovation 1: Hybrid]
    C. + Sarcasm features                             [Innovation 2: Enhanced]
    D. + Hybrid + Sarcasm features combined
    E. + Hybrid + Sarcasm + Stacked Ensemble          [Innovation 3: Ensemble]

    Returns dict of {config_name: metrics}.
    """
    print("\n" + "#" * 100)
    print("# Q5 INNOVATIONS: HYBRID + SARCASM-AWARE + ENSEMBLE (ABLATION STUDY)")
    print("#" * 100)

    ablation_configs = [
        {
            'name': 'A. Baseline (TF-IDF + SVM)',
            'use_hybrid': False, 'use_sarcasm': False, 'use_ensemble': False,
        },
        {
            'name': 'B. + Hybrid (VADER+Stats)',
            'use_hybrid': True, 'use_sarcasm': False, 'use_ensemble': False,
        },
        {
            'name': 'C. + Sarcasm Features',
            'use_hybrid': False, 'use_sarcasm': True, 'use_ensemble': False,
        },
        {
            'name': 'D. + Hybrid + Sarcasm',
            'use_hybrid': True, 'use_sarcasm': True, 'use_ensemble': False,
        },
        {
            'name': 'E. + Hybrid + Sarcasm + Ensemble',
            'use_hybrid': True, 'use_sarcasm': True, 'use_ensemble': True,
        },
    ]

    ablation_results = {}

    for config in ablation_configs:
        name = config['name']
        print(f"\n  {'='*90}")
        print(f"  [Ablation] {name}")
        print(f"  {'='*90}")

        result = _run_cv_with_features(
            texts, labels, label_names, args,
            use_hybrid=config['use_hybrid'],
            use_sarcasm=config['use_sarcasm'],
            use_ensemble=config['use_ensemble'],
            config_name=name,
        )

        agg = result['aggregate']
        ablation_results[name] = {
            'macro_f1': agg['macro_f1'],
            'accuracy': agg['accuracy'],
            'macro_precision': agg['macro_precision'],
            'macro_recall': agg['macro_recall'],
            'weighted_f1': agg['weighted_f1'],
            'records_per_second': result['performance']['records_per_second'],
            'full_result': result,
        }

    # ============================================================
    # Print ablation summary
    # ============================================================
    print("\n" + "=" * 100)
    print("ABLATION STUDY RESULTS (Sentiment 3-class)")
    print("=" * 100)
    print(f"\n  {'Configuration':<40} {'Acc':>8} {'M-Prec':>8} {'M-Rec':>8} "
          f"{'M-F1':>8} {'W-F1':>8} {'Delta':>8}")
    print(f"  {'─'*40} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    baseline_f1 = ablation_results[ablation_configs[0]['name']]['macro_f1']

    for config in ablation_configs:
        name = config['name']
        m = ablation_results[name]
        delta = m['macro_f1'] - baseline_f1
        sign = "+" if delta >= 0 else ""
        delta_str = f"{sign}{delta:.4f}" if name != ablation_configs[0]['name'] else "  base"
        print(f"  {name:<40} "
              f"{m['accuracy']:>8.4f} "
              f"{m['macro_precision']:>8.4f} "
              f"{m['macro_recall']:>8.4f} "
              f"{m['macro_f1']:>8.4f} "
              f"{m['weighted_f1']:>8.4f} "
              f"{delta_str:>8}")

    # Best config
    best_name = max(ablation_results, key=lambda k: ablation_results[k]['macro_f1'])
    best_f1 = ablation_results[best_name]['macro_f1']
    total_gain = best_f1 - baseline_f1
    print(f"\n  Best: {best_name}")
    print(f"  Baseline F1: {baseline_f1:.4f} -> Best F1: {best_f1:.4f} "
          f"(total gain: {'+' if total_gain >= 0 else ''}{total_gain:.4f})")

    return ablation_results
