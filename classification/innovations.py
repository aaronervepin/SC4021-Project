"""
Q5 Innovations for enhancing sentiment classification.

Ablation study with progressive innovations:
A. Baseline:    BoW (CountVectorizer) + Multinomial Naive Bayes  (most primitive)
B. Innovation 1: TF-IDF + Linear SVM                            (improved representation + classifier)
C. Innovation 2: + NLP Preprocessing (lemmatization + stopword removal + POS features)
D. Innovation 3: + Hybrid features (VADER + stats)              (symbolic + subsymbolic)
E. Innovation 4: + Sarcasm features                             (enhanced classification)
F. Innovation 5: Full features + Stacked Ensemble               (ensemble classification)

Shows the incremental contribution of each innovation via ablation.
"""

import re
import time
import numpy as np
from typing import List, Dict
from collections import Counter

from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import MinMaxScaler

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
        vs = sid.polarity_scores(text)
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        avg_word_len = np.mean([len(w) for w in words]) if words else 0

        excl_count = text.count('!')
        ques_count = text.count('?')
        caps_ratio = sum(1 for c in text if c.isupper()) / max(char_count, 1)

        emoji_pattern = re.compile(
            r'[:;][-(]?[)D(P/\\|]|[<>]3|[\U0001F600-\U0001F64F]', re.UNICODE
        )
        emoji_count = len(emoji_pattern.findall(text))

        negation_words = {'not', 'no', 'never', 'neither', 'nobody', 'nothing',
                          "n't", 'cant', 'cannot', 'wont', 'dont', 'doesnt',
                          'isnt', 'arent', 'wasnt', 'werent', 'havent', 'hasnt'}
        neg_count = sum(1 for w in words if w.lower().strip("'\".,!?") in negation_words)

        features.append([
            vs['neg'], vs['neu'], vs['pos'], vs['compound'],
            char_count, word_count, avg_word_len,
            excl_count, ques_count, caps_ratio,
            emoji_count, neg_count,
        ])

    return np.array(features, dtype=np.float64)


# ============================================================================
# Sarcasm-aware features (Enhanced classification)
# ============================================================================

def extract_sarcasm_features(texts: List[str]) -> np.ndarray:
    """
    Extract sarcasm-indicative features:
    - Contrast indicators, hyperbole markers, Singlish sarcasm markers
    - Punctuation patterns, sentiment contrast, Reddit /s tag
    """
    sid = _get_vader()

    contrast_words = {'but', 'however', 'although', 'though', 'yet',
                      'actually', 'technically', 'supposedly', 'apparently'}
    hyperbole_words = {'totally', 'absolutely', 'literally', 'obviously',
                       'clearly', 'definitely', 'surely', 'completely',
                       'utterly', 'perfectly', 'amazing', 'incredible',
                       'brilliant', 'genius', 'wonderful'}
    sg_sarcasm = {'right', 'sure', 'wow', 'wah', 'yah', 'lor', 'meh', 'hor'}

    features = []
    for text in texts:
        words = [w.lower().strip("'\".,!?") for w in text.split()]
        vs = sid.polarity_scores(text)

        features.append([
            sum(1 for w in words if w in contrast_words),
            sum(1 for w in words if w in hyperbole_words),
            sum(1 for w in words if w in sg_sarcasm),
            text.count('"') + text.count("'") // 2,
            text.count('...'),
            abs(vs['pos'] - vs['neg']),
            1.0 if '/s' in text.lower() else 0.0,
        ])

    return np.array(features, dtype=np.float64)


# ============================================================================
# NLP Preprocessing: Lemmatization + Stopword Removal
# ============================================================================

def _get_nlp_tools():
    """Lazy-load NLTK NLP tools."""
    import nltk
    for res in ['wordnet', 'averaged_perceptron_tagger_eng', 'stopwords', 'punkt_tab']:
        nltk.download(res, quiet=True)
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    return WordNetLemmatizer(), set(stopwords.words('english'))


def preprocess_texts_nlp(texts: List[str]) -> List[str]:
    """
    Apply NLP-based preprocessing:
    1. Tokenize
    2. Remove stopwords
    3. Lemmatize remaining words
    Returns cleaned text list.
    """
    import nltk
    lemmatizer, stop_words = _get_nlp_tools()

    processed = []
    for text in texts:
        tokens = nltk.word_tokenize(text.lower())
        # Remove stopwords and non-alphabetic tokens
        tokens = [lemmatizer.lemmatize(w) for w in tokens
                  if w.isalpha() and w not in stop_words]
        processed.append(' '.join(tokens))
    return processed


# ============================================================================
# POS-tag based features (linguistic structure features)
# ============================================================================

def extract_pos_features(texts: List[str]) -> np.ndarray:
    """
    Extract POS-tag distribution features:
    - Adjective ratio (JJ/JJR/JJS): sentiment-bearing words
    - Adverb ratio (RB/RBR/RBS): intensity modifiers
    - Verb ratio (VB*): action/state indicators
    - Noun ratio (NN*): topic/entity words
    - Pronoun ratio (PRP*): subjectivity indicator (1st person = opinionated)
    - Interjection count (UH): emotional exclamations
    """
    import nltk

    features = []
    for text in texts:
        tokens = nltk.word_tokenize(text)
        if not tokens:
            features.append([0.0] * 6)
            continue

        tagged = nltk.pos_tag(tokens)
        n = len(tagged)

        adj_count = sum(1 for _, t in tagged if t.startswith('JJ'))
        adv_count = sum(1 for _, t in tagged if t.startswith('RB'))
        verb_count = sum(1 for _, t in tagged if t.startswith('VB'))
        noun_count = sum(1 for _, t in tagged if t.startswith('NN'))
        pron_count = sum(1 for _, t in tagged if t.startswith('PRP'))
        intj_count = sum(1 for _, t in tagged if t == 'UH')

        features.append([
            adj_count / n,
            adv_count / n,
            verb_count / n,
            noun_count / n,
            pron_count / n,
            intj_count,
        ])

    return np.array(features, dtype=np.float64)


# ============================================================================
# Core CV runner supporting all configurations
# ============================================================================

def _run_cv_experiment(
    texts: List[str],
    labels: np.ndarray,
    label_names: List[str],
    args,
    vectorizer_type: str = 'tfidf',   # 'bow' or 'tfidf'
    classifier_type: str = 'svm',      # 'nb' or 'svm'
    use_nlp_preprocess: bool = False,
    use_pos_features: bool = False,
    use_hybrid: bool = False,
    use_sarcasm: bool = False,
    use_ensemble: bool = False,
    config_name: str = "Baseline",
) -> Dict:
    """
    Run 5-fold CV with configurable vectorizer, classifier, and feature augmentation.
    """
    skf = StratifiedKFold(
        n_splits=args.n_folds, shuffle=True, random_state=args.seed
    )

    # Apply NLP preprocessing if requested (lemmatization + stopword removal)
    texts_for_vectorizer = texts
    if use_nlp_preprocess:
        print(f"    Applying NLP preprocessing (lemmatization + stopword removal)...")
        texts_for_vectorizer = preprocess_texts_nlp(texts)

    # Pre-compute extra features
    hybrid_feats = extract_hybrid_features(texts) if use_hybrid else None
    sarcasm_feats = extract_sarcasm_features(texts) if use_sarcasm else None
    pos_feats = extract_pos_features(texts) if use_pos_features else None

    all_y_true = []
    all_y_pred = []
    total_train_time = 0
    total_predict_time = 0
    total_predict_samples = 0

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(texts, labels)):
        train_texts = [texts_for_vectorizer[i] for i in train_idx]
        test_texts = [texts_for_vectorizer[i] for i in test_idx]
        y_train = labels[train_idx]
        y_test = labels[test_idx]

        # Build vectorizer
        if vectorizer_type == 'bow':
            vectorizer = CountVectorizer(
                max_features=args.max_features,
                ngram_range=(1, 1),  # unigrams only for primitive baseline
                min_df=args.min_df,
                max_df=args.max_df,
                lowercase=True,
                token_pattern=r'(?u)\b\w+\b',
            )
        else:
            vectorizer = build_tfidf_vectorizer(args)

        X_train = vectorizer.fit_transform(train_texts)
        X_test = vectorizer.transform(test_texts)

        # Augment with extra features
        if use_hybrid or use_sarcasm or use_pos_features:
            extra_train_parts = []
            extra_test_parts = []

            if use_hybrid:
                scaler_h = MinMaxScaler()
                h_train = scaler_h.fit_transform(hybrid_feats[train_idx])
                h_test = scaler_h.transform(hybrid_feats[test_idx])
                extra_train_parts.append(csr_matrix(h_train))
                extra_test_parts.append(csr_matrix(h_test))

            if use_pos_features:
                scaler_p = MinMaxScaler()
                p_train = scaler_p.fit_transform(pos_feats[train_idx])
                p_test = scaler_p.transform(pos_feats[test_idx])
                extra_train_parts.append(csr_matrix(p_train))
                extra_test_parts.append(csr_matrix(p_test))

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
            X_train_nn = X_train.copy()
            X_test_nn = X_test.copy()
            X_train_nn[X_train_nn < 0] = 0
            X_test_nn[X_test_nn < 0] = 0

            estimators = [
                ('lr', LogisticRegression(max_iter=1000, C=1.0,
                                          solver='lbfgs', random_state=args.seed,
                                          class_weight='balanced')),
                ('nb', MultinomialNB(alpha=0.1)),
                ('svm', LinearSVC(max_iter=2000, C=1.0, random_state=args.seed,
                                  class_weight='balanced')),
                ('rf', RandomForestClassifier(n_estimators=100, random_state=args.seed,
                                              n_jobs=-1)),
            ]
            clf = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(max_iter=500, random_state=args.seed,
                                                   class_weight='balanced'),
                cv=3, n_jobs=-1,
            )
            X_train_use, X_test_use = X_train_nn, X_test_nn
        elif classifier_type == 'nb':
            clf = MultinomialNB(alpha=0.1)
            X_train_use, X_test_use = X_train, X_test
        else:
            clf = LinearSVC(max_iter=2000, C=1.0, random_state=args.seed,
                            class_weight='balanced')
            X_train_use, X_test_use = X_train, X_test

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

    Progressive configurations:
    A. Baseline:      BoW + Naive Bayes (most primitive ML approach)
    B. Innovation 1:  TF-IDF + Linear SVM (improved representation + classifier)
    C. Innovation 2:  + NLP Preprocessing (lemmatization + stopwords) + POS features
    D. Innovation 3:  + Hybrid features (VADER + stats) (symbolic + subsymbolic)
    E. Innovation 4:  + Sarcasm features (enhanced classification)
    F. Innovation 5:  Full features + Stacked Ensemble (ensemble classification)

    Returns dict of {config_name: metrics}.
    """
    print("\n" + "#" * 100)
    print("# Q5 INNOVATIONS: ABLATION STUDY (BoW+NB -> TF-IDF+SVM -> NLP -> Hybrid -> Ensemble)")
    print("#" * 100)

    ablation_configs = [
        {
            'name': 'A. Baseline (BoW + NB)',
            'vectorizer_type': 'bow', 'classifier_type': 'nb',
            'use_nlp_preprocess': False, 'use_pos_features': False,
            'use_hybrid': False, 'use_sarcasm': False, 'use_ensemble': False,
        },
        {
            'name': 'B. TF-IDF + SVM',
            'vectorizer_type': 'tfidf', 'classifier_type': 'svm',
            'use_nlp_preprocess': False, 'use_pos_features': False,
            'use_hybrid': False, 'use_sarcasm': False, 'use_ensemble': False,
        },
        {
            'name': 'C. + NLP Preprocess + POS',
            'vectorizer_type': 'tfidf', 'classifier_type': 'svm',
            'use_nlp_preprocess': True, 'use_pos_features': True,
            'use_hybrid': False, 'use_sarcasm': False, 'use_ensemble': False,
        },
        {
            'name': 'D. + Hybrid (VADER+Stats)',
            'vectorizer_type': 'tfidf', 'classifier_type': 'svm',
            'use_nlp_preprocess': True, 'use_pos_features': True,
            'use_hybrid': True, 'use_sarcasm': False, 'use_ensemble': False,
        },
        {
            'name': 'E. + Sarcasm Features',
            'vectorizer_type': 'tfidf', 'classifier_type': 'svm',
            'use_nlp_preprocess': True, 'use_pos_features': True,
            'use_hybrid': True, 'use_sarcasm': True, 'use_ensemble': False,
        },
        {
            'name': 'F. + Ensemble',
            'vectorizer_type': 'tfidf', 'classifier_type': 'svm',
            'use_nlp_preprocess': True, 'use_pos_features': True,
            'use_hybrid': True, 'use_sarcasm': True, 'use_ensemble': True,
        },
    ]

    ablation_results = {}

    for config in ablation_configs:
        name = config['name']
        print(f"\n  {'='*90}")
        print(f"  [Ablation] {name}")
        print(f"  {'='*90}")

        result = _run_cv_experiment(
            texts, labels, label_names, args,
            vectorizer_type=config['vectorizer_type'],
            classifier_type=config['classifier_type'],
            use_nlp_preprocess=config.get('use_nlp_preprocess', False),
            use_pos_features=config.get('use_pos_features', False),
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
    print(f"\n  {'Configuration':<42} {'Acc':>8} {'M-Prec':>8} {'M-Rec':>8} "
          f"{'M-F1':>8} {'W-F1':>8} {'Delta':>8}")
    print(f"  {'─'*42} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    baseline_f1 = ablation_results[ablation_configs[0]['name']]['macro_f1']

    for config in ablation_configs:
        name = config['name']
        m = ablation_results[name]
        delta = m['macro_f1'] - baseline_f1
        sign = "+" if delta >= 0 else ""
        delta_str = f"{sign}{delta:.4f}" if name != ablation_configs[0]['name'] else "  base"
        print(f"  {name:<42} "
              f"{m['accuracy']:>8.4f} "
              f"{m['macro_precision']:>8.4f} "
              f"{m['macro_recall']:>8.4f} "
              f"{m['macro_f1']:>8.4f} "
              f"{m['weighted_f1']:>8.4f} "
              f"{delta_str:>8}")

    best_name = max(ablation_results, key=lambda k: ablation_results[k]['macro_f1'])
    best_f1 = ablation_results[best_name]['macro_f1']
    total_gain = best_f1 - baseline_f1
    print(f"\n  Best: {best_name}")
    print(f"  Baseline F1: {baseline_f1:.4f} -> Best F1: {best_f1:.4f} "
          f"(total gain: {'+' if total_gain >= 0 else ''}{total_gain:.4f})")

    return ablation_results
