"""
Post-training prediction on external (crawled) data.
"""

import os
import csv
import json
import time
import numpy as np
from collections import Counter
from typing import List, Dict

from .config import CLASSIFIER_NAMES
from .data_loader import load_data, load_prediction_csv
from .models import get_classifier, build_tfidf_vectorizer


def run_best_sentiment_prediction(comparison_data: List[Dict], args):
    """
    Find the best sentiment model, retrain on full data, and predict external data.
    """
    sentiment_entries = [
        e for e in comparison_data
        if e.get('label_col') == 'sentiment_final'
    ]

    if not sentiment_entries:
        print("\n[Prediction skipped] No sentiment_final experiment result found.")
        return

    best = max(sentiment_entries, key=lambda e: e['aggregate']['macro_f1'])
    best_text_col = best['text_col']
    best_clf_key = best['classifier_key']
    best_clf_name = CLASSIFIER_NAMES.get(best_clf_key, best_clf_key)

    print("\n" + "=" * 100)
    print("POST-TRAINING PREDICTION ON EXTERNAL DATA")
    print("=" * 100)
    print(f"Best sentiment model: {best_clf_name}")
    print(f"Training text column: {best_text_col}")
    print(f"Best macro-F1: {best['aggregate']['macro_f1']:.4f}")

    train_texts, train_labels, label_names = load_data(
        args.input, best_text_col, 'sentiment_final'
    )
    train_labels = np.array(train_labels)

    vectorizer = build_tfidf_vectorizer(args)
    X_train = vectorizer.fit_transform(train_texts)
    clf = get_classifier(best_clf_key, args.seed)

    t0 = time.time()
    clf.fit(X_train, train_labels)
    fit_time = time.time() - t0
    print(f"Re-trained on full sentiment dataset: {len(train_texts)} samples in {fit_time:.2f}s")

    if not os.path.exists(args.predict_input):
        print(f"[Prediction skipped] File not found: {args.predict_input}")
        return

    rows, fieldnames, predict_texts, non_empty_indices = load_prediction_csv(
        args.predict_input, args.predict_text_col
    )

    output1 = ['' for _ in rows]
    output2 = ['' for _ in rows]
    pred_details = []

    if predict_texts:
        X_pred = vectorizer.transform(predict_texts)
        y_pred = clf.predict(X_pred)

        for idx, pred in zip(non_empty_indices, y_pred):
            pred_int = int(pred)
            output1[idx] = pred_int
            output2[idx] = pred_int - 1

    output1_int = [v for v in output1 if v != '']
    dist_output1 = {str(k): 0 for k in [0, 1, 2]}
    for k, v in Counter(output1_int).items():
        dist_output1[str(int(k))] = int(v)

    dist_output2 = {
        '-1': dist_output1['0'],
        '0':  dist_output1['1'],
        '1':  dist_output1['2'],
    }

    # Save CSV
    output_fields = list(rows[0].keys()) if rows else []
    if 'Output_1' not in output_fields:
        output_fields.append('Output_1')
    if 'Output_2' not in output_fields:
        output_fields.append('Output_2')

    with open(args.predict_output, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            row['Output_1'] = output1[i]
            row['Output_2'] = output2[i]
            writer.writerow(row)
            detail = {
                'row_index': i,
                'Output_1': output1[i],
                'Output_2': output2[i],
            }
            if 'id' in row:
                detail['id'] = row['id']
            pred_details.append(detail)

    # Save JSON
    json_payload = {
        'selection_criterion': 'macro_f1',
        'best_model': {
            'label_task': 'sentiment_final',
            'classifier_key': best_clf_key,
            'classifier_name': best_clf_name,
            'text_col': best_text_col,
            'macro_f1': best['aggregate']['macro_f1'],
            'accuracy': best['aggregate']['accuracy'],
        },
        'prediction_input': {
            'file': args.predict_input,
            'text_col': args.predict_text_col,
            'total_rows': len(rows),
            'predicted_rows': len(output1_int),
        },
        'label_distribution': {
            'Output_1': {k: v for k, v in dist_output1.items()},
            'Output_2': {k: v for k, v in dist_output2.items()},
        },
        'predictions': pred_details,
    }

    with open(args.predict_json_output, 'w', encoding='utf-8') as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)

    print(f"Prediction output: {args.predict_output}")
    print(f"Predicted records: {len(predict_texts)}/{len(rows)}")
    print(f"Output_1 distribution (0/1/2): "
          f"0={dist_output1['0']}, 1={dist_output1['1']}, 2={dist_output1['2']}")
    print(f"Label order used by model: {label_names}")
