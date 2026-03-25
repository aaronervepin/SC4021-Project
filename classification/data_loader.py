"""
Data loading and preprocessing utilities.
"""

import csv
import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import LabelEncoder


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
                f"Text column '{text_col}' not found. Available: {available_cols}"
            )
        if label_col not in available_cols:
            raise ValueError(
                f"Label column '{label_col}' not found. Available: {available_cols}"
            )

        for row in reader:
            text = row[text_col]
            label = row[label_col]
            if text and label != '':
                texts.append(text.strip())
                raw_labels.append(str(label).strip())

    le = LabelEncoder()
    labels_int = le.fit_transform(raw_labels)
    label_names = list(le.classes_)

    return texts, labels_int, label_names


def load_prediction_csv(filepath: str, text_col: str):
    """Load CSV for prediction, return (rows, fieldnames, texts, non_empty_indices)."""
    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if text_col not in fieldnames:
            raise ValueError(
                f"Prediction text column '{text_col}' not found. Available: {fieldnames}"
            )
        for row in reader:
            rows.append(row)

    non_empty_indices = []
    predict_texts = []
    for i, row in enumerate(rows):
        txt = (row.get(text_col) or '').strip()
        if txt != '':
            non_empty_indices.append(i)
            predict_texts.append(txt)

    return rows, fieldnames, predict_texts, non_empty_indices
