"""
Console report printing utilities.
"""

import numpy as np
from typing import List, Dict

from .config import LABEL_COL_DISPLAY, TEXT_COL_DISPLAY, CLASSIFIER_NAMES


def print_experiment_result(result: Dict, label_names: List[str]):
    """Print detailed metrics for one experiment run."""
    agg = result['aggregate']
    perf = result['performance']
    cm = np.array(result['confusion_matrix'])

    print(f"    Accuracy:           {agg['accuracy']:.4f}")
    print(f"    Macro Precision:    {agg['macro_precision']:.4f}")
    print(f"    Macro Recall:       {agg['macro_recall']:.4f}")
    print(f"    Macro F1:           {agg['macro_f1']:.4f}")
    print(f"    Weighted F1:        {agg['weighted_f1']:.4f}")
    print(f"    Speed:              {perf['records_per_second']:.0f} records/sec")

    print(f"\n    Confusion Matrix:")
    header = "    " + " " * 14 + "  ".join(f"{n:>10}" for n in label_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>10}" for v in row)
        print(f"    {label_names[i]:>12}  {row_str}")

    per_class = result['per_class_report']
    print(f"\n    Per-Class Report:")
    print(f"    {'Class':>14}  {'Prec':>8}  {'Recall':>8}  {'F1':>8}  {'Support':>8}")
    for name in label_names:
        if name in per_class:
            r = per_class[name]
            print(f"    {name:>14}  {r['precision']:>8.4f}  "
                  f"{r['recall']:>8.4f}  {r['f1-score']:>8.4f}  "
                  f"{r['support']:>8.0f}")


def print_comparison_table(comparison_data: List[Dict], text_cols: List[str]):
    """Print comparison table across all experiments."""
    print("\n" + "=" * 100)
    print("COMPARISON TABLE: original_text vs normalized_text")
    print("=" * 100)

    grouped = {}
    for entry in comparison_data:
        key = (entry['label_col'], entry['classifier_key'])
        if key not in grouped:
            grouped[key] = {}
        grouped[key][entry['text_col']] = entry['aggregate']

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

        clf_display = CLASSIFIER_NAMES.get(clf_key, clf_key)

        for text_col in text_cols:
            text_display = TEXT_COL_DISPLAY.get(text_col, text_col)
            if text_col in text_results:
                m = text_results[text_col]
                print(f"  {clf_display:<22} │ {text_display:<28} │ "
                      f"{m['accuracy']:>7.4f} {m['macro_precision']:>7.4f} "
                      f"{m['macro_recall']:>7.4f} {m['macro_f1']:>7.4f} "
                      f"{m['weighted_f1']:>7.4f}")


def print_delta_analysis(comparison_data: List[Dict]):
    """Print normalization improvement/regression for each task+classifier."""
    print("\n" + "=" * 100)
    print("DELTA ANALYSIS: Normalized vs Original (Macro-F1 difference)")
    print("=" * 100)

    grouped = {}
    for entry in comparison_data:
        key = (entry['label_col'], entry['classifier_key'])
        if key not in grouped:
            grouped[key] = {}
        grouped[key][entry['text_col']] = entry['aggregate']

    print(f"\n  {'Task':<28} {'Classifier':<22} "
          f"{'Orig F1':>9} {'Norm F1':>9} {'Delta':>9} {'Change':>9}")
    print(f"  {'─'*28} {'─'*22} {'─'*9} {'─'*9} {'─'*9} {'─'*9}")

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


def print_best_per_task(comparison_data: List[Dict], label_cols: List[str]):
    """Print the best classifier + text combination per task."""
    print("\n" + "=" * 100)
    print("BEST CONFIGURATION PER TASK")
    print("=" * 100)

    by_task = {}
    for entry in comparison_data:
        lc = entry['label_col']
        if lc not in by_task:
            by_task[lc] = []
        by_task[lc].append(entry)

    print(f"\n  {'Task':<28} {'Best Classifier':<22} {'Text':<28} "
          f"{'Macro-F1':>9} {'Accuracy':>9}")
    print(f"  {'─'*28} {'─'*22} {'─'*28} {'─'*9} {'─'*9}")

    for label_col in label_cols:
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
