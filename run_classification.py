"""
SC4021 Information Retrieval - Classification Pipeline Runner
=============================================================
Main entry point that orchestrates the full classification workflow:
1. Load and evaluate multiple classifiers on multiple tasks (Q4)
2. Generate comparison reports and visualizations
3. Run ensemble innovation experiments with ablation study (Q5)
4. Predict on external crawled data

Usage:
    conda activate mdp
    python run_classification.py
    python run_classification.py --skip_innovations --skip_visualizations
    python run_classification.py --classifiers logistic svm nb

Author: Zhang
Date: 2026-03
"""

import os
import sys
import time
import json
import numpy as np
from collections import Counter

from classification.config import (
    parse_args, LABEL_COL_DISPLAY, TEXT_COL_DISPLAY, CLASSIFIER_NAMES
)
from classification.data_loader import load_data
from classification.evaluation import run_5fold
from classification.report_printer import (
    print_experiment_result, print_comparison_table,
    print_delta_analysis, print_best_per_task
)
from classification.prediction import run_best_sentiment_prediction
from classification.visualization import (
    generate_all_visualizations, plot_ablation_study
)
from classification.innovations import run_ablation_study


def main():
    args = parse_args()

    print("=" * 100)
    print("SC4021 - Classification Pipeline")
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

    # ================================================================
    # Pre-load data
    # ================================================================
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
                      f"{len(texts)} samples, labels={label_names}")
            except Exception as e:
                print(f"  ERROR loading [{text_col}] x [{label_col}]: {e}")

    # ================================================================
    # Run all base experiments (Q4)
    # ================================================================
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
                        texts, labels, label_names, clf_name, args, verbose=True
                    )

                    print(f"\n    --- Aggregate ({args.n_folds}-Fold) ---")
                    print_experiment_result(result, label_names)

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

    # ================================================================
    # Comparison outputs
    # ================================================================
    if comparison_data:
        print_comparison_table(comparison_data, args.text_cols)
        print_delta_analysis(comparison_data)
        print_best_per_task(comparison_data, args.label_cols)

    # Per-task ranking
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

    # ================================================================
    # Q5: Innovation experiments (Ensemble + Ablation)
    # ================================================================
    ablation_results = None
    if not args.skip_innovations:
        # Use sentiment_final with normalized_text for ablation
        ablation_key = ('normalized_text', 'sentiment_final')
        if ablation_key not in data_cache:
            ablation_key = ('original_text', 'sentiment_final')

        if ablation_key in data_cache:
            abl_data = data_cache[ablation_key]
            ablation_results = run_ablation_study(
                abl_data['texts'], abl_data['labels'],
                abl_data['label_names'], args
            )

    # ================================================================
    # Visualizations
    # ================================================================
    if not args.skip_visualizations:
        os.makedirs(args.figures_dir, exist_ok=True)
        generate_all_visualizations(
            comparison_data, all_experiment_results,
            data_cache, args.figures_dir
        )
        if ablation_results:
            plot_ablation_study(
                {k: v for k, v in ablation_results.items() if 'full_result' not in v or True},
                args.figures_dir
            )

    # ================================================================
    # Save JSON Report
    # ================================================================
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

    if ablation_results:
        report['ablation_study'] = {
            k: {key: val for key, val in v.items() if key != 'full_result'}
            for k, v in ablation_results.items()
        }

    total_elapsed = time.time() - global_start_time
    report['total_time_sec'] = round(total_elapsed, 1)

    with open(args.output_report, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ================================================================
    # Post-training prediction
    # ================================================================
    if not args.skip_prediction:
        run_best_sentiment_prediction(comparison_data, args)

    print(f"\n\nReport saved to: {args.output_report}")
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
