"""
Visualization module: generates plots for classification results.
All figures are saved to the figures/ directory.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict

from .config import LABEL_COL_DISPLAY, TEXT_COL_DISPLAY, CLASSIFIER_NAMES


def setup_style():
    """Set consistent plot style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
    })


def plot_confusion_matrix(cm, label_names, title, save_path):
    """Plot and save a confusion matrix heatmap."""
    setup_style()
    fig, ax = plt.subplots(figsize=(max(6, len(label_names) * 1.2),
                                     max(5, len(label_names) * 1.0)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_classifier_comparison(comparison_data: List[Dict], figures_dir: str):
    """Bar chart comparing classifiers across tasks by Macro-F1."""
    setup_style()

    # Group by label_col
    by_task = {}
    for entry in comparison_data:
        lc = entry['label_col']
        if lc not in by_task:
            by_task[lc] = []
        by_task[lc].append(entry)

    for label_col, entries in by_task.items():
        task_name = LABEL_COL_DISPLAY.get(label_col, label_col)

        # Organize data: classifier x text_col
        classifiers = sorted(set(e['classifier_key'] for e in entries))
        text_cols = sorted(set(e['text_col'] for e in entries))

        fig, ax = plt.subplots(figsize=(max(8, len(classifiers) * 2), 5))

        x = np.arange(len(classifiers))
        width = 0.35

        for i, tc in enumerate(text_cols):
            f1_vals = []
            for clf_key in classifiers:
                match = [e for e in entries
                         if e['classifier_key'] == clf_key and e['text_col'] == tc]
                f1_vals.append(match[0]['aggregate']['macro_f1'] if match else 0)

            tc_display = TEXT_COL_DISPLAY.get(tc, tc)
            offset = (i - len(text_cols) / 2 + 0.5) * width
            bars = ax.bar(x + offset, f1_vals, width * 0.9, label=tc_display)

            for bar, val in zip(bars, f1_vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)

        clf_labels = [CLASSIFIER_NAMES.get(c, c) for c in classifiers]
        ax.set_xticks(x)
        ax.set_xticklabels(clf_labels, rotation=15, ha='right')
        ax.set_ylabel('Macro F1')
        ax.set_title(f'Classifier Comparison - {task_name}')
        ax.legend()
        ax.set_ylim(0, 1.05)
        plt.tight_layout()

        fname = f'comparison_{label_col}.png'
        fig.savefig(os.path.join(figures_dir, fname), bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {fname}")


def plot_delta_analysis(comparison_data: List[Dict], figures_dir: str):
    """Bar chart showing F1 improvement from text normalization."""
    setup_style()

    grouped = {}
    for entry in comparison_data:
        key = (entry['label_col'], entry['classifier_key'])
        if key not in grouped:
            grouped[key] = {}
        grouped[key][entry['text_col']] = entry['aggregate']

    tasks = []
    classifiers = []
    deltas = []

    for (label_col, clf_key), text_results in sorted(grouped.items()):
        orig_f1 = text_results.get('original_text', {}).get('macro_f1', None)
        norm_f1 = text_results.get('normalized_text', {}).get('macro_f1', None)
        if orig_f1 is not None and norm_f1 is not None:
            tasks.append(LABEL_COL_DISPLAY.get(label_col, label_col))
            classifiers.append(CLASSIFIER_NAMES.get(clf_key, clf_key))
            deltas.append(norm_f1 - orig_f1)

    if not deltas:
        return

    fig, ax = plt.subplots(figsize=(max(10, len(deltas) * 0.8), 5))
    labels = [f"{t}\n{c}" for t, c in zip(tasks, classifiers)]
    colors = ['#2ecc71' if d >= 0 else '#e74c3c' for d in deltas]

    bars = ax.bar(range(len(deltas)), deltas, color=colors)
    ax.set_xticks(range(len(deltas)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Macro F1 Delta (Normalized - Original)')
    ax.set_title('Impact of Text Normalization on Classification')
    ax.axhline(y=0, color='black', linewidth=0.5)

    for bar, val in zip(bars, deltas):
        y_pos = bar.get_height() + (0.002 if val >= 0 else -0.008)
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f'{val:+.4f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'delta_normalization.png'), bbox_inches='tight')
    plt.close(fig)
    print("  Saved: delta_normalization.png")


def plot_label_distributions(data_cache: Dict, figures_dir: str):
    """Pie/bar charts for label distributions in the evaluation dataset."""
    setup_style()

    # Collect unique label_cols
    label_cols_seen = {}
    for (text_col, label_col), data in data_cache.items():
        if text_col == 'original_text':  # only need one text col for distribution
            label_cols_seen[label_col] = data

    n = len(label_cols_seen)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (label_col, data) in zip(axes, label_cols_seen.items()):
        task_name = LABEL_COL_DISPLAY.get(label_col, label_col)
        label_names = data['label_names']
        dist = data['dist']

        counts = [dist.get(i, 0) for i in range(len(label_names))]
        colors = sns.color_palette('Set2', len(label_names))
        ax.bar(label_names, counts, color=colors)
        ax.set_title(task_name)
        ax.set_ylabel('Count')
        for i, c in enumerate(counts):
            ax.text(i, c + 10, str(c), ha='center', fontsize=8)

    plt.suptitle('Label Distribution in Evaluation Dataset', fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'label_distributions.png'), bbox_inches='tight')
    plt.close(fig)
    print("  Saved: label_distributions.png")


def plot_performance_comparison(comparison_data: List[Dict], figures_dir: str):
    """Bar chart of classification speed (records/sec) per classifier."""
    setup_style()

    by_task = {}
    for entry in comparison_data:
        lc = entry['label_col']
        if lc not in by_task:
            by_task[lc] = []
        by_task[lc].append(entry)

    # Use sentiment_final as representative task
    target_task = 'sentiment_final'
    if target_task not in by_task:
        target_task = list(by_task.keys())[0] if by_task else None
    if target_task is None:
        return

    entries = by_task[target_task]
    classifiers = sorted(set(e['classifier_key'] for e in entries))
    text_cols = sorted(set(e['text_col'] for e in entries))

    fig, ax = plt.subplots(figsize=(max(8, len(classifiers) * 2), 5))
    x = np.arange(len(classifiers))
    width = 0.35

    for i, tc in enumerate(text_cols):
        speeds = []
        for clf_key in classifiers:
            match = [e for e in entries
                     if e['classifier_key'] == clf_key and e['text_col'] == tc]
            speeds.append(match[0]['performance']['records_per_second'] if match else 0)

        tc_display = TEXT_COL_DISPLAY.get(tc, tc)
        offset = (i - len(text_cols) / 2 + 0.5) * width
        bars = ax.bar(x + offset, speeds, width * 0.9, label=tc_display)

        for bar, val in zip(bars, speeds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=8)

    clf_labels = [CLASSIFIER_NAMES.get(c, c) for c in classifiers]
    ax.set_xticks(x)
    ax.set_xticklabels(clf_labels, rotation=15, ha='right')
    ax.set_ylabel('Records / Second')
    ax.set_title(f'Classification Speed - {LABEL_COL_DISPLAY.get(target_task, target_task)}')
    ax.legend()
    plt.tight_layout()

    fig.savefig(os.path.join(figures_dir, 'performance_speed.png'), bbox_inches='tight')
    plt.close(fig)
    print("  Saved: performance_speed.png")


def plot_all_confusion_matrices(all_experiment_results: Dict, figures_dir: str):
    """Generate confusion matrix heatmaps for all experiments."""
    setup_style()
    cm_dir = os.path.join(figures_dir, 'confusion_matrices')
    os.makedirs(cm_dir, exist_ok=True)

    for exp_key, result in all_experiment_results.items():
        parts = exp_key.split('__')
        if len(parts) == 3:
            text_col, label_col, clf_key = parts
        else:
            continue

        cm = np.array(result['confusion_matrix'])
        label_names = list(result['per_class_report'].keys())
        if not label_names:
            continue

        text_display = TEXT_COL_DISPLAY.get(text_col, text_col)
        task_display = LABEL_COL_DISPLAY.get(label_col, label_col)
        clf_display = CLASSIFIER_NAMES.get(clf_key, clf_key)

        title = f'{clf_display}\n{task_display} | {text_display}'
        fname = f'cm_{label_col}_{clf_key}_{text_col}.png'
        plot_confusion_matrix(cm, label_names, title,
                              os.path.join(cm_dir, fname))

    print(f"  Saved confusion matrices to {cm_dir}/")


def plot_ablation_study(ablation_results: Dict, figures_dir: str):
    """Plot ablation study results as grouped bar chart with baseline reference."""
    setup_style()

    configs = list(ablation_results.keys())
    if not configs:
        return

    metrics = ['macro_f1', 'accuracy']
    fig, axes = plt.subplots(1, len(metrics), figsize=(max(7 * len(metrics), 12), 6))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        values = [ablation_results[c].get(metric, 0) for c in configs]
        colors = sns.color_palette('viridis', len(configs))
        bars = ax.bar(range(len(configs)), values, color=colors)
        ax.set_xticks(range(len(configs)))
        # Use shorter labels for readability
        short_labels = [c.split('. ', 1)[-1] if '. ' in c else c for c in configs]
        ax.set_xticklabels(short_labels, rotation=35, ha='right', fontsize=8)
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Ablation Study - {metric.replace("_", " ").title()}')

        # Baseline reference line
        if values:
            ax.axhline(y=values[0], color='red', linewidth=0.8,
                        linestyle='--', alpha=0.7, label='Baseline')
            ax.legend(fontsize=8)

        for bar, val in zip(bars, values):
            delta = val - values[0]
            delta_str = f'{val:.4f}\n({"+{:.4f}" if delta >= 0 else "{:.4f}"})'.format(delta) \
                        if bar != bars[0] else f'{val:.4f}'
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    delta_str, ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'ablation_study.png'), bbox_inches='tight')
    plt.close(fig)
    print("  Saved: ablation_study.png")


def generate_all_visualizations(comparison_data, all_experiment_results,
                                 data_cache, figures_dir):
    """Generate all visualization plots."""
    os.makedirs(figures_dir, exist_ok=True)
    print("\nGenerating visualizations...")

    plot_label_distributions(data_cache, figures_dir)
    plot_classifier_comparison(comparison_data, figures_dir)
    plot_delta_analysis(comparison_data, figures_dir)
    plot_performance_comparison(comparison_data, figures_dir)
    plot_all_confusion_matrices(all_experiment_results, figures_dir)

    print("All visualizations saved.")
