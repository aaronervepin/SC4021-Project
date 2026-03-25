"""
Configuration: CLI arguments, constants, and display mappings.
"""

import argparse
import builtins
import warnings

warnings.filterwarnings('ignore')

# Force flush on print
_original_print = builtins.print
def print(*a, **kw):
    kw.setdefault('flush', True)
    _original_print(*a, **kw)
builtins.print = print


# ============================================================================
# Display name mappings
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
    'ensemble': 'Stacked Ensemble',
}


# ============================================================================
# CLI Arguments
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="SC4021 Classification Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data
    parser.add_argument('--input', type=str, default='eval_preprocessed.csv',
                        help='Input preprocessed CSV file')
    parser.add_argument('--output_report', type=str, default='classification_comparison_report.json',
                        help='Path to save JSON comparison report')
    parser.add_argument('--figures_dir', type=str, default='figures',
                        help='Directory to save visualization figures')

    # Post-training prediction
    parser.add_argument('--predict_input', type=str, default='crawled_clean.csv',
                        help='CSV file to run prediction on')
    parser.add_argument('--predict_output', type=str, default='crawled_clean_with_predictions.csv',
                        help='Output CSV path with predictions')
    parser.add_argument('--predict_json_output', type=str, default='crawled_clean_predictions.json',
                        help='Output JSON path with prediction details')
    parser.add_argument('--predict_text_col', type=str, default='body',
                        help='Text column in predict_input')
    parser.add_argument('--skip_prediction', action='store_true',
                        help='Skip post-training prediction step')

    # Experiment selection
    parser.add_argument('--text_cols', type=str, nargs='+',
                        default=['original_text', 'normalized_text'],
                        help='Text columns to compare')
    parser.add_argument('--label_cols', type=str, nargs='+',
                        default=['sentiment_final', 'sarcasm_final',
                                 'subjectivity_final', 'emotion_final'],
                        help='Label columns to evaluate')

    # TF-IDF parameters
    parser.add_argument('--max_features', type=int, default=50000)
    parser.add_argument('--ngram_min', type=int, default=1)
    parser.add_argument('--ngram_max', type=int, default=2)
    parser.add_argument('--min_df', type=int, default=2)
    parser.add_argument('--max_df', type=float, default=0.95)
    parser.add_argument('--sublinear_tf', action='store_true', default=True)
    parser.add_argument('--use_idf', action='store_true', default=True)

    # Classifier selection
    parser.add_argument('--classifiers', type=str, nargs='+',
                        default=['logistic', 'svm', 'nb', 'rf'],
                        choices=['logistic', 'svm', 'nb', 'rf', 'gb'],
                        help='Base classifiers to evaluate')

    # Cross-validation
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)

    # Preprocessing
    parser.add_argument('--lowercase', action='store_true', default=True)
    parser.add_argument('--strip_accents', type=str, default='unicode',
                        choices=['unicode', 'ascii', 'none'])

    # Innovations
    parser.add_argument('--skip_innovations', action='store_true',
                        help='Skip Q5 innovation experiments')
    parser.add_argument('--skip_visualizations', action='store_true',
                        help='Skip generating visualization plots')

    return parser.parse_args()
