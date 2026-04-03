"""
Classifier factory and TF-IDF vectorizer builder.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    StackingClassifier, VotingClassifier
)


def get_classifier(name: str, seed: int):
    """Return classifier instance by name."""
    classifiers = {
        'logistic': LogisticRegression(
            max_iter=1000, C=1.0, solver='lbfgs', random_state=seed,
            class_weight='balanced'
        ),
        'svm': LinearSVC(
            max_iter=2000, C=1.0, random_state=seed,
            class_weight='balanced'
        ),
        'nb': MultinomialNB(alpha=0.1),
        'rf': RandomForestClassifier(
            n_estimators=200, max_depth=None,
            random_state=seed, n_jobs=-1
        ),
        'gb': GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=seed
        ),
    }
    return classifiers[name]


def get_stacking_ensemble(seed: int):
    """Return a stacking ensemble classifier (LR + SVM + NB + RF -> LR meta)."""
    estimators = [
        ('lr', LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=seed, class_weight='balanced')),
        ('nb', MultinomialNB(alpha=0.1)),
        ('svm', LinearSVC(max_iter=2000, C=1.0, random_state=seed, class_weight='balanced')),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=None, random_state=seed, n_jobs=-1)),
    ]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=500, random_state=seed, class_weight='balanced'),
        cv=3,
        n_jobs=-1,
    )


def get_voting_ensemble(seed: int):
    """Return a soft voting ensemble classifier."""
    estimators = [
        ('lr', LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=seed)),
        ('nb', MultinomialNB(alpha=0.1)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=None, random_state=seed, n_jobs=-1)),
    ]
    return VotingClassifier(
        estimators=estimators,
        voting='soft',
        n_jobs=-1,
    )


def build_tfidf_vectorizer(args):
    """Build TF-IDF vectorizer from args."""
    strip_val = None if args.strip_accents == 'none' else args.strip_accents
    return TfidfVectorizer(
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
