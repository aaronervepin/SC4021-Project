"""
Sentiment Labeler for Singapore Exam Data

This module provides automated sentiment labeling based on:
1. Keyword-based heuristics
2. Rule-based classification
3. Manual labeling support

Sentiment Labels:
- 0: Negative (exam was difficult, stressful, failed)
- 1: Neutral (informational, questions, unclear sentiment)
- 2: Positive (exam was manageable, did well, confident)
"""

import re
from typing import Dict, List, Tuple, Optional
from collections import Counter

# Sentiment keywords with weights
POSITIVE_KEYWORDS = {
    # Difficulty - easy
    "easy": 2.0, "manageable": 2.0, "doable": 2.0, "straightforward": 2.0,
    "simple": 1.5, "not hard": 2.0, "not difficult": 2.0, "not bad": 1.5,
    "easier than": 2.0, "easier": 1.5, "easiest": 2.0,
    
    # Performance - good
    "passed": 2.0, "pass": 1.5, "aced": 2.5, "scored well": 2.5,
    "did well": 2.0, "did good": 2.0, "nailed": 2.0, "smashed": 2.0,
    "got a": 1.0, "distinction": 2.5, "a1": 2.0, "a2": 1.5,
    
    # Emotions - positive
    "happy": 2.0, "relieved": 2.0, "confident": 2.0, "satisfied": 1.5,
    "excited": 1.5, "glad": 1.5, "proud": 2.0, "pleased": 1.5,
    
    # Evaluation - positive
    "good": 1.5, "great": 2.0, "excellent": 2.0, "amazing": 2.0,
    "wonderful": 2.0, "fantastic": 2.0, "awesome": 2.0, "best": 1.5,
    "love": 1.5, "loved": 1.5, "enjoyed": 1.5, "fun": 1.5,
    "interesting": 1.0, "fair": 2.0, "reasonable": 2.0,
    
    # Expressions
    "thank god": 2.0, "finally": 1.0, "yay": 2.0, "woohoo": 2.0,
    "yes": 1.0, "nice": 1.5, "lucky": 1.5, "blessed": 1.5,
    "well prepared": 1.5, "comfortable": 1.5, "smooth": 1.5,
    "expected": 1.0, "as expected": 1.5, "went well": 2.0,
    "no problem": 2.0, "no issues": 2.0, "piece of cake": 2.5,
    "breeze": 2.0, "cinch": 2.0,
}

NEGATIVE_KEYWORDS = {
    # Difficulty - hard
    "hard": 2.0, "difficult": 2.0, "tough": 2.0, "killer": 2.5,
    "impossible": 2.5, "insane": 2.0, "crazy": 1.5, "brutal": 2.5,
    "harder than": 2.0, "harder": 1.5, "hardest": 2.0,
    "challenging": 1.5, "tricky": 1.5, "confusing": 2.0,
    "complicated": 1.5, "complex": 1.0,
    
    # Performance - bad
    "failed": 2.5, "fail": 2.0, "flunked": 2.5, "bombed": 2.5,
    "screwed": 2.0, "destroyed": 2.0, "murdered": 2.0, "wrecked": 2.0,
    "messed up": 2.0, "f9": 2.5, "u grade": 2.5, "ungraded": 2.0,
    
    # Emotions - negative
    "stressed": 2.0, "anxious": 2.0, "worried": 1.5, "scared": 1.5,
    "panic": 2.0, "panicked": 2.0, "nervous": 1.5, "terrified": 2.0,
    "crying": 2.0, "cried": 2.0, "depressed": 2.0, "sad": 1.5,
    "disappointed": 2.0, "upset": 1.5, "frustrated": 2.0, "angry": 1.5,
    "devastated": 2.5, "hopeless": 2.0, "despair": 2.0,
    
    # Expressions
    "wtf": 2.0, "what the": 1.5, "oh no": 1.5, "damn": 1.5,
    "shit": 2.0, "fml": 2.5, "gg": 2.0, "gone case": 2.5,
    "rip": 2.0, "died": 2.0, "dead": 1.5, "death": 1.5,
    "no time": 2.0, "not enough time": 2.5, "ran out of time": 2.5,
    "rushed": 1.5, "couldn't finish": 2.5, "didn't finish": 2.5,
    "careless": 1.5, "mistake": 1.5, "wrong": 1.0, "regret": 2.0,
    
    # Evaluation - negative
    "unfair": 2.5, "unreasonable": 2.0, "unexpected": 1.5,
    "terrible": 2.0, "horrible": 2.0, "awful": 2.0, "worst": 2.0,
    "bad": 1.5, "sucks": 2.0, "sucked": 2.0, "hate": 2.0, "hated": 2.0,
}

NEUTRAL_KEYWORDS = {
    # Uncertainty
    "okay": 1.5, "ok": 1.5, "alright": 1.5, "average": 2.0,
    "moderate": 1.5, "so-so": 2.0, "50-50": 2.0, "mixed": 1.5,
    "not sure": 2.0, "don't know": 1.5, "unsure": 1.5,
    "maybe": 1.0, "perhaps": 1.0, "could be": 1.0,
    "depends": 1.5, "varies": 1.5, "some parts": 1.0,
    
    # Questions/Information seeking
    "anyone": 1.5, "any tips": 2.0, "advice": 1.5, "help": 1.0,
    "question": 2.0, "asking": 1.5, "wondering": 1.5, "curious": 1.5,
    "how to": 1.5, "what is": 1.5, "when is": 1.5, "where is": 1.5,
    "how long": 1.5, "how much": 1.0,
    
    # Waiting/Results
    "waiting": 1.5, "results": 1.0, "when": 0.5,
    "release": 1.0, "coming out": 1.0,
    
    # Informational
    "fyi": 1.5, "information": 1.5, "update": 1.0,
    "schedule": 1.5, "timetable": 1.5, "syllabus": 1.5,
}

# Negation words that flip sentiment
NEGATION_WORDS = [
    "not", "no", "never", "neither", "nobody", "nothing",
    "nowhere", "none", "nor", "cannot", "can't", "couldn't",
    "shouldn't", "wouldn't", "won't", "don't", "doesn't",
    "didn't", "isn't", "aren't", "wasn't", "weren't",
    "hardly", "scarcely", "barely", "seldom", "rarely"
]

# Intensifiers that increase weight
INTENSIFIERS = {
    "very": 1.5, "really": 1.5, "extremely": 2.0, "super": 1.5,
    "so": 1.3, "too": 1.3, "incredibly": 2.0, "absolutely": 2.0,
    "totally": 1.5, "completely": 1.5, "utterly": 2.0,
    "quite": 1.2, "rather": 1.1, "pretty": 1.2,
}


def preprocess_text(text: str) -> str:
    """Preprocess text for sentiment analysis."""
    text = text.lower()
    # Keep some punctuation for context
    text = re.sub(r'[^\w\s\'!?.-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def find_keywords_with_context(text: str, keywords: Dict[str, float], 
                                window: int = 3) -> List[Tuple[str, float, bool]]:
    """
    Find keywords in text with negation detection.
    Returns list of (keyword, weight, is_negated)
    """
    text = preprocess_text(text)
    words = text.split()
    found = []
    
    for keyword, weight in keywords.items():
        # Handle multi-word keywords
        if ' ' in keyword:
            if keyword in text:
                # Check for negation before the phrase
                pattern = r'(\w+\s+){0,3}' + re.escape(keyword)
                matches = re.finditer(pattern, text)
                for match in matches:
                    context = match.group()
                    is_negated = any(neg in context for neg in NEGATION_WORDS)
                    
                    # Check for intensifiers
                    final_weight = weight
                    for intensifier, mult in INTENSIFIERS.items():
                        if intensifier in context:
                            final_weight *= mult
                            break
                    
                    found.append((keyword, final_weight, is_negated))
        else:
            # Single word keywords
            for i, word in enumerate(words):
                if keyword == word or word.startswith(keyword):
                    # Check for negation in window before
                    start_idx = max(0, i - window)
                    context_words = words[start_idx:i]
                    is_negated = any(neg in context_words for neg in NEGATION_WORDS)
                    
                    # Check for intensifiers
                    final_weight = weight
                    for intensifier, mult in INTENSIFIERS.items():
                        if intensifier in context_words:
                            final_weight *= mult
                            break
                    
                    found.append((keyword, final_weight, is_negated))
    
    return found


def calculate_sentiment_score(text: str) -> Tuple[float, Dict]:
    """
    Calculate sentiment score for text.
    Returns (score, details) where:
    - score > 0.3: Positive
    - score < -0.3: Negative
    - otherwise: Neutral
    """
    positive_hits = find_keywords_with_context(text, POSITIVE_KEYWORDS)
    negative_hits = find_keywords_with_context(text, NEGATIVE_KEYWORDS)
    neutral_hits = find_keywords_with_context(text, NEUTRAL_KEYWORDS)
    
    positive_score = 0
    negative_score = 0
    neutral_score = 0
    
    for keyword, weight, is_negated in positive_hits:
        if is_negated:
            negative_score += weight * 0.8  # Negated positive -> negative
        else:
            positive_score += weight
    
    for keyword, weight, is_negated in negative_hits:
        if is_negated:
            positive_score += weight * 0.8  # Negated negative -> positive
        else:
            negative_score += weight
    
    for keyword, weight, is_negated in neutral_hits:
        neutral_score += weight
    
    total = positive_score + negative_score + neutral_score + 0.001  # Avoid division by zero
    
    # Normalize and calculate final score
    norm_positive = positive_score / total
    norm_negative = negative_score / total
    norm_neutral = neutral_score / total
    
    # Score ranges from -1 (very negative) to 1 (very positive)
    final_score = norm_positive - norm_negative
    
    details = {
        "positive_score": positive_score,
        "negative_score": negative_score,
        "neutral_score": neutral_score,
        "normalized_positive": norm_positive,
        "normalized_negative": norm_negative,
        "normalized_neutral": norm_neutral,
        "positive_keywords": [(k, w) for k, w, n in positive_hits if not n],
        "negative_keywords": [(k, w) for k, w, n in negative_hits if not n],
        "neutral_keywords": [(k, w) for k, w, n in neutral_hits],
        "negated_positive": [(k, w) for k, w, n in positive_hits if n],
        "negated_negative": [(k, w) for k, w, n in negative_hits if n],
    }
    
    return final_score, details


def classify_sentiment(text: str, positive_threshold: float = 0.25,
                       negative_threshold: float = -0.25) -> Tuple[int, float, Dict]:
    """
    Classify text sentiment.
    
    Returns:
    - label: 0 (negative), 1 (neutral), 2 (positive)
    - confidence: absolute value of score
    - details: scoring breakdown
    """
    score, details = calculate_sentiment_score(text)
    
    if score >= positive_threshold:
        label = 2  # Positive
    elif score <= negative_threshold:
        label = 0  # Negative
    else:
        label = 1  # Neutral
    
    confidence = abs(score)
    
    return label, confidence, details


def batch_classify(texts: List[str], 
                   positive_threshold: float = 0.25,
                   negative_threshold: float = -0.25) -> List[Tuple[int, float]]:
    """Classify a batch of texts."""
    results = []
    for text in texts:
        label, confidence, _ = classify_sentiment(text, positive_threshold, negative_threshold)
        results.append((label, confidence))
    return results


def get_sentiment_distribution(labels: List[int]) -> Dict[str, int]:
    """Get distribution of sentiment labels."""
    counter = Counter(labels)
    return {
        "negative": counter.get(0, 0),
        "neutral": counter.get(1, 0),
        "positive": counter.get(2, 0),
        "total": len(labels)
    }


def adjust_thresholds_for_balance(texts: List[str], 
                                   target_ratio: float = 0.33) -> Tuple[float, float]:
    """
    Find thresholds that produce roughly balanced distribution.
    Returns (positive_threshold, negative_threshold)
    """
    # Calculate scores for all texts
    scores = [calculate_sentiment_score(text)[0] for text in texts]
    scores.sort()
    
    n = len(scores)
    target_count = int(n * target_ratio)
    
    # Find thresholds at the target percentiles
    negative_threshold = scores[target_count] if target_count < n else -0.25
    positive_threshold = scores[n - target_count - 1] if target_count < n else 0.25
    
    return positive_threshold, negative_threshold


# Example usage
if __name__ == "__main__":
    test_texts = [
        "The A level math paper was so hard, I couldn't finish in time!",
        "O level English was manageable, I think I did okay.",
        "Anyone knows when the results will be released?",
        "NUS finals were killer, I definitely failed.",
        "The exam was easier than expected, thank god!",
        "Not sure how I did, some parts were tricky.",
        "I aced my poly exams! So happy!",
        "The paper was not difficult at all, very straightforward.",
    ]
    
    print("Sentiment Classification Examples:")
    print("=" * 60)
    
    for text in test_texts:
        label, confidence, details = classify_sentiment(text)
        label_name = {0: "Negative", 1: "Neutral", 2: "Positive"}[label]
        print(f"\nText: {text}")
        print(f"Label: {label_name} ({label}), Confidence: {confidence:.2f}")
        print(f"  Positive keywords: {details['positive_keywords']}")
        print(f"  Negative keywords: {details['negative_keywords']}")
