"""
Create Evaluation Dataset (eval.xls) for Singapore Exam Sentiment Analysis

This script:
1. Loads all crawled Reddit data
2. Removes duplicates
3. Labels sentiment (positive, negative, neutral)
4. Balances the dataset
5. Exports to eval.xls in standard sentiment benchmark format

Output format:
- text: The Reddit post/comment text
- label: 0 = negative, 1 = neutral, 2 = positive
"""

import json
import os
import re
import hashlib
from typing import List, Dict, Tuple
from collections import Counter
import pandas as pd
from datetime import datetime

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_FILE = os.path.join(BASE_DIR, "eval.xls")

# Sentiment keywords
POSITIVE_KEYWORDS = {
    # Exam difficulty - easy
    'easy', 'manageable', 'doable', 'straightforward', 'simple',
    'not hard', 'not difficult', 'easier', 'easiest',
    
    # Performance - good
    'passed', 'aced', 'scored', 'distinction', 'a1', 'a2', 'b3', 'b4',
    'did well', 'nailed', 'smashed', 'killed it',
    
    # Emotions - positive
    'happy', 'relieved', 'confident', 'satisfied', 'excited',
    'glad', 'proud', 'pleased', 'grateful', 'thankful',
    
    # Evaluation - positive
    'good', 'great', 'excellent', 'amazing', 'wonderful',
    'fantastic', 'awesome', 'best', 'love', 'enjoyed',
    'interesting', 'fair', 'reasonable', 'expected',
    
    # Expressions
    'thank god', 'finally', 'yay', 'woohoo', 'nice', 'lucky',
    'went well', 'no problem', 'piece of cake', 'breeze',
}

NEGATIVE_KEYWORDS = {
    # Exam difficulty - hard
    'hard', 'difficult', 'tough', 'killer', 'impossible',
    'insane', 'crazy', 'brutal', 'harder', 'hardest',
    'challenging', 'tricky', 'confusing', 'complicated',
    
    # Performance - bad
    'failed', 'fail', 'flunked', 'bombed', 'screwed',
    'destroyed', 'murdered', 'wrecked', 'messed up',
    'f9', 'u grade', 'ungraded', 'retained',
    
    # Emotions - negative
    'stressed', 'anxious', 'worried', 'scared', 'panic',
    'nervous', 'terrified', 'crying', 'cried', 'depressed',
    'sad', 'disappointed', 'upset', 'frustrated', 'angry',
    'devastated', 'hopeless', 'despair',
    
    # Expressions
    'wtf', 'what the', 'oh no', 'damn', 'shit', 'fml',
    'gg', 'gone case', 'rip', 'died', 'dead', 'sian',
    'no time', 'ran out of time', 'rushed', 'couldnt finish',
    'careless mistake', 'regret', 'jialat',
    
    # Evaluation - negative
    'unfair', 'unreasonable', 'unexpected', 'terrible',
    'horrible', 'awful', 'worst', 'bad', 'sucks', 'hate',
}

NEUTRAL_KEYWORDS = {
    # Uncertainty
    'okay', 'ok', 'alright', 'average', 'moderate', 'so-so',
    'not sure', 'dont know', 'unsure', 'maybe', 'perhaps',
    'depends', 'varies', 'mixed',
    
    # Questions
    'anyone', 'any tips', 'advice', 'help', 'question',
    'asking', 'wondering', 'curious', 'how to', 'what is',
    'when is', 'where', 'which', 'who',
    
    # Information seeking
    'waiting', 'results', 'release', 'schedule', 'timetable',
    'syllabus', 'topics', 'format', 'papers',
}

NEGATION_WORDS = ['not', 'no', 'never', 'neither', 'nobody', 'nothing',
                  'cant', 'couldnt', 'shouldnt', 'wouldnt', 'wont',
                  'dont', 'doesnt', 'didnt', 'isnt', 'arent', 'wasnt']


def load_all_data() -> List[Dict]:
    """Load all crawled data from JSON files."""
    all_data = []
    seen_ids = set()
    
    for filename in os.listdir(RAW_DATA_DIR):
        if filename.endswith('.json'):
            filepath = os.path.join(RAW_DATA_DIR, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    records = data.get('data', [])
                    
                    for record in records:
                        rid = record.get('id', '')
                        if rid and rid not in seen_ids:
                            seen_ids.add(rid)
                            all_data.append(record)
                    
                    print(f"Loaded {len(records)} records from {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    print(f"\nTotal unique records: {len(all_data)}")
    return all_data


def clean_text(text: str) -> str:
    """Clean text for sentiment analysis."""
    if not text:
        return ""
    
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    
    # Remove Reddit formatting
    text = re.sub(r'\[deleted\]|\[removed\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'/u/\w+|u/\w+', '', text)
    text = re.sub(r'/r/\w+|r/\w+', '', text)
    
    # Remove markdown
    text = re.sub(r'\*\*|\*|__|_|~~|`', '', text)
    text = re.sub(r'#{1,6}\s', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def get_text_hash(text: str) -> str:
    """Generate hash for duplicate detection."""
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    return hashlib.md5(normalized.encode()).hexdigest()


def remove_duplicates(data: List[Dict]) -> List[Dict]:
    """Remove duplicate texts."""
    seen_hashes = set()
    unique_data = []
    
    for record in data:
        text = record.get('text', '')
        if not text or len(text) < 20:
            continue
        
        text_hash = get_text_hash(text)
        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            unique_data.append(record)
    
    print(f"After deduplication: {len(unique_data)} records")
    return unique_data


def has_negation_before(text: str, keyword: str) -> bool:
    """Check if there's a negation word before the keyword."""
    text_lower = text.lower()
    keyword_lower = keyword.lower()
    
    idx = text_lower.find(keyword_lower)
    if idx == -1:
        return False
    
    # Check 5 words before
    before_text = text_lower[:idx]
    words = before_text.split()[-5:]
    
    return any(neg in words for neg in NEGATION_WORDS)


def classify_sentiment(text: str) -> Tuple[int, float]:
    """
    Classify text sentiment.
    Returns (label, confidence) where:
    - label: 0 = negative, 1 = neutral, 2 = positive
    - confidence: 0.0 to 1.0
    """
    text_lower = text.lower()
    
    positive_score = 0
    negative_score = 0
    neutral_score = 0
    
    # Check positive keywords
    for keyword in POSITIVE_KEYWORDS:
        if keyword in text_lower:
            if has_negation_before(text, keyword):
                negative_score += 1  # Negated positive -> negative
            else:
                positive_score += 1
    
    # Check negative keywords
    for keyword in NEGATIVE_KEYWORDS:
        if keyword in text_lower:
            if has_negation_before(text, keyword):
                positive_score += 1  # Negated negative -> positive
            else:
                negative_score += 1
    
    # Check neutral keywords
    for keyword in NEUTRAL_KEYWORDS:
        if keyword in text_lower:
            neutral_score += 1
    
    # Determine label
    total = positive_score + negative_score + neutral_score + 0.001
    
    if positive_score > negative_score and positive_score > neutral_score:
        label = 2  # Positive
        confidence = positive_score / total
    elif negative_score > positive_score and negative_score > neutral_score:
        label = 0  # Negative
        confidence = negative_score / total
    else:
        label = 1  # Neutral
        confidence = max(neutral_score, 1) / total
    
    return label, confidence


def label_data(data: List[Dict]) -> List[Dict]:
    """Add sentiment labels to data."""
    labeled_data = []
    
    for record in data:
        text = clean_text(record.get('text', ''))
        if not text or len(text) < 20:
            continue
        
        label, confidence = classify_sentiment(text)
        
        labeled_data.append({
            'text': text,
            'label': label,
            'confidence': confidence,
            'subreddit': record.get('subreddit', ''),
            'type': record.get('type', ''),
            'word_count': len(text.split())
        })
    
    return labeled_data


def balance_dataset(data: List[Dict], target_per_class: int = None) -> List[Dict]:
    """
    Balance the dataset to have equal numbers of each sentiment class.
    """
    # Group by label
    by_label = {0: [], 1: [], 2: []}
    for record in data:
        by_label[record['label']].append(record)
    
    print(f"\nBefore balancing:")
    print(f"  Negative (0): {len(by_label[0])}")
    print(f"  Neutral (1): {len(by_label[1])}")
    print(f"  Positive (2): {len(by_label[2])}")
    
    # Find the minimum count
    min_count = min(len(by_label[0]), len(by_label[1]), len(by_label[2]))
    
    if target_per_class:
        target = min(target_per_class, min_count)
    else:
        target = min_count
    
    # Sample from each class
    balanced = []
    for label in [0, 1, 2]:
        # Sort by confidence (higher first) and take top N
        sorted_records = sorted(by_label[label], key=lambda x: -x['confidence'])
        balanced.extend(sorted_records[:target])
    
    print(f"\nAfter balancing (target {target} per class):")
    by_label_after = Counter(r['label'] for r in balanced)
    print(f"  Negative (0): {by_label_after[0]}")
    print(f"  Neutral (1): {by_label_after[1]}")
    print(f"  Positive (2): {by_label_after[2]}")
    
    return balanced


def create_eval_dataset(min_records: int = 10000, min_words: int = 100000):
    """Create the final eval.xls dataset."""
    print("=" * 60)
    print("Creating Evaluation Dataset")
    print("=" * 60)
    
    # 1. Load all data
    print("\n[1/5] Loading data...")
    all_data = load_all_data()
    
    if not all_data:
        print("ERROR: No data found!")
        return None
    
    # 2. Remove duplicates
    print("\n[2/5] Removing duplicates...")
    unique_data = remove_duplicates(all_data)
    
    # 3. Label sentiment
    print("\n[3/5] Labeling sentiment...")
    labeled_data = label_data(unique_data)
    print(f"Labeled {len(labeled_data)} records")
    
    # 4. Balance dataset
    print("\n[4/5] Balancing dataset...")
    balanced_data = balance_dataset(labeled_data)
    
    # 5. Create eval.xls
    print("\n[5/5] Creating eval.xls...")
    
    # Standard sentiment benchmark format: text, label
    df = pd.DataFrame([
        {'text': r['text'], 'label': r['label']}
        for r in balanced_data
    ])
    
    # Calculate statistics
    total_records = len(df)
    total_words = sum(len(t.split()) for t in df['text'])
    label_counts = df['label'].value_counts().sort_index()
    
    print(f"\n" + "=" * 60)
    print("FINAL DATASET STATISTICS")
    print("=" * 60)
    print(f"Total records: {total_records}")
    print(f"Total words: {total_words}")
    print(f"\nLabel distribution:")
    print(f"  Negative (0): {label_counts.get(0, 0)}")
    print(f"  Neutral (1): {label_counts.get(1, 0)}")
    print(f"  Positive (2): {label_counts.get(2, 0)}")
    
    # Check requirements
    print(f"\nRequirements check:")
    print(f"  Records >= {min_records}: {'✓' if total_records >= min_records else '✗'} ({total_records})")
    print(f"  Words >= {min_words}: {'✓' if total_words >= min_words else '✗'} ({total_words})")
    
    # Save to Excel
    df.to_excel(OUTPUT_FILE, index=False, engine='openpyxl')
    print(f"\nSaved to: {OUTPUT_FILE}")
    
    # Also save as CSV for backup
    csv_file = OUTPUT_FILE.replace('.xls', '.csv')
    df.to_csv(csv_file, index=False)
    print(f"Backup CSV: {csv_file}")
    
    # Show sample
    print(f"\nSample records:")
    for label in [0, 1, 2]:
        sample = df[df['label'] == label].head(1)
        if not sample.empty:
            label_name = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}[label]
            text = sample.iloc[0]['text'][:100]
            print(f"\n  {label_name}: {text}...")
    
    return df


def main():
    """Main entry point."""
    df = create_eval_dataset(min_records=10000, min_words=100000)
    
    if df is not None:
        print("\n" + "=" * 60)
        print("Dataset creation complete!")
        print("=" * 60)


if __name__ == "__main__":
    main()
