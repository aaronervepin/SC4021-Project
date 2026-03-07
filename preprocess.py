"""
SC4021 Information Retrieval - Preprocessing & LLM Labeling Script
==================================================================
Features:
1. Singlish -> standard English normalization (dual-model)
2. 3-class sentiment labels (GPT-5 + Gemini -> majority vote with original)
3. Three subtasks: Sarcasm Detection / Subjectivity Detection / Emotion Detection
4. Checkpoint resume, batch processing, robust LLM output parsing

Usage:
    python preprocess_and_label.py --input eval.csv --output eval_preprocessed.csv
    python preprocess_and_label.py --input eval.csv --resume

API config follows the reference code pattern: api.txt for key, apiyi.com endpoint.

Author: Zhang
Date: 2026-03
"""

import os
import csv
import json
import time
import re
import argparse
import requests
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from collections import Counter
from datetime import datetime


# ============================================================================
# CLI Arguments
# ============================================================================
parser = argparse.ArgumentParser(
    description="Preprocess eval.csv: Singlish normalization + dual-LLM labeling + subtasks",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--input', type=str, default='eval.csv',
                    help='Input CSV file (must have "text" and "label" columns)')
parser.add_argument('--output', type=str, default='eval_preprocessed.csv',
                    help='Output preprocessed CSV file')
parser.add_argument('--api_key_path', type=str, default='api.txt',
                    help='Path to API key file')
parser.add_argument('--base_url', type=str,
                    default='https://api.apiyi.com/v1/chat/completions',
                    help='API base URL')
parser.add_argument('--model_a', type=str, default='gpt-5',
                    help='First LLM model (e.g. gpt-5)')
parser.add_argument('--model_b', type=str, default='gemini-2.5-pro',
                    help='Second LLM model (e.g. gemini-2.5-pro)')
parser.add_argument('--batch_size', type=int, default=5,
                    help='Number of texts to process per API call')
parser.add_argument('--max_samples', type=int, default=None,
                    help='Limit number of rows to process (None=all)')
parser.add_argument('--checkpoint_path', type=str, default='preprocess_checkpoint.json',
                    help='Checkpoint file for resume')
parser.add_argument('--resume', action='store_true',
                    help='Resume from last checkpoint')
parser.add_argument('--max_retries', type=int, default=3,
                    help='Max API retry attempts')
parser.add_argument('--retry_delay', type=float, default=2.0,
                    help='Base delay between retries (exponential backoff)')
parser.add_argument('--request_delay', type=float, default=1.0,
                    help='Delay between API requests')
parser.add_argument('--timeout', type=int, default=120,
                    help='API request timeout in seconds')
parser.add_argument('--temperature', type=float, default=0.3,
                    help='LLM temperature (low for consistent labeling)')
parser.add_argument('--debug', action='store_true',
                    help='Verbose debug output')

args = parser.parse_args()


# ============================================================================
# Label Mappings
# ============================================================================
SENTIMENT_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}
SENTIMENT_REVERSE = {"negative": 0, "neutral": 1, "positive": 2}
SARCASM_MAP = {"not_sarcastic": 0, "sarcastic": 1}
SUBJECTIVITY_MAP = {"objective": 0, "subjective": 1}
EMOTION_MAP = {
    "neutral": 0, "anger": 1, "joy": 2,
    "sadness": 3, "surprise": 4, "fear": 5, "disgust": 6
}


# ============================================================================
# API Calls (strictly following reference code pattern)
# ============================================================================

def load_api_key(api_key_path: str) -> Optional[str]:
    """Load API Key from file"""
    if os.path.exists(api_key_path):
        with open(api_key_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return None


def call_llm_api(
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_retries: int = 3,
    timeout: int = 120
) -> Optional[str]:
    """
    Call LLM API (strictly following reference call_vision_api pattern)
    Uses requests.post with Bearer auth to apiyi.com endpoint.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 2000,
        "temperature": temperature
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(
                base_url, headers=headers,
                json=payload, timeout=timeout
            )
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            elif response.status_code == 429:
                wait_time = args.retry_delay * (2 ** attempt)
                print(f"      Rate limited, waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                print(f"      API error {response.status_code}: "
                      f"{response.text[:200]}")
                if attempt < max_retries - 1:
                    time.sleep(args.retry_delay * (2 ** attempt))
        except requests.exceptions.Timeout:
            print(f"      Timeout (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(args.retry_delay)
        except requests.exceptions.RequestException as e:
            print(f"      Request failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(args.retry_delay * (2 ** attempt))

    return None


# ============================================================================
# Prompt Templates
# ============================================================================

NORMALIZE_AND_LABEL_SYSTEM = """You are an expert NLP annotator for Singaporean English (Singlish) text analysis.
You will process a batch of texts. For EACH text, provide ALL of the following in strict JSON format.

Your tasks:
1. **Singlish Normalization**: Rewrite the text into standard English. Expand Singlish particles (lah, lor, leh, sia, meh, etc.), slang, abbreviations, and colloquialisms into proper English while preserving the original meaning. Fix grammar but keep the tone.
2. **Sentiment Label**: Classify overall sentiment as exactly one of: Negative, Neutral, Positive.
3. **Sarcasm Detection**: Is the text sarcastic? Exactly one of: sarcastic, not_sarcastic.
4. **Subjectivity Detection**: Is the text subjective (opinion/personal) or objective (factual/informational)? Exactly one of: subjective, objective.
5. **Emotion Detection**: Primary emotion. Exactly one of: neutral, anger, joy, sadness, surprise, fear, disgust.

CRITICAL OUTPUT FORMAT - respond with ONLY a JSON array, no markdown fences, no explanation:
[
  {
    "idx": 0,
    "normalized_text": "...",
    "sentiment": "Negative|Neutral|Positive",
    "sarcasm": "sarcastic|not_sarcastic",
    "subjectivity": "subjective|objective",
    "emotion": "neutral|anger|joy|sadness|surprise|fear|disgust"
  },
  ...
]

Rules:
- Output ONLY valid JSON. No markdown backticks, no preamble, no explanation.
- Use EXACT label strings as specified above (case-sensitive for sentiment).
- Process ALL texts in the batch, maintaining idx order.
- If a text is ambiguous, make your best judgment.
"""


def build_batch_prompt(texts: List[Tuple[int, str]], max_chars: int = 500) -> str:
    """Build user prompt for a batch of texts"""
    lines = []
    for idx, text in texts:
        truncated = text[:max_chars] + "..." if len(text) > max_chars else text
        truncated = truncated.replace('\\', '\\\\').replace('"', '\\"')
        lines.append(f'[Text {idx}]: """{truncated}"""')
    return "Process the following texts:\n\n" + "\n\n".join(lines)


# ============================================================================
# LLM Output Parsing (robust normalization)
# ============================================================================

def parse_llm_response(response_text: str, expected_count: int) -> List[Dict]:
    """
    Parse and normalize LLM JSON output.
    Handles: markdown fences, extra text, trailing commas, etc.
    """
    if not response_text:
        return []

    # Step 1: strip markdown code blocks
    cleaned = response_text.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    cleaned = cleaned.strip()

    # Step 2: extract JSON array
    start = cleaned.find('[')
    end = cleaned.rfind(']')
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start:end + 1]

    # Step 3: try parsing
    try:
        results = json.loads(cleaned)
        if isinstance(results, list):
            return _validate_results(results, expected_count)
    except json.JSONDecodeError:
        pass

    # Step 4: fix common JSON errors (trailing commas)
    try:
        fixed = re.sub(r',\s*([}\]])', r'\1', cleaned)
        results = json.loads(fixed)
        if isinstance(results, list):
            return _validate_results(results, expected_count)
    except json.JSONDecodeError:
        pass

    # Step 5: extract individual objects
    try:
        objects = re.findall(r'\{[^{}]+\}', cleaned)
        results = [json.loads(obj) for obj in objects]
        if results:
            return _validate_results(results, expected_count)
    except (json.JSONDecodeError, Exception):
        pass

    if args.debug:
        print(f"    [DEBUG] Failed to parse: {response_text[:200]}")
    return []


def _validate_results(results: List[Dict], expected_count: int) -> List[Dict]:
    """Validate and normalize parsed results"""
    validated = []
    for r in results:
        entry = {}
        entry['normalized_text'] = r.get('normalized_text', '')

        # Sentiment normalization
        sent = str(r.get('sentiment', '')).strip().lower()
        if sent in SENTIMENT_REVERSE:
            entry['sentiment'] = sent.capitalize()
        elif sent in ['neg', 'negative']:
            entry['sentiment'] = 'Negative'
        elif sent in ['pos', 'positive']:
            entry['sentiment'] = 'Positive'
        elif sent in ['neu', 'neutral']:
            entry['sentiment'] = 'Neutral'
        else:
            entry['sentiment'] = 'Neutral'

        # Sarcasm
        sarc = str(r.get('sarcasm', '')).strip().lower()
        if 'not' in sarc or sarc == 'no':
            entry['sarcasm'] = 'not_sarcastic'
        elif 'sarcas' in sarc or sarc == 'yes':
            entry['sarcasm'] = 'sarcastic'
        else:
            entry['sarcasm'] = 'not_sarcastic'

        # Subjectivity
        subj = str(r.get('subjectivity', '')).strip().lower()
        if 'obj' in subj:
            entry['subjectivity'] = 'objective'
        elif 'subj' in subj:
            entry['subjectivity'] = 'subjective'
        else:
            entry['subjectivity'] = 'subjective'

        # Emotion
        emo = str(r.get('emotion', '')).strip().lower()
        if emo in EMOTION_MAP:
            entry['emotion'] = emo
        else:
            for valid_emo in EMOTION_MAP:
                if valid_emo in emo:
                    entry['emotion'] = valid_emo
                    break
            else:
                entry['emotion'] = 'neutral'

        entry['idx'] = r.get('idx', len(validated))
        validated.append(entry)

    return validated


# ============================================================================
# Majority Voting
# ============================================================================

def majority_vote_sentiment(
    original_label: int,
    model_a_label: str,
    model_b_label: str
) -> Tuple[int, float]:
    """
    3-way majority vote: original_label + model_a + model_b
    Returns: (final_label, agreement_score)
    """
    orig_str = SENTIMENT_MAP[original_label]
    votes = [orig_str, model_a_label, model_b_label]
    vote_counts = Counter(votes)
    winner = vote_counts.most_common(1)[0]
    final_label = SENTIMENT_REVERSE[winner[0].lower()]
    agreement = winner[1] / 3.0
    return final_label, round(agreement, 4)


def majority_vote_subtask(label_a: str, label_b: str) -> Tuple[str, float]:
    """Dual-model vote for subtasks; if disagreement take model_a"""
    if label_a == label_b:
        return label_a, 1.0
    else:
        return label_a, 0.5


# ============================================================================
# Checkpoint
# ============================================================================

def save_checkpoint(checkpoint_path: str, processed_indices: set, results: dict):
    data = {
        'processed_indices': sorted(list(processed_indices)),
        'results': results,
        'timestamp': datetime.now().isoformat()
    }
    temp = checkpoint_path + '.tmp'
    with open(temp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(temp, checkpoint_path)


def load_checkpoint(checkpoint_path: str) -> Tuple[set, dict]:
    if not os.path.exists(checkpoint_path):
        return set(), {}
    try:
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        indices = set(data.get('processed_indices', []))
        results = data.get('results', {})
        print(f"Loaded checkpoint: {len(indices)} rows already processed")
        return indices, results
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return set(), {}


# ============================================================================
# Main
# ============================================================================

def process_batch(
    api_key: str,
    batch: List[Tuple[int, str]],
    model: str,
    base_url: str
) -> List[Dict]:
    """Process one batch of texts with a given model"""
    user_prompt = build_batch_prompt(batch)
    response = call_llm_api(
        api_key, base_url, model,
        NORMALIZE_AND_LABEL_SYSTEM,
        user_prompt,
        temperature=args.temperature,
        max_retries=args.max_retries,
        timeout=args.timeout
    )
    if response is None:
        return []
    results = parse_llm_response(response, len(batch))
    return results


def main():
    print("=" * 70)
    print("SC4021 Preprocessing & Dual-LLM Labeling")
    print("=" * 70)
    print(f"Input:    {args.input}")
    print(f"Output:   {args.output}")
    print(f"Model A:  {args.model_a}")
    print(f"Model B:  {args.model_b}")
    print(f"Batch:    {args.batch_size}")
    print()

    # Load API key
    api_key = load_api_key(args.api_key_path)
    if api_key is None:
        print(f"Error: Cannot load API key from {args.api_key_path}")
        return
    print("API key loaded")

    # Load data
    rows = []
    with open(args.input, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    print(f"Loaded {len(rows)} rows from {args.input}")

    if args.max_samples:
        rows = rows[:args.max_samples]
        print(f"  (Limited to first {args.max_samples} rows)")

    # Load checkpoint
    processed_indices = set()
    results_store = {}
    if args.resume:
        processed_indices, results_store = load_checkpoint(args.checkpoint_path)

    # Determine unprocessed rows
    total = len(rows)
    unprocessed = [(i, rows[i]['text']) for i in range(total)
                   if i not in processed_indices]
    print(f"Remaining: {len(unprocessed)} rows to process\n")

    # Build batches
    batches = []
    for start in range(0, len(unprocessed), args.batch_size):
        batches.append(unprocessed[start:start + args.batch_size])

    try:
        for batch_idx, batch in enumerate(batches):
            indices_in_batch = [item[0] for item in batch]
            print(f"Batch {batch_idx + 1}/{len(batches)} "
                  f"(rows {indices_in_batch[0]}-{indices_in_batch[-1]})...", end=' ')

            # ---- Model A ----
            results_a = process_batch(api_key, batch, args.model_a, args.base_url)
            time.sleep(args.request_delay)

            # ---- Model B ----
            results_b = process_batch(api_key, batch, args.model_b, args.base_url)
            time.sleep(args.request_delay)

            # ---- Merge results ----
            for j, (global_idx, text) in enumerate(batch):
                orig_label = int(rows[global_idx]['label'])

                ra = results_a[j] if j < len(results_a) else {}
                rb = results_b[j] if j < len(results_b) else {}

                # Normalized text
                norm_a = ra.get('normalized_text', '')
                norm_b = rb.get('normalized_text', '')
                normalized_text = norm_a if norm_a else (norm_b if norm_b else text)

                # Sentiment: 3-way majority vote
                sent_a = ra.get('sentiment', SENTIMENT_MAP[orig_label])
                sent_b = rb.get('sentiment', SENTIMENT_MAP[orig_label])
                final_sentiment, sent_agreement = majority_vote_sentiment(
                    orig_label, sent_a, sent_b
                )

                # Sarcasm
                sarc_a = ra.get('sarcasm', 'not_sarcastic')
                sarc_b = rb.get('sarcasm', 'not_sarcastic')
                final_sarcasm, sarc_agreement = majority_vote_subtask(sarc_a, sarc_b)

                # Subjectivity
                subj_a = ra.get('subjectivity', 'subjective')
                subj_b = rb.get('subjectivity', 'subjective')
                final_subjectivity, subj_agreement = majority_vote_subtask(subj_a, subj_b)

                # Emotion
                emo_a = ra.get('emotion', 'neutral')
                emo_b = rb.get('emotion', 'neutral')
                final_emotion, emo_agreement = majority_vote_subtask(emo_a, emo_b)

                result = {
                    'original_text': text,
                    'normalized_text': normalized_text,
                    'original_label': orig_label,
                    'sentiment_model_a': sent_a,
                    'sentiment_model_b': sent_b,
                    'sentiment_final': final_sentiment,
                    'sentiment_agreement': sent_agreement,
                    'sarcasm_model_a': sarc_a,
                    'sarcasm_model_b': sarc_b,
                    'sarcasm_final': SARCASM_MAP.get(final_sarcasm, 0),
                    'sarcasm_agreement': sarc_agreement,
                    'subjectivity_model_a': subj_a,
                    'subjectivity_model_b': subj_b,
                    'subjectivity_final': SUBJECTIVITY_MAP.get(final_subjectivity, 1),
                    'subjectivity_agreement': subj_agreement,
                    'emotion_model_a': emo_a,
                    'emotion_model_b': emo_b,
                    'emotion_final': EMOTION_MAP.get(final_emotion, 0),
                    'emotion_agreement': emo_agreement,
                }

                results_store[str(global_idx)] = result
                processed_indices.add(global_idx)

            mark_a = 'ok' if results_a else 'FAIL'
            mark_b = 'ok' if results_b else 'FAIL'
            print(f"A:{mark_a} B:{mark_b}")

            # Periodic checkpoint
            if (batch_idx + 1) % 10 == 0:
                save_checkpoint(args.checkpoint_path, processed_indices, results_store)
                print(f"  [Checkpoint: {len(processed_indices)}/{total}]")

    except KeyboardInterrupt:
        print("\n\nInterrupted, saving checkpoint...")
        save_checkpoint(args.checkpoint_path, processed_indices, results_store)
        print(f"Saved. Resume with --resume. ({len(processed_indices)}/{total})")
        return
    except Exception as e:
        print(f"\n\nError: {e}")
        traceback.print_exc()
        save_checkpoint(args.checkpoint_path, processed_indices, results_store)
        raise

    # Final checkpoint
    save_checkpoint(args.checkpoint_path, processed_indices, results_store)

    # ============================================================
    # Write output CSV
    # ============================================================
    print(f"\nWriting output to {args.output}...")

    fieldnames = [
        'original_text', 'normalized_text', 'original_label',
        'sentiment_model_a', 'sentiment_model_b',
        'sentiment_final', 'sentiment_agreement',
        'sarcasm_model_a', 'sarcasm_model_b',
        'sarcasm_final', 'sarcasm_agreement',
        'subjectivity_model_a', 'subjectivity_model_b',
        'subjectivity_final', 'subjectivity_agreement',
        'emotion_model_a', 'emotion_model_b',
        'emotion_final', 'emotion_agreement',
    ]

    with open(args.output, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(total):
            key = str(i)
            if key in results_store:
                writer.writerow(results_store[key])
            else:
                writer.writerow({
                    'original_text': rows[i]['text'],
                    'normalized_text': rows[i]['text'],
                    'original_label': rows[i]['label'],
                    'sentiment_model_a': '', 'sentiment_model_b': '',
                    'sentiment_final': rows[i]['label'],
                    'sentiment_agreement': 0.33,
                    'sarcasm_model_a': '', 'sarcasm_model_b': '',
                    'sarcasm_final': 0, 'sarcasm_agreement': 0,
                    'subjectivity_model_a': '', 'subjectivity_model_b': '',
                    'subjectivity_final': 1, 'subjectivity_agreement': 0,
                    'emotion_model_a': '', 'emotion_model_b': '',
                    'emotion_final': 0, 'emotion_agreement': 0,
                })

    # ============================================================
    # Print Statistics
    # ============================================================
    print("\n" + "=" * 60)
    print("Preprocessing Statistics")
    print("=" * 60)
    print(f"Total rows:     {total}")
    print(f"Processed:      {len(processed_indices)}")

    if results_store:
        agreements = [v['sentiment_agreement'] for v in results_store.values()
                      if isinstance(v.get('sentiment_agreement'), (int, float))]
        if agreements:
            avg_agr = sum(agreements) / len(agreements)
            full_agr = sum(1 for a in agreements if a >= 0.99)
            print(f"\nSentiment Inter-Annotator Agreement:")
            print(f"  Average:       {avg_agr:.2%}")
            print(f"  Full (3/3):    {full_agr}/{len(agreements)} "
                  f"({full_agr/len(agreements):.1%})")

        changed = sum(1 for v in results_store.values()
                      if v.get('sentiment_final') != v.get('original_label'))
        print(f"  Label changed: {changed}/{len(results_store)} "
              f"({changed/max(len(results_store),1):.1%})")

        for task_name, task_key in [('Sarcasm', 'sarcasm_final'),
                                     ('Subjectivity', 'subjectivity_final'),
                                     ('Emotion', 'emotion_final')]:
            dist = Counter(v.get(task_key) for v in results_store.values())
            print(f"\n{task_name} distribution: {dict(dist)}")

    print(f"\nOutput saved to: {args.output}")
    print("Done!")


if __name__ == '__main__':
    main()