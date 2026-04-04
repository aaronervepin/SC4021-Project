"""
SC4021 Information Retrieval - IR Label Enrichment Script
=========================================================
Reads crawled_clean_with_predictions.csv, calls dual-LLM to extract
structured labels from each post's body+title, then appends new columns.

New columns added (all designed to improve TF-IDF & IR):
  1. subjectivity       : objective / subjective
  2. sarcasm            : sarcastic / not_sarcastic
  3. education_level    : psle / nlevel / olevel / alevel / jc / poly / ite / uni / masters / general
  4. subject            : english / math / physics / chemistry / biology / history / geography /
                          economics / gp / chinese / literature / computing / art / music / poa / none
  5. intent             : advice_seeking / advice_giving / rant / discussion / resource_sharing /
                          experience_sharing / question / celebration / commiseration / comparison
  6. topic              : exam_prep / results / school_choice / study_tips / mental_health /
                          career / daily_life / admissions / academic_policy / extracurricular / other
  7. emotion            : neutral / anxiety / joy / sadness / anger / frustration / hope / fear / disgust / surprise
  8. school_mentioned   : comma-separated school names or "none"
  9. specificity        : high / medium / low   (how specific/actionable the post is)
  10. temporal_context   : before_exam / during_exam / after_exam / results_day / enrollment_period /
                          semester / holiday / general
  11. year               : p6 / sec1 / sec2 / sec3 / sec4 / sec5 / jc1 / jc2 /
                          year1 / year2 / year3 / year4 / none

API config follows the reference code pattern: api.txt for key, apiyi.com endpoint.

Usage:
    python ir_label_enrichment.py --input crawled_clean_with_predictions.csv --output enriched.csv
    python ir_label_enrichment.py --input crawled_clean_with_predictions.csv --resume

Author: Zhang
Date: 2026-04
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
    description="IR Label Enrichment: extract structured labels from SGExams/NUS posts",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--input', type=str, default='crawled_clean_with_predictions.csv',
                    help='Input CSV file')
parser.add_argument('--output', type=str, default='crawled_enriched.csv',
                    help='Output enriched CSV file')
parser.add_argument('--api_key_path', type=str, default='api.txt',
                    help='Path to API key file')
parser.add_argument('--base_url', type=str,
                    default='https://api.apiyi.com/v1/chat/completions',
                    help='API base URL')
parser.add_argument('--model_a', type=str, default='gpt-5',
                    help='First LLM model')
parser.add_argument('--model_b', type=str, default='gemini-2.5-pro',
                    help='Second LLM model')
parser.add_argument('--batch_size', type=int, default=5,
                    help='Number of texts to process per API call')
parser.add_argument('--max_samples', type=int, default=None,
                    help='Limit number of rows to process (None=all)')
parser.add_argument('--checkpoint_path', type=str, default='ir_enrich_checkpoint.json',
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
parser.add_argument('--temperature', type=float, default=0.2,
                    help='LLM temperature (low for consistent labeling)')
parser.add_argument('--debug', action='store_true',
                    help='Verbose debug output')

args = parser.parse_args()


# ============================================================================
# Valid Label Enums (for validation & fallback)
# ============================================================================

VALID_LABELS = {
    "subjectivity": ["objective", "subjective"],
    "sarcasm": ["sarcastic", "not_sarcastic"],
    "education_level": [
        "psle", "nlevel", "olevel", "alevel", "jc", "poly", "ite",
        "uni", "masters", "general"
    ],
    "subject": [
        "english", "math", "physics", "chemistry", "biology",
        "history", "geography", "economics", "gp", "chinese",
        "literature", "computing", "art", "music", "poa", "none"
    ],
    "intent": [
        "advice_seeking", "advice_giving", "rant", "discussion",
        "resource_sharing", "experience_sharing", "question",
        "celebration", "commiseration", "comparison"
    ],
    "topic": [
        "exam_prep", "results", "school_choice", "study_tips",
        "mental_health", "career", "daily_life", "admissions",
        "academic_policy", "extracurricular", "other"
    ],
    "emotion": [
        "neutral", "anxiety", "joy", "sadness", "anger",
        "frustration", "hope", "fear", "disgust", "surprise"
    ],
    "specificity": ["high", "medium", "low"],
    "temporal_context": [
        "before_exam", "during_exam", "after_exam", "results_day",
        "enrollment_period", "semester", "holiday", "general"
    ],
    "year": [
        "p6",
        "sec1", "sec2", "sec3", "sec4", "sec5",
        "jc1", "jc2",
        "year1", "year2", "year3", "year4",
        "none"
    ],
}

# Default fallback values
DEFAULTS = {
    "subjectivity": "subjective",
    "sarcasm": "not_sarcastic",
    "education_level": "general",
    "subject": "none",
    "intent": "discussion",
    "topic": "other",
    "emotion": "neutral",
    "school_mentioned": "none",
    "specificity": "medium",
    "temporal_context": "general",
    "year": "none",
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
    temperature: float = 0.2,
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
        "max_tokens": 4000,
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

IR_LABEL_SYSTEM_PROMPT = """You are an expert NLP annotator for Singapore education forum posts (r/SGExams, r/nus).
You will process a batch of posts. For EACH post, extract structured labels to improve Information Retrieval.

Context: These posts are from Singaporean students discussing exams (PSLE, N/O/A Levels), school life, JC/Poly/ITE/Uni, admissions, and academic topics. Many posts use Singlish (lah, lor, sia, meh etc.).

For EACH post, provide ALL of the following labels:

1. **subjectivity**: Is this factual/informational or opinion/personal?
   Values: "objective" | "subjective"

2. **sarcasm**: Is the author being sarcastic or ironic?
   Values: "sarcastic" | "not_sarcastic"

3. **education_level**: Which education level does this post primarily relate to?
   Values: "psle" | "nlevel" | "olevel" | "alevel" | "jc" | "poly" | "ite" | "uni" | "masters" | "general"
   - Use the MOST SPECIFIC applicable level. "jc" = currently in JC life, "alevel" = about A Level exams specifically.
   - If the title mentions "O Level" or "A Level" explicitly, use that.
   - "general" only if truly no specific level can be inferred.

4. **subject**: Which academic subject is primarily discussed?
   Values: "english" | "math" | "physics" | "chemistry" | "biology" | "history" | "geography" | "economics" | "gp" | "chinese" | "literature" | "computing" | "art" | "music" | "poa" | "none"
   - "gp" = General Paper (A Level). "poa" = Principles of Accounting.
   - "none" if no specific subject is the focus.

5. **intent**: What is the author trying to do?
   Values: "advice_seeking" | "advice_giving" | "rant" | "discussion" | "resource_sharing" | "experience_sharing" | "question" | "celebration" | "commiseration" | "comparison"

6. **topic**: What broader category does this post belong to?
   Values: "exam_prep" | "results" | "school_choice" | "study_tips" | "mental_health" | "career" | "daily_life" | "admissions" | "academic_policy" | "extracurricular" | "other"

7. **emotion**: What is the dominant emotion expressed?
   Values: "neutral" | "anxiety" | "joy" | "sadness" | "anger" | "frustration" | "hope" | "fear" | "disgust" | "surprise"

8. **school_mentioned**: List ALL schools/institutions explicitly mentioned. Use standard abbreviations.
   Common ones: "RI" (Raffles Institution), "HCI" (Hwa Chong), "VJC" (Victoria JC), "ACJC", "NJC", "NYJC", "CJC", "SAJC", "TJC", "TMJC", "EJC" (Eunoia JC), "ASRJC", "DHS" (Dunman High), "NUS", "NTU", "SMU", "SUTD", "SIT", "SUSS", "SP" (Singapore Poly), "NP" (Ngee Ann Poly), "TP" (Temasek Poly), "NYP", "RP" (Republic Poly).
   If none mentioned, use "none". Comma-separated if multiple.

9. **specificity**: How specific/actionable is the content for someone searching?
   Values: "high" (specific advice, concrete data, detailed experience) | "medium" (some useful info) | "low" (very brief, vague, or just emotional reaction)

10. **temporal_context**: What time period relative to exams does this relate to?
    Values: "before_exam" | "during_exam" | "after_exam" | "results_day" | "enrollment_period" | "semester" | "holiday" | "general"

11. **year**: Which academic year within their education level does this post relate to?
    Values: "p6" | "sec1" | "sec2" | "sec3" | "sec4" | "sec5" | "jc1" | "jc2" | "year1" | "year2" | "year3" | "year4" | "none"
    - "p6" for Primary 6 / PSLE level posts.
    - "sec1"–"sec5" for Secondary school year levels (Sec 5 = N-level repeat year).
    - "jc1" / "jc2" for Junior College year levels.
    - "year1"–"year4" for Poly, ITE, or University year levels.
    - "none" if no specific year within the level can be inferred.

CRITICAL OUTPUT FORMAT - respond with ONLY a JSON array, no markdown fences, no explanation:
[
  {
    "idx": 0,
    "subjectivity": "subjective",
    "sarcasm": "not_sarcastic",
    "education_level": "olevel",
    "subject": "english",
    "intent": "advice_seeking",
    "topic": "exam_prep",
    "emotion": "anxiety",
    "school_mentioned": "none",
    "specificity": "medium",
    "temporal_context": "before_exam",
    "year": "sec4"
  },
  ...
]

Rules:
- Output ONLY valid JSON. No markdown backticks, no preamble, no explanation.
- Use EXACT label strings as specified above.
- Process ALL texts in the batch, maintaining idx order.
- Use both the TITLE and BODY to determine labels. Title often contains the exam level.
- If ambiguous, make your best judgment — never leave a field empty.
"""


def build_batch_prompt(batch: List[Tuple[int, str, str]], max_body_chars: int = 500) -> str:
    """
    Build user prompt for a batch of (idx, title, body) tuples.
    We send both title and body since title often contains key context.
    """
    lines = []
    for idx, title, body in batch:
        truncated_body = body[:max_body_chars] + "..." if len(body) > max_body_chars else body
        truncated_body = truncated_body.replace('\\', '\\\\').replace('"', '\\"')
        safe_title = title.replace('\\', '\\\\').replace('"', '\\"')
        lines.append(
            f'[Post {idx}]:\n'
            f'  Title: """{safe_title}"""\n'
            f'  Body: """{truncated_body}"""'
        )
    return "Process the following posts:\n\n" + "\n\n".join(lines)


# ============================================================================
# LLM Output Parsing (robust)
# ============================================================================

def parse_llm_response(response_text: str, expected_count: int) -> List[Dict]:
    """Parse and normalize LLM JSON output with robust fallbacks."""
    if not response_text:
        return []

    cleaned = response_text.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    cleaned = cleaned.strip()

    # Extract JSON array
    start = cleaned.find('[')
    end = cleaned.rfind(']')
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start:end + 1]

    # Attempt 1: direct parse
    try:
        results = json.loads(cleaned)
        if isinstance(results, list):
            return _validate_results(results)
    except json.JSONDecodeError:
        pass

    # Attempt 2: fix trailing commas
    try:
        fixed = re.sub(r',\s*([}\]])', r'\1', cleaned)
        results = json.loads(fixed)
        if isinstance(results, list):
            return _validate_results(results)
    except json.JSONDecodeError:
        pass

    # Attempt 3: extract individual JSON objects
    try:
        objects = re.findall(r'\{[^{}]+\}', cleaned)
        results = [json.loads(obj) for obj in objects]
        if results:
            return _validate_results(results)
    except (json.JSONDecodeError, Exception):
        pass

    if args.debug:
        print(f"    [DEBUG] Failed to parse LLM response: {response_text[:300]}")
    return []


def _normalize_field(value: str, field_name: str) -> str:
    """Normalize a single field value against valid labels."""
    v = str(value).strip().lower().replace(' ', '_').replace('-', '_')

    # Direct match
    if field_name in VALID_LABELS:
        if v in VALID_LABELS[field_name]:
            return v
        # Fuzzy: check if any valid label is a substring
        for valid in VALID_LABELS[field_name]:
            if valid in v or v in valid:
                return valid

    return DEFAULTS.get(field_name, v)


def _validate_results(results: List[Dict]) -> List[Dict]:
    """Validate and normalize all parsed results."""
    validated = []
    for r in results:
        entry = {"idx": r.get("idx", len(validated))}

        # Normalize each label field
        for field in VALID_LABELS:
            raw = r.get(field, DEFAULTS[field])
            entry[field] = _normalize_field(raw, field)

        # school_mentioned is free-text, just clean it
        schools = str(r.get("school_mentioned", "none")).strip()
        if not schools or schools.lower() in ["none", "n/a", "na", ""]:
            entry["school_mentioned"] = "none"
        else:
            entry["school_mentioned"] = schools

        validated.append(entry)

    return validated


# ============================================================================
# Dual-Model Voting
# ============================================================================

def merge_labels(result_a: Dict, result_b: Dict) -> Dict:
    """
    Merge labels from two models.
    Strategy: if both agree -> use that; if disagree -> prefer model_a (gpt-5).
    Also record per-field agreement.
    """
    merged = {}
    fields = list(VALID_LABELS.keys()) + ["school_mentioned"]
    total_agree = 0

    for field in fields:
        val_a = result_a.get(field, DEFAULTS.get(field, ""))
        val_b = result_b.get(field, DEFAULTS.get(field, ""))

        merged[f"{field}_a"] = val_a
        merged[f"{field}_b"] = val_b

        if field == "school_mentioned":
            # Union of mentioned schools
            schools_a = set(s.strip() for s in val_a.split(",") if s.strip().lower() != "none")
            schools_b = set(s.strip() for s in val_b.split(",") if s.strip().lower() != "none")
            all_schools = schools_a | schools_b
            merged[field] = ", ".join(sorted(all_schools)) if all_schools else "none"
            if schools_a == schools_b:
                total_agree += 1
        else:
            if val_a == val_b:
                merged[field] = val_a
                total_agree += 1
            else:
                merged[field] = val_a  # prefer model_a

    merged["label_agreement"] = round(total_agree / len(fields), 4)
    return merged


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
# Main Processing
# ============================================================================

def process_batch(
    api_key: str,
    batch: List[Tuple[int, str, str]],
    model: str,
    base_url: str
) -> List[Dict]:
    """Process one batch with a given model."""
    user_prompt = build_batch_prompt(batch)
    response = call_llm_api(
        api_key, base_url, model,
        IR_LABEL_SYSTEM_PROMPT,
        user_prompt,
        temperature=args.temperature,
        max_retries=args.max_retries,
        timeout=args.timeout
    )
    if response is None:
        return []
    return parse_llm_response(response, len(batch))


def main():
    print("=" * 70)
    print("SC4021 IR Label Enrichment - Dual-LLM Labeling")
    print("=" * 70)
    print(f"Input:    {args.input}")
    print(f"Output:   {args.output}")
    print(f"Model A:  {args.model_a}")
    print(f"Model B:  {args.model_b}")
    print(f"Batch:    {args.batch_size}")
    print()

    # --- Label Taxonomy Summary ---
    print("Labels to extract:")
    for field, values in VALID_LABELS.items():
        print(f"  {field:20s} -> {', '.join(values)}")
    print(f"  {'school_mentioned':20s} -> free-text (comma-separated)")
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
        original_fieldnames = reader.fieldnames
        for r in reader:
            rows.append(r)
    print(f"Loaded {len(rows)} rows from {args.input}")
    print(f"Original columns: {original_fieldnames}")

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
    unprocessed = [
        (i, rows[i]['title'], rows[i]['body'])
        for i in range(total)
        if i not in processed_indices
    ]
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

            # ---- Merge ----
            for j, (global_idx, title, body) in enumerate(batch):
                ra = results_a[j] if j < len(results_a) else {}
                rb = results_b[j] if j < len(results_b) else {}

                merged = merge_labels(ra, rb)
                results_store[str(global_idx)] = merged
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
    # Write output CSV: original columns + new label columns
    # ============================================================
    print(f"\nWriting output to {args.output}...")

    # New columns to append
    new_label_fields = [
        "subjectivity", "sarcasm", "education_level", "subject",
        "intent", "topic", "emotion", "school_mentioned",
        "specificity", "temporal_context", "year", "label_agreement",
    ]
    # Also include per-model raw labels for transparency
    per_model_fields = []
    for field in list(VALID_LABELS.keys()) + ["school_mentioned"]:
        per_model_fields.append(f"{field}_a")
        per_model_fields.append(f"{field}_b")

    all_new_fields = new_label_fields + per_model_fields
    fieldnames = list(original_fieldnames) + all_new_fields

    with open(args.output, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(total):
            row = dict(rows[i])
            key = str(i)
            if key in results_store:
                for col in all_new_fields:
                    row[col] = results_store[key].get(col, "")
            else:
                # Unprocessed: fill defaults
                for col in new_label_fields:
                    row[col] = DEFAULTS.get(col, "")
                for col in per_model_fields:
                    row[col] = ""
            writer.writerow(row)

    # ============================================================
    # Statistics
    # ============================================================
    print("\n" + "=" * 60)
    print("IR Label Enrichment Statistics")
    print("=" * 60)
    print(f"Total rows:     {total}")
    print(f"Processed:      {len(processed_indices)}")

    if results_store:
        # Agreement
        agreements = [v['label_agreement'] for v in results_store.values()
                      if isinstance(v.get('label_agreement'), (int, float))]
        if agreements:
            avg_agr = sum(agreements) / len(agreements)
            full_agr = sum(1 for a in agreements if a >= 0.99)
            print(f"\nOverall inter-model agreement:")
            print(f"  Average:       {avg_agr:.2%}")
            print(f"  Full (10/10):  {full_agr}/{len(agreements)} "
                  f"({full_agr / len(agreements):.1%})")

        # Distribution for each label
        for field in VALID_LABELS:
            dist = Counter(v.get(field) for v in results_store.values())
            print(f"\n{field} distribution:")
            for label, count in dist.most_common():
                pct = count / len(results_store) * 100
                print(f"  {label:25s} {count:6d}  ({pct:5.1f}%)")

    print(f"\nOutput saved to: {args.output}")
    print("Done!")


if __name__ == '__main__':
    main()
