"""
add_year_column.py
==================
Patches crawled_enriched.csv with a new 'year' column (academic year)
by calling the LLM on each post's title+body.

Uses the same API pattern as ir_label_enrichment.py.

Usage:
    python add_year_column.py
    python add_year_column.py --input crawled_enriched.csv --output crawled_enriched.csv
    python add_year_column.py --resume
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
from typing import Optional, Dict, List, Tuple
from datetime import datetime

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input',  type=str, default='crawled_enriched.csv')
parser.add_argument('--output', type=str, default='crawled_enriched.csv')
parser.add_argument('--api_key_path', type=str, default='api.txt')
parser.add_argument('--base_url', type=str, default='https://api.apiyi.com/v1/chat/completions')
parser.add_argument('--model', type=str, default='gpt-5')
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--checkpoint_path', type=str, default='year_column_checkpoint.json')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--max_retries', type=int, default=3)
parser.add_argument('--retry_delay', type=float, default=2.0)
parser.add_argument('--request_delay', type=float, default=1.0)
parser.add_argument('--timeout', type=int, default=120)
parser.add_argument('--temperature', type=float, default=0.1)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

VALID_YEAR = [
    "p6",
    "sec1", "sec2", "sec3", "sec4", "sec5",
    "jc1", "jc2",
    "year1", "year2", "year3", "year4",
    "none"
]

SYSTEM_PROMPT = """You are an expert NLP annotator for Singapore education forum posts (r/SGExams, r/nus).

For EACH post, infer the academic year within the author's education level.

**year** values:
- "p6"          → Primary 6 / PSLE
- "sec1"–"sec5" → Secondary 1–5 (Sec 5 = N-level repeat year)
- "jc1" / "jc2" → Junior College Year 1 / Year 2
- "year1"–"year4" → Poly / ITE / University Year 1–4
- "none"        → Cannot be inferred from the post

Clues to look for: mentions of "Sec 3", "JC1", "Year 2", "first year", "second year", "3rd year",
"graduating", "prelims" (usually Sec 4/JC2), "mid-years" (Sec 3/JC1), etc.

CRITICAL OUTPUT FORMAT — respond with ONLY a JSON array, no markdown, no explanation:
[
  {"idx": 0, "year": "sec4"},
  {"idx": 1, "year": "jc2"},
  ...
]
"""


def load_api_key(path: str) -> Optional[str]:
    if os.path.exists(path):
        with open(path, 'r') as f:
            return f.read().strip()
    return None


def call_llm(api_key: str, user_prompt: str) -> Optional[str]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 1000,
        "temperature": args.temperature
    }
    for attempt in range(args.max_retries):
        try:
            r = requests.post(args.base_url, headers=headers, json=payload, timeout=args.timeout)
            if r.status_code == 200:
                return r.json()['choices'][0]['message']['content']
            elif r.status_code == 429:
                time.sleep(args.retry_delay * (2 ** attempt))
            else:
                print(f"  API error {r.status_code}: {r.text[:200]}")
                if attempt < args.max_retries - 1:
                    time.sleep(args.retry_delay * (2 ** attempt))
        except requests.exceptions.RequestException as e:
            print(f"  Request failed: {e}")
            if attempt < args.max_retries - 1:
                time.sleep(args.retry_delay * (2 ** attempt))
    return None


def build_prompt(batch: List[Tuple[int, str, str]]) -> str:
    lines = []
    for idx, title, body in batch:
        truncated = body[:400] + "..." if len(body) > 400 else body
        truncated = truncated.replace('\\', '\\\\').replace('"', '\\"')
        safe_title = title.replace('\\', '\\\\').replace('"', '\\"')
        lines.append(f'[Post {idx}]:\n  Title: """{safe_title}"""\n  Body: """{truncated}"""')
    return "Label the academic year for each post:\n\n" + "\n\n".join(lines)


def parse_response(text: str) -> Dict[int, str]:
    if not text:
        return {}
    cleaned = text.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    start, end = cleaned.find('['), cleaned.rfind(']')
    if start != -1 and end > start:
        cleaned = cleaned[start:end + 1]
    try:
        items = json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            items = json.loads(re.sub(r',\s*([}\]])', r'\1', cleaned))
        except json.JSONDecodeError:
            if args.debug:
                print(f"  [DEBUG] Parse failed: {text[:200]}")
            return {}

    result = {}
    for item in items:
        idx = item.get('idx')
        year = str(item.get('year', 'none')).strip().lower().replace(' ', '').replace('-', '')
        if year not in VALID_YEAR:
            # fuzzy match
            matched = next((v for v in VALID_YEAR if v in year or year in v), 'none')
            year = matched
        result[idx] = year
    return result


def save_checkpoint(processed: dict):
    tmp = args.checkpoint_path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump({'results': processed, 'timestamp': datetime.now().isoformat()}, f)
    os.replace(tmp, args.checkpoint_path)


def load_checkpoint() -> dict:
    if not os.path.exists(args.checkpoint_path):
        return {}
    try:
        with open(args.checkpoint_path) as f:
            return json.load(f).get('results', {})
    except Exception:
        return {}


def main():
    print("=" * 60)
    print("Add 'year' column to crawled_enriched.csv")
    print("=" * 60)

    api_key = load_api_key(args.api_key_path)
    if not api_key:
        print(f"Error: cannot load API key from {args.api_key_path}")
        return

    # Load CSV
    rows = []
    with open(args.input, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        for r in reader:
            rows.append(r)
    print(f"Loaded {len(rows)} rows from {args.input}")

    # If 'year' column already exists, we're patching it; otherwise insert after 'subject'
    if 'year' not in fieldnames:
        # Insert after 'subject' if present, else append before 'label_agreement'
        if 'subject' in fieldnames:
            idx = fieldnames.index('subject') + 1
            fieldnames.insert(idx, 'year')
        elif 'label_agreement' in fieldnames:
            idx = fieldnames.index('label_agreement')
            fieldnames.insert(idx, 'year')
        else:
            fieldnames.append('year')
        print("  Inserting 'year' column after 'subject'")
    else:
        print("  'year' column already exists — overwriting values")

    # Load checkpoint
    results_store = load_checkpoint() if args.resume else {}
    print(f"Checkpoint: {len(results_store)} rows already labeled")

    unprocessed = [
        (i, rows[i].get('title', ''), rows[i].get('body', ''))
        for i in range(len(rows))
        if str(i) not in results_store
    ]
    print(f"Remaining:  {len(unprocessed)} rows to process\n")

    batches = [unprocessed[s:s + args.batch_size] for s in range(0, len(unprocessed), args.batch_size)]

    try:
        for b_idx, batch in enumerate(batches):
            print(f"Batch {b_idx + 1}/{len(batches)} "
                  f"(rows {batch[0][0]}-{batch[-1][0]})...", end=' ', flush=True)
            prompt = build_prompt(batch)
            response = call_llm(api_key, prompt)
            parsed = parse_response(response) if response else {}

            for j, (global_idx, _, _) in enumerate(batch):
                results_store[str(global_idx)] = parsed.get(j, 'none')

            print(f"ok ({len(parsed)}/{len(batch)} labeled)")
            time.sleep(args.request_delay)

            if (b_idx + 1) % 20 == 0:
                save_checkpoint(results_store)
                print(f"  [Checkpoint saved: {len(results_store)}/{len(rows)}]")

    except KeyboardInterrupt:
        print("\nInterrupted — saving checkpoint...")
        save_checkpoint(results_store)
        print(f"Resume with --resume. ({len(results_store)}/{len(rows)})")
        return
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
        save_checkpoint(results_store)
        raise

    save_checkpoint(results_store)

    # Write output
    print(f"\nWriting to {args.output}...")
    with open(args.output, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for i, row in enumerate(rows):
            row['year'] = results_store.get(str(i), 'none')
            writer.writerow(row)

    # Stats
    from collections import Counter
    dist = Counter(results_store.values())
    print("\nyear distribution:")
    for label, count in dist.most_common():
        print(f"  {label:10s}  {count:6d}  ({count / len(rows) * 100:.1f}%)")

    print(f"\nDone! Output saved to {args.output}")


if __name__ == '__main__':
    main()
