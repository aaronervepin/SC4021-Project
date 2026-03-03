"""Stats, cleaning, and eval sample generation."""

import json
import os
import re

import pandas as pd

import config

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")

# Regex patterns for filtering relevant records
# Need BOTH an exam term AND a sentiment term to keep a record

_EXAM_PATTERNS = re.compile(
    r"|".join([
        r"\bo[\s-]?levels?\b", r"\ba[\s-]?levels?\b", r"\bpsle\b", r"\bn[\s-]?levels?\b",
        r"\bh[123]\b", r"\bgce\b",
        r"\bexams?\b", r"\bmidterms?\b", r"\bfinals?\b",
        r"\bprelims?\b", r"\bpromos?\b", r"\bblock\s*test",
        r"\bGPA\b", r"\bCAP\b", r"\bl1r[45]\b", r"\bbell\s*curve",
        r"\bmodule\b",
        r"\ba[\s-]?math\b", r"\be[\s-]?math\b", r"\bamath\b", r"\bemath\b",
        r"\bpure\s+math", r"\badditional\s+math",
        r"\bpure\s+chem", r"\bpure\s+physics", r"\bpure\s+bio",
        r"\bcombined\s+sci", r"\bcombined\s+humanities",
        r"\bsocial\s+studies\b",
        r"\bpoa\b", r"\bd&t\b", r"\bdesign\s+and\s+tech",
        r"\bfood\s+and\s+nutrition", r"\bfnn\b",
        r"\bhigher\s+chinese\b", r"\bhigher\s+malay\b", r"\bhigher\s+tamil\b",
        r"\bmother\s+tongue\b", r"\bmtl\b",
        r"\bh2\s+math", r"\bh1\s+math", r"\bh2\s+physics", r"\bh2\s+chem",
        r"\bh2\s+bio", r"\bh2\s+econs", r"\bh2\s+hist", r"\bh2\s+geog",
        r"\bh2\s+lit", r"\bh2\s+art", r"\bh2\s+computing",
        r"\bh1\s+gp\b", r"\bgeneral\s+paper\b", r"\bGP\b",
        r"\bproject\s+work\b",
        r"\bfurther\s+math", r"\bh3\s+math",
        r"\bknowledge\s+&?\s*inquiry\b",
        r"\bpsle\s+math", r"\bpsle\s+sci", r"\bpsle\s+eng",
        r"\bfoundation\s+math", r"\bstandard\s+math",
        r"\bchem\b", r"\bbio\b", r"\becons?\b",
        r"\bgeog\b", r"\blit\b", r"\bcomputing\b",
        r"\bmath\b", r"\bphysics\b",
    ]),
    re.IGNORECASE,
)

_SENTIMENT_PATTERNS = re.compile(
    r"|".join([
        # Difficulty
        r"\bhard\b", r"\beasy\b", r"\bdifficult", r"\btough\b", r"\bkiller\b",
        r"\bdoable\b", r"\bimpossible\b", r"\btricky\b",
        # Emotion
        r"\bstress", r"\banxi", r"\bworr", r"\bdisappoint", r"\bfrustrat",
        r"\breliev", r"\bscar[ey]", r"\bdepressed", r"\bcried?\b", r"\bcrying\b",
        r"\bproud\b", r"\bconfident\b",
        # Performance
        r"\bfail", r"\bpass(ed)?\b", r"\bflunk", r"\bace[ds]?\b",
        # Study / prep
        r"\bstud(y|ied|ying)\b", r"\brevise\b", r"\brevision\b",
        r"\bmugg(ing|ed)\b", r"\bprepar", r"\bpractice\b",
        # Grades
        r"\bgrade[ds]?\b", r"\bscore[ds]?\b", r"\bmark[sed]?\b",
        r"\bresult", r"\brp\b",
        # Opinion
        r"\bfeedback\b", r"\bfeel(s|ing)?\b", r"\brant\b", r"\btips?\b",
    ]),
    re.IGNORECASE,
)


_OFF_TOPIC_TITLE = re.compile(
    r"(?i)(layoff|salary|tech\s+sector|dating\s+someone|tiktok|brain[\s-]dead"
    r"|misandry|misogyny|empathy|racism|body\s+image|insurance|cpf\b"
    r"|housing|bto\b|cryptocurrency|crypto|invest|stock)"
)
_EXAM_IN_BODY = re.compile(
    r"(?i)\b(levels?|exams?|midterms?|finals?\b|prelim|promo|test|paper"
    r"|grade|score|marks?|result|GPA|CAP|rp\b|bell\s*curve|math|physics"
    r"|chem|bio|econs?|computing|geog|lit|GP\b|module|h[12]\b|psle"
    r"|studied|revision|mugging|flunk|fail|pass)"
)


def is_relevant(row) -> bool:
    """Check if a record is about SG exam difficulty / student sentiment.

    Requires BOTH an exam/subject term AND a sentiment/opinion term.
    Also excludes records with off-topic titles that lack exam terms in body.
    Works with both pd.Series and plain dict.
    """
    title = str(row.get("title", "") or "")
    body = str(row.get("body", "") or "")
    text = title + " " + body
    if not (bool(_EXAM_PATTERNS.search(text)) and bool(_SENTIMENT_PATTERNS.search(text))):
        return False
    # Exclude off-topic titles when body has no exam context
    if _OFF_TOPIC_TITLE.search(title) and not _EXAM_IN_BODY.search(body):
        return False
    return True


def load_and_clean(csv_path: str = config.RAW_CSV_PATH) -> pd.DataFrame:
    """Load raw CSV, deduplicate, remove invalid rows, and filter for relevance."""
    df = pd.read_csv(csv_path)
    # Deduplicate by ID (safety net)
    df = df.drop_duplicates(subset=["id"], keep="first")
    # Drop rows with missing/empty body
    df = df[df["body"].fillna("").str.strip().str.len() >= config.MIN_BODY_LENGTH]
    # Drop [deleted]/[removed] bodies that slipped through
    df = df[~df["body"].isin(["[deleted]", "[removed]"])]
    # Deduplicate by body text (removes bot/mod duplicate messages)
    df = df.drop_duplicates(subset=["body"], keep="first")
    # Keep only records relevant to exams / difficulty / student sentiment
    df = df[df.apply(is_relevant, axis=1)]
    df = df.reset_index(drop=True)
    return df


def save_clean_csv(df: pd.DataFrame, path: str = config.CLEAN_CSV_PATH) -> str:
    """Save cleaned DataFrame to CSV."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    return path


def calculate_stats(csv_path: str = config.RAW_CSV_PATH) -> dict:
    """Calculate record count, word count, and unique word count."""
    df = load_and_clean(csv_path)

    all_text = df["title"].fillna("").astype(str) + " " + df["body"].fillna("").astype(str)

    all_words = []
    for text in all_text:
        all_words.extend(WORD_RE.findall(text.lower()))

    total_words = len(all_words)
    unique_words = len(set(all_words))

    type_counts = df["post_type"].value_counts().to_dict()
    subreddit_counts = df["subreddit"].value_counts().to_dict()

    stats = {
        "total_records": len(df),
        "total_words": total_words,
        "unique_words": unique_words,
        "submissions": type_counts.get("submission", 0),
        "comments": type_counts.get("comment", 0),
        "subreddit_breakdown": subreddit_counts,
        "average_score": round(df["score"].mean(), 2),
    }

    os.makedirs(os.path.dirname(config.STATS_PATH) or ".", exist_ok=True)
    with open(config.STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    # Also save the clean CSV
    save_clean_csv(df)

    return stats


def generate_eval_sample(
    csv_path: str = config.RAW_CSV_PATH,
    sample_size: int = 4000,
) -> str:
    """
    Generate a random sample for manual annotation in .xls format.

    Format follows standard sentiment benchmarks:
      - id: unique identifier
      - text: the text to be annotated (title + body for submissions, body for comments)
      - sentiment_label: blank column for annotators to fill (positive/negative/neutral)
      - annotator: blank column for annotator name
      - source_url: link to original post
      - subreddit: source subreddit
      - post_type: submission or comment
      - score: Reddit score
      - created_utc: timestamp
    """
    df = load_and_clean(csv_path)
    # Only include records with meaningful text (>= 20 chars)
    df = df[df["body"].fillna("").str.len() >= 20]

    # Use RoBERTa (twitter-roberta-base-sentiment-latest) for sentiment.
    # Much better than VADER at handling informal text, sarcasm, and slang.
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    import numpy as np

    _MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
    _model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
    _model.eval()
    _LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

    def _roberta_bucket(text: str) -> str:
        text = str(text)[:512]  # RoBERTa max context
        inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = _model(**inputs).logits
        probs = torch.softmax(logits, dim=1).numpy()[0]
        return _LABEL_MAP[int(np.argmax(probs))]

    # Build combined text (title + body) for all records so eval text is self-contained
    df["_full_text"] = df.apply(
        lambda r: (f"{r['title']}. {r['body']}" if pd.notna(r["title"])
                   and str(r["title"]).strip() else str(r["body"])),
        axis=1,
    )

    # Deduplicate by full text BEFORE classification to avoid wasted work
    df = df.drop_duplicates(subset=["_full_text"], keep="first")

    print("Classifying sentiment with RoBERTa (this may take a few minutes)...")
    df["_sentiment"] = df["_full_text"].apply(_roberta_bucket)

    # --- Subreddit minimum quotas with balanced sentiment ---
    # Ensure at least 500 NUS, 500 NTU, 2000 SGExams — all 1:1:1 balanced
    SUB_QUOTAS = {"SGExams": 2000, "nus": 1000, "NTU": 1000}
    parts = []
    used_ids = set()

    def _balanced_sample(pool, n, seed=42):
        """Sample n records with 1:1:1 sentiment balance from pool."""
        n3 = n // 3
        n_extra = n - 3 * n3
        p = pool[pool["_sentiment"] == "positive"]
        ne = pool[pool["_sentiment"] == "negative"]
        nu = pool[pool["_sentiment"] == "neutral"]
        picked = pd.concat([
            p.sample(n=min(n3, len(p)), random_state=seed),
            ne.sample(n=min(n3, len(ne)), random_state=seed),
            nu.sample(n=min(n3 + n_extra, len(nu)), random_state=seed),
        ])
        # Top up if any bucket was short
        if len(picked) < n:
            remaining = pool[~pool["id"].isin(picked["id"])]
            extra = remaining.sample(
                n=min(len(remaining), n - len(picked)), random_state=seed
            )
            picked = pd.concat([picked, extra])
        return picked

    for sub_name, quota in SUB_QUOTAS.items():
        sub_pool = df[df["subreddit"].str.lower() == sub_name.lower()]
        sub_pool = sub_pool[~sub_pool["id"].isin(used_ids)]
        n_take = min(quota, len(sub_pool))
        picked = _balanced_sample(sub_pool, n_take)
        parts.append(picked)
        used_ids.update(picked["id"].tolist())

    sample = pd.concat(parts)
    sample = sample.drop_duplicates(subset=["id"], keep="first")

    # If we're short of sample_size, top up from any remaining records
    if len(sample) < sample_size:
        leftover = df[~df["id"].isin(sample["id"])]
        extra = leftover.sample(
            n=min(len(leftover), sample_size - len(sample)), random_state=42
        )
        sample = pd.concat([sample, extra])

    sample = sample.sample(frac=1, random_state=42).reset_index(drop=True)

    # Clean illegal characters for Excel (control chars except tab/newline)
    _ILLEGAL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

    def _clean_col(series):
        return series.apply(
            lambda x: _ILLEGAL_CHARS.sub("", str(x)) if isinstance(x, str) else x
        ).str[:32000]

    # --- eval.xls: title + body (full context version) ---
    eval_df = pd.DataFrame()
    eval_df["text"] = _clean_col(sample["_full_text"])
    eval_df["sentiment_label"] = sample["_sentiment"].str.upper().tolist()

    os.makedirs(config.EVAL_DIR, exist_ok=True)

    eval_xls_path = os.path.join(config.EVAL_DIR, "eval.xls")
    eval_df.to_excel(eval_xls_path, index=False, engine="openpyxl")
    eval_df.to_csv(config.EVAL_SAMPLE_PATH, index=False)

    # --- eval_benchmark.xls: body-only, no headers (matches benchmark format) ---
    bench_df = pd.DataFrame()
    bench_df[0] = _clean_col(sample["body"].fillna("").astype(str))
    bench_df[1] = sample["_sentiment"].str.upper().tolist()

    bench_xls_path = os.path.join(config.EVAL_DIR, "eval_benchmark.xls")
    bench_df.to_excel(bench_xls_path, index=False, header=False, engine="openpyxl")

    return eval_xls_path
