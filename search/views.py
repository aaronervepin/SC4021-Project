"""
search/views.py
================
Django search view — now powered by TF-IDF lnc.ltc ranked retrieval
(see search/tfidf_engine.py) with Solr kept as a fallback.

Changes from original:
  • Imports and uses TFIDFEngine for ranking (replaces Solr score ordering)
  • Passes `tfidf_score` (float 0–1) and `rank` (int, 1-based) per result
  • Sentiment stats are now computed from the full TF-IDF hit list
  • Solr is used only as a fallback when the TF-IDF engine is unavailable
"""

import time
import logging
import requests

from django.shortcuts import render

logger = logging.getLogger(__name__)

SOLR_URL = "http://localhost:8983/solr/reddit_opinions/select"

# ── Synonym expansion (unchanged from original) ───────────────────────────────
SYNONYM_GROUPS = {
    # E Math
    "emath": ["emath", "e math", "e-math", "emaths", "e-maths", "elementary math", "elementary mathematics"],
    "e math": ["emath", "e math", "e-math", "emaths", "e-maths", "elementary math", "elementary mathematics"],
    "emaths": ["emath", "e math", "e-math", "emaths", "e-maths", "elementary math", "elementary mathematics"],
    "elementary math": ["emath", "e math", "e-math", "emaths", "e-maths", "elementary math", "elementary mathematics"],
    # A Math
    "amath": ["amath", "a math", "a-math", "amaths", "a-maths", "additional math", "additional mathematics"],
    "a math": ["amath", "a math", "a-math", "amaths", "a-maths", "additional math", "additional mathematics"],
    "additional math": ["amath", "a math", "a-math", "amaths", "a-maths", "additional math", "additional mathematics"],
    # O Level
    "olevel": ["olevel", "o level", "o-level", "olevels", "o levels", "o-levels"],
    "o level": ["olevel", "o level", "o-level", "olevels", "o levels", "o-levels"],
    "olevels": ["olevel", "o level", "o-level", "olevels", "o levels", "o-levels"],
    # A Level
    "alevel": ["alevel", "a level", "a-level", "alevels", "a levels", "a-levels"],
    "a level": ["alevel", "a level", "a-level", "alevels", "a levels", "a-levels"],
    "alevels": ["alevel", "a level", "a-level", "alevels", "a levels", "a-levels"],
    # Subjects
    "phy": ["phy", "phys", "physics"],
    "phys": ["phy", "phys", "physics"],
    "physics": ["phy", "phys", "physics"],
    "chem": ["chem", "chemistry"],
    "chemistry": ["chem", "chemistry"],
    "bio": ["bio", "biol", "biology"],
    "biology": ["bio", "biol", "biology"],
    "gp": ["gp", "general paper"],
    "general paper": ["gp", "general paper"],
    "econs": ["econs", "econ", "economics"],
    "economics": ["econs", "econ", "economics"],
    "geo": ["geo", "geog", "geography"],
    "geography": ["geo", "geog", "geography"],
    "hist": ["hist", "history"],
    "history": ["hist", "history"],
    "lit": ["lit", "literature"],
    "literature": ["lit", "literature"],
    "ss": ["ss", "social studies"],
    "social studies": ["ss", "social studies"],
}


def expand_query(query: str) -> str:
    """Expand Singlish / shorthand synonyms for Solr fallback queries."""
    words = query.lower().split()
    matched = set()
    expanded = []

    for i in range(len(words)):
        for length in range(3, 0, -1):
            phrase = " ".join(words[i:i + length])
            if phrase in SYNONYM_GROUPS and phrase not in matched:
                matched.add(phrase)
                variants = SYNONYM_GROUPS[phrase]
                escaped = [f'"{v}"' if " " in v else v for v in variants]
                expanded.append(f'({" OR ".join(escaped)})')
                break

    return " AND ".join(expanded) if expanded else query


# ── Sentiment helper ──────────────────────────────────────────────────────────
def _sentiment_from_solr_docs(docs):
    counts = {"1": 0, "0": 0, "-1": 0}
    for doc in docs:
        raw = doc.get("Output_2", ["0"])
        val = str(raw[0] if isinstance(raw, list) else raw).strip()
        if val in counts:
            counts[val] += 1
    total = sum(counts.values())
    if total == 0:
        return None
    return {
        "positive": round(counts["1"]  / total * 100),
        "neutral":  round(counts["0"]  / total * 100),
        "negative": round(counts["-1"] / total * 100),
        "total":    total,
    }


# ── TF-IDF ranked search ──────────────────────────────────────────────────────
def _tfidf_search(query, page, rows):
    """
    Run TF-IDF lnc.ltc ranked retrieval.
    Returns (results_page, num_found, qtime_ms, sentiment_dict).
    `results_page` is a list of dicts with keys:
        tfidf_score, rank, title, body, subreddit, author, score, url, Output_2, …
    """
    from search.tfidf_engine import get_engine

    t0 = time.time()
    engine = get_engine()

    # Expand synonyms in query text before vectorising
    expanded = expand_query(query) if any(w in SYNONYM_GROUPS for w in query.lower().split()) else query

    all_hits = engine.search(expanded, top_k=500)   # [(doc, cosine_score), …]
    qtime    = round((time.time() - t0) * 1000)

    num_found = len(all_hits)
    start     = (page - 1) * rows
    page_hits = all_hits[start: start + rows]

    results = []
    for rank, (doc, score) in enumerate(page_hits, start=start + 1):
        results.append({
            **doc,
            "tfidf_score":       score,
            "tfidf_score_pct":   round(score * 100, 1),   # for CSS width
            "rank":              rank,
        })

    sentiment = engine.sentiment_stats(all_hits)
    return results, num_found, qtime, sentiment


# ── Solr fallback search ──────────────────────────────────────────────────────
def _solr_search(query, page, rows):
    """Original Solr-based search, kept as fallback."""
    start  = (page - 1) * rows
    params = {
        "q":    expand_query(query),
        "wt":   "json",
        "rows": rows,
        "start": start,
    }
    resp      = requests.get(SOLR_URL, params=params, timeout=5)
    data      = resp.json()
    docs      = data["response"]["docs"]
    num_found = data["response"]["numFound"]
    qtime     = data["responseHeader"]["QTime"]

    # Attach placeholder rank/score so template stays consistent
    results = []
    for rank, doc in enumerate(docs, start=start + 1):
        results.append({**doc, "tfidf_score": None, "tfidf_score_pct": 0, "rank": rank})

    # Sentiment from full result set
    all_params = {**params, "rows": num_found, "fl": "Output_2"}
    all_docs   = requests.get(SOLR_URL, params=all_params, timeout=5).json()["response"]["docs"]
    sentiment  = _sentiment_from_solr_docs(all_docs)

    return results, num_found, qtime, sentiment


# ── Main view ─────────────────────────────────────────────────────────────────
def search(request):
    query     = request.GET.get("q", "").strip()
    page      = max(1, int(request.GET.get("page", 1)))
    rows      = 10
    results   = []
    num_found = 0
    qtime     = 0
    num_pages = 0
    sentiment = None
    engine_used = None

    if query:
        # ── Primary: TF-IDF ranked retrieval ──────────────────────────────
        try:
            results, num_found, qtime, sentiment = _tfidf_search(query, page, rows)
            engine_used = "tfidf"
        except Exception as exc:
            logger.warning("[TF-IDF] Engine failed (%s); falling back to Solr.", exc)

            # ── Fallback: Solr ─────────────────────────────────────────────
            try:
                results, num_found, qtime, sentiment = _solr_search(query, page, rows)
                engine_used = "solr"
            except Exception as solr_exc:
                logger.error("[Solr] Fallback also failed: %s", solr_exc)

        num_pages = (num_found + rows - 1) // rows

    return render(request, "search/search.html", {
        "results":     results,
        "query":       query,
        "num_found":   num_found,
        "qtime":       qtime,
        "page":        page,
        "num_pages":   num_pages,
        "prev_page":   page - 1 if page > 1 else None,
        "next_page":   page + 1 if page < num_pages else None,
        "sentiment":   sentiment,
        "engine_used": engine_used,
    })