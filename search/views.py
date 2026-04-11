"""
search/views.py
================
Django search view — powered by TF-IDF lnc.ltc ranked retrieval
with Solr as fallback, filter support, search history, and
spell correction (Levenshtein + trigram pre-filtering, Lecture 3).
"""

import re
import time
import math
import logging
import datetime
import requests
from collections import Counter

from django.shortcuts import render
from django.http import JsonResponse

logger = logging.getLogger(__name__)

SOLR_URL = "http://localhost:8983/solr/reddit_opinions/select"

SYNONYM_GROUPS = {
    "emath": ["emath", "e math", "e-math", "emaths", "e-maths", "elementary math", "elementary mathematics"],
    "e math": ["emath", "e math", "e-math", "emaths", "e-maths", "elementary math", "elementary mathematics"],
    "emaths": ["emath", "e math", "e-math", "emaths", "e-maths", "elementary math", "elementary mathematics"],
    "elementary math": ["emath", "e math", "e-math", "emaths", "e-maths", "elementary math", "elementary mathematics"],
    "amath": ["amath", "a math", "a-math", "amaths", "a-maths", "additional math", "additional mathematics"],
    "a math": ["amath", "a math", "a-math", "amaths", "a-maths", "additional math", "additional mathematics"],
    "additional math": ["amath", "a math", "a-math", "amaths", "a-maths", "additional math", "additional mathematics"],
    "olevel": ["olevel", "o level", "o-level", "olevels", "o levels", "o-levels"],
    "o level": ["olevel", "o level", "o-level", "olevels", "o levels", "o-levels"],
    "olevels": ["olevel", "o level", "o-level", "olevels", "o levels", "o-levels"],
    "alevel": ["alevel", "a level", "a-level", "alevels", "a levels", "a-levels"],
    "a level": ["alevel", "a level", "a-level", "alevels", "a levels", "a-levels"],
    "alevels": ["alevel", "a level", "a-level", "alevels", "a levels", "a-levels"],
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
    "chinese": ["chinese", "chi", "cl", "mandarin", "higher chinese", "hcl"],
    "chi": ["chinese", "chi", "cl", "mandarin", "higher chinese", "hcl"],
    "cl": ["chinese", "chi", "cl", "mandarin", "higher chinese", "hcl"],
    "mandarin": ["chinese", "chi", "cl", "mandarin", "higher chinese", "hcl"],
    "higher chinese": ["chinese", "chi", "cl", "mandarin", "higher chinese", "hcl"],
    "hcl": ["chinese", "chi", "cl", "mandarin", "higher chinese", "hcl"],
    "malay": ["malay", "bm", "bahasa melayu", "bahasa"],
    "bm": ["malay", "bm", "bahasa melayu", "bahasa"],
    "bahasa melayu": ["malay", "bm", "bahasa melayu", "bahasa"],
    "bahasa": ["malay", "bm", "bahasa melayu", "bahasa"],
    "tamil": ["tamil", "tml"],
    "tml": ["tamil", "tml"],
    "english": ["english", "eng"],
    "eng": ["english", "eng"],
    "mother tongue": ["mother tongue", "mt", "mtl"],
    "mt": ["mother tongue", "mt", "mtl"],
    "mtl": ["mother tongue", "mt", "mtl"],
    "psle": ["psle", "primary school leaving examination", "primary school leaving exam"],
    "primary school leaving examination": ["psle", "primary school leaving examination", "primary school leaving exam"],
    "primary school leaving exam": ["psle", "primary school leaving examination", "primary school leaving exam"],
    "prelim": ["prelim", "prelims", "preliminary"],
    "prelims": ["prelim", "prelims", "preliminary"],
    "preliminary": ["prelim", "prelims", "preliminary"],
    "jc": ["jc", "junior college"],
    "junior college": ["jc", "junior college"],
    "poly": ["poly", "polytechnic"],
    "polytechnic": ["poly", "polytechnic"],
    "bell curve": ["bell curve", "bell-curve"],
    "bell-curve": ["bell curve", "bell-curve"],
    "tuition": ["tuition", "tution"],
    "tution": ["tuition", "tution"],
    "mug": ["mug", "mugger", "mugging"],
    "mugger": ["mug", "mugger", "mugging"],
    "mugging": ["mug", "mugger", "mugging"],
}

FILTER_OPTIONS = {
    "education_level": [
        ("alevel", "A Level"),
        ("olevel", "O Level"),
        ("psle", "PSLE"),
        ("uni", "University"),
        ("jc", "JC"),
        ("poly", "Polytechnic"),
        ("nlevel", "N Level"),
        ("ite", "ITE"),
        ("masters", "Masters"),
        ("general", "General"),
    ],
    "subject": [
        ("math", "Math"),
        ("gp", "GP"),
        ("computing", "Computing"),
        ("physics", "Physics"),
        ("biology", "Biology"),
        ("chemistry", "Chemistry"),
        ("chinese", "Chinese"),
        ("economics", "Economics"),
        ("english", "English"),
        ("literature", "Literature"),
        ("geography", "Geography"),
        ("history", "History"),
        ("art", "Art"),
    ],
    "topic": [
        ("results", "Results"),
        ("admissions", "Admissions"),
        ("mental_health", "Mental Health"),
        ("academic_policy", "Academic Policy"),
        ("exam_prep", "Exam Prep"),
        ("daily_life", "Daily Life"),
        ("study_tips", "Study Tips"),
        ("school_choice", "School Choice"),
        ("career", "Career"),
        ("extracurricular", "Extracurricular"),
    ],
    "emotion": [
        ("neutral", "Neutral"),
        ("hope", "Hope"),
        ("frustration", "Frustration"),
        ("anxiety", "Anxiety"),
        ("joy", "Joy"),
        ("sadness", "Sadness"),
        ("anger", "Anger"),
        ("surprise", "Surprise"),
        ("disgust", "Disgust"),
        ("fear", "Fear"),
    ],
    "subjectivity": [
        ("subjective", "Subjective"),
        ("objective", "Objective"),
    ],
    "sarcasm": [
        ("sarcastic", "Sarcastic"),
        ("not_sarcastic", "Not Sarcastic"),
    ],
}


def expand_query(query: str) -> str:
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


def _sentiment_from_docs(docs):
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


def apply_filters(hits, filters, date_from=None, date_to=None):
    filtered = []
    for doc, score in hits:
        if filters.get("education_level") and doc.get("education_level", "").strip() not in filters["education_level"]:
            continue
        if filters.get("subject") and doc.get("subject", "").strip() not in filters["subject"]:
            continue
        if filters.get("topic") and doc.get("topic", "").strip() not in filters["topic"]:
            continue
        if filters.get("emotion") and doc.get("emotion", "").strip() not in filters["emotion"]:
            continue
        if filters.get("subjectivity") and doc.get("subjectivity", "").strip() not in filters["subjectivity"]:
            continue
        if filters.get("sarcasm") and doc.get("sarcasm", "").strip() not in filters["sarcasm"]:
            continue
        # Timeline filter
        if date_from or date_to:
            try:
                ts = float(doc.get("created_utc", 0))
                if date_from and ts < date_from:
                    continue
                if date_to and ts > date_to:
                    continue
            except (ValueError, TypeError):
                continue
        filtered.append((doc, score))
    return filtered


def _tfidf_search(query, page, rows, filters, date_from=None, date_to=None, sort_by="relevance"):
    from search.tfidf_engine import get_engine
    t0 = time.time()
    engine = get_engine()
    expanded = expand_query(query) if any(w in SYNONYM_GROUPS for w in query.lower().split()) else query
    all_hits = engine.search(expanded, top_k=500)
    all_hits = apply_filters(all_hits, filters, date_from, date_to)

    # Sort results
    if sort_by == "date_newest":
        all_hits.sort(key=lambda x: float(x[0].get("created_utc", 0)), reverse=True)
    elif sort_by == "date_oldest":
        all_hits.sort(key=lambda x: float(x[0].get("created_utc", 0)))
    elif sort_by == "upvotes":
        all_hits.sort(key=lambda x: int(x[0].get("score", 0)), reverse=True)
    # else: default "relevance" — already sorted by TF-IDF score

    qtime = round((time.time() - t0) * 1000)
    num_found = len(all_hits)
    start = (page - 1) * rows
    page_hits = all_hits[start: start + rows]
    results = []
    for rank, (doc, score) in enumerate(page_hits, start=start + 1):
        # Format date for display
        try:
            date_str = datetime.datetime.fromtimestamp(float(doc.get("created_utc", 0))).strftime("%d %b %Y")
        except (ValueError, TypeError, OSError):
            date_str = ""
        results.append({
            **doc,
            "tfidf_score":     score,
            "tfidf_score_pct": round(score * 100, 1),
            "rank":            rank,
            "date_display":    date_str,
        })
    sentiment = engine.sentiment_stats(all_hits)
    word_cloud = _build_word_cloud(all_hits)
    return results, num_found, qtime, sentiment, word_cloud


def _solr_search(query, page, rows, filters):
    start = (page - 1) * rows
    params = {
        "q":     expand_query(query),
        "wt":    "json",
        "rows":  rows,
        "start": start,
    }
    fq = []
    for field, values in filters.items():
        if values:
            fq.append(f'{field}:({" OR ".join(values)})')
    if fq:
        params["fq"] = fq
    resp = requests.get(SOLR_URL, params=params, timeout=5)
    data = resp.json()
    docs = data["response"]["docs"]
    num_found = data["response"]["numFound"]
    qtime = data["responseHeader"]["QTime"]
    results = []
    for rank, doc in enumerate(docs, start=start + 1):
        results.append({**doc, "tfidf_score": None, "tfidf_score_pct": 0, "rank": rank})
    all_params = {**params, "rows": num_found, "fl": "Output_2"}
    all_docs = requests.get(SOLR_URL, params=all_params, timeout=5).json()["response"]["docs"]
    sentiment = _sentiment_from_docs(all_docs)
    return results, num_found, qtime, sentiment


# ── Word cloud helper ────────────────────────────────────────────────────────

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Extend sklearn's stopwords with Reddit/web-specific noise
_STOPWORDS = ENGLISH_STOP_WORDS.union({
    "amp", "https", "http", "www", "com", "reddit", "deleted", "removed",
    "really", "just", "like", "got", "get", "going", "don", "didn", "doesn",
})


def _build_word_cloud(hits, max_words=40):
    """Extract top terms from result bodies for word cloud display."""
    word_counts = Counter()
    for doc, _ in hits:
        text = (doc.get("body") or "") + " " + (doc.get("title") or "")
        words = re.findall(r'[a-zA-Z]{3,}', text.lower())
        word_counts.update(w for w in words if w not in _STOPWORDS)

    if not word_counts:
        return []

    top = word_counts.most_common(max_words)
    max_count = top[0][1]
    cloud = []
    for word, count in top:
        size = 12 + round(24 * math.log(1 + count) / math.log(1 + max_count))
        cloud.append({"word": word, "count": count, "size": size})

    import random
    random.shuffle(cloud)
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#e67e22",
              "#1abc9c", "#e84393", "#0984e3", "#d63031", "#6c5ce7",
              "#00b894", "#fdcb6e", "#e17055", "#0652DD", "#009432"]
    for i, item in enumerate(cloud):
        item["color"] = colors[i % len(colors)]
    return cloud


# ── Spell-correction helper ──────────────────────────────────────────────────

def _get_spell_suggestion(query: str):
    """
    Return a spelling suggestion for *query*, or None.

    Strategy (as described in the slides):
      • Run the raw query first.
      • Ask the SpellCorrector whether any token looks misspelled.
      • Only surface the suggestion — never silently rewrite the query —
        so the user stays in control (mirrors Google's "Did you mean?" UX).
    """
    try:
        from search.spell_correction import get_corrector
        corrector = get_corrector()
        suggestion = corrector.correct_query(query)
        # Don't show a suggestion if it's identical to the original
        if suggestion and suggestion.lower() != query.lower().strip():
            return suggestion
    except Exception as exc:
        logger.warning("[SpellCorrect] Suggestion failed: %s", exc)
    return None


# ── Main search view ─────────────────────────────────────────────────────────

def search(request):
    query       = request.GET.get("q", "").strip()
    page        = max(1, int(request.GET.get("page", 1)))
    rows        = 10
    results     = []
    num_found   = 0
    qtime       = 0
    num_pages   = 0
    sentiment   = None
    engine_used = None
    did_you_mean = None          # ← spell-correction suggestion
    word_cloud   = []

    # Search history
    if "history" not in request.session:
        request.session["history"] = []

    filters = {}
    for field in FILTER_OPTIONS:
        selected = request.GET.getlist(field)
        if selected:
            filters[field] = selected

    # Timeline search — parse date range
    date_from_str = request.GET.get("date_from", "").strip()
    date_to_str = request.GET.get("date_to", "").strip()
    date_from_ts = None
    date_to_ts = None
    try:
        if date_from_str:
            date_from_ts = datetime.datetime.strptime(date_from_str, "%Y-%m-%d").timestamp()
        if date_to_str:
            date_to_ts = datetime.datetime.strptime(date_to_str, "%Y-%m-%d").replace(
                hour=23, minute=59, second=59
            ).timestamp()
    except ValueError:
        pass

    sort_by = request.GET.get("sort", "relevance")

    if query:
        # Update history unless this is a reload-after-remove
        if not request.GET.get('no_history'):
            history = request.session["history"]
            if query in history:
                history.remove(query)
            history.insert(0, query)
            request.session["history"] = history[:20]
            request.session.modified = True

        # ── Primary search ────────────────────────────────────────────────
        try:
            results, num_found, qtime, sentiment, word_cloud = _tfidf_search(query, page, rows, filters, date_from_ts, date_to_ts, sort_by)
            engine_used = "tfidf"
        except Exception as exc:
            logger.warning("[TF-IDF] Engine failed (%s); falling back to Solr.", exc)
            try:
                results, num_found, qtime, sentiment = _solr_search(query, page, rows, filters)
                engine_used = "solr"
            except Exception as solr_exc:
                logger.error("[Solr] Fallback also failed: %s", solr_exc)

        num_pages = (num_found + rows - 1) // rows

        # ── Spell correction ──────────────────────────────────────────────
        # Surface a "Did you mean?" suggestion whenever:
        #   (a) results are zero or very few, OR
        #   (b) the corrector detects a likely misspelling regardless.
        # The suggestion is shown as a clickable link — we never force a
        # redirect so the user's original intent is always respected.
        did_you_mean = _get_spell_suggestion(query)

    return render(request, "search/search.html", {
        "results":        results,
        "query":          query,
        "num_found":      num_found,
        "qtime":          qtime,
        "page":           page,
        "num_pages":      num_pages,
        "prev_page":      page - 1 if page > 1 else None,
        "next_page":      page + 1 if page < num_pages else None,
        "sentiment":      sentiment,
        "engine_used":    engine_used,
        "filter_options": FILTER_OPTIONS,
        "active_filters": filters,
        "history":        request.session.get("history", []),
        "did_you_mean":   did_you_mean,
        "date_from":      date_from_str,
        "date_to":        date_to_str,
        "word_cloud":     word_cloud,
        "sort_by":        sort_by,
    })


def autocomplete(request):
    """Return JSON list of vocabulary terms matching the prefix."""
    prefix = request.GET.get("term", "").strip().lower()
    if len(prefix) < 2:
        return JsonResponse([], safe=False)
    try:
        from search.tfidf_engine import get_engine
        engine = get_engine()
        vocab = engine._doc_vec.vocabulary_
        matches = [w for w in vocab if w.startswith(prefix) and " " not in w]
        matches.sort(key=lambda w: (-vocab[w], w))  # sort by frequency (index)
        return JsonResponse(matches[:8], safe=False)
    except Exception:
        return JsonResponse([], safe=False)


def remove_history(request):
    if request.method == "POST":
        query = request.POST.get("query", "")
        history = request.session.get("history", [])
        if query in history:
            history.remove(query)
            request.session["history"] = history
            request.session.modified = True
    return JsonResponse({"status": "ok"})


def clear_history(request):
    if request.method == 'POST':
        request.session['history'] = []
        request.session.modified = True
    return JsonResponse({'status': 'ok'})