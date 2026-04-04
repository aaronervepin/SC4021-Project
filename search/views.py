"""
search/views.py
================
Django search view — powered by TF-IDF lnc.ltc ranked retrieval
with Solr as fallback, filter support, and search history.
ADDED:
search/views.py — Extended from original with: synonym expansion for SG education terms (emath/e math etc.),
filter support (education level, subject, topic, emotion, subjectivity, sarcasm), and session-based search
history with add/remove/clear functionality.

"""

import time
import logging
import requests

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import ensure_csrf_cookie

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


def apply_filters(hits, filters):
    if not filters:
        return hits
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
        filtered.append((doc, score))
    return filtered


def _tfidf_search(query, page, rows, filters):
    from search.tfidf_engine import get_engine
    t0 = time.time()
    engine = get_engine()
    expanded = expand_query(query) if any(w in SYNONYM_GROUPS for w in query.lower().split()) else query

    # Use total corpus size so num_found is never artificially capped
    all_hits = engine.search(expanded, top_k=len(engine.docs))
    all_hits = apply_filters(all_hits, filters)

    qtime = round((time.time() - t0) * 1000)
    num_found = len(all_hits)
    start = (page - 1) * rows
    page_hits = all_hits[start: start + rows]

    results = []
    for rank, (doc, score) in enumerate(page_hits, start=start + 1):
        results.append({
            **doc,
            "tfidf_score":     score,
            "tfidf_score_pct": round(score * 100, 1),
            "rank":            rank,
        })

    sentiment = engine.sentiment_stats(all_hits)
    return results, num_found, qtime, sentiment


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


@ensure_csrf_cookie
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

    # Search history
    if "history" not in request.session:
        request.session["history"] = []

    filters = {}
    for field in FILTER_OPTIONS:
        selected = request.GET.getlist(field)
        if selected:
            filters[field] = selected

    if query:
        # Only update history if user explicitly searched (not a reload after remove)
        if not request.GET.get('no_history'):
            history = request.session["history"]
            if query in history:
                history.remove(query)
            history.insert(0, query)
            request.session["history"] = history[:20]
            request.session.modified = True

        try:
            results, num_found, qtime, sentiment = _tfidf_search(query, page, rows, filters)
            engine_used = "tfidf"
        except Exception as exc:
            logger.warning("[TF-IDF] Engine failed (%s); falling back to Solr.", exc)
            try:
                results, num_found, qtime, sentiment = _solr_search(query, page, rows, filters)
                engine_used = "solr"
            except Exception as solr_exc:
                logger.error("[Solr] Fallback also failed: %s", solr_exc)

        num_pages = (num_found + rows - 1) // rows

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
    })


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