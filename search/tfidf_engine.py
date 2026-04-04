"""
TF-IDF Ranked Retrieval Engine
================================
Implements the lnc.ltc weighting scheme described in the RAG slides (Lecture 2):
  - Documents: log(1+tf)  · no IDF  · L2-cosine normalise  (lnc)
  - Queries:   log(1+tf)  · IDF     · L2-cosine normalise  (ltc)

Cosine similarity between the (ltc) query vector and every (lnc) document vector
gives the ranked score in [0, 1] for each result.

Usage (singleton, lazy-loaded at first request):
    from search.tfidf_engine import get_engine
    hits = get_engine().search("a level math", top_k=500)
    # hits → [(doc_dict, score), ...] sorted by descending score
"""

import csv
import os
import time
import logging

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ── Path resolution: two levels up from this file → project root ────────────
CSV_PATH = '/Users/wanxuan/Downloads/crawled_enriched.csv'


# ── lnc vectorizer (documents) ──────────────────────────────────────────────
def _make_doc_vectorizer():
    """
    lnc weighting for documents:
      l → sublinear_tf=True  (log(1+tf) instead of raw tf)
      n → use_idf=False      (no IDF component)
      c → norm='l2'          (cosine / L2 normalisation)
    Matches Classification-JH vocabulary settings for consistency.
    """
    return TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,   # log(1+tf)
        use_idf=False,       # lnc: no IDF on documents
        norm="l2",           # cosine normalise
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",  # keeps single chars (e.g. "A" level)
    )


# ── ltc vectorizer (queries) ─────────────────────────────────────────────────
def _make_query_vectorizer(doc_vectorizer):
    """
    ltc weighting for queries, re-using the vocabulary fitted on documents:
      l → sublinear_tf=True
      t → use_idf=True  with IDF computed from the document corpus
      c → norm='l2'
    We copy vocabulary + IDF weights from the already-fitted doc vectorizer.
    """
    qv = TfidfVectorizer(
        vocabulary=doc_vectorizer.vocabulary_,
        sublinear_tf=True,
        use_idf=True,
        norm="l2",
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    # Inject the pre-computed IDF vector so transform() works without fitting
    qv.idf_ = doc_vectorizer.idf_ if hasattr(doc_vectorizer, "idf_") else np.ones(len(doc_vectorizer.vocabulary_))
    return qv


class TFIDFEngine:
    """In-memory TF-IDF index with lnc.ltc cosine ranked retrieval."""

    def __init__(self):
        self._doc_vec = _make_doc_vectorizer()
        self._qry_vec = None          # built after doc_vec is fitted
        self._matrix  = None          # shape (N, V), sparse, lnc-weighted
        self.docs      = []            # list of raw dicts from CSV
        self.built     = False

    # ── Index construction ───────────────────────────────────────────────────
    def build(self, csv_path: str = CSV_PATH) -> None:
        """
        Read CSV → concatenate title+body → fit lnc TF-IDF matrix.
        Called once at application startup (lazy via get_engine()).
        """
        t0 = time.time()
        logger.info("[TF-IDF] Reading corpus from %s …", csv_path)

        with open(csv_path, encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                title = (row.get("title") or "").strip()
                body  = (row.get("body")  or "").strip()
                self.docs.append({
                    # preserve every column so the template can use them
                    **row,
                    # pre-computed combined text for vectorisation
                    "_text": f"{title} {body}",
                })

        texts = [d["_text"] for d in self.docs]

        # Fit document matrix (lnc)
        self._matrix = self._doc_vec.fit_transform(texts)

        # Build query vectorizer (ltc) — reuses vocabulary + IDF from above
        # We need IDF, so compute it separately with a full vectorizer then copy
        _idf_vec = TfidfVectorizer(
            vocabulary=self._doc_vec.vocabulary_,
            sublinear_tf=True,
            use_idf=True,
            norm=None,          # don't normalise; we only want idf_
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
        )
        _idf_vec.fit(texts)
        self._qry_vec = _make_query_vectorizer(_idf_vec)

        elapsed = time.time() - t0
        logger.info(
            "[TF-IDF] Index built: %d docs | vocab %d | %.1fs",
            len(self.docs), len(self._doc_vec.vocabulary_), elapsed,
        )
        self.built = True

    # ── Query execution ──────────────────────────────────────────────────────
    def search(self, query: str, top_k: int = 500):
        """
        Compute lnc.ltc cosine similarity for `query` against all documents.

        Returns
        -------
        list of (doc_dict, float) — up to top_k entries, sorted by
        descending cosine similarity score.  Only docs with score > 0 appear.
        """
        if not self.built:
            raise RuntimeError("TFIDFEngine.build() must be called before search().")

        # Encode query as ltc vector
        q_vec = self._qry_vec.transform([query])  # (1, V)

        # Cosine similarity against all document vectors (lnc)
        scores = cosine_similarity(q_vec, self._matrix)[0]  # (N,)

        # Filter to non-zero, sort descending, take top_k
        nonzero_idx = np.flatnonzero(scores)
        sorted_idx  = nonzero_idx[np.argsort(-scores[nonzero_idx])][:top_k]

        return [(self.docs[i], round(float(scores[i]), 4)) for i in sorted_idx]

    # ── Sentiment helper ─────────────────────────────────────────────────────
    def sentiment_stats(self, hits):
        """
        Compute positive / neutral / negative breakdown from TF-IDF result hits.
        `hits` is the list of (doc, score) tuples.
        """
        counts = {"1": 0, "0": 0, "-1": 0}
        for doc, _ in hits:
            val = str(doc.get("Output_2", "0")).strip()
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


# ── Module-level singleton ───────────────────────────────────────────────────
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = TFIDFEngine()
        _engine.build()
    return _engine
