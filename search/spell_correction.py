"""
search/spell_correction.py
===========================
Isolated-word spell correction as described in rag-slides.pdf (Lecture 3).

Pipeline (two-stage, as recommended in the slides):
  1. Trigram overlap  — cheap pre-filter to get candidate words from the
                        corpus vocabulary without scanning every entry.
  2. Levenshtein distance — precise edit-distance ranking of candidates.

The lexicon is the single-word portion of the TF-IDF engine's fitted
vocabulary (min_df ≥ 2, so every word appears in at least 2 documents —
this naturally filters out random OCR/OCR noise that appears only once).

Usage (singleton, lazy-loaded):
    from search.spell_correction import get_corrector
    suggestion = get_corrector().correct_query("a lveel maths")
    # → "a level maths"  (or None if nothing needs fixing)
"""

import re
import logging
from collections import defaultdict
from typing import Optional, List

logger = logging.getLogger(__name__)


# ── Levenshtein distance ─────────────────────────────────────────────────────

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Standard dynamic-programming edit distance.
    Operations: insert, delete, replace — each costs 1.
    Space: O(min(|s1|, |s2|)) using a rolling 1-D array.
    """
    if s1 == s2:
        return 0
    # Keep s2 as the shorter string for the inner loop
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    m, n = len(s1), len(s2)

    # Early-exit: if length difference exceeds threshold we can skip later
    dp = list(range(n + 1))

    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev                          # no cost
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])  # replace / delete / insert
            prev = temp

    return dp[n]


# ── Trigram helpers ──────────────────────────────────────────────────────────

def _trigrams(word: str) -> List[str]:
    """
    Return all trigrams for *word* with $$ boundary padding.
    e.g. "cat" → ["$$c", "$ca", "cat", "at$", "t$$"]
    """
    padded = f"$${word}$$"
    return [padded[i: i + 3] for i in range(len(padded) - 2)]


# ── SpellCorrector ───────────────────────────────────────────────────────────

class SpellCorrector:
    """
    Two-stage spell corrector:
      Stage 1 — trigram inverted index narrows the search space from the
                full vocabulary down to a handful of candidates.
      Stage 2 — Levenshtein distance selects the closest match.
    """

    # Words in this set are treated as always-correct (SG edu jargon etc.)
    _DOMAIN_WHITELIST = {
        "psle", "olevel", "alevel", "nlevel", "emath", "amath",
        "gp", "jc", "ite", "poly", "hcl", "mtl", "bm", "tml",
        "prelim", "prelims", "tuition", "mugger", "mugging",
        "bellcurve", "sgexam", "reddit",
    }

    def __init__(self, max_dist: int = 2):
        self.max_dist = max_dist
        self.vocab: set = set()
        self._trigram_idx: dict = defaultdict(set)  # trigram → {word, …}
        self.built = False

    # ── Index construction ────────────────────────────────────────────────

    def build(self, words) -> None:
        """
        Populate vocabulary and trigram index from an iterable of words.
        Only pure-alphabetic words of length ≥ 3 are indexed.
        """
        count = 0
        for raw in words:
            w = raw.lower().strip()
            # Keep only plain alphabetic tokens of reasonable length
            if len(w) < 3 or len(w) > 25 or not re.match(r'^[a-z]+$', w):
                continue
            self.vocab.add(w)
            for tg in _trigrams(w):
                self._trigram_idx[tg].add(w)
            count += 1

        logger.info("[SpellCorrector] Built with %d words in lexicon.", count)
        self.built = True

    # ── Candidate retrieval (Stage 1) ─────────────────────────────────────

    def _candidates(self, word: str) -> set:
        """
        Return all vocabulary words that share ≥ 1 trigram with *word*.
        The slide suggests thresholding by overlap count; we keep the bar
        low (≥ 1) here and let Levenshtein do the real ranking in Stage 2.
        """
        overlap: dict = defaultdict(int)
        for tg in _trigrams(word):
            for cand in self._trigram_idx.get(tg, set()):
                overlap[cand] += 1

        # Require at least ceil(len/3) shared trigrams for longer words
        min_shared = max(1, len(_trigrams(word)) // 4)
        return {w for w, cnt in overlap.items() if cnt >= min_shared}

    # ── Single-word correction (Stage 2) ──────────────────────────────────

    def correct_word(self, word: str) -> Optional[str]:
        """
        Return the best vocabulary correction for *word*, or None if:
          • the word is already in the vocabulary, or
          • no candidate is within self.max_dist edits.
        """
        w = word.lower()

        # Already correct or in whitelist — nothing to do
        if w in self.vocab or w in self._DOMAIN_WHITELIST:
            return None

        # Very short words / numbers — don't try to correct
        if len(w) < 3 or not re.match(r'^[a-z]+$', w):
            return None

        candidates = self._candidates(w)
        if not candidates:
            return None

        best: Optional[str] = None
        best_d: int = self.max_dist + 1

        for cand in candidates:
            # Skip candidates whose length differs too much (quick prune)
            if abs(len(cand) - len(w)) > self.max_dist:
                continue
            d = levenshtein_distance(w, cand)
            if d < best_d:
                best_d, best = d, cand

        return best  # None if nothing within max_dist

    # ── Full query correction ─────────────────────────────────────────────

    def correct_query(self, query: str) -> Optional[str]:
        """
        Attempt to correct every token in *query*.
        Returns the corrected query string, or None if no changes were made.

        Only tokens that are clearly misspelled (not in vocab) are touched;
        correctly-spelled tokens are passed through unchanged.
        """
        if not self.built:
            return None

        tokens = query.strip().lower().split()
        corrected = []
        changed = False

        for tok in tokens:
            fix = self.correct_word(tok)
            if fix and fix != tok:
                corrected.append(fix)
                changed = True
            else:
                corrected.append(tok)

        return " ".join(corrected) if changed else None


# ── Module-level singleton ───────────────────────────────────────────────────

_corrector: Optional[SpellCorrector] = None


def get_corrector() -> SpellCorrector:
    """
    Lazy-load the SpellCorrector singleton.
    Vocabulary is sourced from the already-fitted TF-IDF engine so that
    the same corpus lexicon backs both retrieval and spell correction.
    """
    global _corrector
    if _corrector is None:
        _corrector = SpellCorrector(max_dist=2)

        # Grab single-word tokens from the TF-IDF vocabulary (no bigrams)
        try:
            from search.tfidf_engine import get_engine
            engine = get_engine()
            single_words = [
                term for term in engine._doc_vec.vocabulary_
                if " " not in term  # exclude bigrams
            ]
            _corrector.build(single_words)
            logger.info("[SpellCorrector] Loaded %d single-word vocab terms.", len(single_words))
        except Exception as exc:
            logger.warning("[SpellCorrector] Could not load corpus vocab: %s", exc)
            _corrector.built = True  # mark built even if empty, to avoid retry loop

    return _corrector