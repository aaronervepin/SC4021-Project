"""
Custom template filters for query highlighting and KWIC snippets.
"""

import re
from django import template
from django.utils.safestring import mark_safe
from django.utils.html import escape

register = template.Library()


@register.filter
def highlight(text, query):
    """Wrap query terms in <mark> tags for highlighting."""
    if not query or not text:
        return text
    escaped_text = escape(text)
    terms = set(query.lower().split())
    for term in terms:
        if len(term) < 2:
            continue
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        escaped_text = pattern.sub(
            lambda m: f'<mark>{m.group(0)}</mark>', escaped_text
        )
    return mark_safe(escaped_text)


@register.filter
def kwic_snippet(text, query):
    """
    Extract a keyword-in-context snippet (up to 300 chars) centered
    around the first occurrence of a query term. Falls back to the
    start of the text if no match is found.
    """
    if not text:
        return ""
    if not query:
        return text[:300] + ("..." if len(text) > 300 else "")

    text_lower = text.lower()
    terms = query.lower().split()
    best_pos = -1

    for term in terms:
        if len(term) < 2:
            continue
        pos = text_lower.find(term)
        if pos != -1 and (best_pos == -1 or pos < best_pos):
            best_pos = pos

    snippet_len = 300
    if best_pos == -1:
        snippet = text[:snippet_len]
        prefix = ""
    else:
        # Center the snippet around the match
        start = max(0, best_pos - snippet_len // 3)
        snippet = text[start:start + snippet_len]
        prefix = "..." if start > 0 else ""

    suffix = "..." if len(text) > len(prefix) + len(snippet) else ""
    return prefix + snippet + suffix
