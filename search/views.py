import requests
from django.shortcuts import render

SOLR_URL = "http://localhost:8983/solr/reddit_opinions/select"

SYNONYM_GROUPS = {
    'emath': ['emath', 'e math', 'emaths', 'elementary math', 'elementary mathematics'],
    'e math': ['emath', 'e math', 'emaths', 'elementary math', 'elementary mathematics'],
    'emaths': ['emath', 'e math', 'emaths', 'elementary math', 'elementary mathematics'],
    'elementary math': ['emath', 'e math', 'emaths', 'elementary math', 'elementary mathematics'],
    'elementary mathematics': ['emath', 'e math', 'emaths', 'elementary math', 'elementary mathematics'],
    'amath': ['amath', 'a math', 'amaths', 'additional math', 'additional mathematics'],
    'a math': ['amath', 'a math', 'amaths', 'additional math', 'additional mathematics'],
    'amaths': ['amath', 'a math', 'amaths', 'additional math', 'additional mathematics'],
    'additional math': ['amath', 'a math', 'amaths', 'additional math', 'additional mathematics'],
    'additional mathematics': ['amath', 'a math', 'amaths', 'additional math', 'additional mathematics'],
    'olevel': ['olevel', 'o level', 'o-level', 'olevels', 'o levels'],
    'o level': ['olevel', 'o level', 'o-level', 'olevels', 'o levels'],
    'o-level': ['olevel', 'o level', 'o-level', 'olevels', 'o levels'],
    'olevels': ['olevel', 'o level', 'o-level', 'olevels', 'o levels'],
    'o levels': ['olevel', 'o level', 'o-level', 'olevels', 'o levels'],
    'alevel': ['alevel', 'a level', 'a-level', 'alevels', 'a levels'],
    'a level': ['alevel', 'a level', 'a-level', 'alevels', 'a levels'],
    'a-level': ['alevel', 'a level', 'a-level', 'alevels', 'a levels'],
    'alevels': ['alevel', 'a level', 'a-level', 'alevels', 'a levels'],
    'a levels': ['alevel', 'a level', 'a-level', 'alevels', 'a levels'],
    'phy': ['phy', 'physics'],
    'physics': ['phy', 'physics'],
    'chem': ['chem', 'chemistry'],
    'chemistry': ['chem', 'chemistry'],
    'bio': ['bio', 'biology'],
    'biology': ['bio', 'biology'],
    'gp': ['gp', 'general paper'],
    'general paper': ['gp', 'general paper'],
    'econs': ['econs', 'econ', 'economics'],
    'econ': ['econs', 'econ', 'economics'],
    'economics': ['econs', 'econ', 'economics'],
    'geo': ['geo', 'geography'],
    'geography': ['geo', 'geography'],
    'hist': ['hist', 'history'],
    'history': ['hist', 'history'],
    'lit': ['lit', 'literature'],
    'literature': ['lit', 'literature'],
    'jc': ['jc', 'junior college'],
    'junior college': ['jc', 'junior college'],
    'poly': ['poly', 'polytechnic'],
    'polytechnic': ['poly', 'polytechnic'],
    'prelim': ['prelim', 'prelims', 'preliminary'],
    'prelims': ['prelim', 'prelims', 'preliminary'],
    'preliminary': ['prelim', 'prelims', 'preliminary'],
    'bell curve': ['bell curve', 'bell-curve'],
    'bell-curve': ['bell curve', 'bell-curve'],
    'tuition': ['tuition', 'tution'],
    'tution': ['tuition', 'tution'],
    'mug': ['mug', 'mugger'],
    'mugger': ['mug', 'mugger'],
    'ss': ['ss', 'social studies'],
    'social studies': ['ss', 'social studies'],
}

SORTED_SYNONYMS = sorted(SYNONYM_GROUPS.keys(), key=len, reverse=True)

def expand_query(query):
    q = query.lower().strip()
    matched_terms = []
    remaining = q

    while remaining:
        matched = False
        for key in SORTED_SYNONYMS:
            if remaining.startswith(key):
                variants = SYNONYM_GROUPS[key]
                group_query = ' OR '.join([f'title:"{v}" OR body:"{v}"' for v in variants])
                matched_terms.append(f'({group_query})')
                remaining = remaining[len(key):].strip()
                matched = True
                break
        if not matched:
            parts = remaining.split(' ', 1)
            word = parts[0]
            matched_terms.append(f'(title:"{word}" OR body:"{word}")')
            remaining = parts[1].strip() if len(parts) > 1 else ''

    return ' AND '.join(matched_terms)

def search(request):
    query = request.GET.get('q', '')
    page = int(request.GET.get('page', 1))
    rows = 10
    start = (page - 1) * rows
    results = []
    num_found = 0
    qtime = 0
    num_pages = 0

    if query:
        params = {
            'q': expand_query(query),
            'wt': 'json',
            'rows': rows,
            'start': start,
        }
        response = requests.get(SOLR_URL, params=params)
        data = response.json()
        results = data['response']['docs']
        num_found = data['response']['numFound']
        qtime = data['responseHeader']['QTime']
        num_pages = (num_found + rows - 1) // rows

    return render(request, 'search/search.html', {
        'results': results,
        'query': query,
        'num_found': num_found,
        'qtime': qtime,
        'page': page,
        'num_pages': num_pages,
        'prev_page': page - 1 if page > 1 else None,
        'next_page': page + 1 if page < num_pages else None,
    })