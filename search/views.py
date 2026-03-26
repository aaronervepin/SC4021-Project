import requests
from django.shortcuts import render

SOLR_URL = "http://localhost:8983/solr/reddit_opinions/select"

SYNONYM_GROUPS = {
    # E Math
    'emath': ['emath', 'e math', 'e-math', 'emaths', 'e-maths', 'elementary math', 'elementary mathematics'],
    'e math': ['emath', 'e math', 'e-math', 'emaths', 'e-maths', 'elementary math', 'elementary mathematics'],
    'e-math': ['emath', 'e math', 'e-math', 'emaths', 'e-maths', 'elementary math', 'elementary mathematics'],
    'emaths': ['emath', 'e math', 'e-math', 'emaths', 'e-maths', 'elementary math', 'elementary mathematics'],
    'e-maths': ['emath', 'e math', 'e-math', 'emaths', 'e-maths', 'elementary math', 'elementary mathematics'],
    'elementary math': ['emath', 'e math', 'e-math', 'emaths', 'e-maths', 'elementary math', 'elementary mathematics'],
    'elementary mathematics': ['emath', 'e math', 'e-math', 'emaths', 'e-maths', 'elementary math', 'elementary mathematics'],

    # A Math
    'amath': ['amath', 'a math', 'a-math', 'amaths', 'a-maths', 'additional math', 'additional mathematics'],
    'a math': ['amath', 'a math', 'a-math', 'amaths', 'a-maths', 'additional math', 'additional mathematics'],
    'a-math': ['amath', 'a math', 'a-math', 'amaths', 'a-maths', 'additional math', 'additional mathematics'],
    'amaths': ['amath', 'a math', 'a-math', 'amaths', 'a-maths', 'additional math', 'additional mathematics'],
    'a-maths': ['amath', 'a math', 'a-math', 'amaths', 'a-maths', 'additional math', 'additional mathematics'],
    'additional math': ['amath', 'a math', 'a-math', 'amaths', 'a-maths', 'additional math', 'additional mathematics'],
    'additional mathematics': ['amath', 'a math', 'a-math', 'amaths', 'a-maths', 'additional math', 'additional mathematics'],

    # O Level
    'olevel': ['olevel', 'o level', 'o-level', 'olevels', 'o levels', 'o-levels'],
    'o level': ['olevel', 'o level', 'o-level', 'olevels', 'o levels', 'o-levels'],
    'o-level': ['olevel', 'o level', 'o-level', 'olevels', 'o levels', 'o-levels'],
    'olevels': ['olevel', 'o level', 'o-level', 'olevels', 'o levels', 'o-levels'],
    'o levels': ['olevel', 'o level', 'o-level', 'olevels', 'o levels', 'o-levels'],
    'o-levels': ['olevel', 'o level', 'o-level', 'olevels', 'o levels', 'o-levels'],

    # A Level
    'alevel': ['alevel', 'a level', 'a-level', 'alevels', 'a levels', 'a-levels'],
    'a level': ['alevel', 'a level', 'a-level', 'alevels', 'a levels', 'a-levels'],
    'a-level': ['alevel', 'a level', 'a-level', 'alevels', 'a levels', 'a-levels'],
    'alevels': ['alevel', 'a level', 'a-level', 'alevels', 'a levels', 'a-levels'],
    'a levels': ['alevel', 'a level', 'a-level', 'alevels', 'a levels', 'a-levels'],
    'a-levels': ['alevel', 'a level', 'a-level', 'alevels', 'a levels', 'a-levels'],

    # Physics
    'phy': ['phy', 'phys', 'physics'],
    'phys': ['phy', 'phys', 'physics'],
    'physics': ['phy', 'phys', 'physics'],

    # Chemistry
    'chem': ['chem', 'chemistry'],
    'chemistry': ['chem', 'chemistry'],

    # Biology
    'bio': ['bio', 'biol', 'biology'],
    'biol': ['bio', 'biol', 'biology'],
    'biology': ['bio', 'biol', 'biology'],

    # General Paper
    'gp': ['gp', 'general paper'],
    'general paper': ['gp', 'general paper'],

    # Economics
    'econs': ['econs', 'econ', 'economics'],
    'econ': ['econs', 'econ', 'economics'],
    'economics': ['econs', 'econ', 'economics'],

    # Geography
    'geo': ['geo', 'geog', 'geography'],
    'geog': ['geo', 'geog', 'geography'],
    'geography': ['geo', 'geog', 'geography'],

    # History
    'hist': ['hist', 'history'],
    'history': ['hist', 'history'],

    # Literature
    'lit': ['lit', 'literature'],
    'literature': ['lit', 'literature'],

    # Social Studies
    'ss': ['ss', 'social studies'],
    'social studies': ['ss', 'social studies'],

    # Combined Science
    'combined science': ['combined science', 'combined sci', 'comb sci'],
    'combined sci': ['combined science', 'combined sci', 'comb sci'],
    'comb sci': ['combined science', 'combined sci', 'comb sci'],

    # Combined Humanities
    'combined humanities': ['combined humanities', 'comb hum'],
    'comb hum': ['combined humanities', 'comb hum'],

    # JC
    'jc': ['jc', 'junior college'],
    'junior college': ['jc', 'junior college'],

    # Poly
    'poly': ['poly', 'polys', 'polytechnic'],
    'polys': ['poly', 'polys', 'polytechnic'],
    'polytechnic': ['poly', 'polys', 'polytechnic'],

    # Prelim
    'prelim': ['prelim', 'prelims', 'preliminary'],
    'prelims': ['prelim', 'prelims', 'preliminary'],
    'preliminary': ['prelim', 'prelims', 'preliminary'],

    # Bell curve
    'bell curve': ['bell curve', 'bell-curve'],
    'bell-curve': ['bell curve', 'bell-curve'],

    # Tuition
    'tuition': ['tuition', 'tution'],
    'tution': ['tuition', 'tution'],

    # Mugging
    'mug': ['mug', 'mugger', 'mugging'],
    'mugger': ['mug', 'mugger', 'mugging'],
    'mugging': ['mug', 'mugger', 'mugging'],

    # Secondary / Primary
    'sec': ['sec', 'secondary'],
    'secondary': ['sec', 'secondary'],
    'pri': ['pri', 'primary'],
    'primary': ['pri', 'primary'],

    # Chinese
    'chinese': ['chinese', 'chi', 'cl', 'mandarin', 'higher chinese', 'hcl'],
    'chi': ['chinese', 'chi', 'cl', 'mandarin', 'higher chinese', 'hcl'],
    'cl': ['chinese', 'chi', 'cl', 'mandarin', 'higher chinese', 'hcl'],
    'mandarin': ['chinese', 'chi', 'cl', 'mandarin', 'higher chinese', 'hcl'],
    'higher chinese': ['chinese', 'chi', 'cl', 'mandarin', 'higher chinese', 'hcl'],
    'hcl': ['chinese', 'chi', 'cl', 'mandarin', 'higher chinese', 'hcl'],

    # Malay
    'malay': ['malay', 'bm', 'bahasa melayu', 'bahasa'],
    'bm': ['malay', 'bm', 'bahasa melayu', 'bahasa'],
    'bahasa melayu': ['malay', 'bm', 'bahasa melayu', 'bahasa'],
    'bahasa': ['malay', 'bm', 'bahasa melayu', 'bahasa'],

    # Tamil
    'tamil': ['tamil', 'tml'],
    'tml': ['tamil', 'tml'],

    # English
    'english': ['english', 'eng'],
    'eng': ['english', 'eng'],

    # Mother Tongue
    'mother tongue': ['mother tongue', 'mt', 'mtl'],
    'mt': ['mother tongue', 'mt', 'mtl'],
    'mtl': ['mother tongue', 'mt', 'mtl'],

    # PSLE
    'psle': ['psle', 'primary school leaving examination', 'primary school leaving exam', 'pri sch leaving exam'],
    'primary school leaving examination': ['psle', 'primary school leaving examination', 'primary school leaving exam', 'pri sch leaving exam'],
    'primary school leaving exam': ['psle', 'primary school leaving examination', 'primary school leaving exam', 'pri sch leaving exam'],
    'pri sch leaving exam': ['psle', 'primary school leaving examination', 'primary school leaving exam', 'pri sch leaving exam'],
    'pri sch leaving examination': ['psle', 'primary school leaving examination', 'primary school leaving exam'],

}

SORTED_SYNONYMS = sorted(SYNONYM_GROUPS.keys(), key=len, reverse=True)

def expand_query(query):
    q = query.lower().strip()
    # Normalize hyphens for synonym lookup
    q_normalized = q.replace('-', ' ')
    if q_normalized != q and q_normalized in SYNONYM_GROUPS:
        q = q_normalized
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

def get_sentiment_stats(docs):
    counts = {'1': 0, '0': 0, '-1': 0}
    for doc in docs:
        val = str(doc.get('Output_2', ['0'])[0] if isinstance(doc.get('Output_2'), list) else doc.get('Output_2', '0')).strip()
        if val in counts:
            counts[val] += 1
    total = sum(counts.values())
    if total == 0:
        return None
    return {
        'positive': round(counts['1'] / total * 100),
        'neutral': round(counts['0'] / total * 100),
        'negative': round(counts['-1'] / total * 100),
        'total': total,
    }

def search(request):
    query = request.GET.get('q', '')
    page = int(request.GET.get('page', 1))
    rows = 10
    start = (page - 1) * rows
    results = []
    num_found = 0
    qtime = 0
    num_pages = 0
    sentiment = None

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

        all_params = {
            'q': expand_query(query),
            'wt': 'json',
            'rows': num_found,
            'fl': 'Output_2',
        }
        all_response = requests.get(SOLR_URL, params=all_params)
        all_docs = all_response.json()['response']['docs']
        sentiment = get_sentiment_stats(all_docs)

    return render(request, 'search/search.html', {
        'results': results,
        'query': query,
        'num_found': num_found,
        'qtime': qtime,
        'page': page,
        'num_pages': num_pages,
        'prev_page': page - 1 if page > 1 else None,
        'next_page': page + 1 if page < num_pages else None,
        'sentiment': sentiment,
    })
