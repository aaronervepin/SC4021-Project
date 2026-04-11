# 🔍 Reddit Opinion Search
### SC4021 Information Retrieval — NTU

A Django-based search engine for Singapore education opinions scraped from Reddit, powered by TF-IDF lnc.ltc ranked retrieval with Apache Solr as a fallback.

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.9+ | |
| Java (JDK/JRE) | 11+ | Required by Solr |
| Apache Solr | 9.x | [Download here](https://solr.apache.org/downloads.html) |
| pip | bundled with Python | |
| Git | any | |

---

## Setup Guide

### Step 1 — Clone the Repository

```bash
git clone https://github.com/aaronervepin/SC4021-Project.git
cd SC4021-Project
```

---

### Step 2 — Install Python Dependencies

```bash
pip install django scikit-learn numpy requests
```

Packages used:
- `django` — web framework and session management
- `scikit-learn` — TF-IDF vectorisation and cosine similarity
- `numpy` — array operations used by the search engine
- `requests` — HTTP calls to the Solr fallback endpoint

---

### Step 3 — Download and Start Apache Solr

#### 3.1 Download Solr

Download the latest binary release from https://solr.apache.org/downloads.html, then extract it:

```bash
# macOS / Linux
tar -xzf solr-9.x.x.tgz
cd solr-9.x.x

# Windows (PowerShell)
Expand-Archive solr-9.x.x.zip
cd solr-9.x.x
```

#### 3.2 Start the Solr Server

```bash
# macOS / Linux
bin/solr start

# Windows
bin\solr.cmd start
```

Solr starts on port **8983** by default. Confirm it is running by opening:
```
http://localhost:8983/solr/
```

> **Tip:** If you see a connection error, verify that Java 11+ is installed and on your PATH by running `java -version`.

#### 3.3 Create the Search Core

```bash
# macOS / Linux
bin/solr create -c reddit_opinions

# Windows
bin\solr.cmd create -c reddit_opinions
```

Verify the core was created at:
```
http://localhost:8983/solr/#/reddit_opinions
```

---

### Step 4 — Prepare the Dataset

The application expects the dataset at the following path relative to the project root:

```
data/crawled_enriched.csv
```

Create the directory and place the file there:

```bash
mkdir -p data
cp /path/to/crawled_enriched.csv data/
```

> **Note:** The TF-IDF engine reads this file at startup via `search/tfidf_engine.py`. If the file is missing, the engine will fail to build and search will fall back to Solr.

---

### Step 5 — Index the CSV into Solr

#### 5.1 Post the CSV

Run the following from your Solr installation directory:

```bash
# macOS / Linux
bin/solr post -c reddit_opinions \
  -type text/csv \
  /absolute/path/to/data/crawled_enriched.csv

# Windows (PowerShell)
java -Dtype=text/csv \
  -Durl=http://localhost:8983/solr/reddit_opinions/update \
  -jar example\exampledocs\post.jar \
  C:\absolute\path\to\data\crawled_enriched.csv
```

> **Note:** Use the absolute path to `crawled_enriched.csv`. Relative paths can fail depending on where you run the command.

#### 5.2 Verify Indexing

```
http://localhost:8983/solr/reddit_opinions/select?q=*:*&rows=1
```

The response should show `numFound` greater than 0. You can also commit pending changes explicitly:

```bash
curl http://localhost:8983/solr/reddit_opinions/update?commit=true
```

---

### Step 6 — Run Django Migrations

From the project root (where `manage.py` lives):

```bash
python manage.py migrate
```

This creates `db.sqlite3` and sets up the tables used by Django's session middleware for search history storage.

---

### Step 7 — Start the Development Server

```bash
python manage.py runserver
```

Open your browser and navigate to:
```
http://127.0.0.1:8000/
```

On first startup, the TF-IDF engine will automatically build its index from `crawled_enriched.csv`. Expect a short delay of a few seconds.

> **Tip:** You will see `[TF-IDF] Engine pre-built successfully.` in the terminal once the index is ready.

---

## Project Structure

```
SC4021-Project/
├── data/
│   └── crawled_enriched.csv       # Dataset (place here)
├── manage.py                      # Django entry point
├── opinion_search/
│   ├── settings.py
│   └── urls.py
├── search/
│   ├── tfidf_engine.py            # lnc.ltc ranked retrieval
│   ├── spell_correction.py        # Trigram + Levenshtein corrector
│   ├── views.py                   # Main search view
│   └── urls.py
└── templates/
    └── search/
        └── search.html            # Frontend
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| TF-IDF engine fails to build | Check that `data/crawled_enriched.csv` exists at the correct path |
| Solr connection refused | Run `bin/solr status` — make sure Solr is running on port 8983 |
| `[TF-IDF] Failed to pre-build engine` | Check terminal for the specific error; usually a missing CSV or pip package |
| No results returned | Ensure the CSV was indexed into Solr (Step 5) and the TF-IDF engine built (Step 7 log) |
| Port 8000 already in use | Run `python manage.py runserver 8080` to use a different port |
| Java not found / Solr won't start | Install Java 11+, verify with `java -version` |
