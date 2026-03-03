"""Crawl configuration constants."""

import os
# from dotenv import load_dotenv

# load_dotenv("credentials.env")

# # --- Reddit API credentials (PRAW approach) ---
# REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
# REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
# REDDIT_USERNAME = os.getenv("REDDIT_USERNAME")
# REDDIT_PASSWORD = os.getenv("REDDIT_PASSWORD")
# REDDIT_USER_AGENT = os.getenv(
#     "REDDIT_USER_AGENT",
#     f"python:sg-exam-crawler:v1.0 (by u/{REDDIT_USERNAME})",
# )

# --- Public JSON API settings ---
BASE_URL = "https://www.reddit.com"
REQUEST_HEADERS = {
    "User-Agent": "sg-exam-crawler/1.0 (IR course assignment)"
}
REQUEST_DELAY = 2.0  # seconds between requests (conservative for unauthenticated)

# --- Target subreddits ---
SUBREDDITS = [
    "SGExams", "singapore", "NUS", "NTU", "SGAcademia",
]

# --- Search queries ---
SEARCH_QUERIES = [
    # Exam levels
    "O Level", "A Level", "H1 H2", "H3", "GCE", "PSLE",
    "N Level", "O level results", "A level results",
    # University exams
    "NUS exam", "NUS midterm", "NUS finals",
    "NTU exam", "NTU midterm", "NTU finals",
    "SMU exam", "SUTD exam", "SIT exam", "SUSS exam",
    "midterm exam", "finals exam", "exam paper",
    "exam review", "exam feedback", "exam stress",
    # Difficulty
    "exam hard", "exam easy", "exam difficult",
    "paper hard", "paper easy", "paper difficult",
    "killer paper", "bell curve", "exam was tough",
    "hardest paper", "easiest paper",
    # Subjects
    "math paper", "physics paper", "chemistry paper",
    "biology paper", "economics paper", "GP paper",
    "math exam", "physics exam", "chemistry exam",
    "computing paper", "history paper", "geography paper",
    "english paper", "chinese paper", "literature paper",
    # Sentiment-rich
    "failed exam", "aced exam", "exam results",
    "study tips", "how to study", "exam preparation",
    "exam anxiety", "exam grade", "CAP GPA",
    "passed exam", "flunked", "dean's list",
    # O/A Level specific subjects
    "A math O level", "E math O level", "additional math",
    "pure chemistry", "pure physics", "pure biology",
    "combined science", "combined humanities",
    "social studies O level", "POA O level",
    "H2 math", "H2 physics", "H2 chemistry", "H2 biology",
    "H2 economics", "H1 GP", "general paper A level",
    "project work A level", "H2 computing",
    "H2 history", "H2 geography", "H2 literature",
    "further math A level", "knowledge inquiry",
    # Exam megathreads and results
    "exam megathread", "results megathread", "prelim results",
    "promo results", "block test results",
    "L1R5", "L1R4", "rank points", "cut off point",
    # Study and sentiment
    "mugging for exam", "revision tips", "study plan",
    "failed midterm", "failed finals", "barely passed",
    "studying stress", "exam burnout", "GPA stress",
]

# --- Subreddit-specific browse strategies ---
# r/SGExams: browse everything (nearly all posts are exam-related)
# r/singapore: search only (most posts are unrelated to exams)
BROWSE_SUBREDDITS = ["SGExams", "NUS", "NTU"]
BROWSE_SORTS = ["hot", "new", "top"]
BROWSE_TIME_FILTERS = ["all", "year", "month"]
BROWSE_LIMIT = 1000

# --- Data schema ---
CSV_COLUMNS = [
    "id", "subreddit", "title", "body", "author", "score",
    "num_comments", "created_utc", "url", "post_type", "parent_id",
]

# --- Paths ---
DATA_DIR = "data"
RAW_CSV_PATH = os.path.join(DATA_DIR, "crawled_raw.csv")
CLEAN_CSV_PATH = os.path.join(DATA_DIR, "crawled_clean.csv")
PROGRESS_PATH = os.path.join(DATA_DIR, "crawl_progress.json")
STATS_PATH = os.path.join(DATA_DIR, "corpus_stats.json")
EVAL_DIR = "eval"
EVAL_SAMPLE_PATH = os.path.join(EVAL_DIR, "eval_sample.csv")

# --- Crawl settings ---
# COMMENTS_REPLACE_MORE_LIMIT = 0  # PRAW setting: 0 = fetch ALL nested comments
MIN_BODY_LENGTH = 10  # Skip [deleted], [removed], and very short text
MAX_PAGES_PER_TASK = 15  # Max pages to paginate per listing/search (25 posts per page)

# --- Rate limit settings ---
MAX_RETRIES = 3
BACKOFF_BASE = 60  # seconds — base wait on 429
BACKOFF_MULTIPLIER = 2  # exponential multiplier
