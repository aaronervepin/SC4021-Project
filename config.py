"""
Configuration settings for Reddit crawler focused on Singapore exams.
"""

import os

# Project directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Target subreddits for Singapore exam-related content
SUBREDDITS = [
    "SGExams",           # Main Singapore exams subreddit
    "singapore",         # General Singapore subreddit
    "NUS",               # National University of Singapore
    "NTU",               # Nanyang Technological University
    "SMU",               # Singapore Management University
    "SIT",               # Singapore Institute of Technology
    "SUSS",              # Singapore University of Social Sciences
    "nus",               # NUS alternative
]

# Search keywords for exam-related content in Singapore context
EXAM_KEYWORDS = [
    # National exams
    "O level", "O-level", "Olevel", "O levels",
    "A level", "A-level", "Alevel", "A levels",
    "PSLE",
    "N level", "N-level",
    
    # School types
    "JC", "junior college", "poly", "polytechnic",
    "ITE", "secondary school", "sec school",
    
    # Universities
    "NUS", "NTU", "SMU", "SUTD", "SIT", "SUSS",
    "university", "uni",
    
    # Exam-related terms
    "exam", "examination", "paper", "test",
    "midterm", "mid-term", "finals", "final exam",
    "prelim", "preliminary", "mock exam",
    
    # Difficulty-related terms
    "difficult", "difficulty", "hard", "easy",
    "tough", "killer", "manageable", "doable",
    "challenging", "tricky", "straightforward",
    
    # Subject-specific
    "math", "maths", "mathematics",
    "physics", "chemistry", "biology",
    "english", "chinese", "malay", "tamil",
    "history", "geography", "social studies",
    "economics", "econs", "accounting",
    "computing", "h1", "h2", "h3",
    
    # Module/course related
    "module", "course", "GPA", "CAP",
    "bell curve", "bellcurve", "grade",
    
    # Sentiment indicators
    "how was", "thoughts on", "what do you think",
    "feedback", "review", "experience",
    "struggled", "aced", "failed", "passed",
]

# Combined search queries (subreddit + keyword combinations)
SEARCH_QUERIES = [
    "O level exam difficulty",
    "A level paper hard",
    "A level paper easy",
    "O level math difficult",
    "A level physics hard",
    "A level chemistry killer",
    "JC exam experience",
    "poly exam feedback",
    "NUS exam difficult",
    "NTU finals hard",
    "SMU midterm",
    "university exam Singapore",
    "prelim exam hard",
    "PSLE difficult",
    "O level results",
    "A level results",
    "GCE exam feedback",
    "Singapore exam paper",
    "bell curve NUS",
    "CAP NTU",
    "module review NUS",
    "course feedback NTU",
    "exam tips Singapore",
    "study tips O level",
    "study tips A level",
]

# Pushshift API (archived Reddit data)
PUSHSHIFT_BASE_URL = "https://api.pushshift.io/reddit"
PUSHSHIFT_SEARCH_URL = f"{PUSHSHIFT_BASE_URL}/search"

# Reddit old.reddit.com for scraping (more scraping-friendly)
REDDIT_OLD_URL = "https://old.reddit.com"
REDDIT_WWW_URL = "https://www.reddit.com"

# Request settings
REQUEST_DELAY = 2  # Seconds between requests
MAX_RETRIES = 3
TIMEOUT = 30

# User agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
]

# Dataset settings
MIN_RECORDS = 10000
MIN_WORDS = 100000
MIN_TEXT_LENGTH = 20  # Minimum characters per entry
MAX_TEXT_LENGTH = 5000  # Maximum characters per entry

# Sentiment labels (following standard benchmark format)
SENTIMENT_LABELS = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

# Keywords for sentiment inference
POSITIVE_INDICATORS = [
    "easy", "manageable", "doable", "straightforward", "simple",
    "not hard", "not difficult", "passed", "aced", "scored well",
    "happy", "relieved", "confident", "good", "great", "excellent",
    "love", "enjoyed", "fun", "interesting", "fair", "reasonable",
    "well prepared", "comfortable", "smooth", "expected", "as expected",
    "thank god", "finally", "yay", "woohoo", "yes", "nice",
]

NEGATIVE_INDICATORS = [
    "hard", "difficult", "tough", "killer", "impossible",
    "failed", "struggled", "died", "rip", "gg", "gone case",
    "wtf", "what the", "screwed", "destroyed", "murdered",
    "unfair", "unreasonable", "unexpected", "tricky", "confusing",
    "stressed", "anxious", "worried", "scared", "panic",
    "no time", "not enough time", "rushed", "couldn't finish",
    "careless", "mistake", "wrong", "messed up", "regret",
    "crying", "depressed", "sad", "disappointed", "upset",
]

NEUTRAL_INDICATORS = [
    "okay", "ok", "alright", "average", "moderate",
    "not sure", "don't know", "maybe", "could be",
    "depends", "varies", "mixed", "some parts",
    "waiting", "results", "when", "how long",
    "anyone", "any tips", "advice", "help",
    "question", "asking", "wondering", "curious",
]

# Output file
OUTPUT_FILE = os.path.join(BASE_DIR, "eval.xls")
