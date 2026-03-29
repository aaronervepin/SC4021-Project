"""
Extended Reddit Crawler for Singapore Exam Sentiment Analysis
This script aggressively crawls Reddit to collect more data.
"""

import requests
import json
import time
import random
import re
import os
import logging
from datetime import datetime
from typing import List, Dict, Optional, Set
from bs4 import BeautifulSoup
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# User agents
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
]

class ExtendedRedditCrawler:
    """Extended crawler with more aggressive data collection."""
    
    def __init__(self):
        self.session = requests.Session()
        self.collected_data: List[Dict] = []
        self.seen_ids: Set[str] = set()
        self.seen_texts: Set[str] = set()
        self.total_words = 0
        self.load_existing_data()
        
    def load_existing_data(self):
        """Load existing crawled data to avoid duplicates."""
        files = [
            'crawled_data.json',
            'comprehensive_crawled_data.json',
            'selenium_crawled_data.json',
            'extended_crawled_data.json'
        ]
        
        for filename in files:
            filepath = os.path.join(RAW_DATA_DIR, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        for record in data.get('data', []):
                            self.seen_ids.add(record.get('id', ''))
                            # Also track text hash to avoid similar content
                            text = record.get('text', '')
                            if text:
                                self.seen_texts.add(hash(text[:100].lower()))
                                self.collected_data.append(record)
                                self.total_words += record.get('word_count', len(text.split()))
                    logger.info(f"Loaded {len(self.seen_ids)} existing records from {filename}")
                except Exception as e:
                    logger.warning(f"Could not load {filename}: {e}")
        
        logger.info(f"Starting with {len(self.collected_data)} existing records, {self.total_words} words")
    
    def _get_headers(self) -> Dict[str, str]:
        return {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
    
    def _make_request(self, url: str, params: Optional[Dict] = None) -> Optional[requests.Response]:
        """Make HTTP request with retries."""
        for attempt in range(3):
            try:
                time.sleep(2 + random.uniform(0.5, 1.5))
                response = self.session.get(
                    url,
                    headers=self._get_headers(),
                    params=params,
                    timeout=30
                )
                if response.status_code == 200:
                    return response
                elif response.status_code == 429:
                    logger.warning(f"Rate limited, waiting...")
                    time.sleep(60 * (attempt + 1))
                else:
                    logger.warning(f"Status {response.status_code} for {url}")
            except Exception as e:
                logger.error(f"Request error: {e}")
                time.sleep(5)
        return None
    
    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'\[deleted\]|\[removed\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _is_valid(self, text: str) -> bool:
        if not text or len(text) < 20 or len(text) > 5000:
            return False
        if text.lower() in ['[deleted]', '[removed]', 'deleted', 'removed']:
            return False
        # Check for duplicate content
        text_hash = hash(text[:100].lower())
        if text_hash in self.seen_texts:
            return False
        return True
    
    def _is_sg_exam_related(self, text: str) -> bool:
        text_lower = text.lower()
        
        sg_exam_terms = [
            'o level', 'a level', 'psle', 'n level', 'gce',
            'jc', 'junior college', 'poly', 'polytechnic', 'ite',
            'nus', 'ntu', 'smu', 'sutd', 'sit', 'suss',
            'sgexams', 'singapore', 'seab', 'cambridge',
            'h1', 'h2', 'h3', 'prelim', 'mye', 'promo',
            'bell curve', 'cap', 'gpa', 'module',
            'exam', 'paper', 'test', 'finals', 'midterm'
        ]
        
        return any(term in text_lower for term in sg_exam_terms)
    
    def _add_record(self, text: str, source: str, post_id: str, 
                    subreddit: str, post_type: str, **kwargs) -> bool:
        if post_id in self.seen_ids:
            return False
        
        cleaned_text = self._clean_text(text)
        if not self._is_valid(cleaned_text):
            return False
        if not self._is_sg_exam_related(cleaned_text):
            return False
        
        self.seen_ids.add(post_id)
        self.seen_texts.add(hash(cleaned_text[:100].lower()))
        
        word_count = len(cleaned_text.split())
        self.total_words += word_count
        
        record = {
            "id": post_id,
            "text": cleaned_text,
            "source": source,
            "subreddit": subreddit,
            "type": post_type,
            "word_count": word_count,
            "crawled_at": datetime.now().isoformat(),
            **kwargs
        }
        self.collected_data.append(record)
        return True
    
    def crawl_subreddit(self, subreddit: str, sort: str = "top", 
                        time_filter: str = "all", pages: int = 10) -> int:
        """Crawl a subreddit using JSON API."""
        logger.info(f"Crawling r/{subreddit} ({sort}/{time_filter})...")
        added = 0
        after = None
        
        for page in range(pages):
            url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
            params = {"t": time_filter, "limit": 100, "raw_json": 1}
            if after:
                params["after"] = after
            
            response = self._make_request(url, params)
            if not response:
                break
            
            try:
                data = response.json()
                posts = data.get("data", {}).get("children", [])
                
                if not posts:
                    break
                
                for post in posts:
                    pd = post.get("data", {})
                    post_id = pd.get("id", "")
                    title = pd.get("title", "")
                    selftext = pd.get("selftext", "")
                    full_text = f"{title}. {selftext}" if selftext else title
                    
                    if self._add_record(
                        text=full_text,
                        source="reddit_json",
                        post_id=f"post_{post_id}",
                        subreddit=subreddit,
                        post_type="submission",
                        score=pd.get("score", 0),
                        title=title
                    ):
                        added += 1
                    
                    # Get comments for this post
                    added += self.crawl_comments(subreddit, post_id)
                
                after = data.get("data", {}).get("after")
                if not after:
                    break
                    
            except Exception as e:
                logger.error(f"Error: {e}")
                break
        
        logger.info(f"Added {added} from r/{subreddit}")
        return added
    
    def crawl_comments(self, subreddit: str, post_id: str) -> int:
        """Crawl comments from a post."""
        url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json"
        params = {"limit": 500, "depth": 10, "raw_json": 1}
        
        response = self._make_request(url, params)
        if not response:
            return 0
        
        added = 0
        try:
            data = response.json()
            if len(data) < 2:
                return 0
            
            def process_comments(comments):
                nonlocal added
                for comment in comments:
                    if comment.get("kind") != "t1":
                        continue
                    cd = comment.get("data", {})
                    comment_id = cd.get("id", "")
                    body = cd.get("body", "")
                    
                    if self._add_record(
                        text=body,
                        source="reddit_json",
                        post_id=f"comment_{comment_id}",
                        subreddit=subreddit,
                        post_type="comment",
                        score=cd.get("score", 0)
                    ):
                        added += 1
                    
                    # Process replies
                    replies = cd.get("replies", "")
                    if isinstance(replies, dict):
                        reply_comments = replies.get("data", {}).get("children", [])
                        process_comments(reply_comments)
            
            comments = data[1].get("data", {}).get("children", [])
            process_comments(comments)
            
        except Exception as e:
            logger.error(f"Comment error: {e}")
        
        return added
    
    def search_reddit(self, query: str, subreddit: Optional[str] = None) -> int:
        """Search Reddit for specific query."""
        logger.info(f"Searching: '{query}'" + (f" in r/{subreddit}" if subreddit else ""))
        added = 0
        after = None
        
        for _ in range(5):
            if subreddit:
                url = f"https://www.reddit.com/r/{subreddit}/search.json"
                params = {"q": query, "restrict_sr": "on", "sort": "relevance", 
                         "t": "all", "limit": 100, "raw_json": 1}
            else:
                url = "https://www.reddit.com/search.json"
                params = {"q": query, "sort": "relevance", "t": "all", 
                         "limit": 100, "raw_json": 1}
            
            if after:
                params["after"] = after
            
            response = self._make_request(url, params)
            if not response:
                break
            
            try:
                data = response.json()
                posts = data.get("data", {}).get("children", [])
                
                if not posts:
                    break
                
                for post in posts:
                    pd = post.get("data", {})
                    post_id = pd.get("id", "")
                    sub = pd.get("subreddit", "unknown")
                    title = pd.get("title", "")
                    selftext = pd.get("selftext", "")
                    full_text = f"{title}. {selftext}" if selftext else title
                    
                    if self._add_record(
                        text=full_text,
                        source="reddit_search",
                        post_id=f"search_{post_id}",
                        subreddit=sub,
                        post_type="submission",
                        score=pd.get("score", 0),
                        title=title
                    ):
                        added += 1
                        added += self.crawl_comments(sub, post_id)
                
                after = data.get("data", {}).get("after")
                if not after:
                    break
                    
            except Exception as e:
                logger.error(f"Search error: {e}")
                break
        
        return added
    
    def save_progress(self):
        """Save current data."""
        filepath = os.path.join(RAW_DATA_DIR, "extended_crawled_data.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "total_records": len(self.collected_data),
                    "total_words": self.total_words,
                    "crawled_at": datetime.now().isoformat()
                },
                "data": self.collected_data
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(self.collected_data)} records ({self.total_words} words)")
    
    def run(self, target_records: int = 10000):
        """Run the extended crawl."""
        logger.info(f"Target: {target_records} records")
        logger.info(f"Current: {len(self.collected_data)} records, {self.total_words} words")
        
        # Primary subreddits
        subreddits = [
            "SGExams", "singapore", "NUS", "NTU", "SMU", 
            "askSingapore", "SingaporeRaw"
        ]
        
        # Search queries for Singapore exams
        search_queries = [
            # O/A Levels
            "O level exam", "A level exam", "O level results",
            "A level results", "O level difficult", "A level hard",
            "O level math", "O level english", "O level science",
            "A level H2", "A level H1", "A level GP",
            "GCE O level", "GCE A level", "Cambridge exam",
            
            # JC
            "JC exam", "JC promos", "JC prelims", "junior college exam",
            "JC H2 math", "JC H2 physics", "JC H2 chemistry",
            "JC stress", "JC mugging", "JC results",
            
            # Poly
            "poly exam", "polytechnic exam", "poly GPA",
            "SP exam", "NP exam", "TP exam", "NYP exam", "RP exam",
            "poly finals", "poly stress",
            
            # University
            "NUS exam", "NUS finals", "NUS midterm", "NUS module",
            "NTU exam", "NTU finals", "NTU module", "NTU bell curve",
            "SMU exam", "SMU finals", "SMU class participation",
            "SUTD exam", "SIT exam", "SUSS exam",
            "uni CAP", "university GPA", "bell curve Singapore",
            
            # General
            "Singapore exam stress", "Singapore exam tips",
            "PSLE exam", "N level exam", "ITE exam",
            "SEAB exam", "MOE exam",
            "prelim exam Singapore", "MYE exam Singapore"
        ]
        
        # 1. Crawl subreddits with different sorts
        for subreddit in subreddits:
            if len(self.collected_data) >= target_records:
                break
            
            for sort in ["top", "hot", "new"]:
                for time_filter in ["all", "year", "month"]:
                    self.crawl_subreddit(subreddit, sort, time_filter, pages=5)
                    self.save_progress()
                    
                    if len(self.collected_data) >= target_records:
                        break
        
        # 2. Search queries
        for query in tqdm(search_queries, desc="Searching"):
            if len(self.collected_data) >= target_records:
                break
            
            self.search_reddit(query)
            
            # Also search in specific subreddits
            for sub in ["SGExams", "singapore", "NUS"]:
                self.search_reddit(query, sub)
            
            self.save_progress()
        
        # Final save
        self.save_progress()
        
        logger.info("=" * 60)
        logger.info("CRAWLING COMPLETE")
        logger.info(f"Total records: {len(self.collected_data)}")
        logger.info(f"Total words: {self.total_words}")
        logger.info("=" * 60)
        
        return len(self.collected_data), self.total_words


def main():
    crawler = ExtendedRedditCrawler()
    records, words = crawler.run(target_records=10000)
    
    print(f"\n{'='*60}")
    print(f"FINAL: {records} records, {words} words")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
