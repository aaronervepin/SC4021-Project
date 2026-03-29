"""
Old Reddit HTML Crawler for Singapore Exam Sentiment Analysis
Uses old.reddit.com which is more accessible than the JSON API.
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

# More legitimate looking headers
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}


class OldRedditCrawler:
    """Crawler using old.reddit.com HTML scraping."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.collected_data: List[Dict] = []
        self.seen_ids: Set[str] = set()
        self.seen_texts: Set[str] = set()
        self.total_words = 0
        self.load_existing_data()
    
    def load_existing_data(self):
        """Load existing crawled data."""
        files = [
            'crawled_data.json',
            'comprehensive_crawled_data.json',
            'selenium_crawled_data.json',
            'extended_crawled_data.json',
            'old_reddit_data.json'
        ]
        
        for filename in files:
            filepath = os.path.join(RAW_DATA_DIR, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        for record in data.get('data', []):
                            rid = record.get('id', '')
                            if rid and rid not in self.seen_ids:
                                self.seen_ids.add(rid)
                                text = record.get('text', '')
                                if text:
                                    self.seen_texts.add(hash(text[:100].lower()))
                                    self.collected_data.append(record)
                                    self.total_words += record.get('word_count', len(text.split()))
                    logger.info(f"Loaded from {filename}")
                except Exception as e:
                    logger.warning(f"Could not load {filename}: {e}")
        
        logger.info(f"Starting with {len(self.collected_data)} records, {self.total_words} words")
    
    def _make_request(self, url: str) -> Optional[requests.Response]:
        """Make HTTP request with retries."""
        for attempt in range(3):
            try:
                # Random delay to be respectful
                time.sleep(3 + random.uniform(1, 3))
                
                response = self.session.get(url, timeout=30)
                
                if response.status_code == 200:
                    return response
                elif response.status_code == 429:
                    logger.warning(f"Rate limited, waiting 60s...")
                    time.sleep(60)
                elif response.status_code == 403:
                    logger.warning(f"Forbidden: {url}")
                    return None
                else:
                    logger.warning(f"Status {response.status_code}: {url}")
                    
            except Exception as e:
                logger.error(f"Request error: {e}")
                time.sleep(10)
        
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
        text_hash = hash(text[:100].lower())
        if text_hash in self.seen_texts:
            return False
        return True
    
    def _is_sg_exam_related(self, text: str) -> bool:
        text_lower = text.lower()
        
        sg_terms = [
            'o level', 'a level', 'psle', 'n level', 'gce',
            'jc', 'junior college', 'poly', 'polytechnic', 'ite',
            'nus', 'ntu', 'smu', 'sutd', 'sit', 'suss',
            'sgexams', 'singapore', 'seab', 'cambridge',
            'h1', 'h2', 'h3', 'prelim', 'mye', 'promo',
            'bell curve', 'cap', 'gpa', 'module',
            'exam', 'paper', 'test', 'finals', 'midterm'
        ]
        
        return any(term in text_lower for term in sg_terms)
    
    def _add_record(self, text: str, post_id: str, subreddit: str, 
                    post_type: str, **kwargs) -> bool:
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
            "source": "old_reddit_html",
            "subreddit": subreddit,
            "type": post_type,
            "word_count": word_count,
            "crawled_at": datetime.now().isoformat(),
            **kwargs
        }
        self.collected_data.append(record)
        return True
    
    def crawl_subreddit_page(self, subreddit: str, url: str = None) -> tuple:
        """Crawl a single page of a subreddit. Returns (added_count, next_url)."""
        if url is None:
            url = f"https://old.reddit.com/r/{subreddit}"
        
        response = self._make_request(url)
        if not response:
            return 0, None
        
        added = 0
        next_url = None
        
        try:
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Find all posts
            posts = soup.find_all('div', class_='thing', attrs={'data-type': 'link'})
            
            for post in posts:
                try:
                    post_id = post.get('data-fullname', '').replace('t3_', '')
                    if not post_id:
                        continue
                    
                    # Get title
                    title_elem = post.find('a', class_='title')
                    title = title_elem.get_text().strip() if title_elem else ""
                    
                    # Get selftext if it's a text post
                    selftext = ""
                    expando = post.find('div', class_='expando')
                    if expando:
                        usertext = expando.find('div', class_='usertext-body')
                        if usertext:
                            selftext = usertext.get_text().strip()
                    
                    full_text = f"{title}. {selftext}" if selftext else title
                    
                    if self._add_record(
                        text=full_text,
                        post_id=f"old_{post_id}",
                        subreddit=subreddit,
                        post_type="submission",
                        title=title
                    ):
                        added += 1
                    
                    # Also get comments from this post
                    comments_link = post.find('a', class_='comments')
                    if comments_link:
                        comment_url = comments_link.get('href')
                        if comment_url:
                            added += self.crawl_post_comments(comment_url, subreddit)
                    
                except Exception as e:
                    logger.debug(f"Error parsing post: {e}")
                    continue
            
            # Find next page
            next_button = soup.find('span', class_='next-button')
            if next_button:
                next_link = next_button.find('a')
                if next_link:
                    next_url = next_link.get('href')
            
        except Exception as e:
            logger.error(f"Error parsing page: {e}")
        
        return added, next_url
    
    def crawl_post_comments(self, url: str, subreddit: str) -> int:
        """Crawl comments from a post page."""
        # Make sure we're using old.reddit.com
        if 'old.reddit.com' not in url:
            url = url.replace('www.reddit.com', 'old.reddit.com')
            url = url.replace('reddit.com', 'old.reddit.com')
        
        response = self._make_request(url)
        if not response:
            return 0
        
        added = 0
        
        try:
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Find all comments
            comments = soup.find_all('div', class_='comment')
            
            for comment in comments:
                try:
                    comment_id = comment.get('data-fullname', '').replace('t1_', '')
                    if not comment_id:
                        continue
                    
                    # Get comment text
                    usertext = comment.find('div', class_='usertext-body')
                    if not usertext:
                        continue
                    
                    text = usertext.get_text().strip()
                    
                    if self._add_record(
                        text=text,
                        post_id=f"old_comment_{comment_id}",
                        subreddit=subreddit,
                        post_type="comment"
                    ):
                        added += 1
                        
                except Exception as e:
                    logger.debug(f"Error parsing comment: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error parsing comments: {e}")
        
        return added
    
    def crawl_subreddit(self, subreddit: str, pages: int = 20) -> int:
        """Crawl multiple pages of a subreddit."""
        logger.info(f"Crawling r/{subreddit} ({pages} pages)...")
        
        total_added = 0
        url = None
        
        for page in range(pages):
            added, next_url = self.crawl_subreddit_page(subreddit, url)
            total_added += added
            
            logger.info(f"  Page {page + 1}: +{added} records (total: {len(self.collected_data)})")
            
            if not next_url:
                break
            url = next_url
            
            # Save progress periodically
            if page % 5 == 0:
                self.save_progress()
        
        logger.info(f"  r/{subreddit} total: +{total_added} records")
        return total_added
    
    def crawl_search(self, query: str, subreddit: str = None, pages: int = 5) -> int:
        """Search using old.reddit.com search."""
        logger.info(f"Searching: '{query}'" + (f" in r/{subreddit}" if subreddit else ""))
        
        total_added = 0
        
        if subreddit:
            base_url = f"https://old.reddit.com/r/{subreddit}/search"
        else:
            base_url = "https://old.reddit.com/search"
        
        url = f"{base_url}?q={requests.utils.quote(query)}&restrict_sr=on&sort=relevance&t=all"
        
        for page in range(pages):
            response = self._make_request(url)
            if not response:
                break
            
            try:
                soup = BeautifulSoup(response.text, 'lxml')
                
                # Find search results
                results = soup.find_all('div', class_='search-result-link')
                if not results:
                    # Try alternate structure
                    results = soup.find_all('div', class_='thing')
                
                for result in results:
                    try:
                        post_id = result.get('data-fullname', '').replace('t3_', '')
                        if not post_id:
                            continue
                        
                        title_elem = result.find('a', class_='search-title') or result.find('a', class_='title')
                        title = title_elem.get_text().strip() if title_elem else ""
                        
                        # Get subreddit
                        sub_elem = result.find('a', class_='search-subreddit-link')
                        sub = sub_elem.get_text().replace('/r/', '').strip() if sub_elem else subreddit or "unknown"
                        
                        if self._add_record(
                            text=title,
                            post_id=f"search_{post_id}",
                            subreddit=sub,
                            post_type="submission",
                            title=title
                        ):
                            total_added += 1
                            
                    except Exception as e:
                        continue
                
                # Find next page
                next_button = soup.find('span', class_='next-button')
                if next_button:
                    next_link = next_button.find('a')
                    if next_link:
                        url = next_link.get('href')
                    else:
                        break
                else:
                    break
                    
            except Exception as e:
                logger.error(f"Search error: {e}")
                break
        
        return total_added
    
    def save_progress(self):
        """Save current data."""
        filepath = os.path.join(RAW_DATA_DIR, "old_reddit_data.json")
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
        """Run the crawl."""
        logger.info(f"Target: {target_records} records")
        logger.info(f"Current: {len(self.collected_data)} records")
        
        # Primary subreddits
        subreddits = [
            "SGExams", "singapore", "NUS", "NTU", "SMU",
            "askSingapore", "SingaporeRaw"
        ]
        
        # Search queries
        search_queries = [
            "O level", "A level", "PSLE", "N level",
            "JC exam", "poly exam", "university exam",
            "NUS finals", "NTU exam", "SMU test",
            "exam difficult", "exam hard", "exam easy",
            "results release", "bell curve", "CAP GPA",
            "prelim exam", "promos exam", "MYE exam"
        ]
        
        # 1. Crawl subreddits
        for subreddit in subreddits:
            if len(self.collected_data) >= target_records:
                break
            
            self.crawl_subreddit(subreddit, pages=30)
            self.save_progress()
        
        # 2. Search queries
        for query in tqdm(search_queries, desc="Searching"):
            if len(self.collected_data) >= target_records:
                break
            
            self.crawl_search(query)
            
            for sub in ["SGExams", "singapore"]:
                self.crawl_search(query, sub)
            
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
    crawler = OldRedditCrawler()
    records, words = crawler.run(target_records=10000)
    
    print(f"\n{'='*60}")
    print(f"FINAL: {records} records, {words} words")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
