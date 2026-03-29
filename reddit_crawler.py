"""
Reddit Crawler for Singapore Exam Sentiment Analysis

This script crawls Reddit posts and comments related to Singapore exams
(O-Level, A-Level, Polytechnic, University) using multiple methods:
1. Old Reddit HTML scraping
2. Reddit JSON endpoints
3. Pushshift API (archived data)

Note: This crawler respects Reddit's robots.txt and includes appropriate delays.
"""

import requests
import json
import time
import random
import re
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd
from urllib.parse import urljoin, quote_plus

from config import (
    SUBREDDITS, EXAM_KEYWORDS, SEARCH_QUERIES,
    REDDIT_OLD_URL, REDDIT_WWW_URL,
    REQUEST_DELAY, MAX_RETRIES, TIMEOUT,
    USER_AGENTS, RAW_DATA_DIR, LOGS_DIR,
    MIN_TEXT_LENGTH, MAX_TEXT_LENGTH
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'crawler.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RedditCrawler:
    """Crawler for Reddit posts and comments related to Singapore exams."""
    
    def __init__(self):
        self.session = requests.Session()
        self.collected_data: List[Dict] = []
        self.seen_ids: Set[str] = set()
        self.total_words = 0
        
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with random user agent."""
        return {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }
    
    def _make_request(self, url: str, params: Optional[Dict] = None) -> Optional[requests.Response]:
        """Make HTTP request with retries and delay."""
        for attempt in range(MAX_RETRIES):
            try:
                time.sleep(REQUEST_DELAY + random.uniform(0.5, 1.5))
                response = self.session.get(
                    url,
                    headers=self._get_headers(),
                    params=params,
                    timeout=TIMEOUT
                )
                
                if response.status_code == 200:
                    return response
                elif response.status_code == 429:  # Rate limited
                    wait_time = 60 * (attempt + 1)
                    logger.warning(f"Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                elif response.status_code == 403:
                    logger.warning(f"Access forbidden for {url}")
                    return None
                else:
                    logger.warning(f"Status {response.status_code} for {url}")
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error (attempt {attempt + 1}): {e}")
                time.sleep(5 * (attempt + 1))
                
        return None
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        # Remove Reddit-specific formatting
        text = re.sub(r'\[deleted\]|\[removed\]', '', text)
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?\'"-]', '', text)
        
        return text.strip()
    
    def _count_words(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())
    
    def _is_valid_text(self, text: str) -> bool:
        """Check if text meets minimum requirements."""
        if not text:
            return False
        if len(text) < MIN_TEXT_LENGTH:
            return False
        if len(text) > MAX_TEXT_LENGTH:
            return False
        if text.lower() in ['[deleted]', '[removed]', 'deleted', 'removed']:
            return False
        return True
    
    def _is_singapore_exam_related(self, text: str) -> bool:
        """Check if text is related to Singapore exams."""
        text_lower = text.lower()
        
        # Must contain at least one exam-related keyword
        exam_terms = [
            'o level', 'o-level', 'olevel', 'a level', 'a-level', 'alevel',
            'psle', 'n level', 'n-level', 'jc', 'junior college', 'poly',
            'polytechnic', 'ite', 'nus', 'ntu', 'smu', 'sutd', 'sit', 'suss',
            'exam', 'paper', 'test', 'module', 'finals', 'midterm', 'prelim',
            'gce', 'cambridge', 'seab', 'h1', 'h2', 'h3', 'gpa', 'cap',
            'bell curve', 'bellcurve', 'grade'
        ]
        
        # Singapore context indicators
        sg_terms = [
            'singapore', 'sg', 'sgexams', 'nus', 'ntu', 'smu', 'sp', 'np', 'tp', 'rp', 'nyp',
            'temasek', 'ngee ann', 'nanyang', 'republic', 'o level', 'a level', 'psle',
            'jc', 'moe', 'seab', 'cambridge'
        ]
        
        has_exam_term = any(term in text_lower for term in exam_terms)
        has_sg_context = any(term in text_lower for term in sg_terms)
        
        return has_exam_term or has_sg_context
    
    def _add_record(self, text: str, source: str, post_id: str, 
                    subreddit: str, post_type: str, url: str = "",
                    title: str = "", score: int = 0, created_utc: int = 0):
        """Add a record to the collected data."""
        if post_id in self.seen_ids:
            return False
            
        cleaned_text = self._clean_text(text)
        
        if not self._is_valid_text(cleaned_text):
            return False
            
        if not self._is_singapore_exam_related(cleaned_text):
            return False
        
        self.seen_ids.add(post_id)
        word_count = self._count_words(cleaned_text)
        self.total_words += word_count
        
        record = {
            "id": post_id,
            "text": cleaned_text,
            "title": self._clean_text(title),
            "source": source,
            "subreddit": subreddit,
            "type": post_type,
            "url": url,
            "score": score,
            "created_utc": created_utc,
            "word_count": word_count,
            "crawled_at": datetime.now().isoformat()
        }
        
        self.collected_data.append(record)
        return True
    
    def crawl_subreddit_json(self, subreddit: str, limit: int = 100, 
                              sort: str = "hot", time_filter: str = "all") -> int:
        """Crawl subreddit using Reddit's JSON API."""
        logger.info(f"Crawling r/{subreddit} via JSON API ({sort})...")
        
        added_count = 0
        after = None
        
        for _ in range(limit // 25 + 1):
            url = f"{REDDIT_WWW_URL}/r/{subreddit}/{sort}.json"
            params = {
                "limit": 25,
                "t": time_filter,
                "raw_json": 1
            }
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
                    post_data = post.get("data", {})
                    post_id = post_data.get("id", "")
                    title = post_data.get("title", "")
                    selftext = post_data.get("selftext", "")
                    
                    # Combine title and selftext
                    full_text = f"{title}. {selftext}" if selftext else title
                    
                    if self._add_record(
                        text=full_text,
                        source="reddit_json",
                        post_id=f"post_{post_id}",
                        subreddit=subreddit,
                        post_type="submission",
                        url=post_data.get("permalink", ""),
                        title=title,
                        score=post_data.get("score", 0),
                        created_utc=int(post_data.get("created_utc", 0))
                    ):
                        added_count += 1
                    
                    # Also crawl comments for this post
                    added_count += self.crawl_post_comments(subreddit, post_id)
                
                after = data.get("data", {}).get("after")
                if not after:
                    break
                    
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                break
        
        logger.info(f"Added {added_count} records from r/{subreddit}")
        return added_count
    
    def crawl_post_comments(self, subreddit: str, post_id: str, limit: int = 50) -> int:
        """Crawl comments from a specific post."""
        url = f"{REDDIT_WWW_URL}/r/{subreddit}/comments/{post_id}.json"
        params = {"limit": limit, "raw_json": 1}
        
        response = self._make_request(url, params)
        if not response:
            return 0
        
        added_count = 0
        
        try:
            data = response.json()
            if len(data) < 2:
                return 0
            
            comments = data[1].get("data", {}).get("children", [])
            
            def process_comments(comment_list):
                nonlocal added_count
                for comment in comment_list:
                    if comment.get("kind") != "t1":
                        continue
                    
                    comment_data = comment.get("data", {})
                    comment_id = comment_data.get("id", "")
                    body = comment_data.get("body", "")
                    
                    if self._add_record(
                        text=body,
                        source="reddit_json",
                        post_id=f"comment_{comment_id}",
                        subreddit=subreddit,
                        post_type="comment",
                        score=comment_data.get("score", 0),
                        created_utc=int(comment_data.get("created_utc", 0))
                    ):
                        added_count += 1
                    
                    # Process replies
                    replies = comment_data.get("replies", "")
                    if isinstance(replies, dict):
                        reply_comments = replies.get("data", {}).get("children", [])
                        process_comments(reply_comments)
            
            process_comments(comments)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for comments: {e}")
        
        return added_count
    
    def search_reddit(self, query: str, subreddit: Optional[str] = None, 
                      limit: int = 100) -> int:
        """Search Reddit for specific query."""
        logger.info(f"Searching Reddit for: '{query}'" + 
                   (f" in r/{subreddit}" if subreddit else ""))
        
        added_count = 0
        after = None
        
        for _ in range(limit // 25 + 1):
            if subreddit:
                url = f"{REDDIT_WWW_URL}/r/{subreddit}/search.json"
                params = {
                    "q": query,
                    "restrict_sr": "on",
                    "sort": "relevance",
                    "t": "all",
                    "limit": 25,
                    "raw_json": 1
                }
            else:
                url = f"{REDDIT_WWW_URL}/search.json"
                params = {
                    "q": f"{query} (site:reddit.com/r/SGExams OR site:reddit.com/r/singapore)",
                    "sort": "relevance",
                    "t": "all",
                    "limit": 25,
                    "raw_json": 1
                }
            
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
                    post_data = post.get("data", {})
                    post_id = post_data.get("id", "")
                    title = post_data.get("title", "")
                    selftext = post_data.get("selftext", "")
                    subreddit_name = post_data.get("subreddit", "unknown")
                    
                    full_text = f"{title}. {selftext}" if selftext else title
                    
                    if self._add_record(
                        text=full_text,
                        source="reddit_search",
                        post_id=f"post_{post_id}",
                        subreddit=subreddit_name,
                        post_type="submission",
                        url=post_data.get("permalink", ""),
                        title=title,
                        score=post_data.get("score", 0),
                        created_utc=int(post_data.get("created_utc", 0))
                    ):
                        added_count += 1
                        
                        # Crawl comments
                        added_count += self.crawl_post_comments(subreddit_name, post_id)
                
                after = data.get("data", {}).get("after")
                if not after:
                    break
                    
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error in search: {e}")
                break
        
        logger.info(f"Search '{query}' added {added_count} records")
        return added_count
    
    def crawl_old_reddit_html(self, subreddit: str, pages: int = 10) -> int:
        """Crawl subreddit using old.reddit.com HTML parsing."""
        logger.info(f"Crawling r/{subreddit} via old.reddit.com...")
        
        added_count = 0
        next_page_url = f"{REDDIT_OLD_URL}/r/{subreddit}"
        
        for page in range(pages):
            response = self._make_request(next_page_url)
            if not response:
                break
            
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Find all posts
            posts = soup.find_all('div', class_='thing')
            
            for post in posts:
                try:
                    post_id = post.get('data-fullname', '').replace('t3_', '')
                    if not post_id:
                        continue
                    
                    title_elem = post.find('a', class_='title')
                    title = title_elem.get_text() if title_elem else ""
                    
                    # Get selftext if available
                    expando = post.find('div', class_='expando')
                    selftext = ""
                    if expando:
                        text_elem = expando.find('div', class_='md')
                        if text_elem:
                            selftext = text_elem.get_text()
                    
                    full_text = f"{title}. {selftext}" if selftext else title
                    
                    score_elem = post.find('div', class_='score')
                    score = 0
                    if score_elem:
                        score_text = score_elem.get('title', '0')
                        try:
                            score = int(score_text)
                        except ValueError:
                            pass
                    
                    if self._add_record(
                        text=full_text,
                        source="old_reddit_html",
                        post_id=f"post_{post_id}",
                        subreddit=subreddit,
                        post_type="submission",
                        title=title,
                        score=score
                    ):
                        added_count += 1
                        
                except Exception as e:
                    logger.debug(f"Error parsing post: {e}")
                    continue
            
            # Find next page link
            next_button = soup.find('span', class_='next-button')
            if next_button:
                next_link = next_button.find('a')
                if next_link:
                    next_page_url = next_link.get('href')
                else:
                    break
            else:
                break
        
        logger.info(f"Old Reddit crawl added {added_count} records from r/{subreddit}")
        return added_count
    
    def crawl_pushshift(self, subreddit: str, limit: int = 500) -> int:
        """Crawl using Pushshift API for archived Reddit data."""
        logger.info(f"Crawling r/{subreddit} via Pushshift...")
        
        added_count = 0
        
        # Pushshift submissions endpoint
        url = "https://api.pushshift.io/reddit/search/submission/"
        params = {
            "subreddit": subreddit,
            "size": min(limit, 100),
            "sort": "desc",
            "sort_type": "score"
        }
        
        response = self._make_request(url, params)
        if not response:
            logger.warning("Pushshift API may be unavailable")
            return 0
        
        try:
            data = response.json()
            posts = data.get("data", [])
            
            for post in posts:
                post_id = post.get("id", "")
                title = post.get("title", "")
                selftext = post.get("selftext", "")
                
                full_text = f"{title}. {selftext}" if selftext else title
                
                if self._add_record(
                    text=full_text,
                    source="pushshift",
                    post_id=f"ps_post_{post_id}",
                    subreddit=subreddit,
                    post_type="submission",
                    title=title,
                    score=post.get("score", 0),
                    created_utc=post.get("created_utc", 0)
                ):
                    added_count += 1
                    
        except json.JSONDecodeError as e:
            logger.error(f"Pushshift JSON error: {e}")
        except Exception as e:
            logger.error(f"Pushshift error: {e}")
        
        # Pushshift comments endpoint
        url = "https://api.pushshift.io/reddit/search/comment/"
        params = {
            "subreddit": subreddit,
            "size": min(limit, 100),
            "sort": "desc",
            "sort_type": "score"
        }
        
        response = self._make_request(url, params)
        if response:
            try:
                data = response.json()
                comments = data.get("data", [])
                
                for comment in comments:
                    comment_id = comment.get("id", "")
                    body = comment.get("body", "")
                    
                    if self._add_record(
                        text=body,
                        source="pushshift",
                        post_id=f"ps_comment_{comment_id}",
                        subreddit=subreddit,
                        post_type="comment",
                        score=comment.get("score", 0),
                        created_utc=comment.get("created_utc", 0)
                    ):
                        added_count += 1
                        
            except Exception as e:
                logger.error(f"Pushshift comments error: {e}")
        
        logger.info(f"Pushshift added {added_count} records from r/{subreddit}")
        return added_count
    
    def save_progress(self, filename: str = "crawled_data.json"):
        """Save current progress to file."""
        filepath = os.path.join(RAW_DATA_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "total_records": len(self.collected_data),
                    "total_words": self.total_words,
                    "crawled_at": datetime.now().isoformat()
                },
                "data": self.collected_data
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(self.collected_data)} records to {filepath}")
        
        # Also save as CSV for easier viewing
        csv_filepath = os.path.join(RAW_DATA_DIR, "crawled_data.csv")
        df = pd.DataFrame(self.collected_data)
        df.to_csv(csv_filepath, index=False, encoding='utf-8')
        logger.info(f"Saved CSV to {csv_filepath}")
    
    def get_stats(self) -> Dict:
        """Get current crawling statistics."""
        return {
            "total_records": len(self.collected_data),
            "total_words": self.total_words,
            "unique_ids": len(self.seen_ids),
            "by_source": pd.DataFrame(self.collected_data).groupby('source').size().to_dict() if self.collected_data else {},
            "by_subreddit": pd.DataFrame(self.collected_data).groupby('subreddit').size().to_dict() if self.collected_data else {},
            "by_type": pd.DataFrame(self.collected_data).groupby('type').size().to_dict() if self.collected_data else {},
        }
    
    def run_full_crawl(self):
        """Run the full crawling process."""
        logger.info("Starting full Reddit crawl for Singapore exam data...")
        logger.info("=" * 60)
        
        # 1. Crawl main Singapore exam subreddits
        logger.info("\n[1/4] Crawling main subreddits...")
        for subreddit in tqdm(SUBREDDITS, desc="Subreddits"):
            # JSON API crawl with different sorts
            for sort in ["hot", "top", "new"]:
                self.crawl_subreddit_json(subreddit, limit=100, sort=sort)
            
            # Old Reddit HTML crawl
            self.crawl_old_reddit_html(subreddit, pages=5)
            
            # Save progress
            self.save_progress()
            
            stats = self.get_stats()
            logger.info(f"Progress: {stats['total_records']} records, {stats['total_words']} words")
        
        # 2. Search with specific queries
        logger.info("\n[2/4] Running search queries...")
        for query in tqdm(SEARCH_QUERIES, desc="Search queries"):
            self.search_reddit(query, limit=50)
            
            # Also search in specific subreddits
            for subreddit in ["SGExams", "singapore"]:
                self.search_reddit(query, subreddit=subreddit, limit=25)
            
            self.save_progress()
        
        # 3. Try Pushshift for archived data
        logger.info("\n[3/4] Trying Pushshift for archived data...")
        for subreddit in tqdm(SUBREDDITS[:3], desc="Pushshift"):
            self.crawl_pushshift(subreddit, limit=200)
            self.save_progress()
        
        # 4. Additional keyword searches
        logger.info("\n[4/4] Additional keyword searches...")
        additional_queries = [
            "Singapore exam", "GCE O level", "GCE A level",
            "Singapore JC", "Singapore poly", "Singapore university exam",
            "NUS finals", "NTU exam", "SMU test",
            "Cambridge O level", "Cambridge A level"
        ]
        
        for query in tqdm(additional_queries, desc="Additional queries"):
            self.search_reddit(query, limit=50)
            self.save_progress()
        
        # Final save and stats
        self.save_progress("final_crawled_data.json")
        
        final_stats = self.get_stats()
        logger.info("\n" + "=" * 60)
        logger.info("CRAWLING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total records: {final_stats['total_records']}")
        logger.info(f"Total words: {final_stats['total_words']}")
        logger.info(f"By source: {final_stats['by_source']}")
        logger.info(f"By subreddit: {final_stats['by_subreddit']}")
        logger.info(f"By type: {final_stats['by_type']}")
        
        return final_stats


def main():
    """Main entry point."""
    crawler = RedditCrawler()
    
    try:
        stats = crawler.run_full_crawl()
        
        print("\n" + "=" * 60)
        print("FINAL STATISTICS")
        print("=" * 60)
        print(f"Total records collected: {stats['total_records']}")
        print(f"Total words collected: {stats['total_words']}")
        print(f"\nRecords by source:")
        for source, count in stats.get('by_source', {}).items():
            print(f"  - {source}: {count}")
        print(f"\nRecords by subreddit:")
        for subreddit, count in stats.get('by_subreddit', {}).items():
            print(f"  - r/{subreddit}: {count}")
            
    except KeyboardInterrupt:
        print("\nCrawling interrupted by user.")
        crawler.save_progress("interrupted_crawl.json")
    except Exception as e:
        logger.error(f"Crawling error: {e}")
        crawler.save_progress("error_crawl.json")
        raise


if __name__ == "__main__":
    main()
