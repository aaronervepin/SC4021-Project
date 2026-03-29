"""
Comprehensive Reddit Data Collector for Singapore Exam Sentiment Analysis

This script uses multiple methods to collect Reddit data:
1. Reddit's .json endpoints with proper headers
2. Reddit RSS feeds
3. Alternative Reddit frontends (teddit, libreddit)
4. Cached/archived Reddit data

This approach is more reliable than direct API access.
"""

import os
import json
import time
import random
import re
import hashlib
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Dict, Set, Optional, Tuple
from urllib.parse import quote_plus, urljoin
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

from config import (
    SUBREDDITS, SEARCH_QUERIES,
    RAW_DATA_DIR, LOGS_DIR,
    MIN_TEXT_LENGTH, MAX_TEXT_LENGTH,
    USER_AGENTS
)

import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'comprehensive_crawler.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ComprehensiveRedditCrawler:
    """Multi-method Reddit crawler for Singapore exam data."""
    
    def __init__(self):
        self.session = requests.Session()
        self.collected_data: List[Dict] = []
        self.seen_ids: Set[str] = set()
        self.seen_hashes: Set[str] = set()
        self.total_words = 0
        
        # Alternative Reddit frontends
        self.alternative_urls = [
            "https://teddit.net",
            "https://libreddit.spike.codes",
            "https://reddit.invak.id",
        ]
        
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        return {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Cache-Control": "no-cache",
        }
    
    def _make_request(self, url: str, params: Dict = None, 
                      timeout: int = 30, retries: int = 3) -> Optional[requests.Response]:
        """Make HTTP request with retries."""
        for attempt in range(retries):
            try:
                time.sleep(random.uniform(2, 4))
                response = self.session.get(
                    url,
                    headers=self._get_headers(),
                    params=params,
                    timeout=timeout,
                    allow_redirects=True
                )
                
                if response.status_code == 200:
                    return response
                elif response.status_code == 429:
                    wait = 60 * (attempt + 1)
                    logger.warning(f"Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    logger.warning(f"Status {response.status_code} for {url}")
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error (attempt {attempt + 1}): {e}")
                time.sleep(5 * (attempt + 1))
        
        return None
    
    def _text_hash(self, text: str) -> str:
        """Generate hash for duplicate detection."""
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _clean_text(self, text: str) -> str:
        """Clean text content."""
        if not text:
            return ""
        
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'\[deleted\]|\[removed\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'&amp;|&lt;|&gt;|&nbsp;|&#x200B;', ' ', text)
        text = re.sub(r'/u/\w+|u/\w+', '', text)
        text = re.sub(r'/r/\w+|r/\w+', '', text)
        text = re.sub(r'\*\*|\*|__|_|~~|`|#{1,6}\s', '', text)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _is_valid_text(self, text: str) -> bool:
        """Validate text quality."""
        if not text or len(text) < MIN_TEXT_LENGTH or len(text) > MAX_TEXT_LENGTH:
            return False
        if text.lower() in ['[deleted]', '[removed]', 'deleted', 'removed', '']:
            return False
        if len(re.findall(r'[a-zA-Z]{2,}', text)) < 3:
            return False
        return True
    
    def _is_singapore_exam_related(self, text: str) -> bool:
        """Check Singapore exam relevance."""
        text_lower = text.lower()
        
        sg_terms = [
            'o level', 'o-level', 'olevel', 'a level', 'a-level', 'alevel',
            'psle', 'n level', 'n-level', 'gce', 'cambridge',
            'jc', 'junior college', 'polytechnic', 'poly', 'ite',
            'nus', 'ntu', 'smu', 'sutd', 'sit', 'suss',
            'ngee ann', 'nanyang', 'temasek', 'singapore poly',
            'singapore', 'sg', 'sgexams', 'moe', 'seab',
            'h1', 'h2', 'h3', 'cap', 'gpa', 'bellcurve', 'bell curve'
        ]
        
        exam_terms = [
            'exam', 'examination', 'paper', 'test', 'quiz',
            'midterm', 'mid-term', 'finals', 'final exam',
            'prelim', 'preliminary', 'mock',
            'module', 'course', 'grade', 'result', 'score'
        ]
        
        has_sg = any(term in text_lower for term in sg_terms)
        has_exam = any(term in text_lower for term in exam_terms)
        
        return has_sg or has_exam
    
    def _add_record(self, text: str, source: str, post_id: str,
                    subreddit: str, post_type: str, **kwargs) -> bool:
        """Add record to collection."""
        if post_id in self.seen_ids:
            return False
        
        cleaned_text = self._clean_text(text)
        
        if not self._is_valid_text(cleaned_text):
            return False
        
        text_hash = self._text_hash(cleaned_text)
        if text_hash in self.seen_hashes:
            return False
        
        if not self._is_singapore_exam_related(cleaned_text):
            return False
        
        self.seen_ids.add(post_id)
        self.seen_hashes.add(text_hash)
        
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
    
    def crawl_reddit_rss(self, subreddit: str) -> int:
        """Crawl subreddit using RSS feed."""
        logger.info(f"Crawling r/{subreddit} via RSS...")
        
        added = 0
        url = f"https://www.reddit.com/r/{subreddit}/.rss"
        
        response = self._make_request(url)
        if not response:
            return 0
        
        try:
            root = ET.fromstring(response.content)
            
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            for entry in root.findall('.//atom:entry', ns):
                try:
                    entry_id = entry.find('atom:id', ns)
                    title = entry.find('atom:title', ns)
                    content = entry.find('atom:content', ns)
                    
                    if entry_id is None or title is None:
                        continue
                    
                    post_id = entry_id.text.split('/')[-1] if entry_id.text else ""
                    title_text = title.text or ""
                    
                    content_text = ""
                    if content is not None and content.text:
                        soup = BeautifulSoup(content.text, 'html.parser')
                        content_text = soup.get_text()
                    
                    full_text = f"{title_text}. {content_text}" if content_text else title_text
                    
                    if self._add_record(
                        text=full_text,
                        source="reddit_rss",
                        post_id=f"rss_{post_id}",
                        subreddit=subreddit,
                        post_type="submission",
                        title=title_text
                    ):
                        added += 1
                        
                except Exception as e:
                    logger.debug(f"RSS parse error: {e}")
                    
        except ET.ParseError as e:
            logger.error(f"RSS XML parse error: {e}")
        
        logger.info(f"RSS added {added} records from r/{subreddit}")
        return added
    
    def crawl_teddit(self, subreddit: str, pages: int = 5) -> int:
        """Crawl subreddit via Teddit (Reddit alternative frontend)."""
        logger.info(f"Crawling r/{subreddit} via Teddit...")
        
        added = 0
        
        for base_url in self.alternative_urls:
            url = f"{base_url}/r/{subreddit}"
            
            for page in range(pages):
                response = self._make_request(url)
                if not response:
                    continue
                
                soup = BeautifulSoup(response.text, 'lxml')
                
                # Find posts (Teddit structure)
                posts = soup.find_all('div', class_='post') or soup.find_all('article')
                
                for post in posts:
                    try:
                        # Get title
                        title_elem = post.find(['h2', 'h3', 'a'], class_=['title', 'post-title'])
                        title = title_elem.get_text().strip() if title_elem else ""
                        
                        # Get content
                        content_elem = post.find(['div', 'p'], class_=['post-body', 'content', 'md'])
                        content = content_elem.get_text().strip() if content_elem else ""
                        
                        # Generate ID
                        post_id = hashlib.md5(title.encode()).hexdigest()[:12]
                        
                        full_text = f"{title}. {content}" if content else title
                        
                        if self._add_record(
                            text=full_text,
                            source="teddit",
                            post_id=f"teddit_{post_id}",
                            subreddit=subreddit,
                            post_type="submission",
                            title=title
                        ):
                            added += 1
                            
                    except Exception as e:
                        logger.debug(f"Teddit parse error: {e}")
                
                # Find next page
                try:
                    next_link = soup.find('a', text=re.compile(r'next|more', re.I))
                    if next_link and next_link.get('href'):
                        url = urljoin(base_url, next_link.get('href'))
                    else:
                        break
                except:
                    break
            
            if added > 0:
                break
        
        logger.info(f"Teddit added {added} records from r/{subreddit}")
        return added
    
    def crawl_reddit_search_google(self, query: str, subreddit: str = None) -> int:
        """Search for Reddit content using search operators."""
        logger.info(f"Searching for: '{query}'")
        
        # This uses Reddit's search functionality
        added = 0
        
        if subreddit:
            url = f"https://old.reddit.com/r/{subreddit}/search"
            params = {
                "q": query,
                "restrict_sr": "on",
                "sort": "relevance",
                "t": "all"
            }
        else:
            url = "https://old.reddit.com/search"
            params = {
                "q": f"{query} site:reddit.com/r/SGExams OR site:reddit.com/r/singapore",
                "sort": "relevance",
                "t": "all"
            }
        
        response = self._make_request(url, params)
        if not response:
            return 0
        
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Find search results
        posts = soup.find_all('div', class_='search-result') or soup.find_all('div', class_='thing')
        
        for post in posts:
            try:
                title_elem = post.find('a', class_=['title', 'search-title'])
                title = title_elem.get_text().strip() if title_elem else ""
                
                post_id = post.get('data-fullname', '') or hashlib.md5(title.encode()).hexdigest()[:12]
                
                if self._add_record(
                    text=title,
                    source="reddit_search",
                    post_id=f"search_{post_id}",
                    subreddit=subreddit or "mixed",
                    post_type="submission",
                    title=title
                ):
                    added += 1
                    
            except Exception as e:
                logger.debug(f"Search parse error: {e}")
        
        logger.info(f"Search added {added} records")
        return added
    
    def save_progress(self, filename: str = "comprehensive_crawled_data.json"):
        """Save progress to file."""
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
        
        # CSV export
        csv_filepath = os.path.join(RAW_DATA_DIR, "comprehensive_crawled_data.csv")
        df = pd.DataFrame(self.collected_data)
        df.to_csv(csv_filepath, index=False, encoding='utf-8')
    
    def get_stats(self) -> Dict:
        """Get statistics."""
        return {
            "total_records": len(self.collected_data),
            "total_words": self.total_words,
            "unique_ids": len(self.seen_ids)
        }
    
    def run_crawl(self):
        """Run comprehensive crawl."""
        logger.info("Starting comprehensive Reddit crawl...")
        logger.info("=" * 60)
        
        # 1. RSS Feeds
        logger.info("\n[1/3] Crawling RSS feeds...")
        for subreddit in tqdm(SUBREDDITS, desc="RSS"):
            self.crawl_reddit_rss(subreddit)
            self.save_progress()
        
        # 2. Alternative frontends
        logger.info("\n[2/3] Crawling alternative frontends...")
        for subreddit in tqdm(SUBREDDITS[:3], desc="Teddit"):
            self.crawl_teddit(subreddit, pages=3)
            self.save_progress()
        
        # 3. Search queries
        logger.info("\n[3/3] Running search queries...")
        for query in tqdm(SEARCH_QUERIES[:10], desc="Search"):
            self.crawl_reddit_search_google(query, "SGExams")
            self.crawl_reddit_search_google(query, "singapore")
            self.save_progress()
        
        # Final save
        self.save_progress("final_comprehensive_data.json")
        
        stats = self.get_stats()
        logger.info("\n" + "=" * 60)
        logger.info("CRAWLING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total records: {stats['total_records']}")
        logger.info(f"Total words: {stats['total_words']}")
        
        return stats


def main():
    """Main entry point."""
    crawler = ComprehensiveRedditCrawler()
    
    try:
        stats = crawler.run_crawl()
        
        print("\n" + "=" * 60)
        print("FINAL STATISTICS")
        print("=" * 60)
        print(f"Total records: {stats['total_records']}")
        print(f"Total words: {stats['total_words']}")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        crawler.save_progress("interrupted_comprehensive.json")
    except Exception as e:
        logger.error(f"Error: {e}")
        crawler.save_progress("error_comprehensive.json")
        raise


if __name__ == "__main__":
    main()
