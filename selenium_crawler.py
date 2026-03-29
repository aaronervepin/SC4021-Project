"""
Reddit Selenium Crawler for Singapore Exam Sentiment Analysis

This script uses Selenium to crawl Reddit posts and comments related to 
Singapore exams. Selenium-based approach is more reliable as it simulates
a real browser.

Note: This crawler respects Reddit's rate limits and includes appropriate delays.
"""

import os
import json
import time
import random
import re
import hashlib
from datetime import datetime
from typing import List, Dict, Set, Optional
from tqdm import tqdm
import pandas as pd

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("Selenium not available. Using requests-based fallback.")

from config import (
    SUBREDDITS, SEARCH_QUERIES,
    RAW_DATA_DIR, LOGS_DIR,
    MIN_TEXT_LENGTH, MAX_TEXT_LENGTH,
    REQUEST_DELAY
)

import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'selenium_crawler.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SeleniumRedditCrawler:
    """Selenium-based Reddit crawler for Singapore exam data."""
    
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.driver = None
        self.collected_data: List[Dict] = []
        self.seen_ids: Set[str] = set()
        self.seen_hashes: Set[str] = set()
        self.total_words = 0
        
    def _init_driver(self):
        """Initialize Chrome WebDriver."""
        if not SELENIUM_AVAILABLE:
            raise RuntimeError("Selenium is not available")
        
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless=new")
        
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # Random user agent
        user_agents = [
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        ]
        chrome_options.add_argument(f"user-agent={random.choice(user_agents)}")
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        logger.info("Chrome WebDriver initialized")
    
    def _close_driver(self):
        """Close WebDriver."""
        if self.driver:
            self.driver.quit()
            self.driver = None
            logger.info("Chrome WebDriver closed")
    
    def _random_delay(self, min_delay: float = 1.5, max_delay: float = 3.0):
        """Add random delay between actions."""
        time.sleep(random.uniform(min_delay, max_delay))
    
    def _scroll_page(self, scroll_count: int = 3):
        """Scroll down the page to load more content."""
        for _ in range(scroll_count):
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            self._random_delay(1.0, 2.0)
    
    def _text_hash(self, text: str) -> str:
        """Generate hash for duplicate detection."""
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        # Remove Reddit-specific formatting
        text = re.sub(r'\[deleted\]|\[removed\]', '', text, flags=re.IGNORECASE)
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?\'"-]', '', text)
        
        return text.strip()
    
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
        
        # Check for actual words
        word_count = len(re.findall(r'[a-zA-Z]{2,}', text))
        if word_count < 3:
            return False
            
        return True
    
    def _is_singapore_exam_related(self, text: str) -> bool:
        """Check if text is related to Singapore exams."""
        text_lower = text.lower()
        
        sg_indicators = [
            'o level', 'o-level', 'olevel', 'a level', 'a-level', 'alevel',
            'psle', 'n level', 'n-level', 'gce', 'cambridge',
            'jc', 'junior college', 'polytechnic', 'poly', 'ite',
            'nus', 'ntu', 'smu', 'sutd', 'sit', 'suss',
            'ngee ann', 'nanyang', 'temasek', 'republic poly', 'singapore poly',
            'singapore', 'sg', 'sgexams', 'moe', 'seab',
            'h1', 'h2', 'h3', 'cap', 'gpa',
        ]
        
        exam_terms = [
            'exam', 'examination', 'paper', 'test', 'quiz',
            'midterm', 'mid-term', 'finals', 'final exam',
            'prelim', 'preliminary', 'mock',
            'module', 'course', 'grade', 'result', 'score',
        ]
        
        has_sg = any(term in text_lower for term in sg_indicators)
        has_exam = any(term in text_lower for term in exam_terms)
        
        return has_sg or has_exam
    
    def _add_record(self, text: str, source: str, post_id: str,
                    subreddit: str, post_type: str, title: str = "",
                    url: str = "", score: int = 0) -> bool:
        """Add a record to collected data."""
        # Check for duplicate ID
        if post_id in self.seen_ids:
            return False
        
        cleaned_text = self._clean_text(text)
        
        if not self._is_valid_text(cleaned_text):
            return False
        
        # Check for near-duplicate content
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
            "title": self._clean_text(title),
            "source": source,
            "subreddit": subreddit,
            "type": post_type,
            "url": url,
            "score": score,
            "word_count": word_count,
            "crawled_at": datetime.now().isoformat()
        }
        
        self.collected_data.append(record)
        return True
    
    def crawl_subreddit_old_reddit(self, subreddit: str, pages: int = 5) -> int:
        """Crawl subreddit using old.reddit.com."""
        logger.info(f"Crawling r/{subreddit} via old.reddit.com...")
        
        added_count = 0
        base_url = f"https://old.reddit.com/r/{subreddit}"
        current_url = base_url
        
        for page in range(pages):
            try:
                self.driver.get(current_url)
                self._random_delay(2, 4)
                
                # Wait for posts to load
                try:
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "thing"))
                    )
                except TimeoutException:
                    logger.warning(f"Timeout loading r/{subreddit} page {page}")
                    break
                
                # Find all posts
                posts = self.driver.find_elements(By.CLASS_NAME, "thing")
                
                for post in posts:
                    try:
                        # Get post ID
                        post_id = post.get_attribute("data-fullname")
                        if not post_id:
                            continue
                        post_id = post_id.replace("t3_", "")
                        
                        # Get title
                        try:
                            title_elem = post.find_element(By.CSS_SELECTOR, "a.title")
                            title = title_elem.text
                        except NoSuchElementException:
                            title = ""
                        
                        # Get selftext if available
                        selftext = ""
                        try:
                            expando = post.find_element(By.CLASS_NAME, "expando")
                            text_elem = expando.find_element(By.CLASS_NAME, "md")
                            selftext = text_elem.text
                        except NoSuchElementException:
                            pass
                        
                        # Get score
                        score = 0
                        try:
                            score_elem = post.find_element(By.CLASS_NAME, "score")
                            score_text = score_elem.get_attribute("title") or "0"
                            score = int(score_text)
                        except (NoSuchElementException, ValueError):
                            pass
                        
                        # Get URL
                        url = ""
                        try:
                            url_elem = post.find_element(By.CSS_SELECTOR, "a.bylink")
                            url = url_elem.get_attribute("href")
                        except NoSuchElementException:
                            pass
                        
                        full_text = f"{title}. {selftext}" if selftext else title
                        
                        if self._add_record(
                            text=full_text,
                            source="old_reddit_selenium",
                            post_id=f"post_{post_id}",
                            subreddit=subreddit,
                            post_type="submission",
                            title=title,
                            url=url,
                            score=score
                        ):
                            added_count += 1
                            
                    except Exception as e:
                        logger.debug(f"Error parsing post: {e}")
                        continue
                
                # Find next page
                try:
                    next_button = self.driver.find_element(By.CSS_SELECTOR, "span.next-button a")
                    current_url = next_button.get_attribute("href")
                except NoSuchElementException:
                    logger.info(f"No more pages for r/{subreddit}")
                    break
                    
            except Exception as e:
                logger.error(f"Error crawling r/{subreddit} page {page}: {e}")
                break
        
        logger.info(f"Added {added_count} records from r/{subreddit}")
        return added_count
    
    def crawl_post_comments(self, subreddit: str, post_url: str, post_id: str) -> int:
        """Crawl comments from a specific post."""
        if not post_url.startswith("http"):
            post_url = f"https://old.reddit.com{post_url}"
        
        added_count = 0
        
        try:
            self.driver.get(post_url)
            self._random_delay(2, 3)
            
            # Expand all comments
            try:
                more_comments = self.driver.find_elements(By.CSS_SELECTOR, "a.morecomments")
                for mc in more_comments[:5]:  # Limit to prevent too many clicks
                    try:
                        mc.click()
                        self._random_delay(0.5, 1)
                    except:
                        pass
            except:
                pass
            
            # Find all comments
            comments = self.driver.find_elements(By.CSS_SELECTOR, "div.comment")
            
            for comment in comments:
                try:
                    comment_id = comment.get_attribute("data-fullname")
                    if not comment_id:
                        continue
                    comment_id = comment_id.replace("t1_", "")
                    
                    # Get comment text
                    try:
                        text_elem = comment.find_element(By.CSS_SELECTOR, "div.md")
                        comment_text = text_elem.text
                    except NoSuchElementException:
                        continue
                    
                    # Get score
                    score = 0
                    try:
                        score_elem = comment.find_element(By.CSS_SELECTOR, "span.score")
                        score_text = score_elem.text.replace(" points", "").replace(" point", "")
                        score = int(score_text)
                    except (NoSuchElementException, ValueError):
                        pass
                    
                    if self._add_record(
                        text=comment_text,
                        source="old_reddit_selenium",
                        post_id=f"comment_{comment_id}",
                        subreddit=subreddit,
                        post_type="comment",
                        score=score
                    ):
                        added_count += 1
                        
                except Exception as e:
                    logger.debug(f"Error parsing comment: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error crawling comments for {post_url}: {e}")
        
        return added_count
    
    def search_subreddit(self, subreddit: str, query: str, pages: int = 3) -> int:
        """Search within a subreddit."""
        logger.info(f"Searching r/{subreddit} for: '{query}'")
        
        added_count = 0
        search_url = f"https://old.reddit.com/r/{subreddit}/search?q={query.replace(' ', '+')}&restrict_sr=on&sort=relevance&t=all"
        current_url = search_url
        
        for page in range(pages):
            try:
                self.driver.get(current_url)
                self._random_delay(2, 4)
                
                # Find search results
                posts = self.driver.find_elements(By.CLASS_NAME, "thing")
                
                for post in posts:
                    try:
                        post_id = post.get_attribute("data-fullname")
                        if not post_id:
                            continue
                        post_id = post_id.replace("t3_", "")
                        
                        try:
                            title_elem = post.find_element(By.CSS_SELECTOR, "a.title")
                            title = title_elem.text
                        except NoSuchElementException:
                            title = ""
                        
                        if self._add_record(
                            text=title,
                            source="old_reddit_search",
                            post_id=f"search_post_{post_id}",
                            subreddit=subreddit,
                            post_type="submission",
                            title=title
                        ):
                            added_count += 1
                            
                    except Exception as e:
                        continue
                
                # Find next page
                try:
                    next_button = self.driver.find_element(By.CSS_SELECTOR, "span.next-button a")
                    current_url = next_button.get_attribute("href")
                except NoSuchElementException:
                    break
                    
            except Exception as e:
                logger.error(f"Search error: {e}")
                break
        
        logger.info(f"Search added {added_count} records")
        return added_count
    
    def save_progress(self, filename: str = "selenium_crawled_data.json"):
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
        
        # Also save as CSV
        csv_filepath = os.path.join(RAW_DATA_DIR, "selenium_crawled_data.csv")
        df = pd.DataFrame(self.collected_data)
        df.to_csv(csv_filepath, index=False, encoding='utf-8')
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        return {
            "total_records": len(self.collected_data),
            "total_words": self.total_words,
            "unique_ids": len(self.seen_ids)
        }
    
    def run_crawl(self, max_records: int = 15000):
        """Run the crawling process."""
        logger.info("Starting Selenium-based Reddit crawl...")
        logger.info("=" * 60)
        
        try:
            self._init_driver()
            
            # 1. Crawl main subreddits
            logger.info("\n[1/3] Crawling main subreddits...")
            for subreddit in tqdm(SUBREDDITS[:5], desc="Subreddits"):
                self.crawl_subreddit_old_reddit(subreddit, pages=10)
                self.save_progress()
                
                stats = self.get_stats()
                logger.info(f"Progress: {stats['total_records']} records, {stats['total_words']} words")
                
                if stats['total_records'] >= max_records:
                    break
            
            # 2. Search with queries
            if self.get_stats()['total_records'] < max_records:
                logger.info("\n[2/3] Running search queries...")
                for query in tqdm(SEARCH_QUERIES[:15], desc="Searches"):
                    for subreddit in ["SGExams", "singapore"]:
                        self.search_subreddit(subreddit, query, pages=2)
                    self.save_progress()
                    
                    if self.get_stats()['total_records'] >= max_records:
                        break
            
            # 3. Additional targeted searches
            if self.get_stats()['total_records'] < max_records:
                logger.info("\n[3/3] Additional targeted searches...")
                additional_queries = [
                    "O level results", "A level results",
                    "NUS exam", "NTU finals",
                    "JC exam experience", "poly exam",
                    "PSLE stress", "exam tips Singapore"
                ]
                for query in tqdm(additional_queries, desc="Additional"):
                    self.search_subreddit("SGExams", query, pages=3)
                    self.save_progress()
                    
                    if self.get_stats()['total_records'] >= max_records:
                        break
            
        finally:
            self._close_driver()
        
        # Final save
        self.save_progress("final_selenium_data.json")
        
        stats = self.get_stats()
        logger.info("\n" + "=" * 60)
        logger.info("CRAWLING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total records: {stats['total_records']}")
        logger.info(f"Total words: {stats['total_words']}")
        
        return stats


def main():
    """Main entry point."""
    if not SELENIUM_AVAILABLE:
        print("Selenium is not available. Please install it first.")
        print("pip install selenium webdriver-manager")
        return
    
    crawler = SeleniumRedditCrawler(headless=True)
    
    try:
        stats = crawler.run_crawl(max_records=15000)
        
        print("\n" + "=" * 60)
        print("FINAL STATISTICS")
        print("=" * 60)
        print(f"Total records collected: {stats['total_records']}")
        print(f"Total words collected: {stats['total_words']}")
        
    except KeyboardInterrupt:
        print("\nCrawling interrupted by user.")
        crawler.save_progress("interrupted_selenium_data.json")
    except Exception as e:
        logger.error(f"Crawling error: {e}")
        crawler.save_progress("error_selenium_data.json")
        raise


if __name__ == "__main__":
    main()
