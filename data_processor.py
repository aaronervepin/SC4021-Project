"""
Data Processor for Singapore Exam Sentiment Dataset

This module handles:
1. Loading raw crawled data
2. Cleaning and deduplication
3. Filtering Singapore exam-related content
4. Preparing data for sentiment labeling
"""

import os
import json
import re
import hashlib
from typing import List, Dict, Set, Tuple
from datetime import datetime
import pandas as pd
from tqdm import tqdm

from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR,
    MIN_TEXT_LENGTH, MAX_TEXT_LENGTH
)


class DataProcessor:
    """Process raw crawled Reddit data."""
    
    def __init__(self):
        self.raw_data: List[Dict] = []
        self.processed_data: List[Dict] = []
        self.seen_hashes: Set[str] = set()
        self.stats = {
            "total_raw": 0,
            "after_dedup": 0,
            "after_filter": 0,
            "after_clean": 0,
        }
    
    def _text_hash(self, text: str) -> str:
        """Generate hash for text to detect duplicates."""
        # Normalize text before hashing
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _is_near_duplicate(self, text: str, threshold: float = 0.85) -> bool:
        """Check if text is a near-duplicate of existing entries."""
        text_hash = self._text_hash(text)
        
        if text_hash in self.seen_hashes:
            return True
        
        self.seen_hashes.add(text_hash)
        return False
    
    def _clean_text(self, text: str) -> str:
        """Clean text content."""
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        
        # Remove Reddit-specific content
        text = re.sub(r'\[deleted\]|\[removed\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'&amp;|&lt;|&gt;|&nbsp;', ' ', text)
        text = re.sub(r'/u/\w+|u/\w+', '', text)  # Remove user mentions
        text = re.sub(r'/r/\w+|r/\w+', '', text)  # Remove subreddit mentions
        
        # Remove markdown formatting
        text = re.sub(r'\*\*|\*|__|_|~~|`', '', text)
        text = re.sub(r'#{1,6}\s', '', text)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # [text](link) -> text
        
        # Remove excessive punctuation
        text = re.sub(r'[!?]{3,}', '!', text)
        text = re.sub(r'\.{3,}', '...', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _is_valid_text(self, text: str) -> bool:
        """Check if text meets quality requirements."""
        if not text:
            return False
        
        # Length checks
        if len(text) < MIN_TEXT_LENGTH:
            return False
        if len(text) > MAX_TEXT_LENGTH:
            return False
        
        # Content checks
        if text.lower() in ['[deleted]', '[removed]', 'deleted', 'removed', '']:
            return False
        
        # Must contain actual words (not just numbers/symbols)
        word_count = len(re.findall(r'[a-zA-Z]{2,}', text))
        if word_count < 3:
            return False
        
        return True
    
    def _is_singapore_exam_related(self, text: str, strict: bool = False) -> bool:
        """Check if text is related to Singapore exams."""
        text_lower = text.lower()
        
        # Singapore context indicators (at least one required)
        sg_indicators = [
            # Exams
            'o level', 'o-level', 'olevel', 'a level', 'a-level', 'alevel',
            'psle', 'n level', 'n-level', 'gce', 'cambridge',
            
            # Schools/Unis
            'jc', 'junior college', 'polytechnic', 'poly', 'ite',
            'nus', 'ntu', 'smu', 'sutd', 'sit', 'suss',
            'ngee ann', 'nanyang', 'temasek', 'republic poly', 'singapore poly',
            'hci', 'ri', 'rjc', 'vjc', 'njc', 'tjc', 'acjc', 'sajc',
            
            # Singapore specific
            'singapore', 'sg', 'sgexams', 'moe', 'seab',
            'h1', 'h2', 'h3', 'cap', 'gpa',
            
            # Local terms
            'mugging', 'chiong', 'sian', 'jialat',
        ]
        
        # Exam/academic terms
        exam_terms = [
            'exam', 'examination', 'paper', 'test', 'quiz',
            'midterm', 'mid-term', 'finals', 'final exam',
            'prelim', 'preliminary', 'mock',
            'module', 'course', 'grade', 'result', 'score',
        ]
        
        has_sg_indicator = any(term in text_lower for term in sg_indicators)
        has_exam_term = any(term in text_lower for term in exam_terms)
        
        if strict:
            return has_sg_indicator and has_exam_term
        else:
            return has_sg_indicator or (has_exam_term and len(text) > 50)
    
    def load_raw_data(self, filename: str = None) -> int:
        """Load raw crawled data from file(s)."""
        files_to_load = []
        
        if filename:
            filepath = os.path.join(RAW_DATA_DIR, filename)
            if os.path.exists(filepath):
                files_to_load.append(filepath)
        else:
            # Load all JSON files in raw data directory
            for f in os.listdir(RAW_DATA_DIR):
                if f.endswith('.json'):
                    files_to_load.append(os.path.join(RAW_DATA_DIR, f))
        
        for filepath in files_to_load:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if isinstance(data, dict) and 'data' in data:
                    self.raw_data.extend(data['data'])
                elif isinstance(data, list):
                    self.raw_data.extend(data)
                    
                print(f"Loaded {filepath}")
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
        
        self.stats["total_raw"] = len(self.raw_data)
        print(f"Total raw records loaded: {self.stats['total_raw']}")
        return self.stats["total_raw"]
    
    def deduplicate(self) -> int:
        """Remove duplicate entries."""
        print("Deduplicating data...")
        
        unique_data = []
        self.seen_hashes.clear()
        
        for record in tqdm(self.raw_data, desc="Deduplicating"):
            text = record.get('text', '')
            
            if not self._is_near_duplicate(text):
                unique_data.append(record)
        
        self.raw_data = unique_data
        self.stats["after_dedup"] = len(self.raw_data)
        
        removed = self.stats["total_raw"] - self.stats["after_dedup"]
        print(f"Removed {removed} duplicates. Remaining: {self.stats['after_dedup']}")
        
        return self.stats["after_dedup"]
    
    def filter_relevant(self, strict: bool = False) -> int:
        """Filter for Singapore exam-related content."""
        print("Filtering for Singapore exam content...")
        
        filtered_data = []
        
        for record in tqdm(self.raw_data, desc="Filtering"):
            text = record.get('text', '')
            
            if self._is_singapore_exam_related(text, strict=strict):
                filtered_data.append(record)
        
        self.raw_data = filtered_data
        self.stats["after_filter"] = len(self.raw_data)
        
        removed = self.stats["after_dedup"] - self.stats["after_filter"]
        print(f"Removed {removed} non-relevant. Remaining: {self.stats['after_filter']}")
        
        return self.stats["after_filter"]
    
    def clean_data(self) -> int:
        """Clean text content."""
        print("Cleaning text content...")
        
        cleaned_data = []
        
        for record in tqdm(self.raw_data, desc="Cleaning"):
            text = record.get('text', '')
            cleaned_text = self._clean_text(text)
            
            if self._is_valid_text(cleaned_text):
                record['text'] = cleaned_text
                record['word_count'] = len(cleaned_text.split())
                cleaned_data.append(record)
        
        self.processed_data = cleaned_data
        self.stats["after_clean"] = len(self.processed_data)
        
        removed = self.stats["after_filter"] - self.stats["after_clean"]
        print(f"Removed {removed} invalid. Final count: {self.stats['after_clean']}")
        
        return self.stats["after_clean"]
    
    def process_all(self, strict_filter: bool = False) -> List[Dict]:
        """Run full processing pipeline."""
        print("\n" + "=" * 60)
        print("PROCESSING PIPELINE")
        print("=" * 60)
        
        # Load data
        if not self.raw_data:
            self.load_raw_data()
        
        # Process
        self.deduplicate()
        self.filter_relevant(strict=strict_filter)
        self.clean_data()
        
        # Calculate total words
        total_words = sum(r.get('word_count', 0) for r in self.processed_data)
        
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Original records: {self.stats['total_raw']}")
        print(f"After deduplication: {self.stats['after_dedup']}")
        print(f"After filtering: {self.stats['after_filter']}")
        print(f"After cleaning: {self.stats['after_clean']}")
        print(f"Total words: {total_words}")
        
        return self.processed_data
    
    def save_processed(self, filename: str = "processed_data.json"):
        """Save processed data to file."""
        filepath = os.path.join(PROCESSED_DATA_DIR, filename)
        
        total_words = sum(r.get('word_count', 0) for r in self.processed_data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "total_records": len(self.processed_data),
                    "total_words": total_words,
                    "processed_at": datetime.now().isoformat(),
                    "stats": self.stats
                },
                "data": self.processed_data
            }, f, indent=2, ensure_ascii=False)
        
        print(f"Saved processed data to {filepath}")
        
        # Also save as CSV
        csv_filepath = os.path.join(PROCESSED_DATA_DIR, "processed_data.csv")
        df = pd.DataFrame(self.processed_data)
        df.to_csv(csv_filepath, index=False, encoding='utf-8')
        print(f"Saved CSV to {csv_filepath}")
    
    def get_statistics(self) -> Dict:
        """Get detailed statistics about processed data."""
        if not self.processed_data:
            return {}
        
        df = pd.DataFrame(self.processed_data)
        
        stats = {
            "total_records": len(self.processed_data),
            "total_words": df['word_count'].sum(),
            "avg_word_count": df['word_count'].mean(),
            "min_word_count": df['word_count'].min(),
            "max_word_count": df['word_count'].max(),
            "by_subreddit": df['subreddit'].value_counts().to_dict() if 'subreddit' in df.columns else {},
            "by_type": df['type'].value_counts().to_dict() if 'type' in df.columns else {},
            "by_source": df['source'].value_counts().to_dict() if 'source' in df.columns else {},
        }
        
        return stats


def main():
    """Main entry point."""
    processor = DataProcessor()
    
    # Load and process data
    processed_data = processor.process_all(strict_filter=False)
    
    # Save processed data
    processor.save_processed()
    
    # Print statistics
    stats = processor.get_statistics()
    print("\nDetailed Statistics:")
    print(f"  Total records: {stats.get('total_records', 0)}")
    print(f"  Total words: {stats.get('total_words', 0)}")
    print(f"  Average words per record: {stats.get('avg_word_count', 0):.1f}")
    
    if stats.get('by_subreddit'):
        print("\n  By Subreddit:")
        for sub, count in stats['by_subreddit'].items():
            print(f"    - r/{sub}: {count}")


if __name__ == "__main__":
    main()
