"""Main script - run crawler, stats, clean, or eval commands."""

import argparse
import logging
import sys

import config
from crawler import RedditCrawler
from stats import calculate_stats, generate_eval_sample, load_and_clean, save_clean_csv
from storage import DataStore


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("crawl.log"),
        ],
    )


def cmd_crawl(args):
    setup_logging(verbose=args.verbose)
    store = DataStore()
    crawler = RedditCrawler(store)
    try:
        crawler.run()
    except KeyboardInterrupt:
        logging.info("Interrupted. Flushing buffer...")
        store.flush()
        logging.info("Progress saved. Safe to resume with: python main.py crawl")
    except Exception:
        store.flush()
        raise


def cmd_stats(_args):
    result = calculate_stats()
    print(f"Total records:  {result['total_records']}")
    print(f"Total words:    {result['total_words']}")
    print(f"Unique words:   {result['unique_words']}")
    print(f"Submissions:    {result['submissions']}")
    print(f"Comments:       {result['comments']}")
    print(f"Subreddits:     {result['subreddit_breakdown']}")
    print(f"Avg score:      {result['average_score']}")
    meets = result["total_records"] >= 10000 and result["total_words"] >= 100000
    print(f"\nMeets assignment requirements: {'YES' if meets else 'NO'}")


def cmd_clean(_args):
    df = load_and_clean()
    path = save_clean_csv(df)
    print(f"Cleaned dataset: {len(df)} records saved to {path}")


def cmd_eval(args):
    path = generate_eval_sample(sample_size=args.size)
    print(f"Evaluation dataset written to {path}")
    print("Open it in Excel, fill in the 'sentiment_label' column")
    print("Labels: positive / negative / neutral")
    print("Fill in 'annotator' column with your name")


def main():
    parser = argparse.ArgumentParser(
        description="Reddit crawler for SG exam sentiment corpus"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    crawl_p = sub.add_parser("crawl", help="Run the crawler")
    crawl_p.add_argument("-v", "--verbose", action="store_true", help="Debug logging")

    sub.add_parser("stats", help="Calculate corpus statistics")
    sub.add_parser("clean", help="Deduplicate and clean the raw crawled data")

    eval_p = sub.add_parser("eval", help="Generate evaluation sample for annotation")
    eval_p.add_argument(
        "-n", "--size", type=int, default=4000, help="Sample size (default 4000)"
    )

    args = parser.parse_args()
    {"crawl": cmd_crawl, "stats": cmd_stats, "clean": cmd_clean, "eval": cmd_eval}[args.command](args)


if __name__ == "__main__":
    main()
