"""Reddit crawler using the public JSON API."""

import logging
import time

import requests

import config
from storage import DataStore

logger = logging.getLogger(__name__)


class RedditCrawler:
    """Crawls Reddit posts and comments using .json endpoints."""

    def __init__(self, store: DataStore):
        self.store = store
        self.progress = store.load_progress()
        self._expanded_submissions: set[str] = set()
        self.session = requests.Session()
        self.session.headers.update(config.REQUEST_HEADERS)

    def _get_json(self, url: str, params: dict | None = None) -> dict | list | None:
        """GET request with rate limit handling."""
        time.sleep(config.REQUEST_DELAY)
        for attempt in range(config.MAX_RETRIES):
            try:
                resp = self.session.get(url, params=params, timeout=30)
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 0))
                    wait = max(retry_after, config.BACKOFF_BASE * (config.BACKOFF_MULTIPLIER ** attempt))
                    logger.warning("Rate limited (attempt %d/%d). Sleeping %ds...", attempt + 1, config.MAX_RETRIES, wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.HTTPError:
                if attempt < config.MAX_RETRIES - 1:
                    wait = config.BACKOFF_BASE * (config.BACKOFF_MULTIPLIER ** attempt)
                    logger.warning("HTTP error (attempt %d/%d). Retrying in %ds...", attempt + 1, config.MAX_RETRIES, wait)
                    time.sleep(wait)
                else:
                    logger.warning("Request failed after %d attempts for %s", config.MAX_RETRIES, url)
            except Exception as e:
                logger.warning("Request failed for %s: %s", url, e)
                return None
        return None

    def _task_key(self, *parts: str) -> str:
        return ":".join(parts)

    def _is_done(self, key: str) -> bool:
        return key in self.progress.get("completed_tasks", [])

    def _mark_done(self, key: str) -> None:
        self.progress.setdefault("completed_tasks", []).append(key)
        self.store.save_progress(self.progress)

    def _parse_submission(self, data: dict) -> dict | None:
        """Extract submission data from Reddit JSON."""
        d = data.get("data", data)
        body = d.get("selftext", "")
        title = d.get("title", "")
        if len(body) < config.MIN_BODY_LENGTH and len(title) < config.MIN_BODY_LENGTH:
            return None
        author = d.get("author", "[deleted]")
        if author in ("[deleted]", "[removed]"):
            author = "[deleted]"
        return {
            "id": d.get("id", ""),
            "subreddit": d.get("subreddit", ""),
            "title": title,
            "body": body,
            "author": author,
            "score": d.get("score", 0),
            "num_comments": d.get("num_comments", 0),
            "created_utc": d.get("created_utc", 0),
            "url": f"https://reddit.com{d.get('permalink', '')}",
            "post_type": "submission",
            "parent_id": "",
        }

    def _parse_comment(self, data: dict, subreddit: str, submission_title: str = "") -> dict | None:
        """Extract comment data from Reddit JSON."""
        d = data.get("data", data)
        body = d.get("body", "")
        if len(body) < config.MIN_BODY_LENGTH or body in ("[deleted]", "[removed]"):
            return None
        author = d.get("author", "[deleted]")
        if author in ("[deleted]", "[removed]"):
            author = "[deleted]"
        return {
            "id": d.get("id", ""),
            "subreddit": subreddit,
            "title": submission_title,
            "body": body,
            "author": author,
            "score": d.get("score", 0),
            "num_comments": 0,
            "created_utc": d.get("created_utc", 0),
            "url": f"https://reddit.com{d.get('permalink', '')}",
            "post_type": "comment",
            "parent_id": d.get("parent_id", ""),
        }

    def _fetch_comments(self, permalink: str, submission_id: str, subreddit: str, submission_title: str = "") -> int:
        """Fetch all comments for a submission."""
        if submission_id in self._expanded_submissions:
            return 0
        self._expanded_submissions.add(submission_id)

        url = f"{config.BASE_URL}{permalink}.json"
        data = self._get_json(url, params={"limit": 500, "depth": 10})
        if not data or not isinstance(data, list) or len(data) < 2:
            return 0

        count = 0
        count += self._walk_comment_tree(data[1], subreddit, submission_title)
        return count

    def _walk_comment_tree(self, listing: dict, subreddit: str, submission_title: str = "") -> int:
        """Recursively go through comment tree."""
        count = 0
        children = listing.get("data", {}).get("children", [])
        for child in children:
            kind = child.get("kind", "")
            if kind == "t1":
                rec = self._parse_comment(child, subreddit, submission_title)
                if rec and self.store.add_record(rec):
                    count += 1
                # check replies
                replies = child.get("data", {}).get("replies", "")
                if isinstance(replies, dict):
                    count += self._walk_comment_tree(replies, subreddit, submission_title)
            # "more" = deeper comments, skip these
        return count

    def _fetch_listing(self, url: str, params: dict, max_pages: int) -> list[dict]:
        """Paginate through a Reddit listing."""
        submissions = []
        after = None

        for page in range(max_pages):
            p = {**params}
            if after:
                p["after"] = after
            data = self._get_json(url, params=p)
            if not data:
                break

            children = data.get("data", {}).get("children", [])
            if not children:
                break

            for child in children:
                if child.get("kind") == "t3":
                    submissions.append(child)

            after = data.get("data", {}).get("after")
            if not after:
                break

            logger.debug("Page %d: got %d posts, after=%s", page + 1, len(children), after)

        return submissions

    def _process_submissions(self, submissions: list[dict], label: str) -> tuple[int, int]:
        sub_count = 0
        com_count = 0

        for child in submissions:
            rec = self._parse_submission(child)
            if not rec:
                continue
            if self.store.add_record(rec):
                sub_count += 1

            # get comments too
            permalink = child.get("data", {}).get("permalink", "")
            sid = child.get("data", {}).get("id", "")
            subreddit = child.get("data", {}).get("subreddit", "")
            sub_title = child.get("data", {}).get("title", "")
            if permalink and sid:
                com_count += self._fetch_comments(permalink, sid, subreddit, sub_title)

            total = sub_count + com_count
            if total > 0 and total % 50 == 0:
                logger.info(
                    "[%s] +%d subs, +%d comments (store total: %d)",
                    label, sub_count, com_count, self.store.total_records,
                )

        return sub_count, com_count

    def crawl_browse(self, subreddit_name: str) -> None:
        """Browse hot/new/top posts from a subreddit."""
        for sort in config.BROWSE_SORTS:
            if sort == "top":
                for tf in config.BROWSE_TIME_FILTERS:
                    key = self._task_key("browse", subreddit_name, sort, tf)
                    if self._is_done(key):
                        logger.info("Skip completed: %s", key)
                        continue
                    logger.info("Browsing r/%s top(%s)...", subreddit_name, tf)
                    url = f"{config.BASE_URL}/r/{subreddit_name}/top.json"
                    params = {"t": tf, "limit": 100}
                    subs = self._fetch_listing(url, params, config.MAX_PAGES_PER_TASK)
                    s, c = self._process_submissions(subs, key)
                    logger.info("Done %s: +%d subs, +%d comments", key, s, c)
                    self._mark_done(key)
            else:
                key = self._task_key("browse", subreddit_name, sort)
                if self._is_done(key):
                    logger.info("Skip completed: %s", key)
                    continue
                logger.info("Browsing r/%s %s...", subreddit_name, sort)
                url = f"{config.BASE_URL}/r/{subreddit_name}/{sort}.json"
                params = {"limit": 100}
                subs = self._fetch_listing(url, params, config.MAX_PAGES_PER_TASK)
                s, c = self._process_submissions(subs, key)
                logger.info("Done %s: +%d subs, +%d comments", key, s, c)
                self._mark_done(key)

    def crawl_search(self, subreddit_name: str) -> None:
        """Search a subreddit with our keyword queries."""
        for query in config.SEARCH_QUERIES:
            for sort in ["relevance", "top", "new"]:
                key = self._task_key("search", subreddit_name, query, sort)
                if self._is_done(key):
                    logger.info("Skip completed: %s", key)
                    continue
                logger.info(
                    "Searching r/%s for '%s' sort=%s...",
                    subreddit_name, query, sort,
                )
                url = f"{config.BASE_URL}/r/{subreddit_name}/search.json"
                params = {
                    "q": query,
                    "sort": sort,
                    "t": "all",
                    "restrict_sr": "on",
                    "limit": 100,
                }
                subs = self._fetch_listing(url, params, config.MAX_PAGES_PER_TASK)
                s, c = self._process_submissions(subs, key)
                logger.info("Done %s: +%d subs, +%d comments", key, s, c)
                self._mark_done(key)

    def run(self) -> None:
        """Run the full crawl: browse then search."""
        logger.info("Starting crawl. Existing records: %d", self.store.total_records)

        # browse subreddits first
        for name in config.BROWSE_SUBREDDITS:
            logger.info("Browsing r/%s...", name)
            self.crawl_browse(name)
            self.store.flush()
            logger.info("After browsing r/%s: %d total records", name, self.store.total_records)

        # then search with queries
        for name in config.SUBREDDITS:
            logger.info("Searching r/%s...", name)
            self.crawl_search(name)
            self.store.flush()
            logger.info("After searching r/%s: %d total records", name, self.store.total_records)

        self.store.flush()
        logger.info("Crawl complete. Total records: %d", self.store.total_records)


# =============================================================================
# ORIGINAL PRAW-BASED CRAWLER (commented out — uncomment to use with API creds)
# =============================================================================
#
# import praw
# from praw.models import Comment, Submission
#
# class PRAWRedditCrawler:
#     """
#     Crawls Reddit submissions and comments via PRAW.
#     Requires Reddit API credentials in credentials.env.
#     """
#
#     def __init__(self, store: DataStore):
#         self.reddit = praw.Reddit(
#             client_id=config.REDDIT_CLIENT_ID,
#             client_secret=config.REDDIT_CLIENT_SECRET,
#             username=config.REDDIT_USERNAME,
#             password=config.REDDIT_PASSWORD,
#             user_agent=config.REDDIT_USER_AGENT,
#         )
#         self.store = store
#         self.progress = store.load_progress()
#         self._expanded_submissions: set[str] = set()
#
#     def _task_key(self, *parts: str) -> str:
#         return ":".join(parts)
#
#     def _is_done(self, key: str) -> bool:
#         return key in self.progress.get("completed_tasks", [])
#
#     def _mark_done(self, key: str) -> None:
#         self.progress.setdefault("completed_tasks", []).append(key)
#         self.store.save_progress(self.progress)
#
#     def _submission_record(self, sub: Submission) -> dict | None:
#         body = sub.selftext or ""
#         title = sub.title or ""
#         if len(body) < config.MIN_BODY_LENGTH and len(title) < config.MIN_BODY_LENGTH:
#             return None
#         return {
#             "id": sub.id, "subreddit": str(sub.subreddit), "title": title,
#             "body": body, "author": str(sub.author) if sub.author else "[deleted]",
#             "score": sub.score, "num_comments": sub.num_comments,
#             "created_utc": sub.created_utc,
#             "url": f"https://reddit.com{sub.permalink}",
#             "post_type": "submission", "parent_id": "",
#         }
#
#     def _comment_record(self, comment: Comment) -> dict | None:
#         body = comment.body or ""
#         if len(body) < config.MIN_BODY_LENGTH or body in ("[deleted]", "[removed]"):
#             return None
#         return {
#             "id": comment.id, "subreddit": str(comment.subreddit), "title": "",
#             "body": body, "author": str(comment.author) if comment.author else "[deleted]",
#             "score": comment.score, "num_comments": 0,
#             "created_utc": comment.created_utc,
#             "url": f"https://reddit.com{comment.permalink}",
#             "post_type": "comment", "parent_id": comment.parent_id,
#         }
#
#     def _expand_comments(self, submission: Submission) -> int:
#         if submission.id in self._expanded_submissions:
#             return 0
#         self._expanded_submissions.add(submission.id)
#         count = 0
#         try:
#             submission.comments.replace_more(limit=config.COMMENTS_REPLACE_MORE_LIMIT)
#             for comment in submission.comments.list():
#                 rec = self._comment_record(comment)
#                 if rec and self.store.add_record(rec):
#                     count += 1
#         except Exception as e:
#             logger.warning("Error expanding comments for %s: %s", submission.id, e)
#         return count
#
#     def _process_submissions(self, submissions, label: str) -> tuple[int, int]:
#         sub_count, com_count = 0, 0
#         for submission in submissions:
#             rec = self._submission_record(submission)
#             if rec and self.store.add_record(rec):
#                 sub_count += 1
#             com_count += self._expand_comments(submission)
#             total = sub_count + com_count
#             if total > 0 and total % 200 == 0:
#                 logger.info("[%s] +%d subs, +%d comments (store: %d)",
#                     label, sub_count, com_count, self.store.total_records)
#         return sub_count, com_count
#
#     def crawl_browse(self, subreddit_name: str) -> None:
#         sub = self.reddit.subreddit(subreddit_name)
#         for sort in config.BROWSE_SORTS:
#             if sort == "top":
#                 for tf in config.BROWSE_TIME_FILTERS:
#                     key = self._task_key("browse", subreddit_name, sort, tf)
#                     if self._is_done(key): continue
#                     listings = sub.top(time_filter=tf, limit=config.BROWSE_LIMIT)
#                     s, c = self._process_submissions(listings, key)
#                     self._mark_done(key)
#             else:
#                 key = self._task_key("browse", subreddit_name, sort)
#                 if self._is_done(key): continue
#                 listings = getattr(sub, sort)(limit=config.BROWSE_LIMIT)
#                 s, c = self._process_submissions(listings, key)
#                 self._mark_done(key)
#
#     def crawl_search(self, subreddit_name: str) -> None:
#         sub = self.reddit.subreddit(subreddit_name)
#         for query in config.SEARCH_QUERIES:
#             for sort in ["relevance", "top", "new"]:
#                 key = self._task_key("search", subreddit_name, query, sort)
#                 if self._is_done(key): continue
#                 results = sub.search(query, sort=sort, time_filter="all", limit=None)
#                 s, c = self._process_submissions(results, key)
#                 self._mark_done(key)
#
#     def run(self) -> None:
#         for name in config.BROWSE_SUBREDDITS:
#             self.crawl_browse(name)
#         for name in config.SUBREDDITS:
#             self.crawl_search(name)
#         self.store.flush()
