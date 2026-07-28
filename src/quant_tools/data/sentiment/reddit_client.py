"""Reddit sentiment analysis via praw.

Fetches recent posts mentioning a ticker from finance subreddits
and computes a simple sentiment score based on upvote ratios.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from quant_tools.errors import ConfigError, DataError, NetworkError
from quant_tools.retry import reddit_breaker

logger = logging.getLogger(__name__)

DEFAULT_SUBREDDITS = ["wallstreetbets", "stocks", "investing"]


@dataclass(frozen=True)
class RedditPost:
    """A single Reddit post mentioning the ticker."""

    title: str
    score: int
    upvote_ratio: float
    num_comments: int
    subreddit: str


@dataclass(frozen=True)
class SentimentResult:
    """Aggregated sentiment from Reddit posts."""

    ticker: str
    score: float  # -1.0 (bearish) to 1.0 (bullish)
    sample_size: int
    top_posts: list[RedditPost]


class RedditSentimentClient:
    """Fetch Reddit sentiment for stock tickers.

    Args:
        client_id: Reddit API client ID. Loaded from config if not provided.
        client_secret: Reddit API client secret. Loaded from config if not provided.
    """

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
    ) -> None:
        if client_id is None or client_secret is None:
            from quant_tools.config import cfg
            client_id = client_id or cfg.reddit_client_id
            client_secret = client_secret or cfg.reddit_client_secret

        if not client_id or not client_secret:
            raise ConfigError(
                "REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET are required. "
                "Set them in .env or pass as arguments."
            )
        self._client_id = client_id
        self._client_secret = client_secret
        self._reddit = None

    def _get_reddit(self):
        """Lazy-init praw.Reddit instance."""
        if self._reddit is None:
            import praw
            self._reddit = praw.Reddit(
                client_id=self._client_id,
                client_secret=self._client_secret,
                user_agent="quant-tools/0.1 sentiment-scanner",
            )
        return self._reddit

    def fetch_sentiment(
        self,
        ticker: str,
        subreddits: list[str] | None = None,
        limit: int = 50,
    ) -> SentimentResult:
        """Fetch and score Reddit sentiment for a ticker.

        Args:
            ticker: Stock ticker symbol (e.g. "AAPL").
            subreddits: Subreddits to search. Defaults to WSB, stocks, investing.
            limit: Max posts to fetch per subreddit.

        Returns:
            SentimentResult with aggregated score and top posts.

        Raises:
            DataError: If no posts found or search fails.
            NetworkError: If Reddit API is unreachable.
            ConfigError: If credentials are missing.
        """
        if reddit_breaker.is_open:
            raise NetworkError("Reddit circuit breaker is open. Try again later.")

        if subreddits is None:
            subreddits = DEFAULT_SUBREDDITS

        try:
            reddit = self._get_reddit()
            all_posts: list[RedditPost] = []

            for sub_name in subreddits:
                try:
                    subreddit = reddit.subreddit(sub_name)
                    for submission in subreddit.search(
                        f"${ticker}", sort="new", time_filter="week", limit=limit,
                    ):
                        all_posts.append(RedditPost(
                            title=submission.title,
                            score=submission.score,
                            upvote_ratio=submission.upvote_ratio,
                            num_comments=submission.num_comments,
                            subreddit=sub_name,
                        ))
                except Exception as exc:
                    logger.warning("Failed to search r/%s: %s", sub_name, exc)

            if not all_posts:
                reddit_breaker.record_success()
                raise DataError(f"No Reddit posts found for ${ticker}")

            # Score: weighted average of (upvote_ratio - 0.5) * 2, weighted by log(score+1)
            import math
            total_weight = 0.0
            weighted_sum = 0.0
            for post in all_posts:
                weight = math.log1p(max(post.score, 0))
                sentiment = (post.upvote_ratio - 0.5) * 2  # Map 0-1 to -1..1
                weighted_sum += sentiment * weight
                total_weight += weight

            final_score = weighted_sum / total_weight if total_weight > 0 else 0.0
            final_score = max(-1.0, min(1.0, final_score))

            # Top 5 posts by score
            top_posts = sorted(all_posts, key=lambda p: p.score, reverse=True)[:5]

            reddit_breaker.record_success()
            logger.info(
                "Reddit sentiment for %s: %.2f (n=%d)",
                ticker, final_score, len(all_posts),
            )
            return SentimentResult(
                ticker=ticker,
                score=final_score,
                sample_size=len(all_posts),
                top_posts=top_posts,
            )

        except (ConfigError, DataError):
            raise
        except Exception as exc:
            reddit_breaker.record_failure()
            msg = str(exc).lower()
            if any(kw in msg for kw in ("timeout", "connection", "refused", "network")):
                raise NetworkError(f"Reddit API unreachable: {exc}") from exc
            raise DataError(f"Reddit sentiment failed for {ticker}: {exc}") from exc
