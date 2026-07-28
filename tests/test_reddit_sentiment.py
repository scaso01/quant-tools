"""Tests for Reddit sentiment client.

All external APIs are mocked. No real network calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from quant_tools.errors import ConfigError, DataError, NetworkError


class TestRedditSentimentClient:
    """Tests for RedditSentimentClient."""

    @patch("quant_tools.config.cfg")
    def test_init_no_credentials_raises(self, mock_cfg):
        mock_cfg.reddit_client_id = ""
        mock_cfg.reddit_client_secret = ""
        from quant_tools.data.sentiment.reddit_client import RedditSentimentClient

        with pytest.raises(ConfigError, match="REDDIT_CLIENT_ID"):
            RedditSentimentClient()

    def test_init_explicit_credentials(self):
        from quant_tools.data.sentiment.reddit_client import RedditSentimentClient

        client = RedditSentimentClient(client_id="id", client_secret="secret")
        assert client._client_id == "id"
        assert client._client_secret == "secret"

    @patch("praw.Reddit")
    def test_fetch_sentiment_success(self, mock_reddit_cls):
        from quant_tools.data.sentiment.reddit_client import (
            RedditSentimentClient,
            SentimentResult,
        )
        from quant_tools.retry import reddit_breaker

        reddit_breaker.reset()

        # Build mock submissions
        mock_submission_1 = MagicMock()
        mock_submission_1.title = "AAPL is going up!"
        mock_submission_1.score = 100
        mock_submission_1.upvote_ratio = 0.85
        mock_submission_1.num_comments = 50
        mock_submission_1.subreddit = "stocks"

        mock_submission_2 = MagicMock()
        mock_submission_2.title = "AAPL earnings beat"
        mock_submission_2.score = 200
        mock_submission_2.upvote_ratio = 0.90
        mock_submission_2.num_comments = 120
        mock_submission_2.subreddit = "stocks"

        mock_subreddit = MagicMock()
        mock_subreddit.search.return_value = [mock_submission_1, mock_submission_2]

        mock_reddit = MagicMock()
        mock_reddit.subreddit.return_value = mock_subreddit
        mock_reddit_cls.return_value = mock_reddit

        client = RedditSentimentClient(client_id="id", client_secret="secret")
        result = client.fetch_sentiment("AAPL", subreddits=["stocks"])

        assert isinstance(result, SentimentResult)
        assert result.ticker == "AAPL"
        assert result.sample_size == 2
        assert -1.0 <= result.score <= 1.0
        assert result.score > 0  # Both posts are bullish
        assert len(result.top_posts) <= 5

    @patch("praw.Reddit")
    def test_fetch_sentiment_no_posts(self, mock_reddit_cls):
        from quant_tools.data.sentiment.reddit_client import RedditSentimentClient
        from quant_tools.retry import reddit_breaker

        reddit_breaker.reset()

        mock_subreddit = MagicMock()
        mock_subreddit.search.return_value = []

        mock_reddit = MagicMock()
        mock_reddit.subreddit.return_value = mock_subreddit
        mock_reddit_cls.return_value = mock_reddit

        client = RedditSentimentClient(client_id="id", client_secret="secret")
        with pytest.raises(DataError, match="No Reddit posts found"):
            client.fetch_sentiment("ZZZZZ")

    @patch("praw.Reddit")
    def test_fetch_sentiment_default_subreddits(self, mock_reddit_cls):
        from quant_tools.data.sentiment.reddit_client import (
            DEFAULT_SUBREDDITS,
            RedditSentimentClient,
        )
        from quant_tools.retry import reddit_breaker

        reddit_breaker.reset()

        mock_submission = MagicMock()
        mock_submission.title = "TSLA moon"
        mock_submission.score = 50
        mock_submission.upvote_ratio = 0.70
        mock_submission.num_comments = 30

        mock_subreddit = MagicMock()
        mock_subreddit.search.return_value = [mock_submission]

        mock_reddit = MagicMock()
        mock_reddit.subreddit.return_value = mock_subreddit
        mock_reddit_cls.return_value = mock_reddit

        client = RedditSentimentClient(client_id="id", client_secret="secret")
        result = client.fetch_sentiment("TSLA")

        # Should search all 3 default subreddits
        assert mock_reddit.subreddit.call_count == len(DEFAULT_SUBREDDITS)
        assert result.sample_size == len(DEFAULT_SUBREDDITS)

    @patch("praw.Reddit")
    def test_fetch_sentiment_bearish(self, mock_reddit_cls):
        from quant_tools.data.sentiment.reddit_client import RedditSentimentClient
        from quant_tools.retry import reddit_breaker

        reddit_breaker.reset()

        mock_submission = MagicMock()
        mock_submission.title = "GME crash"
        mock_submission.score = 300
        mock_submission.upvote_ratio = 0.20  # Low ratio = bearish
        mock_submission.num_comments = 200

        mock_subreddit = MagicMock()
        mock_subreddit.search.return_value = [mock_submission]

        mock_reddit = MagicMock()
        mock_reddit.subreddit.return_value = mock_subreddit
        mock_reddit_cls.return_value = mock_reddit

        client = RedditSentimentClient(client_id="id", client_secret="secret")
        result = client.fetch_sentiment("GME", subreddits=["wallstreetbets"])

        assert result.score < 0  # Bearish sentiment

    def test_circuit_breaker_open(self):
        from quant_tools.data.sentiment.reddit_client import RedditSentimentClient
        from quant_tools.retry import reddit_breaker

        reddit_breaker.reset()
        # Force open
        for _ in range(5):
            reddit_breaker.record_failure()

        client = RedditSentimentClient(client_id="id", client_secret="secret")
        with pytest.raises(NetworkError, match="circuit breaker"):
            client.fetch_sentiment("AAPL")

        reddit_breaker.reset()

    @patch("praw.Reddit")
    def test_fetch_sentiment_partial_subreddit_failure(self, mock_reddit_cls):
        """One subreddit failing shouldn't break the whole fetch."""
        from quant_tools.data.sentiment.reddit_client import RedditSentimentClient
        from quant_tools.retry import reddit_breaker

        reddit_breaker.reset()

        mock_submission = MagicMock()
        mock_submission.title = "AAPL buy"
        mock_submission.score = 10
        mock_submission.upvote_ratio = 0.75
        mock_submission.num_comments = 5

        call_count = 0

        def mock_subreddit(name):
            nonlocal call_count
            call_count += 1
            sub = MagicMock()
            if name == "wallstreetbets":
                sub.search.side_effect = Exception("banned")
            else:
                sub.search.return_value = [mock_submission]
            return sub

        mock_reddit = MagicMock()
        mock_reddit.subreddit.side_effect = mock_subreddit
        mock_reddit_cls.return_value = mock_reddit

        client = RedditSentimentClient(client_id="id", client_secret="secret")
        result = client.fetch_sentiment("AAPL", subreddits=["wallstreetbets", "stocks"])

        assert result.sample_size == 1  # Only stocks returned results

    @patch("praw.Reddit")
    def test_network_error_all_subs_fail_raises_data_error(self, mock_reddit_cls):
        """When all subreddits fail with network errors, raises DataError."""
        from quant_tools.data.sentiment.reddit_client import RedditSentimentClient
        from quant_tools.retry import reddit_breaker

        reddit_breaker.reset()

        mock_reddit = MagicMock()
        mock_reddit.subreddit.side_effect = ConnectionError("connection refused")
        mock_reddit_cls.return_value = mock_reddit

        client = RedditSentimentClient(client_id="id", client_secret="secret")
        with pytest.raises(DataError, match="No Reddit posts found"):
            client.fetch_sentiment("AAPL", subreddits=["stocks"])

        reddit_breaker.reset()

    @patch("praw.Reddit")
    def test_top_posts_limited_to_five(self, mock_reddit_cls):
        from quant_tools.data.sentiment.reddit_client import RedditSentimentClient
        from quant_tools.retry import reddit_breaker

        reddit_breaker.reset()

        submissions = []
        for i in range(10):
            sub = MagicMock()
            sub.title = f"Post {i}"
            sub.score = i * 10
            sub.upvote_ratio = 0.7
            sub.num_comments = i
            submissions.append(sub)

        mock_subreddit = MagicMock()
        mock_subreddit.search.return_value = submissions

        mock_reddit = MagicMock()
        mock_reddit.subreddit.return_value = mock_subreddit
        mock_reddit_cls.return_value = mock_reddit

        client = RedditSentimentClient(client_id="id", client_secret="secret")
        result = client.fetch_sentiment("SPY", subreddits=["stocks"])

        assert result.sample_size == 10
        assert len(result.top_posts) == 5
        # Top posts should be sorted by score descending
        scores = [p.score for p in result.top_posts]
        assert scores == sorted(scores, reverse=True)
