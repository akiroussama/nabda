"""
Rate limiting module for Jira API calls.

Provides intelligent rate limiting with exponential backoff and jitter
to respect Jira API rate limits.
"""

import random
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar

from loguru import logger
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

P = ParamSpec("P")
T = TypeVar("T")


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded and all retries failed."""

    def __init__(self, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class JiraAPIError(Exception):
    """Raised for Jira API errors that should trigger retry."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _log_retry(retry_state: RetryCallState) -> None:
    """Log retry attempts."""
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        f"Rate limit retry attempt {retry_state.attempt_number}/3 "
        f"after {retry_state.seconds_since_start:.1f}s. "
        f"Error: {exception}"
    )


def _should_retry(exception: BaseException) -> bool:
    """Determine if an exception should trigger a retry."""
    from jira.exceptions import JIRAError

    if isinstance(exception, JIRAError):
        # Retry on rate limit (429) and server errors (5xx)
        return exception.status_code in (429, 500, 502, 503, 504)
    return isinstance(exception, (JiraAPIError, ConnectionError, TimeoutError))


def rate_limited(
    max_attempts: int = 3,
    initial_wait: float = 1.0,
    max_wait: float = 60.0,
    jitter: float = 1.0,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for rate-limited API calls with exponential backoff.

    Uses tenacity for robust retry logic with exponential backoff and jitter.

    Args:
        max_attempts: Maximum number of retry attempts
        initial_wait: Initial wait time in seconds
        max_wait: Maximum wait time in seconds
        jitter: Random jitter to add to wait time

    Returns:
        Decorated function with retry logic

    Example:
        >>> @rate_limited(max_attempts=3)
        ... def fetch_issues():
        ...     return jira.search_issues(jql)
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential_jitter(
                initial=initial_wait,
                max=max_wait,
                jitter=jitter,
            ),
            retry=retry_if_exception_type((Exception,)),
            before_sleep=_log_retry,
            reraise=True,
        )
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if _should_retry(e):
                    raise
                # Don't retry non-retryable errors
                raise

        return wrapper

    return decorator


class RateLimiter:
    """
    Intelligent rate limiter that tracks API usage.

    Monitors X-RateLimit headers and adjusts request timing accordingly.

    Example:
        >>> limiter = RateLimiter()
        >>> with limiter.limit():
        ...     response = jira.search_issues(jql)
    """

    def __init__(
        self,
        requests_per_second: float = 10.0,
        burst_limit: int = 100,
    ):
        """
        Initialize the rate limiter.

        Args:
            requests_per_second: Target requests per second
            burst_limit: Maximum burst requests allowed
        """
        self.requests_per_second = requests_per_second
        self.burst_limit = burst_limit
        self._last_request_time: float = 0.0
        self._request_count: int = 0
        self._window_start: float = 0.0
        self._remaining: int | None = None
        self._reset_time: float | None = None

    def wait_if_needed(self) -> None:
        """
        Wait if necessary to respect rate limits.

        Checks both local rate tracking and any known API limits.
        """
        current_time = time.time()

        # Reset window if needed (1 second window)
        if current_time - self._window_start >= 1.0:
            self._window_start = current_time
            self._request_count = 0

        # Check if we've hit the per-second limit
        if self._request_count >= self.requests_per_second:
            sleep_time = 1.0 - (current_time - self._window_start)
            if sleep_time > 0:
                logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
                self._window_start = time.time()
                self._request_count = 0

        # Check API-reported remaining requests
        if self._remaining is not None and self._remaining <= 1:
            if self._reset_time and self._reset_time > current_time:
                sleep_time = self._reset_time - current_time + random.uniform(0.1, 0.5)
                logger.warning(f"API rate limit near: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
                self._remaining = None
                self._reset_time = None

        # Minimum spacing between requests
        min_interval = 1.0 / self.requests_per_second
        elapsed = current_time - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        self._last_request_time = time.time()
        self._request_count += 1

    def update_from_headers(self, headers: dict[str, Any]) -> None:
        """
        Update rate limit tracking from response headers.

        Args:
            headers: Response headers from Jira API
        """
        if "X-RateLimit-Remaining" in headers:
            try:
                self._remaining = int(headers["X-RateLimit-Remaining"])
                logger.debug(f"Rate limit remaining: {self._remaining}")
            except (ValueError, TypeError):
                pass

        if "X-RateLimit-Reset" in headers:
            try:
                self._reset_time = float(headers["X-RateLimit-Reset"])
            except (ValueError, TypeError):
                pass

        if "Retry-After" in headers:
            try:
                retry_after = int(headers["Retry-After"])
                self._reset_time = time.time() + retry_after
                logger.warning(f"Retry-After header: {retry_after}s")
            except (ValueError, TypeError):
                pass

    @property
    def remaining_requests(self) -> int | None:
        """Get remaining requests from last API response."""
        return self._remaining

    def limit(self) -> "RateLimitContext":
        """
        Context manager for rate-limited operations.

        Returns:
            RateLimitContext for use with 'with' statement
        """
        return RateLimitContext(self)


class RateLimitContext:
    """Context manager for rate-limited operations."""

    def __init__(self, limiter: RateLimiter):
        self._limiter = limiter

    def __enter__(self) -> RateLimiter:
        self._limiter.wait_if_needed()
        return self._limiter

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


# Global rate limiter instance
_global_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    """Get or create the global rate limiter instance."""
    global _global_limiter
    if _global_limiter is None:
        _global_limiter = RateLimiter()
    return _global_limiter


def reset_rate_limiter() -> None:
    """Reset the global rate limiter."""
    global _global_limiter
    _global_limiter = None
