"""Per-key sliding-window RPM rate limiter and token budget guard."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, Optional, Tuple


class SlidingWindowRateLimiter:
    """Thread-safe 60-second sliding-window RPM rate limiter per virtual key_id."""

    def __init__(self, window_seconds: float = 60.0) -> None:
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        self._history: Dict[str, Deque[float]] = defaultdict(deque)

    def check_and_record(
        self, key_id: str, rpm_limit: int, now: Optional[float] = None
    ) -> Tuple[bool, int]:
        """Check whether key_id is within its RPM quota and record a hit if allowed.

        Returns:
            (allowed, remaining_requests_in_window)
        """
        if rpm_limit <= 0:
            return True, 999999

        current_time = now if now is not None else time.monotonic()
        cutoff = current_time - self.window_seconds

        with self._lock:
            q = self._history[key_id]
            while q and q[0] <= cutoff:
                q.popleft()

            if len(q) >= rpm_limit:
                return False, 0

            q.append(current_time)
            remaining = max(0, rpm_limit - len(q))
            return True, remaining

    def get_remaining(
        self, key_id: str, rpm_limit: int, now: Optional[float] = None
    ) -> int:
        """Inspect remaining requests in the current window without recording a hit."""
        if rpm_limit <= 0:
            return 999999
        current_time = now if now is not None else time.monotonic()
        cutoff = current_time - self.window_seconds
        with self._lock:
            q = self._history[key_id]
            while q and q[0] <= cutoff:
                q.popleft()
            return max(0, rpm_limit - len(q))

    def reset(self, key_id: Optional[str] = None) -> None:
        """Clear rate-limit history for a single key or all keys."""
        with self._lock:
            if key_id is None:
                self._history.clear()
            else:
                self._history.pop(key_id, None)


def check_token_budget(key_record: Dict[str, Any]) -> Tuple[bool, Optional[int]]:
    """Verify whether the key has remaining token budget.

    Returns:
        (allowed, remaining_tokens_or_none_if_unlimited)
    """
    budget = key_record.get("token_budget")
    if budget is None or int(budget) <= 0:
        return True, None
    used = int(key_record.get("tokens_used") or 0)
    remaining = int(budget) - used
    if remaining <= 0:
        return False, 0
    return True, remaining


rate_limiter = SlidingWindowRateLimiter()
