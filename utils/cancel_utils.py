"""
Shared cancellation utilities for long-running operations.
Both app.py and utility modules use this to check for user cancellation.
"""

import threading

# Thread-safe flags (shared across modules)
_cancel_requested = threading.Event()
_paused = threading.Event()


def request_cancel():
    """Set the cancel flag."""
    _cancel_requested.set()


def reset_cancel():
    """Clear all flags."""
    _cancel_requested.clear()
    _paused.clear()


def pause():
    """Set the pause flag."""
    _paused.set()


def resume():
    """Clear the pause flag."""
    _paused.clear()


def is_paused() -> bool:
    """Check if task is paused."""
    return _paused.is_set()


def raise_if_cancelled():
    """
    Check cancel flag; raise TaskCancelledError if set.
    Also blocks while paused.
    Call this periodically inside long-running loops.
    """
    # Block while paused (check cancel every 0.5s so pause is responsive)
    while _paused.is_set():
        if _cancel_requested.is_set():
            break
        try:
            import time
            time.sleep(0.5)
        except KeyboardInterrupt:
            break

    if _cancel_requested.is_set():
        reset_cancel()
        raise TaskCancelledError("任务已被用户取消")


class TaskCancelledError(Exception):
    """Raised when the user cancels a long-running task."""
    pass
