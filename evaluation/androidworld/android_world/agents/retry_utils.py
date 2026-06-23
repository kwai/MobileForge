# Copyright 2025 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Retry utilities for API calls with exponential backoff."""

import time
import random
from typing import Callable, Any, Optional, Tuple


# Default retry configuration
DEFAULT_MAX_RETRIES = None  # None means infinite retries until success
DEFAULT_BASE_DELAY = 2.0  # seconds
DEFAULT_MAX_DELAY = 120.0  # seconds (increased for long waits)
DEFAULT_EXPONENTIAL_BASE = 2.0


def is_rate_limit_error(error: Exception) -> bool:
    """Check if the error is a rate limit (429) error.
    
    Args:
        error: The exception to check.
        
    Returns:
        True if this is a rate limit error that should be retried.
    """
    error_str = str(error).lower()
    return (
        "429" in error_str
        or "rate limit" in error_str
        or "too many requests" in error_str
        or "请求频率过高" in error_str
        or "toomanyrequest" in error_str
    )


def is_retryable_error(error: Exception) -> bool:
    """Check if the error is retryable (rate limit, timeout, server errors).
    
    Args:
        error: The exception to check.
        
    Returns:
        True if this error should be retried.
    """
    error_str = str(error).lower()
    
    # Rate limit errors
    if is_rate_limit_error(error):
        return True
    
    # Server errors (5xx)
    if any(code in error_str for code in ["500", "502", "503", "504"]):
        return True
    
    # Timeout errors
    if any(keyword in error_str for keyword in ["timeout", "timed out", "connection"]):
        return True
    
    return False


def calculate_backoff_delay(
    retry_count: int,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    exponential_base: float = DEFAULT_EXPONENTIAL_BASE,
    jitter: bool = True,
) -> float:
    """Calculate delay with exponential backoff and optional jitter.
    
    Args:
        retry_count: Current retry attempt number (0-indexed).
        base_delay: Base delay in seconds.
        max_delay: Maximum delay in seconds.
        exponential_base: Base for exponential calculation.
        jitter: Whether to add random jitter.
        
    Returns:
        Delay in seconds.
    """
    # Exponential backoff: base_delay * (exponential_base ^ retry_count)
    delay = base_delay * (exponential_base ** retry_count)
    
    # Cap at max delay
    delay = min(delay, max_delay)
    
    # Add jitter (random factor between 0.5 and 1.5)
    if jitter:
        jitter_factor = 0.5 + random.random()
        delay = delay * jitter_factor
    
    return delay


def call_with_retry(
    func: Callable,
    *args,
    max_retries: Optional[int] = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
    **kwargs
) -> Tuple[Any, int]:
    """Call a function with retry logic and exponential backoff.
    
    Args:
        func: The function to call.
        *args: Positional arguments for the function.
        max_retries: Maximum number of retry attempts. None means infinite retries until success.
        base_delay: Base delay in seconds for exponential backoff.
        max_delay: Maximum delay in seconds.
        on_retry: Optional callback called before each retry with (retry_count, error, delay).
        **kwargs: Keyword arguments for the function.
        
    Returns:
        Tuple of (result, retry_count) where retry_count is the number of retries used.
        
    Raises:
        Non-retryable exceptions are raised immediately.
    """
    retry_count = 0
    
    while True:
        try:
            result = func(*args, **kwargs)
            return result, retry_count
            
        except Exception as e:
            # Check if this error is retryable
            if not is_retryable_error(e):
                raise e
            
            # Check if we have retries left (if max_retries is set)
            if max_retries is not None and retry_count >= max_retries:
                raise e
            
            # Calculate delay (cap retry_count for delay calculation to avoid overflow)
            delay = calculate_backoff_delay(
                min(retry_count, 10),  # Cap at 10 for delay calculation
                base_delay=base_delay,
                max_delay=max_delay,
            )
            
            # Call retry callback if provided
            if on_retry:
                on_retry(retry_count + 1, e, delay)
            
            # Wait before retrying
            time.sleep(delay)
            retry_count += 1


class RetryableAPIClient:
    """A wrapper class to add retry capability to API clients.
    
    By default, retries indefinitely until success for rate limit errors.
    """
    
    def __init__(
        self,
        client,
        max_retries: Optional[int] = None,  # None = infinite retries until success
        base_delay: float = DEFAULT_BASE_DELAY,
        max_delay: float = DEFAULT_MAX_DELAY,
        verbose: bool = True,
    ):
        """Initialize the retryable client wrapper.
        
        Args:
            client: The underlying API client (e.g., OpenAI client).
            max_retries: Maximum number of retry attempts. None means infinite retries.
            base_delay: Base delay in seconds.
            max_delay: Maximum delay in seconds.
            verbose: Whether to print retry messages.
        """
        self.client = client
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.verbose = verbose
    
    def _on_retry(self, retry_count: int, error: Exception, delay: float):
        """Callback for retry events."""
        if self.verbose:
            max_str = str(self.max_retries) if self.max_retries is not None else "inf"
            # Extract short error message
            error_str = str(error)
            if "429" in error_str or "请求频率过高" in error_str:
                error_msg = "Rate limit (429)"
            elif "500" in error_str or "502" in error_str or "503" in error_str:
                error_msg = "Server error"
            elif "timeout" in error_str.lower():
                error_msg = "Timeout"
            else:
                error_msg = error_str[:50]
            
            print(
                f"[Retry {retry_count}/{max_str}] "
                f"{error_msg} - Waiting {delay:.1f}s before retry..."
            )
    
    def create_chat_completion(self, **kwargs):
        """Create a chat completion with retry logic.
        
        Retries indefinitely until success for rate limit and server errors.
        Only returns the successful response - intermediate errors are not exposed.
        
        Args:
            **kwargs: Arguments to pass to client.chat.completions.create()
            
        Returns:
            The API response (only successful responses).
        """
        def _call():
            return self.client.chat.completions.create(**kwargs)
        
        result, retry_count = call_with_retry(
            _call,
            max_retries=self.max_retries,  # None = infinite
            base_delay=self.base_delay,
            max_delay=self.max_delay,
            on_retry=self._on_retry,
        )
        
        if retry_count > 0 and self.verbose:
            print(f"[Retry] Success after {retry_count} retries.")
        
        return result

