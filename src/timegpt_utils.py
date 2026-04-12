import time


def _looks_like_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in [
            "429",
            "rate limit",
            "too many requests",
            "requests per minute",
        ]
    )


def _looks_like_transient_network_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in [
            "connection reset by peer",
            "remote disconnected",
            "connection aborted",
            "connection broken",
            "server disconnected",
            "timed out",
            "timeout",
            "temporarily unavailable",
            "max retries exceeded",
            "connection error",
        ]
    )


def _looks_like_transient_server_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in [
            "status_code: 500",
            "status code: 500",
            "status_code: 502",
            "status code: 502",
            "status_code: 503",
            "status code: 503",
            "status_code: 504",
            "status code: 504",
            "internal server error",
            "bad gateway",
            "service unavailable",
            "gateway timeout",
            "could not parse json",
        ]
    )


class TimeGPTRateLimiter:
    """
    Lightweight client-side throttle for TimeGPT API calls.

    The goal is not perfect theoretical throughput. The goal is to keep long
    rolling validations from tripping provider burst limits during notebook runs.
    """

    def __init__(
        self,
        max_requests_per_window: int = 40,
        window_seconds: int = 60,
        cooldown_seconds: int = 90,
        max_retries: int = 3,
    ):
        self.max_requests_per_window = max_requests_per_window
        self.window_seconds = window_seconds
        self.cooldown_seconds = cooldown_seconds
        self.max_retries = max_retries
        self._window_started_at = None
        self._request_count = 0
        self._total_sleep_seconds = 0

    def _maybe_reset_window(self):
        now = time.monotonic()
        if self._window_started_at is None:
            self._window_started_at = now
            self._request_count = 0
            return

        if (now - self._window_started_at) >= self.window_seconds:
            self._window_started_at = now
            self._request_count = 0

    def _before_request(self):
        self._maybe_reset_window()
        if self._request_count >= self.max_requests_per_window:
            self._total_sleep_seconds += self.cooldown_seconds
            print(
                "TimeGPT rate limiter: "
                f"reached {self.max_requests_per_window} requests in ~{self.window_seconds}s; "
                f"sleeping {self.cooldown_seconds}s. "
                f"Added throttle so far: {self._total_sleep_seconds / 60:.1f} min."
            )
            time.sleep(self.cooldown_seconds)
            self._window_started_at = time.monotonic()
            self._request_count = 0

        self._request_count += 1

    def forecast(self, client, **kwargs):
        attempts = 0
        while True:
            self._before_request()
            try:
                return client.forecast(**kwargs)
            except Exception as exc:
                is_rate_limit = _looks_like_rate_limit_error(exc)
                is_transient_network = _looks_like_transient_network_error(exc)
                is_transient_server = _looks_like_transient_server_error(exc)

                if (
                    not is_rate_limit
                    and not is_transient_network
                    and not is_transient_server
                ) or attempts >= self.max_retries:
                    raise

                attempts += 1
                if is_rate_limit:
                    sleep_for = self.cooldown_seconds
                    reason = "provider-side rate limit"
                elif is_transient_server:
                    sleep_for = min(30 * (2 ** (attempts - 1)), self.cooldown_seconds)
                    reason = "transient server error"
                else:
                    sleep_for = min(15 * (2 ** (attempts - 1)), self.cooldown_seconds)
                    reason = "transient network error"
                self._total_sleep_seconds += sleep_for
                print(
                    "TimeGPT rate limiter: "
                    f"{reason} detected; sleeping {sleep_for}s "
                    f"before retry {attempts}/{self.max_retries}. "
                    f"Added throttle so far: {self._total_sleep_seconds / 60:.1f} min."
                )
                time.sleep(sleep_for)
                self._window_started_at = time.monotonic()
                self._request_count = 0


_TIMEGPT_RATE_LIMITER = TimeGPTRateLimiter()


def timegpt_forecast(client, **kwargs):
    return _TIMEGPT_RATE_LIMITER.forecast(client, **kwargs)
