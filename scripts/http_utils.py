import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional, Dict, Any
from urllib.parse import urlparse
import time

# Simple per-host throttling to avoid rate‑limit issues for APIs like Alpha Vantage
_last_call: Dict[str, float] = {}
# Default minimum interval between calls per host (seconds). Override often rate-limited hosts here.
_MIN_INTERVALS: Dict[str, float] = {
    "www.alphavantage.co": 12.0,
    "alphavantage.co": 12.0,
}


def _build_session(retries: int = 3, backoff_factor: float = 0.5, status_forcelist=(429, 500, 502, 503, 504)):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(['GET', 'POST'])
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session


_SESSION = _build_session()


def _throttle(url: str) -> None:
    try:
        host = urlparse(url).netloc.lower()
        min_interval = _MIN_INTERVALS.get(host, 1.0)
        last = _last_call.get(host)
        now = time.monotonic()
        if last is not None:
            elapsed = now - last
            if elapsed < min_interval:
                to_sleep = min_interval - elapsed
                time.sleep(to_sleep)
        _last_call[host] = time.monotonic()
    except Exception:
        # If parsing fails, do not throttle
        return


def get_json(url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, timeout: int = 10):
    _throttle(url)
    resp = _SESSION.get(url, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    try:
        return resp.json()
    except Exception:
        return None


def get_text(url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, timeout: int = 10) -> str:
    _throttle(url)
    resp = _SESSION.get(url, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def post_json(url: str, data: Dict[str, Any], headers: Optional[Dict[str, str]] = None, timeout: int = 10):
    _throttle(url)
    resp = _SESSION.post(url, json=data, headers=headers, timeout=timeout)
    resp.raise_for_status()
    try:
        return resp.json()
    except Exception:
        return None
