# api_utils.py
"""
Gemeinsame Utility-Funktionen für API-Calls, inkl. Retry-Logik mit Exponential Backoff.
"""
import time
import logging
from typing import Callable, Optional, Any
from functools import wraps

logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 2.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    check_result: Optional[Callable[[Any], bool]] = None
):
    """
    Decorator für Retry-Logik mit Exponential Backoff.
    
    Args:
        max_retries: Maximale Anzahl Retries (default: 3)
        initial_delay: Initiale Wartezeit in Sekunden (default: 2.0)
        max_delay: Maximale Wartezeit in Sekunden (default: 60.0)
        exponential_base: Basis für Exponential Backoff (default: 2.0)
        check_result: Optional Funktion zum Prüfen ob Ergebnis gültig ist.
                     Wenn None, wird nur auf Exceptions geprüft.
                     Wenn Funktion False zurückgibt, wird retried.
    
    Example:
        @retry_with_backoff(max_retries=3, initial_delay=2.0)
        def fetch_data():
            return api_call()
        
        @retry_with_backoff(check_result=lambda r: len(r) > 0)
        def fetch_posts():
            return api_call()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    
                    # Prüfe ob Ergebnis gültig ist (falls check_result Funktion vorhanden)
                    if check_result is not None:
                        if not check_result(result):
                            if attempt < max_retries - 1:
                                wait_time = min(
                                    initial_delay * (exponential_base ** attempt),
                                    max_delay
                                )
                                logger.debug(
                                    f"{func.__name__}: Invalid result, retrying in {wait_time:.1f}s "
                                    f"(attempt {attempt + 1}/{max_retries})"
                                )
                                time.sleep(wait_time)
                                continue
                            else:
                                logger.warning(
                                    f"{func.__name__}: Invalid result after {max_retries} attempts"
                                )
                                return result
                    
                    # Erfolgreich
                    if attempt > 0:
                        logger.debug(f"{func.__name__}: Succeeded on attempt {attempt + 1}")
                    return result
                    
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = min(
                            initial_delay * (exponential_base ** attempt),
                            max_delay
                        )
                        logger.debug(
                            f"{func.__name__}: Error '{type(e).__name__}: {str(e)[:100]}', "
                            f"retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(wait_time)
                    else:
                        logger.warning(
                            f"{func.__name__}: Failed after {max_retries} attempts: {e}"
                        )
            
            # Alle Retries fehlgeschlagen
            if last_exception:
                raise last_exception
            return None
        
        return wrapper
    return decorator


def fetch_with_retry(
    func: Callable,
    max_retries: int = 3,
    initial_delay: float = 2.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    check_result: Optional[Callable[[Any], bool]] = None,
    *args,
    **kwargs
) -> Any:
    """
    Wrapper-Funktion für Retry-Logik mit Exponential Backoff.
    Nützlich wenn man Retry-Logik inline verwenden möchte.
    
    Args:
        func: Funktion die aufgerufen werden soll
        max_retries: Maximale Anzahl Retries (default: 3)
        initial_delay: Initiale Wartezeit in Sekunden (default: 2.0)
        max_delay: Maximale Wartezeit in Sekunden (default: 60.0)
        exponential_base: Basis für Exponential Backoff (default: 2.0)
        check_result: Optional Funktion zum Prüfen ob Ergebnis gültig ist
        *args, **kwargs: Argumente für func
    
    Returns:
        Ergebnis von func oder None bei Fehler
    
    Example:
        result = fetch_with_retry(
            api_call,
            max_retries=3,
            check_result=lambda r: r is not None and len(r) > 0,
            ticker="AAPL"
        )
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            result = func(*args, **kwargs)
            
            # Prüfe ob Ergebnis gültig ist
            if check_result is not None:
                if not check_result(result):
                    if attempt < max_retries - 1:
                        wait_time = min(
                            initial_delay * (exponential_base ** attempt),
                            max_delay
                        )
                        logger.debug(
                            f"{func.__name__}: Invalid result, retrying in {wait_time:.1f}s "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.warning(
                            f"{func.__name__}: Invalid result after {max_retries} attempts"
                        )
                        return result
            
            # Erfolgreich
            if attempt > 0:
                logger.debug(f"{func.__name__}: Succeeded on attempt {attempt + 1}")
            return result
            
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = min(
                    initial_delay * (exponential_base ** attempt),
                    max_delay
                )
                logger.debug(
                    f"{func.__name__}: Error '{type(e).__name__}: {str(e)[:100]}', "
                    f"retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(wait_time)
            else:
                logger.warning(
                    f"{func.__name__}: Failed after {max_retries} attempts: {e}"
                )
    
    # Alle Retries fehlgeschlagen
    if last_exception:
        raise last_exception
    return None
