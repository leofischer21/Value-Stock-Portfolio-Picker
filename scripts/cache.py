from pathlib import Path
import json
import time
import hashlib
from typing import Any, Optional

# Root-Verzeichnis bestimmen (funktioniert sowohl von Root als auch von scripts/)
try:
    # Wenn von scripts/ aufgerufen, gehe 2 Ebenen hoch
    ROOT_DIR = Path(__file__).parent.parent
except:
    # Fallback: aktuelles Verzeichnis
    ROOT_DIR = Path.cwd()

CACHE_DIR = ROOT_DIR / "data/cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _key_to_file(key: str) -> Path:
    # Use a hash to keep filenames stable and safe
    h = hashlib.sha256(key.encode('utf-8')).hexdigest()
    return CACHE_DIR / f"{h}.json"


def set(key: str, value: Any, ttl_seconds: int = 24 * 3600) -> None:
    path = _key_to_file(key)
    payload = {
        "expires": int(time.time()) + int(ttl_seconds),
        "value": value,
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    except Exception:
        # Swallow to avoid breaking upstream callers
        return


def get(key: str) -> Optional[Any]:
    path = _key_to_file(key)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        expires = int(payload.get("expires", 0))
        if time.time() > expires:
            try:
                path.unlink()
            except Exception:
                pass
            return None
        return payload.get("value")
    except Exception:
        try:
            path.unlink()
        except Exception:
            pass
        return None


def clear() -> None:
    for p in CACHE_DIR.iterdir():
        if p.is_file() and p.suffix == ".json":
            try:
                p.unlink()
            except Exception:
                pass
