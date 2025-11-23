import json
from pathlib import Path
from typing import Tuple, Dict

# Root-Verzeichnis bestimmen (funktioniert sowohl von Root als auch von scripts/)
try:
    ROOT_DIR = Path(__file__).parent.parent
except:
    ROOT_DIR = Path.cwd()

def _load_json(path: Path):
    try:
        if path.exists():
            return json.load(open(path, encoding='utf-8'))
    except Exception:
        return None
    return None


def load_community_signals(path: Path = None) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    if path is None:
        path = ROOT_DIR / "data/community_data/community_signals.json"
    """Load community signals from JSON.

    Supports two formats:
    - Nested: {"superinvestor_score": {...}, "reddit_score": {...}, "x_score": {...}}
    - Flat: {"AAPL": 0.5, "MSFT": 0.4, ...} in which case the same map is returned for all three.

    Returns (superinvestor_map, reddit_map, x_map)
    """
    data = _load_json(path)
    if not data:
        return {}, {}, {}

    # Nested format
    if isinstance(data, dict) and any(k in data for k in ("superinvestor_score", "reddit_score", "x_score")):
        super_map = data.get("superinvestor_score", {}) or {}
        reddit_map = data.get("reddit_score", {}) or {}
        x_map = data.get("x_score", {}) or {}
        # ensure values are floats
        def norm(d):
            return {k: float(v) for k, v in (d.items() if isinstance(d, dict) else [])}
        return norm(super_map), norm(reddit_map), norm(x_map)

    # Flat format: treat as generic community signal -> use same for all
    if isinstance(data, dict):
        try:
            flat = {k: float(v) for k, v in data.items()}
            return flat, flat, flat
        except Exception:
            return {}, {}, {}

    return {}, {}, {}


def load_ai_moat(path: Path = None) -> Dict[str, float]:
    if path is None:
        path = ROOT_DIR / "data/community_data/ai_moat.json"
    data = _load_json(path)
    if not data:
        return {}
    moat = data.get("ki_moat_score") or data.get("ai_moat_score") or {}
    try:
        return {k: float(v) for k, v in moat.items()} if isinstance(moat, dict) else {}
    except Exception:
        return {}
