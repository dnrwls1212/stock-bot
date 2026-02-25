# src/utils/ollama_client.py
from __future__ import annotations

import os
import json
import requests
from typing import Any, Dict, Optional


def _normalize_base_url(url: str) -> str:
    """
    Accepts:
      - http://localhost:11434
      - http://localhost:11434/
      - http://localhost:11434/api/generate
      - http://localhost:11434/api/chat
    Returns base: http://localhost:11434
    """
    u = (url or "").strip()
    if not u:
        return "http://localhost:11434"
    # strip trailing slash
    u = u.rstrip("/")
    # if someone passed endpoint, strip '/api/...'
    api_idx = u.find("/api/")
    if api_idx >= 0:
        u = u[:api_idx]
    return u.rstrip("/")


def get_ollama_base_url() -> str:
    # NEW canonical key
    url = os.environ.get("OLLAMA_URL", "").strip()

    # Backward-compat (older code used BASE_URL)
    if not url:
        url = os.environ.get("OLLAMA_BASE_URL", "").strip()

    if not url:
        url = "http://localhost:11434"

    return _normalize_base_url(url)


def get_ollama_model(default: str = "qwen2.5:7b-instruct") -> str:
    # NEW canonical key
    m = os.environ.get("OLLAMA_MODEL", "").strip()

    # Backward-compat
    if not m:
        m = os.environ.get("OLLAMA_MODEL_SUMMARY", "").strip()

    return m or default


def get_ollama_timeout(default: float = 120.0) -> float:
    # NEW canonical key
    t = os.environ.get("OLLAMA_TIMEOUT", "").strip()
    if not t:
        t = "0"
    try:
        v = float(t)
        if v <= 0:
            return float(default)
        return v
    except Exception:
        return float(default)


def ollama_generate(
    *,
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.2,
    timeout: Optional[float] = None,
    raw_url: Optional[str] = None,
) -> str:
    """
    Calls Ollama /api/generate (stream=false). Returns response text.
    """
    base = _normalize_base_url(raw_url) if raw_url else get_ollama_base_url()
    m = (model or get_ollama_model()).strip()
    to = float(timeout) if timeout is not None else get_ollama_timeout()

    r = requests.post(
        f"{base}/api/generate",
        json={
            "model": m,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": float(temperature)},
        },
        timeout=to,
    )
    r.raise_for_status()
    j = r.json()
    return str(j.get("response", "") or "")


def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort JSON parse:
    - if whole response is JSON -> ok
    - if response contains a JSON object somewhere -> try extract
    """
    s = (text or "").strip()
    if not s:
        return None
    # direct json
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
    # try slice
    i = s.find("{")
    j = s.rfind("}")
    if i >= 0 and j > i:
        try:
            obj = json.loads(s[i : j + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None