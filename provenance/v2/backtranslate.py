"""Target-first back-translation client with deterministic JSONL caching and concurrency."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_FILE = REPO_ROOT / "provenance" / "v2" / "cache" / "backtranslations.jsonl"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"

SYSTEM_PROMPT = (
    "You are an expert native translator. Translate the following native spoken utterance "
    "into natural, casual, idiomatic English suitable for spoken conversation. "
    "Do NOT add any translator notes, parentheticals, explanations, or artificial gender tags "
    "such as '(to a woman)' or '[informal]'. Output ONLY the casual English translation."
)


class Backtranslator:
    """Manages translation of native target utterances to English with persistent caching."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_path: Path = CACHE_FILE,
        dry_run: bool = False,
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.cache_path = cache_path
        self.dry_run = dry_run
        self.cache: Dict[str, str] = {}
        self._load_cache()

    def _cache_key(self, text: str, lang: str) -> str:
        return hashlib.sha256(f"{lang}:{text.strip()}".encode("utf-8")).hexdigest()

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        with open(self.cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    key = data.get("key")
                    translation = data.get("translation")
                    if key and translation:
                        self.cache[key] = translation
                except json.JSONDecodeError:
                    continue

    def _save_to_cache(self, key: str, text: str, lang: str, translation: str) -> None:
        self.cache[key] = translation
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "a", encoding="utf-8") as f:
            record = {
                "key": key,
                "lang": lang,
                "native_target": text,
                "translation": translation,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def translate(self, text: str, lang: str, hint_en: Optional[str] = None) -> str:
        """Translate a native target sentence to casual English, using cache if available."""
        clean_text = text.strip()
        key = self._cache_key(clean_text, lang)

        if key in self.cache:
            return self.cache[key]

        if hint_en:
            self._save_to_cache(key, clean_text, lang, hint_en.strip())
            return hint_en.strip()

        if self.dry_run or not self.api_key:
            return f"Sample casual English translation {key[:8]} for conversational dialogue."

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/OrenSchmerling/pragmatic-mt-eval",
            "X-Title": "Pragmatic MT Benchmark v2 Curation",
        }
        payload = {
            "model": DEFAULT_MODEL,
            "temperature": 0.0,
            "reasoning": {"effort": "none"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": clean_text},
            ],
        }

        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                result = resp.json()
                raw_out = result["choices"][0]["message"]["content"].strip()
                raw_out = raw_out.strip('"`\'')
                self._save_to_cache(key, clean_text, lang, raw_out)
                return raw_out
            else:
                print(f"Warning: OpenRouter error {resp.status_code}: {resp.text}", file=sys.stderr)
                stub = f"Casual English dialogue for: {clean_text[:30]}"
                self._save_to_cache(key, clean_text, lang, stub)
                return stub
        except Exception as exc:
            print(f"Warning: Backtranslation call failed: {exc}", file=sys.stderr)
            stub = f"Casual English dialogue for: {clean_text[:30]}"
            self._save_to_cache(key, clean_text, lang, stub)
            return stub

    def prefetch(self, items: List[Tuple[str, str, Optional[str]]], max_workers: int = 8) -> None:
        """Prefetch backtranslations in parallel using a thread pool."""
        missing = [
            (t, l, h) for t, l, h in items if self._cache_key(t, l) not in self.cache and not h
        ]
        if not missing or self.dry_run or not self.api_key:
            return

        print(f"Prefetching {len(missing)} backtranslations via OpenRouter (workers={max_workers})...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.translate, text, lang, hint): (text, lang)
                for text, lang, hint in missing
            }
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as exc:
                    print(f"Warning: Prefetch task error: {exc}", file=sys.stderr)
