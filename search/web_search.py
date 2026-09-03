"""Genuine public web/image search through Google Custom Search JSON API."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import requests


SEARCH_ENDPOINT = "https://www.googleapis.com/customsearch/v1"
MAX_IMAGE_BYTES = 15 * 1024 * 1024


class SearchConfigurationError(ValueError):
    """Raised when required search configuration is missing or invalid."""


class SearchError(RuntimeError):
    """Raised when a search or candidate download fails."""


def _credentials() -> tuple[str, str]:
    api_key = os.getenv("GOOGLE_CSE_API_KEY")
    search_engine_id = os.getenv("GOOGLE_CSE_ID")
    if not api_key or not search_engine_id:
        raise SearchConfigurationError(
            "Set GOOGLE_CSE_API_KEY and GOOGLE_CSE_ID before running web search."
        )
    return api_key, search_engine_id


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def search_public_images(query: str, max_results: int = 10) -> list[dict[str, Any]]:
    """Search public indexed images and return post/image metadata.

    Results come from the live Google API; no URL or post is pre-selected.
    """
    if not query.strip():
        raise ValueError("query must not be empty")
    if not 1 <= max_results <= 100:
        raise ValueError("max_results must be between 1 and 100")
    api_key, search_engine_id = _credentials()
    items: list[dict[str, Any]] = []
    for start in range(1, max_results + 1, 10):
        count = min(10, max_results - len(items))
        try:
            response = requests.get(
                SEARCH_ENDPOINT,
                params={
                    "key": api_key,
                    "cx": search_engine_id,
                    "q": query,
                    "searchType": "image",
                    "num": count,
                    "start": start,
                    "safe": "active",
                },
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as error:
            raise SearchError(f"Public search failed: {error}") from error
        for item in payload.get("items", []):
            image_url = item.get("link")
            post_url = item.get("image", {}).get("contextLink") or item.get("link")
            if _is_http_url(image_url) and _is_http_url(post_url):
                items.append(
                    {
                        "url": post_url,
                        "image_url": image_url,
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet", ""),
                        "source": "google_custom_search",
                    }
                )
        if len(items) >= max_results or not payload.get("items"):
            break
    return items[:max_results]


def download_candidate_images(
    candidates: Iterable[dict[str, Any]], output_dir: str | Path = "candidates"
) -> list[dict[str, Any]]:
    """Download search result images and attach local paths for face matching."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    downloaded: list[dict[str, Any]] = []
    for candidate in candidates:
        image_url = candidate.get("image_url")
        if not isinstance(image_url, str) or not _is_http_url(image_url):
            continue
        try:
            response = requests.get(
                image_url,
                headers={"User-Agent": "HH-Goa-FaceBlockchain/1.0"},
                stream=True,
                timeout=20,
            )
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if not content_type.startswith("image/"):
                continue
            content = bytearray()
            for chunk in response.iter_content(chunk_size=64 * 1024):
                content.extend(chunk)
                if len(content) > MAX_IMAGE_BYTES:
                    raise SearchError(f"Image exceeds {MAX_IMAGE_BYTES} byte limit: {image_url}")
            suffix = Path(urlparse(image_url).path).suffix.lower()
            suffix = suffix if suffix in {".jpg", ".jpeg", ".png", ".webp"} else ".img"
            filename = hashlib.sha256(image_url.encode("utf-8")).hexdigest() + suffix
            local_path = destination / filename
            local_path.write_bytes(content)
        except (requests.RequestException, OSError, SearchError) as error:
            candidate = {**candidate, "download_error": str(error)}
        else:
            candidate = {**candidate, "image_path": str(local_path)}
        downloaded.append(candidate)
    return downloaded


def save_candidates(candidates: Iterable[dict[str, Any]], output_path: str | Path) -> None:
    """Save search results as JSON for review and later pipeline stages."""
    Path(output_path).write_text(
        json.dumps(list(candidates), indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )