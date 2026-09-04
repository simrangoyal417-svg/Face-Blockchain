import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv


load_dotenv()


class SearchConfigurationError(Exception):
    """Raised when search configuration is missing."""


def upload_image(image_path):
    """Upload an image to SerpApi and return its temporary image ID."""

    api_key = os.getenv("SERPAPI_KEY")

    if not api_key:
        raise SearchConfigurationError(
            "SERPAPI_KEY not found in .env"
        )

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    with image_path.open("rb") as image_file:
        response = requests.post(
            "https://serpapi.com/image",
            params={"api_key": api_key},
            files={"image": image_file},
            timeout=30,
        )

    response.raise_for_status()

    data = response.json()

    if "image_id" not in data:
        raise RuntimeError(
            f"SerpApi image upload failed: {data}"
        )

    return data["image_id"]


def search_google_lens(image_id, max_results=10):
    """Search the uploaded image using Google Lens."""

    api_key = os.getenv("SERPAPI_KEY")

    if not api_key:
        raise SearchConfigurationError(
            "SERPAPI_KEY not found in .env"
        )

    response = requests.get(
        "https://serpapi.com/search.json",
        params={
            "engine": "google_lens",
            "image_id": image_id,
            "type": "visual_matches",
            "api_key": api_key,
        },
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    if data.get("error"):
        raise RuntimeError(
            f"Google Lens search failed: {data['error']}"
        )

    matches = data.get("visual_matches", [])

    return matches[:max_results]


def search_public_images(image_path, max_results=10):
    """Perform a genuine reverse-image search."""

    image_id = upload_image(image_path)

    matches = search_google_lens(
        image_id,
        max_results=max_results,
    )

    candidates = []

    for match in matches:
        candidates.append(
            {
                "url": match.get("link"),
                "title": match.get("title"),
                "caption": match.get("title"),
                "platform": match.get("source"),
                "image_url": match.get("image"),
                "thumbnail_url": match.get("thumbnail"),
                "position": match.get("position"),
            }
        )

    return candidates


def download_candidate_images(candidates, output_dir):
    """Download candidate images for later face matching."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    downloaded = []

    for index, candidate in enumerate(candidates, start=1):
        urls = [
            candidate.get("image_url"),
            candidate.get("thumbnail_url"),
        ]

        for image_url in urls:
            if not image_url:
                continue

            try:
                response = requests.get(
                    image_url,
                    timeout=30,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 "
                            "(X11; Linux x86_64) "
                            "AppleWebKit/537.36 "
                            "Chrome/120 Safari/537.36"
                        )
                    },
                )

                response.raise_for_status()

                content_type = response.headers.get(
                    "Content-Type",
                    ""
                )

                if not content_type.startswith("image/"):
                    continue

                file_path = output_path / f"candidate_{index}.jpg"

                file_path.write_bytes(response.content)

                candidate_copy = dict(candidate)
                candidate_copy["image_path"] = str(file_path)

                downloaded.append(candidate_copy)

                break

            except requests.RequestException:
                continue

    return downloaded


def save_candidates(candidates, output_file):
    """Save candidate records as JSON."""

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            candidates,
            file,
            indent=2,
            ensure_ascii=False,
        )
