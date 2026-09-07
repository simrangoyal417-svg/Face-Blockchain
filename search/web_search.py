import json
import os
from pathlib import Path

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from html.parser import HTMLParser
from urllib.parse import parse_qs, urlparse
import re


load_dotenv()


class SearchConfigurationError(Exception):
    """Raised when search configuration is missing."""


class SearchFailure(RuntimeError):
    """Raised when SerpApi cannot complete a required request."""

    def __init__(self, message, *, image_id=None, phase="search"):
        super().__init__(message)
        self.image_id = image_id
        self.phase = phase


class SearchResults(list):
    """Normalized candidates plus the raw SerpApi request diagnostics."""

    def __init__(self, candidates, *, image_id, exact_count, visual_count, all_count,
                 secondary_count=0, profile_count=0, post_count=0,
                 zero_messages=None, errors=None):
        super().__init__(candidates)
        self.image_id = image_id
        self.exact_count = exact_count
        self.visual_count = visual_count
        self.all_count = all_count
        self.secondary_count = secondary_count
        self.profile_count = profile_count
        self.post_count = post_count
        self.zero_messages = zero_messages or []
        self.errors = errors or []


class _InstagramMetadataParser(HTMLParser):
    """Collect public metadata values without depending on an HTML library."""

    def __init__(self):
        super().__init__()
        self.values = []
        self.metadata = {}

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "meta":
            return
        attributes = dict(attrs)
        key = attributes.get("property") or attributes.get("name")
        content = attributes.get("content")
        if key and content and key.lower() in {
            "og:title",
            "og:description",
            "description",
            "twitter:title",
            "twitter:description",
        }:
            self.values.append(content)
        if key and content:
            self.metadata[key.lower()] = content


def _response_data(response):
    try:
        return response.json()
    except ValueError:
        return {"raw_response": response.text}


def _print_api_status(response, data, image_id, request_type):
    metadata = data.get("search_metadata") or {}
    print("SERPAPI STATUS:")
    print(f"request_type: {request_type}")
    print(f"HTTP status: {response.status_code}")
    print(f"search_metadata.status: {metadata.get('status')}")
    print(f"search_metadata.id: {metadata.get('id')}")
    print(f"error: {data.get('error')}")
    print(f"image_id used: {image_id}")
    if response.status_code >= 400 or data.get("error"):
        print(f"COMPLETE API RESPONSE: {json.dumps(data, indent=2, ensure_ascii=False)}")


def instagram_profile_url(source_url):
    """Return an explicitly discovered Instagram profile URL, if applicable."""
    if not isinstance(source_url, str):
        return None
    parsed = urlparse(source_url)
    if parsed.netloc.lower() not in {"instagram.com", "www.instagram.com"}:
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) != 1 or parts[0].lower() in {"p", "reel", "tv", "explore"}:
        return None
    return f"{parsed.scheme}://{parsed.netloc}/{parts[0]}"


def _instagram_url(value):
    if not isinstance(value, str):
        return None
    parsed = urlparse(value)
    if parsed.netloc.lower() not in {"instagram.com", "www.instagram.com"}:
        return None
    return value


def _explicit_username(match):
    username = match.get("username")
    if isinstance(username, str) and re.fullmatch(r"[A-Za-z0-9._]{1,30}", username):
        return username
    for value in (match.get("title"), match.get("snippet"), match.get("description")):
        if isinstance(value, str):
            found = re.search(r"@([A-Za-z0-9._]{1,30})", value)
            if found:
                return found.group(1)
    return None


def _username_from_profile_url(profile_url):
    if not profile_url:
        return None
    parts = [part for part in urlparse(profile_url).path.split("/") if part]
    if len(parts) == 1 and re.fullmatch(r"[A-Za-z0-9._]{1,30}", parts[0]):
        return parts[0]
    return None


def _valid_instagram_username(value):
    if not isinstance(value, str):
        return None
    value = value.strip().lstrip("@")
    return value if re.fullmatch(r"[A-Za-z0-9._]{1,30}", value) else None


def _username_from_post_url(source_url):
    """Extract a handle only when the URL explicitly contains one."""
    if not isinstance(source_url, str):
        return None
    parsed = urlparse(source_url)
    if parsed.netloc.lower() not in {"instagram.com", "www.instagram.com"}:
        return None
    parts = [part for part in parsed.path.split("/") if part]
    for marker in ("p", "reel", "tv"):
        if marker in parts:
            marker_index = parts.index(marker)
            if marker_index > 0:
                username = _valid_instagram_username(parts[marker_index - 1])
                if username and username.lower() not in {"www", "instagram"}:
                    return username
            break
    query = parse_qs(parsed.query)
    for key in ("username", "user", "profile", "owner", "author"):
            values = query.get(key, [])
            if values:
                username = _valid_instagram_username(values[0])
                if username:
                    return username
    return None


def _username_from_post_metadata(source_url, metadata=None):
    """Read an explicitly published creator handle from post metadata."""
    values = []
    if isinstance(metadata, dict):
        values.extend(value for value in metadata.values() if isinstance(value, str))
    if isinstance(source_url, str) and source_url.startswith(("http://", "https://")):
        try:
            response = requests.get(
                source_url,
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0 (compatible; FaceSearch/1.0)"},
            )
            if response.ok:
                parser = _InstagramMetadataParser()
                parser.feed(response.text)
                values.extend(parser.values)
                values.append(response.text[:200000])
        except requests.RequestException:
            pass
    for value in values:
        match = re.search(r"@([A-Za-z0-9._]{1,30})", value)
        if match:
            return match.group(1)
        match = re.search(r'"username"\s*:\s*"([A-Za-z0-9._]{1,30})"', value)
        if match:
            return match.group(1)
    return None


def _google_organic_search(query, max_results=20):
    """Run one SerpApi Google Organic query and return its organic results."""
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        raise SearchConfigurationError("SERPAPI_KEY not found in .env")
    response = requests.get(
        "https://serpapi.com/search.json",
        params={
            "engine": "google",
            "q": query,
            "num": max_results,
            "api_key": api_key,
        },
        timeout=15,
    )
    data = _response_data(response)
    _print_api_status(response, data, "not applicable", f"google organic: {query}")
    if response.status_code >= 400 or data.get("error"):
        return [], [f"google organic query {query!r}: {json.dumps(data, ensure_ascii=False)}"]
    return data.get("organic_results") or [], []


def _fetch_profile_page_metadata(profile_url):
    try:
        response = requests.get(
            profile_url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0 (compatible; FaceSearch/1.0)"},
        )
        if not response.ok:
            return {}
        parser = _InstagramMetadataParser()
        parser.feed(response.text)
        return parser.metadata
    except requests.RequestException:
        return {}


def fetch_instagram_profile(profile_url):
    """Fetch one explicit Instagram profile through Google Organic and metadata."""
    profile_url = _instagram_url(profile_url)
    if not profile_url:
        return None
    username = _username_from_profile_url(profile_url)
    if not username:
        return None
    organic_results, _ = _google_organic_search(
        f"site:instagram.com/{username}",
        max_results=10,
    )
    matching_result = next(
        (
            result for result in organic_results
            if _instagram_url(result.get("link"))
            and _username_from_profile_url(result.get("link")) == username
        ),
        {},
    )
    page_metadata = _fetch_profile_page_metadata(profile_url)
    full_name = page_metadata.get("og:title") or matching_result.get("title")
    if isinstance(full_name, str):
        full_name = re.sub(r"\s*\(@?" + re.escape(username) + r"\).*", "", full_name).strip()
    return {
        "username": username,
        "profile_url": profile_url,
        "full_name": full_name,
        "profile_image_url": (
            page_metadata.get("og:image")
            or page_metadata.get("twitter:image")
            or matching_result.get("image")
            or matching_result.get("thumbnail")
        ),
        "source": matching_result.get("source") or "Google Organic",
        "discovery_method": "google_organic",
        "title": matching_result.get("title"),
        "snippet": matching_result.get("snippet"),
    }


def fetch_instagram_profile_image(profile):
    """Return the discovered profile image URL without inventing an image."""
    if not isinstance(profile, dict):
        return None
    return profile.get("profile_image_url") or profile.get("image_url") or profile.get("thumbnail_url")


def discover_instagram_profiles(search_terms, max_results=20):
    """Discover Instagram profiles using only dynamically supplied search terms."""
    profiles = []
    seen = set()
    queries = []
    for term in search_terms:
        if isinstance(term, dict):
            term = term.get("title") or term.get("snippet") or term.get("query")
        if not isinstance(term, str) or not term.strip():
            continue
        phrase = _clean_profile_search_phrase(term)
        if not phrase:
            continue
        query = f'site:instagram.com "{phrase}"'
        if query not in queries:
            queries.append(query)
        if len(queries) >= 3:
            break
    for query in queries:
        organic_results, errors = _google_organic_search(query, max_results)
        if errors:
            print(f"SECONDARY SEARCH WARNING: {errors[0]}")
        if not organic_results:
            continue
        for result in organic_results:
            link = result.get("link")
            profile_url = instagram_profile_url(link)
            if not profile_url:
                username = _username_from_post_url(link) or _explicit_username(result)
                if username:
                    profile_url = f"https://www.instagram.com/{username}/"
            if not profile_url or profile_url in seen:
                continue
            seen.add(profile_url)
            profile = fetch_instagram_profile(profile_url)
            if profile:
                profiles.append(profile)
    return profiles


def _clean_profile_search_phrase(value):
    """Keep a short identity phrase and discard bios, titles, and punctuation."""
    text = re.sub(r"[\"'`]+", " ", value)
    text = re.split(r"\s+[|:\u2022]\s+|\s+-\s+|\s+[\u2013\u2014]\s+", text, maxsplit=1)[0]
    text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[^A-Za-z0-9.@_\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = []
    ignored = {
        "backend", "frontend", "developer", "engineer", "founder", "student",
        "software", "data", "analyst", "manager", "director", "president",
        "ceo", "sde", "at", "the", "and", "official", "profile",
    }
    for token in text.split():
        normalized = token.lstrip("@").strip("._")
        if not normalized or normalized.lower() in ignored:
            continue
        tokens.append(normalized)
        if len(tokens) >= 6:
            break
    return " ".join(tokens)


def _classify_candidate(source_url, profile_url):
    instagram_url = _instagram_url(source_url)
    if not instagram_url and not profile_url:
        return "OTHER_WEBSITE" if source_url else "UNKNOWN"
    path = urlparse(instagram_url or profile_url).path.strip("/").split("/")
    if len(path) == 1 and path[0] and path[0].lower() not in {"explore", "accounts", "direct"}:
        return "INSTAGRAM_PROFILE"
    return "INSTAGRAM_POST"


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

    try:
        with image_path.open("rb") as image_file:
            response = requests.post(
                "https://serpapi.com/image",
                params={"api_key": api_key},
                files={"image": image_file},
                timeout=30,
            )
    except requests.RequestException as error:
        print("SERPAPI STATUS:")
        print("request_type: image upload")
        print("HTTP status: unavailable")
        print("search_metadata.status: unavailable")
        print("search_metadata.id: unavailable")
        print(f"error: {error}")
        print("image_id used: not available")
        print(f"COMPLETE API ERROR: {error}")
        raise SearchFailure(str(error), phase="image upload") from error

    data = _response_data(response)
    _print_api_status(response, data, "not available", "image upload")
    if response.status_code >= 400:
        raise SearchFailure(
            f"SerpApi image upload failed: {json.dumps(data, ensure_ascii=False)}",
            phase="image upload",
        )

    if "image_id" not in data:
        raise SearchFailure(
            f"SerpApi image upload returned no image_id: {json.dumps(data, ensure_ascii=False)}",
            phase="image upload",
        )

    print(f"IMAGE UPLOAD: SUCCESS")
    print(f"IMAGE ID: {data['image_id']}")
    return data["image_id"]


def search_google_lens(image_id, max_results=20):
    """Run all, exact, and visual Lens searches with complete status output."""

    api_key = os.getenv("SERPAPI_KEY")

    if not api_key:
        raise SearchConfigurationError(
            "SERPAPI_KEY not found in .env"
        )

    matches = []
    all_matches = []
    counts = {"all": 0, "exact_matches": 0, "visual_matches": 0}
    zero_messages = []
    errors = []
    for result_type in ("all", "exact_matches", "visual_matches"):
        try:
            response = requests.get(
            "https://serpapi.com/search.json",
            params={
                "engine": "google_lens",
                "image_id": image_id,
                "type": result_type,
                "num": max_results,
                "api_key": api_key,
            },
            timeout=30,
            )
        except requests.RequestException as error:
            message = f"{result_type}: request failed: {error}"
            print(f"SERPAPI ERROR: {message}")
            errors.append(message)
            continue
        data = _response_data(response)
        _print_api_status(response, data, image_id, result_type)
        if response.status_code >= 400:
            errors.append(f"{result_type}: {json.dumps(data, ensure_ascii=False)}")
            continue
        if data.get("error"):
            message = f"{result_type}: {data['error']}"
            print(f"COMPLETE API ERROR: {message}")
            if "hasn't returned any results" in str(data["error"]).lower():
                zero_messages.append(message)
            else:
                errors.append(message)
            continue
        if result_type == "all":
            all_items = []
            for key in ("exact_matches", "visual_matches", "image_results"):
                items = data.get(key) or []
                if isinstance(items, list):
                    for item in items:
                        match_copy = dict(item)
                        match_copy["result_type"] = key
                        match_copy["exact_matches"] = key == "exact_matches"
                        all_matches.append(match_copy)
                    all_items.extend(items)
            counts["all"] = len(all_items)
            if not all_items:
                message = f"all: empty result payload: {json.dumps(data, ensure_ascii=False)}"
                print(f"COMPLETE API MESSAGE: {message}")
                zero_messages.append(message)
            continue
        items = data.get(result_type) or []
        counts[result_type] = len(items)
        if not items:
            message = f"{result_type}: empty result payload: {json.dumps(data, ensure_ascii=False)}"
            print(f"COMPLETE API MESSAGE: {message}")
            zero_messages.append(message)
        for match in items[:max_results]:
            match_copy = dict(match)
            match_copy["result_type"] = result_type
            match_copy["exact_matches"] = result_type == "exact_matches"
            matches.append(match_copy)
    known_keys = {
        (match.get("link"), match.get("image"), match.get("thumbnail"))
        for match in matches
    }
    for match in all_matches:
        key = (match.get("link"), match.get("image"), match.get("thumbnail"))
        if key not in known_keys:
            matches.append(match)
    return matches, counts, zero_messages, errors


def _secondary_queries(matches, profile_targets=(), limit=3):
    queries = []
    seen = set()
    for profile_url in profile_targets:
        parsed = urlparse(profile_url)
        path = parsed.path.strip("/")
        if path:
            query = f"site:instagram.com/{path}"
            queries.append(query)
            seen.add(query)
        if len(queries) >= limit:
            return queries
    ranked_matches = sorted(
        matches,
        key=lambda match: (
            "instagram.com" in str(match.get("link") or "").lower(),
            "@" in str(match.get("title") or ""),
        ),
        reverse=True,
    )
    for match in ranked_matches:
        parts = [
            match.get("title"),
            match.get("snippet"),
        ]
        text = " ".join(part.strip() for part in parts if isinstance(part, str) and part.strip())
        text = _clean_profile_search_phrase(text)
        if not text:
            continue
        query = f'site:instagram.com "{text}"'
        if query not in seen:
            seen.add(query)
            queries.append(query)
        if len(queries) >= limit:
            break
    return queries


def search_instagram_secondary(matches, max_results=20, profile_targets=()):
    """Search Instagram using text returned by Lens, never a hardcoded account."""
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        raise SearchConfigurationError("SERPAPI_KEY not found in .env")

    results = []
    errors = []
    for query in _secondary_queries(matches, profile_targets):
        try:
            response = requests.get(
                "https://serpapi.com/search.json",
                params={
                    "engine": "google",
                    "q": query,
                    "num": max_results,
                    "api_key": api_key,
                },
                timeout=15,
            )
        except requests.RequestException as error:
            errors.append(f"google secondary query {query!r}: {error}")
            print(f"SERPAPI ERROR: {errors[-1]}")
            continue
        data = _response_data(response)
        _print_api_status(response, data, "not applicable", f"secondary google: {query}")
        if response.status_code >= 400 or data.get("error"):
            errors.append(f"google secondary query {query!r}: {json.dumps(data, ensure_ascii=False)}")
            continue
        for item in (data.get("organic_results") or [])[:max_results]:
            item_copy = dict(item)
            item_copy["secondary_query"] = query
            item_copy["result_type"] = "secondary_google"
            results.append(item_copy)
    return results, errors


def _normalize_candidate(match):
    source_url = match.get("link") or match.get("page_url") or match.get("url")
    return {
        "url": source_url,
        "source_url": source_url,
        "page_url": match.get("page_url"),
        "title": match.get("title"),
        "caption": match.get("snippet") or match.get("title"),
        "snippet": match.get("snippet"),
        "platform": match.get("source") or urlparse(source_url or "").netloc or None,
        "image_url": match.get("image") or match.get("original"),
        "profile_image_url": match.get("profile_image_url"),
        "thumbnail_url": match.get("thumbnail"),
        "position": match.get("position"),
        "profile_url": None,
        "username": None,
        "result_type": match.get("result_type"),
        "secondary_query": match.get("secondary_query"),
        "full_name": match.get("full_name"),
        "source": match.get("source") or urlparse(source_url or "").netloc or None,
        "discovery_method": match.get("discovery_method") or "google_lens",
        "candidate_classification": "UNKNOWN" if not source_url else "OTHER_WEBSITE",
        "match_type": "post_or_page",
    }


def search_public_images(image_path, max_results=20):
    """Perform a genuine reverse-image search."""

    image_id = upload_image(image_path)

    matches, counts, zero_messages, errors = search_google_lens(
        image_id,
        max_results=max_results,
    )

    candidates = []

    for match in matches:
        candidates.append(_normalize_candidate(match))

    return SearchResults(
        candidates,
        image_id=image_id,
        exact_count=counts["exact_matches"],
        visual_count=counts["visual_matches"],
        all_count=counts["all"],
        secondary_count=0,
        profile_count=0,
        post_count=0,
        zero_messages=zero_messages,
        errors=errors,
    )


def print_search_diagnostics(candidates):
    """Print every normalized Lens result before download or face filtering."""
    exact_count = getattr(candidates, "exact_count", sum(item.get("result_type") == "exact_matches" for item in candidates))
    visual_count = getattr(candidates, "visual_count", sum(item.get("result_type") == "visual_matches" for item in candidates))
    instagram_candidates = [
        item for item in candidates
        if item.get("candidate_classification") in {"INSTAGRAM_POST", "INSTAGRAM_PROFILE"}
    ]
    print("\nRAW GOOGLE LENS CANDIDATES")
    for index, candidate in enumerate(candidates, start=1):
        source_url = candidate.get("source_url") or candidate.get("url")
        print(f"Raw candidate {index}")
        print(f"  Position: {candidate.get('position')}")
        print(f"  Title: {candidate.get('title')}")
        print(f"  Source/domain: {candidate.get('platform')}")
        print(f"  Source URL: {source_url}")
        print(f"  Image URL: {candidate.get('image_url')}")
        print(f"  Thumbnail URL: {candidate.get('thumbnail_url')}")
        print(f"  Instagram: {'YES' if 'instagram.com' in str(source_url or '').lower() else 'NO'}")
        print(f"  Result type: {candidate.get('result_type')}")
    print(f"Exact matches: {exact_count}")
    print(f"Visual matches: {visual_count}")
    print(f"Google Lens all results: {getattr(candidates, 'all_count', 0)}")
    print(f"Instagram candidates found: {len(instagram_candidates)}")
    for index, candidate in enumerate(instagram_candidates, start=1):
        print(f"Instagram candidate #{index}")
        print(f"URL: {candidate.get('profile_url') or candidate.get('source_url')}")
        print(f"Username: {candidate.get('username')}")
        print(f"Image URL: {candidate.get('image_url')}")
        print(f"Result type: {candidate.get('result_type')}")


def download_candidate_images(candidates, output_dir):
    """Download candidate images for later face matching."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    downloaded = []

    for index, candidate in enumerate(candidates, start=1):
        urls = [
            (candidate.get("profile_image_url"), "profile_image_url"),
            (candidate.get("image_url"), "image_url"),
            (candidate.get("thumbnail_url"), "thumbnail_url"),
        ]

        for image_url, image_kind in urls:
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

                content = response.content
                decoded = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
                content_type = response.headers.get("Content-Type", "")
                if decoded is None:
                    print(
                        f"Skipped candidate {index}: {image_kind} was not a valid full image "
                        f"(content-type={content_type or 'missing'})"
                    )
                    continue

                suffix = Path(urlparse(image_url).path).suffix or ".jpg"
                file_path = output_path / f"candidate_{index}{suffix}"

                file_path.write_bytes(content)

                candidate_copy = dict(candidate)
                candidate_copy["image_path"] = str(file_path)
                candidate_copy["downloaded_from"] = image_url
                candidate_copy["downloaded_image_kind"] = image_kind

                downloaded.append(candidate_copy)
                print(
                    f"Downloaded candidate {index} from {image_kind}: {image_url}"
                )

                break

            except requests.RequestException as error:
                print(f"Skipped candidate {index}: {image_kind} download failed ({error})")
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
