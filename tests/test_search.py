from unittest.mock import Mock, patch

import pytest

from search.web_search import SearchConfigurationError, download_candidate_images, search_public_images


def test_search_requires_credentials(monkeypatch) -> None:
    monkeypatch.delenv("GOOGLE_CSE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CSE_ID", raising=False)
    with pytest.raises(SearchConfigurationError):
        search_public_images("HH Goa 2026")


@patch("search.web_search.requests.get")
def test_search_returns_live_api_metadata(mock_get, monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_CSE_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_CSE_ID", "test-id")
    response = Mock()
    response.json.return_value = {
        "items": [{
            "link": "https://cdn.example/post.jpg",
            "title": "Public post",
            "snippet": "Caption",
            "image": {"contextLink": "https://example.test/post/1"},
        }]
    }
    mock_get.return_value = response
    result = search_public_images("public person", 1)
    assert result[0]["url"] == "https://example.test/post/1"
    assert result[0]["image_url"] == "https://cdn.example/post.jpg"


@patch("search.web_search.requests.get")
def test_downloads_only_images(mock_get, tmp_path) -> None:
    response = Mock()
    response.headers = {"content-type": "image/jpeg"}
    response.iter_content.return_value = [b"image-bytes"]
    mock_get.return_value = response
    result = download_candidate_images(
        [{"url": "https://example.test/post/1", "image_url": "https://cdn.test/a.jpg"}], tmp_path
    )
    assert "image_path" in result[0]
    assert (tmp_path / result[0]["image_path"]).exists()