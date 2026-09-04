import os

import pytest
from dotenv import load_dotenv
from google import genai


def test_gemini_connection() -> None:
    """Optionally check Gemini credentials without breaking offline test runs."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY is not configured")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents="Reply with exactly: Gemini connection successful",
    )
    assert response.text
