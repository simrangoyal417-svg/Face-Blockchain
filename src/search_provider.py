import os
from dotenv import load_dotenv
from google import genai


load_dotenv()


class GeminiSearchProvider:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env")

        self.client = genai.Client(api_key=api_key)

    def analyze_search_results(self, search_text):
        prompt = f"""
You are helping analyze public web search results for a project.

Review the search result information below.

Identify candidate public pages or posts that appear relevant.

Return JSON with this structure:

{{
    "candidates": [
        {{
            "url": "URL if available",
            "caption": "caption or text if available",
            "platform": "website or platform if available",
            "reason": "why this result appears relevant"
        }}
    ]
}}

Do not invent URLs.
Only return URLs that actually appear in the supplied search results.

Search results:
{search_text}
"""

        response = self.client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        return response.text
