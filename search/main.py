"""Command-line interface for genuine public candidate discovery."""

import argparse
import json

from .web_search import download_candidate_images, save_candidates, search_public_images


def main() -> None:
    parser = argparse.ArgumentParser(description="Search public web images for candidate posts.")
    parser.add_argument("query", help="Live search query, for example a name and event")
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--output-dir", default="candidates")
    parser.add_argument("--output-json", default="candidates/results.json")
    arguments = parser.parse_args()

    results = search_public_images(arguments.query, arguments.max_results)
    candidates = download_candidate_images(results, arguments.output_dir)
    save_candidates(candidates, arguments.output_json)
    print(json.dumps(candidates, indent=2))


if __name__ == "__main__":
    main()