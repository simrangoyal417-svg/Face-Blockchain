"""Command-line entry point for local face identification."""

import argparse
import json

from face.matcher import find_best_match


def main() -> None:
	parser = argparse.ArgumentParser(description="Find the best matching candidate face.")
	parser.add_argument("input_image", help="Input face image")
	parser.add_argument("candidate_images", nargs="+", help="Candidate image paths")
	parser.add_argument("--threshold", type=float, default=0.363)
	arguments = parser.parse_args()

	result = find_best_match(
		arguments.input_image, arguments.candidate_images, arguments.threshold
	)
	print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
	main()
