"""End-to-end adapter connecting candidate search, face matching, and ledger verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from blockchain.ledger import BlockchainLedger
from face.matcher import find_best_match

from .models import PipelineResult, candidate_image_path


def run_pipeline(
	input_image: str | Path,
	candidates: Iterable[str | Path | dict[str, Any]],
	ledger_path: str | Path = "blockchain/ledger.json",
	threshold: float | None = None,
) -> PipelineResult:
	"""Match candidates and record/verify the winning candidate when matched.

	``candidates`` is intentionally provider-agnostic. Vedant can supply records
	from any genuine search implementation as long as each record has an
	``image_path`` and may include URL/caption/platform metadata.
	"""
	candidate_list = list(candidates)
	match = find_best_match(input_image, candidate_list, threshold)
	if not match["is_match"]:
		return PipelineResult(match, None, None)

	winner = match["best_match"]
	winner_path = candidate_image_path(winner)
	post_metadata = winner if isinstance(winner, dict) else {"image_path": str(winner)}
	ledger = BlockchainLedger(ledger_path)
	block = ledger.upload(winner_path, post_metadata)
	verification = ledger.verify(winner_path, block)
	return PipelineResult(match, block, verification)


def main() -> None:
	parser = argparse.ArgumentParser(description="Run face matching and blockchain verification.")
	parser.add_argument("input_image")
	parser.add_argument("candidates_json", help="JSON file containing candidate records")
	parser.add_argument("--ledger", default="blockchain/ledger.json")
	parser.add_argument("--threshold", type=float, default=None)
	args = parser.parse_args()
	candidates = json.loads(Path(args.candidates_json).read_text(encoding="utf-8"))
	if not isinstance(candidates, list):
		parser.error("candidates JSON must contain a list")
	result = run_pipeline(args.input_image, candidates, args.ledger, args.threshold)
	print(json.dumps(result.as_dict(), indent=2, default=str))


if __name__ == "__main__":
	main()
