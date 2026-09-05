"""Run genuine image search, face matching, and local ledger verification."""

import argparse
import json
import sys
from pathlib import Path

from blockchain.ledger import BlockchainLedger
from face.detector import detect_faces
from face.encoder import encode_faces
from face.matcher import find_best_match, resolve_threshold
from search.web_search import download_candidate_images, search_public_images


def run_integration(
	input_image: str | Path,
	ledger_path: str | Path = "blockchain/ledger.json",
	candidates_dir: str | Path = "candidates/downloaded",
	threshold: float | None = None,
	max_results: int = 10,
) -> dict:
	"""Run the complete search, match, hash, store, and verify workflow."""
	input_path = Path(input_image)
	print(f"1. Input image loaded: {input_path}")

	faces = detect_faces(input_path)
	print(f"2. Face detected: {len(faces)} face(s)")
	embeddings = encode_faces(input_path)
	print(f"3. Face encoding generated: {len(embeddings)} embedding(s)")

	print("4. Web/social search started")
	search_results = search_public_images(input_path, max_results=max_results)
	print(f"5. Candidate post(s) discovered: {len(search_results)}")

	downloaded_candidates = download_candidate_images(search_results, candidates_dir)
	print(f"   Candidate image(s) downloaded: {len(downloaded_candidates)}")
	if not downloaded_candidates:
		raise RuntimeError("No candidate images could be downloaded from search results")

	match = find_best_match(input_path, downloaded_candidates, threshold)
	print(f"6. Candidate face compared: {len(match['candidates'])} candidate(s)")
	if not match["is_match"] or not match.get("best_match"):
		print("7. Matching post confirmed: NO")
		print("FINAL RESULT: VERIFICATION FAILED")
		return {"face_match": match, "blockchain_block": None, "blockchain_verification": None}

	winner = match["best_match"]
	winner_path = Path(winner["image_path"])
	display_threshold = match.get("threshold")
	if display_threshold is None:
		display_threshold = resolve_threshold()
	print(
		f"7. Matching post confirmed: {winner.get('url') or winner_path} "
		f"(score {match['score']:.2f}%, "
		f"threshold {match.get('threshold_percent', display_threshold * 100):.2f}%)"
	)

	ledger = BlockchainLedger(ledger_path)
	block = ledger.upload(winner_path, winner)
	print(f"8. Post/data hash generated: {block['artifact_sha256']}")
	print(f"9. Blockchain transaction/storage completed: block {block['index']}")

	stored_blocks = ledger._read()
	stored_block = stored_blocks[block["index"]]
	print(f"10. On-chain record retrieved: block {stored_block['index']}")
	print("11. Hash recalculated during verification")
	verification = ledger.verify(winner_path, stored_block)
	hashes_match = verification["verified"]
	print(f"12. Hashes compared: {'MATCH' if hashes_match else 'MISMATCH'}")
	print(f"13. Final result: {'VERIFIED' if hashes_match else 'VERIFICATION FAILED'}")

	return {
		"face_match": match,
		"blockchain_block": stored_block,
		"blockchain_verification": verification,
	}


def main() -> int:
	parser = argparse.ArgumentParser(
		description="Run genuine web search, face matching, and blockchain verification."
	)
	parser.add_argument("input_image", help="Local input face image")
	parser.add_argument("--ledger", default="blockchain/ledger.json")
	parser.add_argument("--candidates-dir", default="candidates/downloaded")
	parser.add_argument("--threshold", type=float, default=None)
	parser.add_argument("--max-results", type=int, default=10)
	args = parser.parse_args()

	try:
		result = run_integration(
			args.input_image,
			args.ledger,
			args.candidates_dir,
			args.threshold,
			args.max_results,
		)
	except Exception as error:
		print(f"INTEGRATION FAILED: {error}", file=sys.stderr)
		return 1
	print("\nFinal JSON result:")
	print(json.dumps(result, indent=2, default=str))
	return 0 if result["blockchain_verification"] and result["blockchain_verification"]["verified"] else 1


if __name__ == "__main__":
	raise SystemExit(main())
