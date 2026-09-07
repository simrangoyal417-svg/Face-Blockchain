"""Run genuine image search, face matching, and local ledger verification."""

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

from blockchain.ledger import BlockchainLedger
from face.detector import detect_faces
from face.encoder import encode_faces
from face.matcher import classify_confidence, find_best_match, resolve_threshold
from search.web_search import (
	download_candidate_images,
	print_search_diagnostics,
	SearchFailure,
	search_public_images,
)


def _source_url(candidate: dict) -> str | None:
	value = candidate.get("source_url") or candidate.get("url")
	return value if isinstance(value, str) and urlparse(value).scheme in {"http", "https"} else None


def _candidate_group(candidate: dict) -> str | None:
	return candidate.get("profile_url") or _source_url(candidate)


def _candidate_key(candidate: dict) -> tuple:
	return (
		candidate.get("source_url") or candidate.get("url"),
		candidate.get("image_url"),
		candidate.get("thumbnail_url"),
		candidate.get("position"),
	)


def _print_instagram_candidate_results(search_results, downloaded_candidates, match):
	"""Print download and face-verification outcomes for every Instagram result."""
	downloaded_by_key = {
		_candidate_key(candidate): candidate for candidate in downloaded_candidates
	}
	matched_by_key = {
		_candidate_key(result["candidate"]): result
		for result in match.get("candidates", [])
		if isinstance(result.get("candidate"), dict)
	}
	instagram_candidates = [
		candidate for candidate in search_results
		if candidate.get("candidate_classification") in {"INSTAGRAM_POST", "INSTAGRAM_PROFILE"}
	]
	print("\nINSTAGRAM CANDIDATE VERIFICATION DETAILS")
	for index, candidate in enumerate(instagram_candidates, start=1):
		downloaded = downloaded_by_key.get(_candidate_key(candidate))
		result = matched_by_key.get(_candidate_key(downloaded or candidate))
		print(f"Instagram candidate #{index}")
		print(f"  Title: {candidate.get('title')}")
		print(f"  Instagram/profile/post URL: {candidate.get('profile_url') or candidate.get('source_url')}")
		print(f"  Username: {candidate.get('username')}")
		print(f"  Image URL: {candidate.get('image_url')}")
		print(f"  Thumbnail URL: {candidate.get('thumbnail_url')}")
		if downloaded:
			print("  Downloaded: YES")
			print(f"  Downloaded from: {downloaded.get('downloaded_from')}")
			print(f"  Download type: {downloaded.get('downloaded_image_kind')}")
		else:
			print("  Downloaded: NO")
		if result and "similarity" in result:
			print(f"  Face detected: YES ({result.get('face_count', 0)} face(s))")
			print(f"  Face similarities: {result.get('face_similarities')}")
			print(f"  Face similarity score: {result['similarity']:.4f}")
			print(f"  Threshold: {result['threshold']:.4f}")
			print(f"  Final decision: {'PASS' if result.get('is_match') else 'REJECT'}")
		elif result:
			print("  Face detected: NO")
			print("  Face similarity score: unavailable")
			print(f"  Final decision: REJECT ({result.get('decision', 'encoding failed')})")
		else:
			print("  Face detected: NO")
			print("  Face similarity score: unavailable")
			print("  Final decision: REJECT (not downloaded)")


def _apply_high_confidence_gate(match: dict) -> dict:
	"""Require the existing face threshold plus a genuine source URL."""
	results = match.get("candidates", [])
	def is_allowed_source(candidate):
		classification = candidate.get("candidate_classification")
		if classification in {"INSTAGRAM_POST", "INSTAGRAM_PROFILE"}:
			return True
		if (
			candidate.get("match_type") == "instagram_profile_picture"
			and candidate.get("profile_url")
		):
			return True
		if classification == "INSTAGRAM_POST":
			return True
		if classification == "INSTAGRAM_PROFILE":
			return True
		if classification == "OTHER_WEBSITE":
			return True
		return "instagram.com" not in str(_source_url(candidate) or "").lower()

	passing = [
		result for result in results
		if result.get("is_match")
		and result.get(
			"confidence",
			classify_confidence(
				result.get("similarity", -1.0),
				result.get("threshold", resolve_threshold()),
			),
		) == "HIGH CONFIDENCE"
		and isinstance(result.get("candidate"), dict)
		and _source_url(result["candidate"])
		and is_allowed_source(result["candidate"])
	]
	group_counts = {}
	for result in passing:
		group = _candidate_group(result["candidate"])
		if group:
			group_counts[group] = group_counts.get(group, 0) + 1

	for result in passing:
		candidate = result["candidate"]
		group = _candidate_group(candidate)
		result["group_count"] = group_counts.get(group, 0)
		result["confidence"] = "HIGH CONFIDENCE MATCH"

	high_confidence = [result for result in passing if result["confidence"] == "HIGH CONFIDENCE MATCH"]
	if not high_confidence:
		updated = dict(match)
		updated["best_match"] = None
		updated["is_match"] = False
		updated["decision"] = "no_reliable_match"
		updated["confidence"] = "NO RELIABLE MATCH FOUND"
		return updated

	winner = max(high_confidence, key=lambda result: result["similarity"])
	updated = dict(match)
	updated["best_match"] = winner["candidate"]
	updated["best_candidate"] = winner["candidate"]
	updated["score"] = winner["score_percent"]
	updated["similarity"] = winner["similarity"]
	updated["is_match"] = True
	updated["decision"] = "same_person"
	updated["confidence"] = "HIGH CONFIDENCE MATCH"
	return updated


def run_integration(
	input_image: str | Path,
	ledger_path: str | Path = "blockchain/ledger.json",
	candidates_dir: str | Path = "candidates/downloaded",
	threshold: float | None = None,
	max_results: int = 50,
) -> dict:
	"""Run the complete search, match, hash, store, and verify workflow."""
	input_path = Path(input_image)
	print(f"1. Input image loaded: {input_path}")

	faces = detect_faces(input_path)
	print(f"2. Face detected: {len(faces)} face(s)")
	embeddings = encode_faces(input_path)
	print(f"3. Face encoding generated: {len(embeddings)} embedding(s)")

	print("4. Web/social search started")
	try:
		search_results = search_public_images(input_path, max_results=max_results)
	except SearchFailure as error:
		print(f"SEARCH FAILED: {error}")
		print(f"IMAGE UPLOAD: {'FAILED' if error.phase == 'image upload' else 'SUCCESS'}")
		print(f"IMAGE ID: {error.image_id or 'not available'}")
		print("GOOGLE LENS ALL: 0")
		print("EXACT MATCHES: 0")
		print("VISUAL MATCHES: 0")
		print("INSTAGRAM CANDIDATES: 0")
		print("DOWNLOADED CANDIDATES: 0")
		print("CANDIDATES WITH FACES: 0")
		print("FACE MATCHES PASSING: 0")
		print("FINAL RESULT: SEARCH FAILED")
		return {
			"face_match": {
				"best_match": None,
				"is_match": False,
				"decision": "search_failed",
				"confidence": "SEARCH FAILED",
				"threshold": resolve_threshold(threshold),
				"error": str(error),
				"candidates": [],
			},
			"blockchain_block": None,
			"blockchain_verification": None,
		}
	print(f"5. Candidate post(s) discovered: {len(search_results)}")
	exact_count = getattr(search_results, "exact_count", sum(item.get("result_type") == "exact_matches" for item in search_results))
	visual_count = getattr(search_results, "visual_count", sum(item.get("result_type") == "visual_matches" for item in search_results))
	all_count = getattr(search_results, "all_count", 0)
	secondary_count = getattr(search_results, "secondary_count", 0)
	profile_count = getattr(search_results, "profile_count", 0)
	post_count = getattr(search_results, "post_count", 0)
	instagram_count = sum(item.get("candidate_classification") in {"INSTAGRAM_POST", "INSTAGRAM_PROFILE"} for item in search_results)
	print(f"RESULT COUNTS: exact_matches = {exact_count}, visual_matches = {visual_count}")
	print(f"GOOGLE LENS ALL: {all_count}")
	print("SEARCH METHOD: SerpApi Google Lens reverse-image search")
	if getattr(search_results, "errors", []):
		print("SEARCH FAILED: one or more Lens requests failed; returned results retained for diagnostics")
		for error in search_results.errors:
			print(f"  {error}")
	elif not search_results and all_count == 0:
		print("SEARCH RETURNED ZERO RESULTS")
	elif instagram_count == 0:
		print("SEARCH RETURNED RESULTS BUT NO INSTAGRAM MATCH")
	else:
		print("SEARCH RETURNED INSTAGRAM RESULTS")
	print_search_diagnostics(search_results)

	downloaded_candidates = download_candidate_images(search_results, candidates_dir)
	print(f"   Candidate image(s) downloaded: {len(downloaded_candidates)}")
	if not downloaded_candidates:
		print("FACE DETECTED: YES")
		print(f"GOOGLE LENS RESULTS: {all_count}")
		print(f"\nEXACT MATCHES: {exact_count}")
		print(f"VISUAL MATCHES: {visual_count}")
		print(f"INSTAGRAM CANDIDATES: {instagram_count}")
		print("DOWNLOADED CANDIDATES: 0")
		print("CANDIDATES WITH FACES: 0")
		print("FACE MATCHES PASSING: 0")
		print("VERIFIED SOURCE:")
		print("BEST VALID MATCH: NONE")
		print("FINAL RESULT: NO RELIABLE MATCH FOUND")
		print("BLOCKCHAIN: NOT CREATED")
		print("BLOCKCHAIN VERIFICATION: FALSE")
		print("IMAGE UPLOAD: SUCCESS")
		print(f"IMAGE ID: {getattr(search_results, 'image_id', 'unknown')}")
		print(f"GOOGLE LENS ALL: {all_count}")
		print("DOWNLOADED CANDIDATES: 0")
		print("CANDIDATES WITH FACES: 0")
		print("FACE MATCHES PASSING: 0")
		print("FINAL RESULT: NO RELIABLE MATCH FOUND" if not getattr(search_results, "errors", []) else "FINAL RESULT: SEARCH FAILED")
		print("7. Matching post confirmed: NO RELIABLE MATCH FOUND")
		print("BEST VALID MATCH: NONE")
		return {
			"face_match": {
				"best_match": None,
				"is_match": False,
				"decision": "no_reliable_match",
				"confidence": "NO RELIABLE MATCH FOUND",
				"threshold": resolve_threshold(threshold),
				"candidates": [],
			},
			"blockchain_block": None,
			"blockchain_verification": None,
		}

	match = find_best_match(input_path, downloaded_candidates, threshold)
	match = _apply_high_confidence_gate(match)
	print(f"6. Candidate face compared: {len(match['candidates'])} candidate(s)")
	print(f"Candidates with faces: {sum('similarity' in item for item in match['candidates'])}")
	print(f"Candidates passing face verification: {sum(item.get('is_match', False) for item in match['candidates'])}")
	for index, candidate_result in enumerate(match["candidates"], start=1):
		candidate = candidate_result["candidate"]
		print(f"Candidate {index}")
		print(f"Platform: {candidate.get('platform') if isinstance(candidate, dict) else 'unknown'}")
		print(f"Source URL: {candidate.get('url') or candidate.get('source_url') if isinstance(candidate, dict) else candidate}")
		print(f"Image URL: {candidate.get('image_url') if isinstance(candidate, dict) else 'unknown'}")
		if "similarity" in candidate_result:
			print(f"Similarity: {candidate_result['similarity']:.4f}")
			print(f"Threshold: {candidate_result['threshold']:.4f}")
			print(f"Face detected: {candidate_result.get('face_count', 0) > 0}")
			print(f"Decision: {'PASS' if candidate_result['is_match'] else 'REJECT'}")
			if candidate_result.get("confidence"):
				print(f"Confidence: {candidate_result['confidence']}")
		else:
			print("Face detected: False")
			print(f"Similarity: unavailable ({candidate_result.get('error', 'encoding failed')})")
			print(f"Threshold: {match['threshold']:.4f}")
			print("Decision: REJECT")
	print(f"\nEXACT MATCHES: {exact_count}")
	print(f"VISUAL MATCHES: {visual_count}")
	print(f"INSTAGRAM CANDIDATES: {instagram_count}")
	print(f"DOWNLOADED CANDIDATES: {len(downloaded_candidates)}")
	print(f"CANDIDATES WITH FACES: {sum('similarity' in item for item in match['candidates'])}")
	print(f"FACE MATCHES PASSING: {sum(item.get('is_match', False) for item in match['candidates'])}")
	print(f"FACE DETECTED: {'YES' if faces else 'NO'}")
	print(f"GOOGLE LENS RESULTS: {all_count}")
	if not match["is_match"] or not match.get("best_match"):
		print("BEST VALID MATCH: NONE")
		print("SIMILARITY: unavailable")
		no_match_threshold = match.get("threshold")
		if no_match_threshold is None:
			no_match_threshold = resolve_threshold(threshold)
		print(f"THRESHOLD: {no_match_threshold:.4f}")
		print("7. Matching post confirmed: NO RELIABLE MATCH FOUND")
		if (
			isinstance(match.get("best_candidate"), dict)
			and match["best_candidate"].get("candidate_classification") is None
		) or any(
			isinstance(item.get("candidate"), dict)
			and item["candidate"].get("candidate_classification") is None
			for item in match.get("candidates", [])
		):
			print("FINAL RESULT: VERIFICATION FAILED")
		print("FINAL RESULT: NO RELIABLE MATCH FOUND")
		print("BLOCKCHAIN: NOT CREATED")
		print("BLOCKCHAIN VERIFICATION: FALSE")
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
	print(
		f"BEST VALID MATCH: {winner.get('profile_url') or winner.get('url') or winner_path} "
		f"(similarity {match['similarity']:.4f})"
	)
	print(f"SIMILARITY: {match['similarity']:.4f}")
	effective_threshold = match.get("threshold")
	if effective_threshold is None:
		effective_threshold = display_threshold
	print(f"THRESHOLD: {effective_threshold:.4f}")
	print("VERIFIED SOURCE:")
	print(f"TITLE: {winner.get('title') or winner.get('caption') or 'N/A'}")
	print(f"PLATFORM: {winner.get('platform') or winner.get('source') or 'N/A'}")
	print(f"SOURCE URL: {winner.get('source_url') or winner.get('url') or 'N/A'}")
	print(f"IMAGE URL: {winner.get('image_url') or winner.get('profile_image_url') or 'N/A'}")
	if winner.get("username"):
		print(f"Username: {winner['username']}")
	if winner.get("profile_url"):
		print(f"Profile URL: {winner['profile_url']}")
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
	print(f"FINAL RESULT: {'VERIFIED' if hashes_match else 'VERIFICATION FAILED'}")
	print("BLOCKCHAIN: CREATED")
	print(f"BLOCKCHAIN VERIFICATION: {str(hashes_match).upper()}")

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
	parser.add_argument("--max-results", type=int, default=50)
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
