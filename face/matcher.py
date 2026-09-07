"""Face embedding comparison and candidate ranking."""

import os
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import numpy as np
from dotenv import load_dotenv

from .encoder import FaceEncodingError, encode_faces


load_dotenv()

DEFAULT_COSINE_THRESHOLD = 0.363
MATCH_THRESHOLD_ENV = "FACE_MATCH_THRESHOLD"
HIGH_CONFIDENCE_MARGIN = 0.02


def classify_confidence(similarity: float, threshold: float) -> str:
    """Classify a score without changing the configured match threshold."""
    if similarity < threshold:
        return "REJECT"
    if similarity >= threshold + HIGH_CONFIDENCE_MARGIN:
        return "HIGH CONFIDENCE"
    return "BORDERLINE"


def _candidate_relevance(candidate: Any) -> int:
    """Return a small metadata tie-breaker after face similarity is accepted."""
    if not isinstance(candidate, dict):
        return 0
    score = 0
    if candidate.get("platform"):
        score += 1
    if candidate.get("url") or candidate.get("source_url"):
        score += 2
        source = candidate.get("url") or candidate.get("source_url")
        if "instagram.com" in urlparse(source).netloc.lower():
            score += 2
    if candidate.get("title") or candidate.get("caption"):
        score += 1
    if candidate.get("position") is not None:
        score += max(0, 20 - int(candidate["position"]))
    return score


def resolve_threshold(threshold: float | None = None) -> float:
    """Resolve and validate the match threshold from an argument or environment."""
    value = threshold
    if value is None:
        configured = os.getenv(MATCH_THRESHOLD_ENV)
        value = DEFAULT_COSINE_THRESHOLD if configured is None else float(configured)
    if not 0 <= value <= 1:
        raise ValueError("threshold must be between 0 and 1")
    return value


def compare_faces(
    face1: np.ndarray,
    face2: np.ndarray,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Compare two SFace embeddings using cosine similarity."""
    threshold = resolve_threshold(threshold)
    first_norm = np.linalg.norm(face1)
    second_norm = np.linalg.norm(face2)
    if first_norm == 0 or second_norm == 0:
        raise ValueError("face embeddings must not be zero vectors")
    similarity = float(np.dot(face1.ravel(), face2.ravel()) / (first_norm * second_norm))
    return {
        "similarity": similarity,
        "score_percent": round(max(0.0, similarity) * 100, 2),
        "threshold": threshold,
        "threshold_percent": round(threshold * 100, 2),
        "is_match": similarity >= threshold,
        "decision": "same_person" if similarity >= threshold else "different_person",
		"confidence": classify_confidence(similarity, threshold),
    }


def find_best_match(
    input_image: str | Path,
    candidate_images: Iterable[str | Path | dict[str, Any]],
    threshold: float | None = None,
) -> dict[str, Any]:
    """Rank candidate images and return the strongest match with metadata."""
    threshold = resolve_threshold(threshold)
    input_embedding = encode_faces(input_image)[0]
    results = []
    for candidate in candidate_images:
        path = Path(candidate["image_path"] if isinstance(candidate, dict) else candidate)
        try:
            candidate_embeddings = encode_faces(path)
            comparisons = [compare_faces(input_embedding, embedding, threshold) for embedding in candidate_embeddings]
            best_index, best = max(
                enumerate(comparisons), key=lambda item: item[1]["similarity"]
            )
            results.append(
                {
                    "candidate": candidate,
                    **best,
                    "face_detected": True,
                    "face_count": len(candidate_embeddings),
                    "face_similarities": [item["similarity"] for item in comparisons],
                    "selected_face_index": best_index,
					"confidence": classify_confidence(best["similarity"], threshold),
                }
            )
        except FaceEncodingError as error:
            message = str(error)
            decision = "no_face_detected" if "No face detected" in message else "error"
            results.append({"candidate": candidate, "error": message, "decision": decision})
    valid_results = [result for result in results if "similarity" in result]
    if not valid_results:
        return {
            "best_match": None,
            "best_candidate": None,
            "score": None,
            "is_match": False,
            "decision": "no_face_detected",
            "threshold": threshold,
            "candidates": results,
        }
    matching_results = [result for result in valid_results if result["is_match"]]
    if not matching_results:
        best_candidate = max(valid_results, key=lambda result: result["similarity"])
        return {
            "best_match": None,
            "best_candidate": best_candidate["candidate"],
            "score": best_candidate["score_percent"],
            "similarity": best_candidate["similarity"],
            "threshold_percent": best_candidate["threshold_percent"],
            "is_match": False,
            "decision": "no_reliable_match",
            "threshold": threshold,
            "candidates": results,
        }
    best = max(
        matching_results,
        key=lambda result: (result["similarity"], _candidate_relevance(result["candidate"])),
    )
    is_match = True
    return {
        "best_match": best["candidate"] if is_match else None,
        "best_candidate": best["candidate"],
        "score": best["score_percent"],
        "similarity": best["similarity"],
        "threshold_percent": best["threshold_percent"],
        "is_match": is_match,
        "decision": "same_person" if is_match else "different_person",
        "threshold": threshold,
        "candidates": results,
    }