"""Evaluate SFace matching thresholds on a labeled local image dataset."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import re
from typing import Iterable

from .encoder import FaceEncodingError, encode_faces
from .matcher import DEFAULT_COSINE_THRESHOLD, compare_faces, resolve_threshold


DEFAULT_THRESHOLDS = tuple(round(0.30 + step * 0.01, 2) for step in range(31))
IMAGE_EXTENSIONS = {".avif", ".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ImageSample:
    path: Path
    identity: str


@dataclass(frozen=True)
class PairResult:
    image1: Path
    image2: Path
    similarity: float
    expected: str


def discover_samples(data_dir: str | Path) -> list[ImageSample]:
    """Discover labeled images under same_person and different_people.

    Each identity is represented by a directory. Files placed directly in
    same_person are treated as one identity; direct files in different_people
    are treated as separate identities. A flat directory is also accepted and
    treated as one same-person identity for quick consistency checks.
    """
    root = Path(data_dir)
    samples = []
    if root.is_dir() and not any((root / category).is_dir() for category in ("same_person", "different_people")):
        for path in sorted(root.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append(ImageSample(path, "same_person/default"))
        return samples

    for category in ("same_person", "different_people"):
        category_dir = root / category
        if not category_dir.is_dir():
            continue
        files = sorted(
            path
            for path in category_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for path in files:
            relative_parent = path.parent.relative_to(category_dir)
            if relative_parent == Path("."):
                identity = f"{category}/{path.stem}" if category == "different_people" else f"{category}/default"
            else:
                identity = f"{category}/{relative_parent.as_posix()}"
            samples.append(ImageSample(path, identity))
    return samples


def encode_samples(samples: Iterable[ImageSample]) -> tuple[dict[ImageSample, object], list[tuple[Path, str]]]:
    """Encode one face per sample and return skipped images with reasons."""
    embeddings = {}
    skipped = []
    for sample in samples:
        try:
            faces = encode_faces(sample.path)
            if len(faces) != 1:
                skipped.append((sample.path, f"expected 1 face, found {len(faces)}"))
                continue
            embeddings[sample] = faces[0]
        except FaceEncodingError as error:
            skipped.append((sample.path, str(error)))
    return embeddings, skipped


def compare_pairs(samples: Iterable[ImageSample], embeddings: dict[ImageSample, object]) -> list[PairResult]:
    """Compare every pair of successfully encoded samples."""
    pairs = []
    available = sorted(embeddings, key=lambda sample: str(sample.path))
    for first, second in combinations(available, 2):
        result = compare_faces(embeddings[first], embeddings[second], threshold=0.0)
        expected = "same_person" if first.identity == second.identity else "different_person"
        pairs.append(PairResult(first.path, second.path, result["similarity"], expected))
    return pairs


def threshold_metrics(pairs: Iterable[PairResult], threshold: float) -> dict[str, float | int]:
    """Calculate confusion-matrix counts and accuracy for one threshold."""
    threshold = resolve_threshold(threshold)
    true_positive = false_positive = true_negative = false_negative = 0
    for pair in pairs:
        predicted_same = pair.similarity >= threshold
        expected_same = pair.expected == "same_person"
        if expected_same and predicted_same:
            true_positive += 1
        elif expected_same:
            false_negative += 1
        elif predicted_same:
            false_positive += 1
        else:
            true_negative += 1
    total = true_positive + false_positive + true_negative + false_negative
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": threshold,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "accuracy": (true_positive + true_negative) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


def recommend_threshold(
    metrics: Iterable[dict[str, float | int]],
    target_accuracy: float | None = None,
) -> dict[str, float | int]:
    """Choose highest accuracy, then fewest false positives, then closest default."""
    available = list(metrics)
    if not available:
        raise ValueError("at least one threshold is required")
    if target_accuracy is not None:
        qualifying = [m for m in available if float(m["accuracy"]) >= target_accuracy]
        if qualifying:
            available = qualifying
    return max(
        available,
        key=lambda result: (
            result["accuracy"],
            -result["false_positive"],
            -abs(float(result["threshold"]) - DEFAULT_COSINE_THRESHOLD),
        ),
    )


def distribution_summary(pairs: Iterable[PairResult]) -> dict[str, float | int | bool | None]:
    """Summarize same/different score distributions and their overlap."""
    same_scores = [pair.similarity for pair in pairs if pair.expected == "same_person"]
    different_scores = [pair.similarity for pair in pairs if pair.expected == "different_person"]
    if not same_scores or not different_scores:
        return {
            "same_count": len(same_scores),
            "different_count": len(different_scores),
            "same_min": min(same_scores) if same_scores else None,
            "same_max": max(same_scores) if same_scores else None,
            "same_average": sum(same_scores) / len(same_scores) if same_scores else None,
            "different_min": min(different_scores) if different_scores else None,
            "different_max": max(different_scores) if different_scores else None,
            "different_average": sum(different_scores) / len(different_scores) if different_scores else None,
            "overlap": False,
        }
    same_min = min(same_scores)
    different_max = max(different_scores)
    return {
        "same_count": len(same_scores),
        "different_count": len(different_scores),
        "same_min": same_min,
        "same_max": max(same_scores),
        "same_average": sum(same_scores) / len(same_scores),
        "different_min": min(different_scores),
        "different_max": different_max,
        "different_average": sum(different_scores) / len(different_scores),
        "overlap": different_max >= same_min,
    }


def recommended_operating_threshold(summary: dict[str, float | int | bool | None]) -> float | None:
    """Return a conservative separating threshold, or None when distributions overlap."""
    if summary["same_min"] is None or summary["different_max"] is None:
        return None
    if summary["different_max"] >= summary["same_min"]:
        return None
    return (float(summary["different_max"]) + float(summary["same_min"])) / 2


def update_env_threshold(threshold: float, env_path: str | Path = ".env") -> None:
    path = Path(env_path)
    content = path.read_text(encoding="utf-8") if path.is_file() else ""
    line = f'FACE_MATCH_THRESHOLD="{threshold:.2f}"'
    if "FACE_MATCH_THRESHOLD=" in content:
        new_content = re.sub(r'FACE_MATCH_THRESHOLD=.*', line, content)
    else:
        new_content = content + ("\n" if content and not content.endswith("\n") else "") + line + "\n"
    path.write_text(new_content, encoding="utf-8")
    print(f"Updated {path} with FACE_MATCH_THRESHOLD={threshold:.2f}")


def print_report(
    pairs: Iterable[PairResult],
    metrics: Iterable[dict[str, float | int]],
    skipped: Iterable[tuple[Path, str]],
    report_threshold: float,
    target_accuracy: float | None = None,
) -> dict[str, float | int]:
    """Print pair-level results and threshold evaluation tables."""
    pair_list = list(pairs)
    metric_list = list(metrics)
    summary = distribution_summary(pair_list)
    print("Calibration data summary")
    print(f"Same-person pairs: {summary['same_count']}")
    print(f"Different-person pairs: {summary['different_count']}")
    if summary["same_count"] == 0 or summary["different_count"] == 0:
        print("CALIBRATION INSUFFICIENT: both same-person and different-person pairs are required.")
    else:
        print(f"Same-person minimum: {summary['same_min']:.4f}")
        print(f"Same-person average: {summary['same_average']:.4f}")
        print(f"Different-person maximum: {summary['different_max']:.4f}")
        print(f"Different-person average: {summary['different_average']:.4f}")
        print(f"Distribution overlap: {'YES' if summary['overlap'] else 'NO'}")
        operating = recommended_operating_threshold(summary)
        print(f"Recommended separating threshold: {operating:.4f}" if operating is not None else "Recommended separating threshold: NONE")
    print(f"Pairwise results at threshold {report_threshold:.3f}")
    print("image 1 | image 2 | similarity | expected | predicted | result")
    for pair in pair_list:
        predicted = "same_person" if pair.similarity >= report_threshold else "different_person"
        result = "correct" if predicted == pair.expected else "incorrect"
        print(
            f"{pair.image1} | {pair.image2} | {pair.similarity:.4f} ({pair.similarity * 100:.2f}%) | "
            f"{pair.expected} | {predicted} | {result}"
        )
    for path, reason in skipped:
        print(f"SKIPPED | {path} | {reason}")

    print("\nThreshold metrics")
    print("threshold | TP | FP | TN | FN | accuracy | precision | recall | f1")
    for result in metric_list:
        print(
            f"{result['threshold']:.2f} | {result['true_positive']} | {result['false_positive']} | "
            f"{result['true_negative']} | {result['false_negative']} | {result['accuracy']:.2%} | "
            f"{result.get('precision', 0.0):.2%} | {result.get('recall', 0.0):.2%} | {result.get('f1_score', 0.0):.2%}"
        )
    if summary["same_count"] == 0 or summary["different_count"] == 0:
        print("\nRecommended threshold: unavailable until both classes have labeled pairs")
        return {}

    recommendation = recommend_threshold(metric_list, target_accuracy=target_accuracy)
    target_str = f" (target >= {target_accuracy:.1%})" if target_accuracy is not None else ""
    print(
        f"\nRecommended threshold{target_str}: {recommendation['threshold']:.2f} "
        f"(accuracy {recommendation['accuracy']:.2%}, "
        f"FP {recommendation['false_positive']}, FN {recommendation['false_negative']})"
    )
    return recommendation


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate SFace match thresholds on labeled images.")
    parser.add_argument("--data-dir", default="calibration_data")
    parser.add_argument("--report-threshold", type=float, default=None)
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--target-accuracy", type=float, default=None, help="Target accuracy (e.g. 0.90 for 90%)")
    parser.add_argument("--update-env", action="store_true", help="Update FACE_MATCH_THRESHOLD in .env with recommendation")
    args = parser.parse_args()

    samples = discover_samples(args.data_dir)
    if len(samples) < 2:
        print("Calibration data summary")
        print(f"Images discovered: {len(samples)}")
        print("CALIBRATION INSUFFICIENT: add labeled images under calibration_data/same_person and calibration_data/different_people.")
        return
    embeddings, skipped = encode_samples(samples)
    pairs = compare_pairs(samples, embeddings)
    if not pairs:
        print("Calibration data summary")
        print("CALIBRATION INSUFFICIENT: no comparable image pairs were found.")
        return
    metrics = [threshold_metrics(pairs, threshold) for threshold in args.thresholds]
    report_threshold = resolve_threshold(args.report_threshold)
    recommendation = print_report(pairs, metrics, skipped, report_threshold, target_accuracy=args.target_accuracy)
    if args.update_env and recommendation and "threshold" in recommendation:
        update_env_threshold(float(recommendation["threshold"]))


if __name__ == "__main__":
    main()