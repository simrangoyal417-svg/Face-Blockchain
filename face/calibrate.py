"""Evaluate SFace matching thresholds on a labeled local image dataset."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

from .encoder import FaceEncodingError, encode_faces
from .matcher import DEFAULT_COSINE_THRESHOLD, compare_faces, resolve_threshold


DEFAULT_THRESHOLDS = (0.30, 0.35, 0.36, 0.37, 0.38, 0.40, 0.42, 0.45, 0.50)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


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
    are treated as separate identities.
    """
    root = Path(data_dir)
    samples = []
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
    return {
        "threshold": threshold,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "accuracy": (true_positive + true_negative) / total if total else 0.0,
    }


def recommend_threshold(metrics: Iterable[dict[str, float | int]]) -> dict[str, float | int]:
    """Choose highest accuracy, then fewest false positives, then closest default."""
    available = list(metrics)
    if not available:
        raise ValueError("at least one threshold is required")
    return max(
        available,
        key=lambda result: (
            result["accuracy"],
            -result["false_positive"],
            -abs(float(result["threshold"]) - DEFAULT_COSINE_THRESHOLD),
        ),
    )


def print_report(
    pairs: Iterable[PairResult],
    metrics: Iterable[dict[str, float | int]],
    skipped: Iterable[tuple[Path, str]],
    report_threshold: float,
) -> dict[str, float | int]:
    """Print pair-level results and threshold evaluation tables."""
    pair_list = list(pairs)
    metric_list = list(metrics)
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
    print("threshold | TP | FP | TN | FN | accuracy")
    for result in metric_list:
        print(
            f"{result['threshold']:.2f} | {result['true_positive']} | {result['false_positive']} | "
            f"{result['true_negative']} | {result['false_negative']} | {result['accuracy']:.2%}"
        )
    recommendation = recommend_threshold(metric_list)
    print(
        f"\nRecommended threshold: {recommendation['threshold']:.2f} "
        f"(accuracy {recommendation['accuracy']:.2%}, "
        f"FP {recommendation['false_positive']}, FN {recommendation['false_negative']})"
    )
    return recommendation


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate SFace match thresholds on labeled images.")
    parser.add_argument("--data-dir", default="calibration_data")
    parser.add_argument("--report-threshold", type=float, default=None)
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    args = parser.parse_args()

    samples = discover_samples(args.data_dir)
    if len(samples) < 2:
        parser.error("add at least two images under calibration_data/same_person or different_people")
    embeddings, skipped = encode_samples(samples)
    pairs = compare_pairs(samples, embeddings)
    if not pairs:
        parser.error("no comparable image pairs were found")
    metrics = [threshold_metrics(pairs, threshold) for threshold in args.thresholds]
    report_threshold = resolve_threshold(args.report_threshold)
    print_report(pairs, metrics, skipped, report_threshold)


if __name__ == "__main__":
    main()