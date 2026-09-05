from pathlib import Path

from face.calibrate import PairResult, discover_samples, recommend_threshold, threshold_metrics


def test_discover_samples_uses_identity_directories(tmp_path) -> None:
    same_dir = tmp_path / "same_person" / "person_a"
    different_dir = tmp_path / "different_people" / "person_b"
    same_dir.mkdir(parents=True)
    different_dir.mkdir(parents=True)
    (same_dir / "one.jpg").write_bytes(b"")
    (same_dir / "two.jpg").write_bytes(b"")
    (different_dir / "one.jpg").write_bytes(b"")

    samples = discover_samples(tmp_path)

    assert len(samples) == 3
    assert {sample.identity for sample in samples} == {
        "different_people/person_b",
        "same_person/person_a",
    }


def test_threshold_metrics_counts_confusion_matrix() -> None:
    pairs = [
        PairResult(Path("same-1.jpg"), Path("same-2.jpg"), 0.80, "same_person"),
        PairResult(Path("same-1.jpg"), Path("different.jpg"), 0.20, "different_person"),
        PairResult(Path("same-2.jpg"), Path("different.jpg"), 0.70, "different_person"),
    ]

    result = threshold_metrics(pairs, 0.50)

    assert result == {
        "threshold": 0.5,
        "true_positive": 1,
        "false_positive": 1,
        "true_negative": 1,
        "false_negative": 0,
        "accuracy": 2 / 3,
    }


def test_recommend_threshold_prefers_accuracy_then_false_positives() -> None:
    result = recommend_threshold(
        [
            {"threshold": 0.30, "accuracy": 0.75, "false_positive": 2},
            {"threshold": 0.40, "accuracy": 0.75, "false_positive": 0},
            {"threshold": 0.50, "accuracy": 0.50, "false_positive": 0},
        ]
    )

    assert result["threshold"] == 0.40