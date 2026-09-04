from pathlib import Path

from src import pipeline


def test_pipeline_links_match_to_blockchain(tmp_path, monkeypatch) -> None:
    candidate_image = tmp_path / "candidate.jpg"
    candidate_image.write_bytes(b"candidate")
    face_result = {
        "best_match": {"image_path": str(candidate_image), "url": "https://example.test/post/1"},
        "score": 95.0,
        "is_match": True,
        "threshold": 0.363,
        "candidates": [],
    }
    monkeypatch.setattr(pipeline, "find_best_match", lambda *args: face_result)

    result = pipeline.run_pipeline("input.jpg", [face_result["best_match"]], tmp_path / "ledger.json")

    assert result.face_match["is_match"] is True
    assert result.blockchain_block["post"]["url"] == "https://example.test/post/1"
    assert result.blockchain_verification["verified"] is True


def test_pipeline_does_not_record_no_match(tmp_path, monkeypatch) -> None:
    no_match = {"best_match": "candidate.jpg", "score": 10.0, "is_match": False}
    monkeypatch.setattr(pipeline, "find_best_match", lambda *args: no_match)

    result = pipeline.run_pipeline("input.jpg", ["candidate.jpg"], tmp_path / "ledger.json")

    assert result.blockchain_block is None
    assert not (tmp_path / "ledger.json").exists()