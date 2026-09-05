from pathlib import Path

import main


def test_run_integration_connects_search_match_and_ledger(tmp_path, monkeypatch, capsys) -> None:
    candidate_path = tmp_path / "candidate.jpg"
    candidate_path.write_bytes(b"candidate image")
    candidate = {
        "image_path": str(candidate_path),
        "url": "https://example.test/post/1",
        "platform": "example",
    }

    monkeypatch.setattr(main, "detect_faces", lambda path: [{"facial_area": {}}])
    monkeypatch.setattr(main, "encode_faces", lambda path: [object()])
    monkeypatch.setattr(main, "search_public_images", lambda path, max_results: [{"url": candidate["url"]}])
    monkeypatch.setattr(main, "download_candidate_images", lambda candidates, output_dir: [candidate])
    monkeypatch.setattr(
        main,
        "find_best_match",
        lambda input_path, candidates, threshold: {
            "best_match": candidate,
            "best_candidate": candidate,
            "score": 50.0,
            "is_match": True,
            "decision": "same_person",
            "threshold": threshold,
            "candidates": [],
        },
    )

    result = main.run_integration("input.png", tmp_path / "ledger.json")

    assert result["blockchain_verification"]["verified"] is True
    assert result["blockchain_block"]["post"]["url"] == candidate["url"]
    output = capsys.readouterr().out
    assert "4. Web/social search started" in output
    assert "13. Final result: VERIFIED" in output


def test_run_integration_does_not_store_below_threshold(tmp_path, monkeypatch, capsys) -> None:
    candidate = {"image_path": str(tmp_path / "candidate.jpg")}
    monkeypatch.setattr(main, "detect_faces", lambda path: [{"facial_area": {}}])
    monkeypatch.setattr(main, "encode_faces", lambda path: [object()])
    monkeypatch.setattr(main, "search_public_images", lambda path, max_results: [{}])
    monkeypatch.setattr(main, "download_candidate_images", lambda candidates, output_dir: [candidate])
    monkeypatch.setattr(
        main,
        "find_best_match",
        lambda input_path, candidates, threshold: {
            "best_match": None,
            "best_candidate": candidate,
            "score": 20.0,
            "is_match": False,
            "decision": "different_person",
            "threshold": threshold,
            "candidates": [],
        },
    )

    result = main.run_integration("input.png", tmp_path / "ledger.json")

    assert result["blockchain_block"] is None
    assert result["blockchain_verification"] is None
    assert not (tmp_path / "ledger.json").exists()
    assert "FINAL RESULT: VERIFICATION FAILED" in capsys.readouterr().out