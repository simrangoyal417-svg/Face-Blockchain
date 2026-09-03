import json

from blockchain.ledger import BlockchainLedger, GENESIS_HASH, sha256_file


def test_upload_and_verify(tmp_path) -> None:
    artifact = tmp_path / "post.jpg"
    artifact.write_bytes(b"consented test artifact")
    ledger = BlockchainLedger(tmp_path / "ledger.json")
    block = ledger.upload(artifact, {"url": "https://example.test/post/1", "caption": "test"})

    result = ledger.verify(artifact, block)
    assert result["verified"] is True
    assert block["previous_hash"] == GENESIS_HASH
    assert block["artifact_sha256"] == sha256_file(artifact)


def test_changed_artifact_fails_verification(tmp_path) -> None:
    artifact = tmp_path / "post.jpg"
    artifact.write_bytes(b"original")
    ledger = BlockchainLedger(tmp_path / "ledger.json")
    block = ledger.upload(artifact, {"url": "https://example.test/post/1"})
    artifact.write_bytes(b"changed")

    result = ledger.verify(artifact, block)
    assert result["verified"] is False
    assert "artifact hash does not match" in result["reasons"]


def test_changed_block_fails_chain_verification(tmp_path) -> None:
    artifact = tmp_path / "post.jpg"
    artifact.write_bytes(b"artifact")
    ledger_path = tmp_path / "ledger.json"
    ledger = BlockchainLedger(ledger_path)
    ledger.upload(artifact, {"url": "https://example.test/post/1"})
    blocks = json.loads(ledger_path.read_text())
    blocks[0]["post"]["url"] = "https://example.test/tampered"
    ledger_path.write_text(json.dumps(blocks))

    result = ledger.verify_chain()
    assert result["verified"] is False