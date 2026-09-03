"""Local tamper-evident ledger for matched social-post artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GENESIS_HASH = "0" * 64


class BlockchainVerificationError(ValueError):
    """Raised when a ledger or artifact fails verification."""


def sha256_file(file_path: str | Path) -> str:
    """Return the SHA-256 digest of a local file without uploading it."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Artifact not found: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash_block(block: dict[str, Any]) -> str:
    block_without_hash = {key: value for key, value in block.items() if key != "block_hash"}
    return hashlib.sha256(_canonical_json(block_without_hash).encode("utf-8")).hexdigest()


class BlockchainLedger:
    """A JSON-backed append-only chain suitable for local demonstrations."""

    def __init__(self, ledger_path: str | Path = "blockchain/ledger.json") -> None:
        self.ledger_path = Path(ledger_path)

    def _read(self) -> list[dict[str, Any]]:
        if not self.ledger_path.exists():
            return []
        try:
            value = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise BlockchainVerificationError(f"Cannot read ledger: {error}") from error
        if not isinstance(value, list):
            raise BlockchainVerificationError("Ledger must contain a JSON list of blocks")
        return value

    def upload(self, artifact_path: str | Path, post: dict[str, Any]) -> dict[str, Any]:
        """Hash a matched artifact and post metadata, then append a block."""
        if not isinstance(post, dict):
            raise TypeError("post metadata must be a dictionary")
        blocks = self._read()
        previous_hash = blocks[-1]["block_hash"] if blocks else GENESIS_HASH
        block = {
            "index": len(blocks),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "artifact_sha256": sha256_file(artifact_path),
            "post": post,
            "previous_hash": previous_hash,
        }
        block["block_hash"] = _hash_block(block)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.ledger_path.write_text(
            json.dumps(blocks + [block], indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        return block

    def verify(self, artifact_path: str | Path, block: dict[str, Any]) -> dict[str, Any]:
        """Recalculate the artifact and block hashes and return verification details."""
        blocks = self._read()
        index = block.get("index")
        reasons: list[str] = []
        if not isinstance(index, int) or index < 0 or index >= len(blocks):
            reasons.append("block is not present at its declared index")
        else:
            stored_block = blocks[index]
            if stored_block != block:
                reasons.append("stored block differs from supplied block")
            expected_previous = GENESIS_HASH if index == 0 else blocks[index - 1].get("block_hash")
            if block.get("previous_hash") != expected_previous:
                reasons.append("previous block hash does not match")
        if block.get("block_hash") != _hash_block(block):
            reasons.append("block hash does not match its contents")
        try:
            actual_artifact_hash = sha256_file(artifact_path)
            if block.get("artifact_sha256") != actual_artifact_hash:
                reasons.append("artifact hash does not match")
        except FileNotFoundError as error:
            reasons.append(str(error))
            actual_artifact_hash = None
        return {
            "verified": not reasons,
            "block_hash": block.get("block_hash"),
            "artifact_sha256": actual_artifact_hash,
            "reasons": reasons,
        }

    def verify_chain(self) -> dict[str, Any]:
        """Verify every block's hash and link without needing the artifact files."""
        blocks = self._read()
        reasons: list[str] = []
        for index, block in enumerate(blocks):
            expected_previous = GENESIS_HASH if index == 0 else blocks[index - 1].get("block_hash")
            if block.get("index") != index:
                reasons.append(f"block {index} has an invalid index")
            if block.get("previous_hash") != expected_previous:
                reasons.append(f"block {index} has an invalid previous hash")
            if block.get("block_hash") != _hash_block(block):
                reasons.append(f"block {index} has an invalid hash")
        return {"verified": not reasons, "blocks": len(blocks), "reasons": reasons}