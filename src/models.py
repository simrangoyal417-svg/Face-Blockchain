from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PipelineResult:
	"""Structured output shared by face matching and blockchain verification."""

	face_match: dict[str, Any]
	blockchain_block: dict[str, Any] | None
	blockchain_verification: dict[str, Any] | None

	def as_dict(self) -> dict[str, Any]:
		return {
			"face_match": self.face_match,
			"blockchain_block": self.blockchain_block,
			"blockchain_verification": self.blockchain_verification,
		}


def candidate_image_path(candidate: str | Path | dict[str, Any]) -> Path:
	"""Extract and validate the local image path from a search candidate."""
	value = candidate.get("image_path") if isinstance(candidate, dict) else candidate
	if not isinstance(value, (str, Path)) or not str(value).strip():
		raise ValueError("Every candidate must contain a non-empty image_path")
	return Path(value)
