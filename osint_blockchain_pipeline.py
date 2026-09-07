"""Gemini-assisted Instagram discovery with a local Web3 recording step.

The Gemini prompt generates text queries from visual context; it does not ask
Gemini to identify a private person from facial features. Search results are
then limited to public Instagram profile URLs before anything is recorded.

Minimal DataVerifier.sol:

    // SPDX-License-Identifier: MIT
    pragma solidity ^0.8.20;
    contract DataVerifier {
        mapping(bytes32 => uint256) public verifiedAt;
        event HashRecorded(bytes32 indexed fingerprint, uint256 timestamp);
        function recordData(bytes32 dataHash) external {
            verifiedAt[dataHash] = block.timestamp;
            emit HashRecorded(dataHash, block.timestamp);
        }
    }
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv


LOGGER = logging.getLogger(__name__)
SEARCH_ENDPOINT = "https://serpapi.com/search.json"
INSTAGRAM_PROFILE_PATH = re.compile(r"^/[A-Za-z0-9._]{1,30}/?$")
INSTAGRAM_RESERVED_PATHS = {"p", "reel", "tv", "explore", "accounts", "direct"}
MOCK_CHAIN_STORAGE: dict[str, dict[str, Any]] = {}


class PipelineError(RuntimeError):
    """Raised when a required pipeline stage cannot produce a result."""


def _require_file(image_path: str | Path) -> Path:
    path = Path(image_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    return path


def _parse_query_list(response_text: str) -> list[str]:
    """Parse Gemini's response as a short, deduplicated list of strings."""
    text = response_text.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            value = ast.literal_eval(text)
        except (SyntaxError, ValueError) as error:
            raise PipelineError("Gemini did not return a valid list of queries") from error
    if not isinstance(value, list):
        raise PipelineError("Gemini response must be a list of search strings")
    queries: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip() and item.strip() not in queries:
            queries.append(item.strip())
    if len(queries) != 3:
        raise PipelineError(
            f"Gemini returned {len(queries)} useful queries; exactly 3 are required"
        )
    return queries


def generate_search_queries(image_path: str | Path) -> list[str]:
    """Use Gemini vision to generate exactly three public-web search queries."""
    path = _require_file(image_path)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise PipelineError("GEMINI_API_KEY is not configured")

    try:
        import google.generativeai as genai
    except ImportError as error:
        raise PipelineError(
            "Install google-generativeai before running the Gemini step"
        ) from error

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = """Analyze this image for public, non-sensitive context only. Look for recognizable public figures, visible text, logos, landmarks, venues, brands, or event clues. Do not infer or identify a private person from facial features. Return ONLY a valid Python list containing exactly 3 concise keyword combinations optimized to find a relevant public social media profile. If there is no useful context, return []."""
    try:
        uploaded_image = genai.upload_file(path=str(path))
        response = model.generate_content([prompt, uploaded_image])
        response_text = response.text or ""
    except Exception as error:
        raise PipelineError(f"Gemini analysis failed: {error}") from error
    return _parse_query_list(response_text)


def _instagram_profile_url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"}:
        return None
    if parsed.netloc.lower() not in {"instagram.com", "www.instagram.com"}:
        return None
    if not INSTAGRAM_PROFILE_PATH.fullmatch(parsed.path):
        return None
    username = parsed.path.strip("/").lower()
    if username in INSTAGRAM_RESERVED_PATHS:
        return None
    return f"https://www.instagram.com{parsed.path.rstrip('/')}/"


def find_instagram_profile(queries: list[str]) -> tuple[str, str] | None:
    """Search each Gemini query and return the first public profile URL."""
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        raise PipelineError("SERPAPI_KEY is not configured")
    for query in queries:
        try:
            response = requests.get(
                SEARCH_ENDPOINT,
                params={
                    "engine": "google",
                    "q": f"{query} site:instagram.com",
                    "api_key": api_key,
                    "num": 10,
                },
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as error:
            LOGGER.warning("SerpApi query failed for %r: %s", query, error)
            continue
        if data.get("error"):
            LOGGER.warning("SerpApi returned an error for %r: %s", query, data["error"])
            continue
        for result in data.get("organic_results", []):
            profile_url = _instagram_profile_url(result.get("link"))
            if profile_url:
                return profile_url, query
    return None


def fingerprint(image_path: str | Path, profile_url: str) -> str:
    """Create a deterministic fingerprint from the URL and original path."""
    path = str(Path(image_path).expanduser().resolve())
    payload = f"{profile_url}\n{path}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def mock_upload_to_blockchain(fingerprint_hash: str) -> dict[str, Any]:
    """Record a hash using Web3 and a local mock when no contract is deployed.

    Set DATA_VERIFIER_ADDRESS and WEB3_PRIVATE_KEY to send a real transaction
    to the local contract. Without them, the same mapping semantics are kept
    in memory so the pipeline remains runnable as a hackathon demonstration.
    """
    try:
        from web3 import Web3
    except ImportError as error:
        raise PipelineError("Install web3 before running the blockchain step") from error

    rpc_url = os.getenv("WEB3_RPC_URL", "http://127.0.0.1:8545")
    web3 = Web3(Web3.HTTPProvider(rpc_url))
    try:
        connected = web3.is_connected()
    except Exception as error:
        LOGGER.warning("Local RPC unavailable at %s: %s", rpc_url, error)
        connected = False
    contract_address = os.getenv("DATA_VERIFIER_ADDRESS")
    private_key = os.getenv("WEB3_PRIVATE_KEY")

    if connected and contract_address and private_key:
        account = web3.eth.account.from_key(private_key)
        contract = web3.eth.contract(
            address=Web3.to_checksum_address(contract_address),
            abi=[{
                "inputs": [{"internalType": "bytes32", "name": "dataHash", "type": "bytes32"}],
                "name": "recordData",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function",
            }],
        )
        transaction = contract.functions.recordData(bytes.fromhex(fingerprint_hash)).build_transaction({
            "from": account.address,
            "nonce": web3.eth.get_transaction_count(account.address),
            "gas": 100_000,
            "gasPrice": web3.eth.gas_price,
            "chainId": web3.eth.chain_id,
        })
        signed = account.sign_transaction(transaction)
        tx_hash = web3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        return {"mode": "web3", "tx_hash": tx_hash.hex(), "block_number": receipt.blockNumber}

    if not connected:
        LOGGER.warning("No local Ethereum node detected; using the in-memory mock registry")
    MOCK_CHAIN_STORAGE[fingerprint_hash] = {
        "timestamp": web3.eth.get_block("latest")["timestamp"] if connected else None
    }
    return {
        "mode": "mock",
        "rpc_connected": connected,
        "fingerprint": fingerprint_hash,
        "mapping_value": MOCK_CHAIN_STORAGE[fingerprint_hash],
    }


def run_pipeline(image_path: str | Path) -> dict[str, Any]:
    """Run Gemini analysis, Instagram discovery, hashing, and recording."""
    load_dotenv()
    queries = generate_search_queries(image_path)
    match = find_instagram_profile(queries)
    if not match:
        raise PipelineError("No verified Instagram profile was found")
    profile_url, matched_query = match
    image_hash = fingerprint(image_path, profile_url)
    blockchain_result = mock_upload_to_blockchain(image_hash)
    return {
        "image_path": str(Path(image_path).expanduser()),
        "queries": queries,
        "matched_query": matched_query,
        "instagram_profile": profile_url,
        "fingerprint": image_hash,
        "blockchain": blockchain_result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_path", type=Path, help="Local image to analyze")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        print(json.dumps(run_pipeline(args.image_path), indent=2))
    except (FileNotFoundError, PipelineError) as error:
        LOGGER.error("Pipeline stopped: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())