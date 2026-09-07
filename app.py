"""Web API server and static file host for the Face Identification & Blockchain UI."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time
from http import HTTPStatus
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from blockchain.ledger import BlockchainLedger
from face.detector import FaceDetectionError, detect_faces
from face.matcher import resolve_threshold
from src.pipeline import run_pipeline

PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_DIR = PROJECT_ROOT / "web"
SCANS_DIR = PROJECT_ROOT / "input" / "scans"
CANDIDATES_JSON = PROJECT_ROOT / "candidates" / "results.json"
LEDGER_JSON = PROJECT_ROOT / "blockchain" / "ledger.json"

SCANS_DIR.mkdir(parents=True, exist_ok=True)


def _load_candidates() -> list[dict]:
    """Load default candidate records or construct fallback from downloaded images."""
    if CANDIDATES_JSON.is_file():
        try:
            candidates = json.loads(CANDIDATES_JSON.read_text(encoding="utf-8"))
            if isinstance(candidates, list) and candidates:
                return candidates
        except Exception:
            pass

    # Fallback to sample candidates
    fallback = [
        {"image_path": "input/sample4.png", "caption": "Candidate A (Person A)", "platform": "Instagram"},
        {"image_path": "input/sample2.png", "caption": "Candidate B (Negative sample)", "platform": "Web"},
        {"image_path": "input/sample3.png", "caption": "Candidate C (Person B)", "platform": "Web"},
        {"image_path": "input/sample5.png", "caption": "Candidate D (Person C)", "platform": "Instagram"},
    ]
    # If person1 exists, add them as well
    person1_dir = PROJECT_ROOT / "input" / "person1"
    if person1_dir.is_dir():
        for img in sorted(person1_dir.glob("*.*")):
            fallback.append({
                "image_path": f"input/person1/{img.name}",
                "caption": f"Person 1 / Wasim ({img.name})",
                "platform": "Instagram",
            })
    return fallback


class AppRequestHandler(SimpleHTTPRequestHandler):
    """Serve the single-page application and handle scanning API requests."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PROJECT_ROOT), **kwargs)

    def _send_json(self, data: dict, status: int = 200) -> None:
        payload = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path in {"", "/"}:
            index_file = STATIC_DIR / "index.html"
            if index_file.is_file():
                content = index_file.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
            self._send_json({"error": "web/index.html not found"}, status=404)
            return

        if path == "/api/candidates":
            candidates = _load_candidates()
            self._send_json({"candidates": candidates, "count": len(candidates)})
            return

        if path == "/api/ledger":
            ledger = BlockchainLedger(str(LEDGER_JSON))
            blocks = ledger._read()
            self._send_json({"blocks": blocks, "count": len(blocks)})
            return

        # Serve static assets from project directory
        return super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.rstrip("/") != "/api/scan":
            self._send_json({"error": "Endpoint not found"}, status=404)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if not content_length:
            self._send_json({"error": "Missing request body"}, status=400)
            return

        body = self.rfile.read(content_length)
        try:
            req_data = json.loads(body.decode("utf-8"))
        except Exception as err:
            self._send_json({"error": f"Invalid JSON body: {err}"}, status=400)
            return

        image_data = req_data.get("image_data")
        image_path = req_data.get("image_path")
        threshold = req_data.get("threshold")
        if threshold is not None:
            try:
                threshold = float(threshold)
            except ValueError:
                threshold = None

        target_file: Path
        if image_data:
            # Decode base64 image data URI
            match = re.match(r"^data:image\/([a-zA-Z0-9]+);base64,(.+)$", image_data)
            if match:
                ext = match.group(1).lower()
                if ext == "jpeg":
                    ext = "jpg"
                raw_bytes = base64.b64decode(match.group(2))
            else:
                try:
                    raw_bytes = base64.b64decode(image_data)
                    ext = "jpg"
                except Exception:
                    self._send_json({"error": "Invalid base64 image string"}, status=400)
                    return

            timestamp = int(time.time() * 1000)
            target_file = SCANS_DIR / f"scan_{timestamp}.{ext}"
            target_file.write_bytes(raw_bytes)
        elif image_path:
            target_file = PROJECT_ROOT / image_path.lstrip("/")
            if not target_file.is_file():
                self._send_json({"error": f"Specified image not found: {image_path}"}, status=404)
                return
        else:
            self._send_json({"error": "Must supply image_data (base64) or image_path"}, status=400)
            return

        # 1. Face Detection
        try:
            faces = detect_faces(target_file)
        except FaceDetectionError as err:
            self._send_json({
                "success": False,
                "face_detected": False,
                "error": str(err),
                "image_url": f"/{target_file.relative_to(PROJECT_ROOT)}",
            })
            return

        # 2. Run Face Matching & Blockchain Pipeline
        candidates = _load_candidates()
        try:
            pipeline_result = run_pipeline(
                input_image=target_file,
                candidates=candidates,
                ledger_path=str(LEDGER_JSON),
                threshold=threshold,
            )
        except Exception as err:
            self._send_json({
                "success": False,
                "face_detected": True,
                "face_count": len(faces),
                "error": f"Pipeline processing failed: {err}",
            }, status=500)
            return

        match = pipeline_result.face_match
        block = pipeline_result.blockchain_block
        verification = pipeline_result.blockchain_verification

        # Construct relative image URLs
        best_match_dict = match.get("best_match")
        best_match_url = None
        if isinstance(best_match_dict, dict) and best_match_dict.get("image_path"):
            best_match_url = f"/{best_match_dict['image_path'].lstrip('/')}"

        self._send_json({
            "success": True,
            "face_detected": True,
            "face_count": len(faces),
            "faces": faces,
            "scan_image_url": f"/{target_file.relative_to(PROJECT_ROOT)}",
            "is_match": match.get("is_match", False),
            "decision": match.get("decision", "no_reliable_match"),
            "confidence": match.get("confidence", "UNKNOWN"),
            "similarity": match.get("similarity"),
            "score_percent": match.get("score"),
            "threshold": match.get("threshold", resolve_threshold(threshold)),
            "threshold_percent": match.get("threshold_percent"),
            "best_match": best_match_dict,
            "best_match_image_url": best_match_url,
            "all_candidates": match.get("candidates", []),
            "blockchain_block": block,
            "blockchain_verification": verification,
        })


def run_server(host: str = "0.0.0.0", port: int = 5000) -> None:
    server_address = (host, port)
    httpd = HTTPServer(server_address, AppRequestHandler)
    print(f"Face Identification & Blockchain Web Server running at http://localhost:{port}/")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer shutting down.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the Face Identification & Blockchain Web Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Port number (default: 5000)")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)
