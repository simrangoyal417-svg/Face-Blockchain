"""Command-line interface for recording and verifying matched posts."""

import argparse
import json

from .ledger import BlockchainLedger


def main() -> None:
    parser = argparse.ArgumentParser(description="Record or verify a local blockchain artifact.")
    parser.add_argument("--ledger", default="blockchain/ledger.json")
    subparsers = parser.add_subparsers(dest="command", required=True)

    upload = subparsers.add_parser("upload")
    upload.add_argument("artifact")
    upload.add_argument("--post-json", required=True, help="JSON object containing post URL and metadata")

    verify = subparsers.add_parser("verify")
    verify.add_argument("artifact")
    verify.add_argument("--block", type=int, required=True)

    args = parser.parse_args()
    ledger = BlockchainLedger(args.ledger)
    if args.command == "upload":
        result = ledger.upload(args.artifact, json.loads(args.post_json))
    else:
        blocks = ledger._read()
        if args.block < 0 or args.block >= len(blocks):
            parser.error(f"block must be between 0 and {len(blocks) - 1}")
        result = ledger.verify(args.artifact, blocks[args.block])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()