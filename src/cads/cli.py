from __future__ import annotations

import argparse
import json
import logging

from cads.config import AppConfig, ensure_directories
from cads.data.pipeline import prepare_data
from cads.data.synthetic import generate_sample_dataset
from cads.inference.replay import replay_from_test_split, simulate_live_alerts
from cads.logging_utils import setup_logging
from cads.models.train import train_and_evaluate

LOGGER = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cads",
        description="Cyber Attack Detection System CLI (Phases 1-3).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate-sample-data", help="Generate synthetic raw network telemetry CSV.")
    gen.add_argument("--rows", type=int, default=5000, help="Number of rows to generate.")
    gen.add_argument("--seed", type=int, default=42, help="Random seed.")

    subparsers.add_parser("prepare-data", help="Run ingestion + preprocessing + feature engineering + data split.")
    subparsers.add_parser("train-models", help="Train and evaluate supervised + anomaly models.")
    replay = subparsers.add_parser("replay-test-alerts", help="Replay processed test records into alert DB.")
    replay.add_argument("--limit", type=int, default=200, help="Maximum test records to replay.")
    replay.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "compare_all", "ensemble"],
        help="Inference mode for replay.",
    )
    live = subparsers.add_parser("simulate-live-alerts", help="Continuously insert alerts for real-time dashboard demo.")
    live.add_argument("--mode", type=str, default="single", choices=["single", "compare_all", "ensemble"])
    live.add_argument("--interval", type=float, default=5.0, help="Seconds between cycles.")
    live.add_argument("--batch-size", type=int, default=3, help="Alerts inserted per cycle.")
    live.add_argument("--cycles", type=int, default=30, help="How many cycles to run.")

    return parser


def main() -> None:
    setup_logging()
    parser = _build_parser()
    args = parser.parse_args()

    config = AppConfig()
    ensure_directories(config.paths)

    if args.command == "generate-sample-data":
        output_path = generate_sample_dataset(config.paths.data_raw, rows=args.rows, seed=args.seed)
        print(json.dumps({"synthetic_data_path": str(output_path)}, indent=2))
        return

    if args.command == "prepare-data":
        result = prepare_data(config)
        print(json.dumps(result, indent=2))
        return

    if args.command == "train-models":
        result = train_and_evaluate(config)
        print(json.dumps(result, indent=2))
        return

    if args.command == "replay-test-alerts":
        result = replay_from_test_split(config, limit=args.limit, mode=args.mode)
        print(json.dumps(result, indent=2))
        return

    if args.command == "simulate-live-alerts":
        result = simulate_live_alerts(
            config,
            mode=args.mode,
            interval_seconds=args.interval,
            batch_size=args.batch_size,
            cycles=args.cycles,
        )
        print(json.dumps(result, indent=2))
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
