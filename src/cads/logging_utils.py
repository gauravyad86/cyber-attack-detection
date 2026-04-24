from __future__ import annotations

import logging
import os


def setup_logging(level: int | None = None) -> None:
    if level is None:
        env_level = os.getenv("CADS_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, env_level, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
