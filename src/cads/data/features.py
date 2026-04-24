from __future__ import annotations

import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

PROTOCOL_MAP: dict[str, int] = {
    "ICMP": 1,
    "TCP": 6,
    "UDP": 17,
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["bytes_per_packet"] = work["byte_count"] / work["packet_count"].clip(lower=1)
    work["packets_per_second"] = work["packet_count"] / work["duration"].clip(lower=1e-3)
    work["bytes_per_second"] = work["byte_count"] / work["duration"].clip(lower=1e-3)
    work["is_well_known_dst_port"] = (work["dst_port"] <= 1024).astype(int)
    work["same_src_dst"] = (work["src_ip"] == work["dst_ip"]).astype(int)
    work["protocol_id"] = work["protocol"].map(PROTOCOL_MAP).fillna(0).astype(int)

    inf_replaced = work.replace([np.inf, -np.inf], np.nan)
    work = inf_replaced.fillna(0.0)
    if len(work) > 1:
        LOGGER.info("Feature engineered dataframe shape: %s", work.shape)
    return work
