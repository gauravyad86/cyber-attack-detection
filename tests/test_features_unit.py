from __future__ import annotations

import numpy as np
import pandas as pd

from cads.data.features import PROTOCOL_MAP, engineer_features


def _feature_input() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": "2026-04-23T11:00:00Z",
                "src_ip": "10.1.1.1",
                "dst_ip": "8.8.8.8",
                "src_port": 50000,
                "dst_port": 443,
                "protocol": "TCP",
                "packet_count": 20,
                "byte_count": 10000,
                "duration": 2.0,
                "label": "benign",
                "src_is_private": 1,
                "dst_is_private": 0,
                "src_ip_low16": 1,
                "dst_ip_low16": 2,
                "hour_of_day": 11,
            }
        ]
    )


def test_protocol_map_contains_expected_values() -> None:
    assert PROTOCOL_MAP["TCP"] == 6
    assert PROTOCOL_MAP["UDP"] == 17
    assert PROTOCOL_MAP["ICMP"] == 1


def test_engineer_features_bytes_per_packet() -> None:
    out = engineer_features(_feature_input())
    assert float(out.iloc[0]["bytes_per_packet"]) == 500.0


def test_engineer_features_rates() -> None:
    out = engineer_features(_feature_input())
    assert float(out.iloc[0]["packets_per_second"]) == 10.0
    assert float(out.iloc[0]["bytes_per_second"]) == 5000.0


def test_engineer_features_well_known_port_flag() -> None:
    out = engineer_features(_feature_input())
    assert int(out.iloc[0]["is_well_known_dst_port"]) == 1


def test_engineer_features_same_src_dst_flag() -> None:
    df = _feature_input()
    df.loc[0, "dst_ip"] = "10.1.1.1"
    out = engineer_features(df)
    assert int(out.iloc[0]["same_src_dst"]) == 1


def test_engineer_features_protocol_id_known() -> None:
    out = engineer_features(_feature_input())
    assert int(out.iloc[0]["protocol_id"]) == 6


def test_engineer_features_protocol_id_unknown() -> None:
    df = _feature_input()
    df.loc[0, "protocol"] = "GRE"
    out = engineer_features(df)
    assert int(out.iloc[0]["protocol_id"]) == 0


def test_engineer_features_replaces_infinite_and_nan() -> None:
    df = _feature_input()
    df["byte_count"] = df["byte_count"].astype(float)
    df.loc[0, "byte_count"] = np.inf
    out = engineer_features(df)
    assert np.isfinite(float(out.iloc[0]["byte_count"]))
