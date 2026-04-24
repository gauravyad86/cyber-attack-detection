from __future__ import annotations

import pandas as pd

from cads.config import DataConfig
from cads.data.preprocess import _as_private_flag, _ip_low16, preprocess_dataframe


def _required() -> tuple[str, ...]:
    return DataConfig().canonical_columns


def _base_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": "2026-04-23T11:00:00Z",
                "src_ip": "10.1.1.1",
                "dst_ip": "8.8.8.8",
                "src_port": 51512,
                "dst_port": 443,
                "protocol": "tcp",
                "packet_count": 20,
                "byte_count": 12000,
                "duration": 1.2,
                "label": "BENIGN",
            }
        ]
    )


def test_as_private_flag_private_ip() -> None:
    assert _as_private_flag("10.0.0.2") == 1


def test_as_private_flag_public_ip() -> None:
    assert _as_private_flag("8.8.8.8") == 0


def test_as_private_flag_invalid_ip() -> None:
    assert _as_private_flag("not-an-ip") == 0


def test_ip_low16_is_deterministic() -> None:
    assert _ip_low16("10.1.2.3") == _ip_low16("10.1.2.3")


def test_preprocess_keeps_valid_row() -> None:
    out = preprocess_dataframe(_base_df(), _required())
    assert len(out) == 1


def test_preprocess_normalizes_protocol_and_label() -> None:
    out = preprocess_dataframe(_base_df(), _required())
    assert out.iloc[0]["protocol"] == "TCP"
    assert out.iloc[0]["label"] == "benign"


def test_preprocess_drops_invalid_duration_and_packet_count() -> None:
    df = _base_df()
    df2 = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {**df.iloc[0].to_dict(), "duration": 0},
                    {**df.iloc[0].to_dict(), "packet_count": 0},
                ]
            ),
        ],
        ignore_index=True,
    )
    out = preprocess_dataframe(df2, _required())
    assert len(out) == 1


def test_preprocess_drops_duplicates() -> None:
    df = pd.concat([_base_df(), _base_df()], ignore_index=True)
    out = preprocess_dataframe(df, _required())
    assert len(out) == 1


def test_preprocess_adds_missing_columns() -> None:
    df = pd.DataFrame([{"timestamp": "2026-04-23T11:00:00Z"}])
    out = preprocess_dataframe(df, _required())
    assert out.empty
    for col in _required():
        assert col in out.columns


def test_preprocess_computes_hour_of_day() -> None:
    out = preprocess_dataframe(_base_df(), _required())
    assert int(out.iloc[0]["hour_of_day"]) == 11


def test_preprocess_computes_private_and_low16_features() -> None:
    out = preprocess_dataframe(_base_df(), _required())
    assert int(out.iloc[0]["src_is_private"]) == 1
    assert int(out.iloc[0]["dst_is_private"]) == 0
    assert int(out.iloc[0]["src_ip_low16"]) >= 0
    assert int(out.iloc[0]["dst_ip_low16"]) >= 0

