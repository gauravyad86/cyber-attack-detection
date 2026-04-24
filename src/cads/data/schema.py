from __future__ import annotations

CANONICAL_ALIASES: dict[str, str] = {
    "time": "timestamp",
    "datetime": "timestamp",
    "ts": "timestamp",
    "source_ip": "src_ip",
    "src": "src_ip",
    "destination_ip": "dst_ip",
    "dest_ip": "dst_ip",
    "dst": "dst_ip",
    "source_port": "src_port",
    "sport": "src_port",
    "destination_port": "dst_port",
    "dest_port": "dst_port",
    "dport": "dst_port",
    "proto": "protocol",
    "packets": "packet_count",
    "packet_len": "byte_count",
    "bytes": "byte_count",
    "flow_duration": "duration",
    "label_name": "label",
    "class": "label",
    "attack_type": "label",
}

