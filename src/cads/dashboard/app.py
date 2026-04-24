from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


SEVERITY_ORDER = ["critical", "high", "medium", "low"]
SEVERITY_COLOR_DOMAIN = ["critical", "high", "medium", "low"]
SEVERITY_COLOR_RANGE = ["#dc2626", "#f97316", "#f59e0b", "#22c55e"]


def _db_path() -> Path:
    return Path(os.getenv("CADS_DB_PATH", "artifacts/reports/alerts.db"))


def _safe_json_load(value: str) -> dict[str, object]:
    try:
        payload = json.loads(value)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _read_alerts(limit: int = 2000) -> pd.DataFrame:
    db = _db_path()
    if not db.exists():
        return pd.DataFrame()

    with sqlite3.connect(db) as conn:
        query = """
        SELECT
          id,
          timestamp,
          src_ip,
          dst_ip,
          predicted_label,
          confidence,
          severity,
          score,
          evidence_json,
          created_at
        FROM alerts
        ORDER BY id DESC
        LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(limit,))

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df["severity"] = df["severity"].astype(str).str.lower()
    df["predicted_label"] = df["predicted_label"].astype(str).str.lower()

    evidence = df["evidence_json"].fillna("{}").map(_safe_json_load)
    df["protocol"] = evidence.map(lambda x: str(x.get("protocol", "N/A")).upper())
    df["packet_count"] = pd.to_numeric(evidence.map(lambda x: x.get("packet_count", 0)), errors="coerce").fillna(0)
    df["byte_count"] = pd.to_numeric(evidence.map(lambda x: x.get("byte_count", 0)), errors="coerce").fillna(0)
    df["duration"] = pd.to_numeric(evidence.map(lambda x: x.get("duration", 0)), errors="coerce").fillna(0)
    df["model"] = evidence.map(lambda x: str(x.get("model", "N/A")))
    return df


def _apply_filters(
    df: pd.DataFrame,
    *,
    time_window: str,
    severity_selected: list[str],
    label_selected: list[str],
    confidence_range: tuple[float, float],
    score_range: tuple[float, float],
    ip_search: str,
) -> pd.DataFrame:
    filtered = df.copy()
    now_utc = pd.Timestamp.now("UTC")
    if time_window != "All":
        mapping = {
            "Last 1 hour": pd.Timedelta(hours=1),
            "Last 6 hours": pd.Timedelta(hours=6),
            "Last 24 hours": pd.Timedelta(hours=24),
            "Last 7 days": pd.Timedelta(days=7),
        }
        start_ts = now_utc - mapping[time_window]
        filtered = filtered[filtered["timestamp"] >= start_ts]

    filtered = filtered[filtered["severity"].isin(severity_selected)]
    filtered = filtered[filtered["predicted_label"].isin(label_selected)]
    filtered = filtered[
        (filtered["confidence"] >= confidence_range[0]) & (filtered["confidence"] <= confidence_range[1])
    ]
    filtered = filtered[(filtered["score"] >= score_range[0]) & (filtered["score"] <= score_range[1])]

    if ip_search:
        mask = filtered["src_ip"].str.lower().str.contains(ip_search) | filtered["dst_ip"].str.lower().str.contains(ip_search)
        filtered = filtered[mask]

    return filtered


def _metric_cards(df: pd.DataFrame) -> None:
    critical_count = int((df["severity"] == "critical").sum())
    attack_like = int((df["predicted_label"] != "benign").sum())
    cols = st.columns(5)
    cols[0].metric("Total Alerts", f"{len(df):,}")
    cols[1].metric("Critical Alerts", f"{critical_count:,}")
    cols[2].metric("Attack-like Alerts", f"{attack_like:,}")
    cols[3].metric("Unique Source IPs", f"{df['src_ip'].nunique():,}")
    cols[4].metric("Avg Confidence", f"{df['confidence'].mean():.3f}")


def _severity_pie(df: pd.DataFrame) -> alt.Chart:
    data = (
        df["severity"]
        .value_counts()
        .rename_axis("severity")
        .reset_index(name="count")
    )
    return (
        alt.Chart(data)
        .mark_arc(innerRadius=70, outerRadius=120)
        .encode(
            theta=alt.Theta("count:Q"),
            color=alt.Color(
                "severity:N",
                scale=alt.Scale(domain=SEVERITY_COLOR_DOMAIN, range=SEVERITY_COLOR_RANGE),
                legend=alt.Legend(title="Severity"),
            ),
            tooltip=["severity:N", "count:Q"],
        )
        .properties(title="Severity Split")
    )


def _label_bar(df: pd.DataFrame) -> alt.Chart:
    data = (
        df["predicted_label"]
        .value_counts()
        .rename_axis("predicted_label")
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("predicted_label:N", sort="-y", title="Predicted Label"),
            y=alt.Y("count:Q", title="Alerts"),
            color=alt.value("#2563eb"),
            tooltip=["predicted_label:N", "count:Q"],
        )
        .properties(title="Label Distribution", height=300)
    )


def _timeline(df: pd.DataFrame, bucket: str = "15min") -> alt.Chart:
    if df.empty:
        return alt.Chart(pd.DataFrame({"timestamp": [], "count": []})).mark_line()

    trend = (
        df.set_index("timestamp")
        .resample(bucket)["id"]
        .count()
        .rename("count")
        .reset_index()
    )
    title_bucket = bucket.replace("min", "-min")
    return (
        alt.Chart(trend)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=alt.X("timestamp:T", title="Time"),
            y=alt.Y("count:Q", title="Alerts"),
            tooltip=["timestamp:T", "count:Q"],
        )
        .properties(title=f"Alert Velocity ({title_bucket} buckets)", height=280)
    )


def _host_bar(df: pd.DataFrame, column: str, title: str, top_n: int) -> alt.Chart:
    top = (
        df[column]
        .value_counts()
        .head(top_n)
        .rename_axis(column)
        .reset_index(name="count")
    )
    return (
        alt.Chart(top)
        .mark_bar(cornerRadiusEnd=6)
        .encode(
            y=alt.Y(f"{column}:N", sort="-x", title=""),
            x=alt.X("count:Q", title="Alerts"),
            color=alt.value("#0ea5e9" if column == "src_ip" else "#8b5cf6"),
            tooltip=[f"{column}:N", "count:Q"],
        )
        .properties(title=title, height=320)
    )


def _risk_scatter(df: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_circle(size=70, opacity=0.65)
        .encode(
            x=alt.X("confidence:Q", title="Confidence"),
            y=alt.Y("score:Q", title="Risk Score"),
            color=alt.Color(
                "severity:N",
                scale=alt.Scale(domain=SEVERITY_COLOR_DOMAIN, range=SEVERITY_COLOR_RANGE),
                legend=alt.Legend(title="Severity"),
            ),
            tooltip=["id:Q", "timestamp:T", "src_ip:N", "dst_ip:N", "predicted_label:N", "confidence:Q", "score:Q"],
        )
        .properties(title="Confidence vs Risk", height=320)
    )


def _protocol_heatmap(df: pd.DataFrame) -> alt.Chart:
    heat = (
        df.groupby(["protocol", "predicted_label"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    return (
        alt.Chart(heat)
        .mark_rect()
        .encode(
            x=alt.X("predicted_label:N", title="Predicted Label"),
            y=alt.Y("protocol:N", title="Protocol"),
            color=alt.Color("count:Q", scale=alt.Scale(scheme="tealblues")),
            tooltip=["protocol:N", "predicted_label:N", "count:Q"],
        )
        .properties(title="Protocol x Label Heatmap", height=240)
    )


def _styled_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    show_cols = [
        "id",
        "timestamp",
        "severity",
        "predicted_label",
        "confidence",
        "score",
        "src_ip",
        "dst_ip",
        "protocol",
        "packet_count",
        "duration",
    ]
    table = df[show_cols].copy().sort_values("id", ascending=False)

    def sev_style(val: str) -> str:
        color_map = {
            "critical": "background-color: #fee2e2; color: #991b1b; font-weight: 700;",
            "high": "background-color: #ffedd5; color: #9a3412; font-weight: 700;",
            "medium": "background-color: #fef3c7; color: #92400e; font-weight: 700;",
            "low": "background-color: #dcfce7; color: #166534; font-weight: 700;",
        }
        return color_map.get(str(val).lower(), "")

    return table.style.format({"confidence": "{:.3f}", "score": "{:.3f}", "packet_count": "{:.0f}", "duration": "{:.3f}"}).map(
        sev_style, subset=["severity"]
    )


def _drilldown(df: pd.DataFrame, live_refresh: bool) -> None:
    st.subheader("Alert Drill-down")
    if df.empty:
        st.info("No data after filters.")
        return

    alert_ids = df["id"].sort_values(ascending=False).tolist()
    if "drilldown_follow_latest" not in st.session_state:
        st.session_state["drilldown_follow_latest"] = bool(live_refresh)
    if "drilldown_selected_id" not in st.session_state or st.session_state["drilldown_selected_id"] not in alert_ids:
        st.session_state["drilldown_selected_id"] = alert_ids[0]

    st.toggle(
        "Follow latest alert",
        key="drilldown_follow_latest",
        help="When ON, drill-down auto-switches to newest alert on every refresh.",
    )
    if st.session_state["drilldown_follow_latest"]:
        st.session_state["drilldown_selected_id"] = alert_ids[0]

    selected = st.selectbox("Select Alert ID", alert_ids, key="drilldown_selected_id")
    row = df[df["id"] == selected].iloc[0]

    a, b, c = st.columns(3)
    a.metric("Severity", str(row["severity"]).upper())
    b.metric("Confidence", f"{float(row['confidence']):.3f}")
    c.metric("Risk Score", f"{float(row['score']):.3f}")

    st.markdown(
        f"""
**Source**: `{row['src_ip']}`  
**Destination**: `{row['dst_ip']}`  
**Predicted Label**: `{row['predicted_label']}`  
**Protocol**: `{row['protocol']}`  
**Model**: `{row['model']}`  
**Timestamp**: `{row['timestamp']}`  
        """
    )

    evidence = _safe_json_load(str(row["evidence_json"]))
    st.code(json.dumps(evidence, indent=2), language="json")


def _inject_css() -> None:
    st.markdown(
        """
<style>
  .main > div { padding-top: 1rem; }
  .block-container { max-width: 1400px; }
  [data-testid="stMetricValue"] { font-size: 1.6rem; }
  [data-testid="stSidebar"] { border-right: 1px solid rgba(148,163,184,0.25); }
</style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Cyber Attack Detection Dashboard", page_icon="🛡️", layout="wide")
    _inject_css()

    st.title("Cyber Attack Detection Command Center")
    st.caption("Interactive SOC-style dashboard for alert triage and investigation.")

    alerts_for_filters = _read_alerts(limit=5000)

    with st.sidebar:
        st.header("Filters")
        st.caption("Tune the view for triage and demo storytelling.")
        time_window = st.selectbox(
            "Time window",
            options=["All", "Last 1 hour", "Last 6 hours", "Last 24 hours", "Last 7 days"],
            index=0,
        )
        severity_default = [sev for sev in SEVERITY_ORDER if sev in set(alerts_for_filters["severity"])] if not alerts_for_filters.empty else SEVERITY_ORDER
        severity_selected = st.multiselect("Severity", options=SEVERITY_ORDER, default=severity_default)
        label_options = sorted(alerts_for_filters["predicted_label"].dropna().unique().tolist()) if not alerts_for_filters.empty else []
        label_selected = st.multiselect("Predicted label", options=label_options, default=label_options)
        confidence_range = st.slider("Confidence", 0.0, 1.0, (0.0, 1.0), 0.01)
        score_range = st.slider("Risk score", 0.0, 1.0, (0.0, 1.0), 0.01)
        ip_search = st.text_input("Search IP (src or dst)", value="").strip().lower()
        top_n = st.slider("Top N hosts/charts", min_value=5, max_value=30, value=10, step=1)
        preview_filtered = _apply_filters(
            alerts_for_filters,
            time_window=time_window,
            severity_selected=severity_selected,
            label_selected=label_selected,
            confidence_range=confidence_range,
            score_range=score_range,
            ip_search=ip_search,
        )
        st.markdown("---")
        st.metric("Records in view", len(preview_filtered))

    col_left, col_live, col_refresh = st.columns([3, 1, 1])
    with col_left:
        st.markdown("Track threat alerts, investigate suspicious flows, and export filtered evidence.")
    with col_live:
        live_refresh = st.toggle("Live Mode", value=False, help="Auto-refresh dashboard data.")
    refresh_interval = st.selectbox(
        "Refresh Every (seconds)",
        options=[3, 5, 8, 10, 15, 30],
        index=1,
        disabled=not live_refresh,
    )
    with col_refresh:
        if st.button("Refresh Data", use_container_width=True):
            st.rerun()

    def render_dashboard() -> None:
        alerts = _read_alerts(limit=5000)
        if alerts.empty:
            st.warning("No alerts found. Run `uv run cads replay-test-alerts --limit 300` first.")
            return

        filtered = _apply_filters(
            alerts,
            time_window=time_window,
            severity_selected=severity_selected,
            label_selected=label_selected,
            confidence_range=confidence_range,
            score_range=score_range,
            ip_search=ip_search,
        )
        if filtered.empty:
            st.warning("No records match the selected filters.")
            return

        _metric_cards(filtered)
        st.caption(f"Last refreshed: {pd.Timestamp.now('UTC').strftime('%Y-%m-%d %H:%M:%S')} UTC")
        st.markdown("---")

        tab_overview, tab_traffic, tab_explorer = st.tabs(["Overview", "Traffic Insights", "Alert Explorer"])

        with tab_overview:
            c1, c2 = st.columns([1, 1.2])
            with c1:
                st.altair_chart(_severity_pie(filtered), use_container_width=True)
            with c2:
                st.altair_chart(_label_bar(filtered), use_container_width=True)

            timeline_bucket = "1min" if live_refresh else "15min"
            st.altair_chart(_timeline(filtered, bucket=timeline_bucket), use_container_width=True)

        with tab_traffic:
            c3, c4 = st.columns(2)
            with c3:
                st.altair_chart(_host_bar(filtered, "src_ip", f"Top {top_n} Source IPs", top_n), use_container_width=True)
            with c4:
                st.altair_chart(_host_bar(filtered, "dst_ip", f"Top {top_n} Destination IPs", top_n), use_container_width=True)

            c5, c6 = st.columns([1.2, 1])
            with c5:
                st.altair_chart(_risk_scatter(filtered), use_container_width=True)
            with c6:
                st.altair_chart(_protocol_heatmap(filtered), use_container_width=True)

        with tab_explorer:
            st.subheader("Filtered Alert Feed")
            st.dataframe(_styled_table(filtered), use_container_width=True, hide_index=True)
            csv = filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Filtered Alerts (CSV)",
                csv,
                file_name="alerts_filtered.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.markdown("---")
            _drilldown(filtered, live_refresh=live_refresh)

    if live_refresh:
        @st.fragment(run_every=f"{refresh_interval}s")
        def live_panel() -> None:
            render_dashboard()

        live_panel()
    else:
        render_dashboard()


if __name__ == "__main__":
    main()
