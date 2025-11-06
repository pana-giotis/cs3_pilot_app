import pandas as pd
import numpy as np
import streamlit as st
import json
import matplotlib.pyplot as plt
import networkx as nx

st.set_page_config(page_title="Cold-Chain KM Pilot", layout="wide")
st.title("Cold-Chain Knowledge Management Pilot")

@st.cache_data
def load_data():
    shipments = pd.read_csv("data/shipments.csv", parse_dates=["origin_ts", "delivered_ts"])
    events = pd.read_csv("data/events.csv", parse_dates=["event_ts"], dtype=str)
    sensors = pd.read_csv("data/sensors.csv", parse_dates=["timestamp"], dtype=str)
    exceptions = pd.read_csv("data/exceptions.csv", parse_dates=["timestamp"], dtype=str)
    catalog = pd.read_csv("data/catalog_entities.csv")
    with open("contracts/contracts.json", "r") as f:
        contracts = json.load(f)
    # Cast sensor numerics after load so we can still display raw strings in Quality checks
    sensors["temp_c"] = pd.to_numeric(sensors["temp_c"], errors="coerce")
    sensors["battery_pct"] = pd.to_numeric(sensors["battery_pct"], errors="coerce")
    return shipments, events, sensors, exceptions, catalog, contracts

shipments, events, sensors, exceptions, catalog, contracts = load_data()
tabs = st.tabs(["Catalog", "Lineage", "Quality", "Analytics", "Exceptions"])

with tabs[0]:
    st.subheader("Catalog")
    st.write("Definitions, units, ranges, and sampling cadence for core entities.")
    st.dataframe(catalog, use_container_width=True)

with tabs[1]:
    st.subheader("Lineage (raw → curated → metrics)")
    st.write("A compact graph that depicts how raw files become curated tables and published metrics.")
    G = nx.DiGraph()
    G.add_edges_from([
        ("raw_shipments.csv", "curated_shipments"),
        ("raw_events.csv", "curated_events"),
        ("raw_sensors.csv", "curated_sensors"),
        ("raw_exceptions.csv", "curated_exceptions"),
        ("curated_shipments", "published_metrics"),
        ("curated_events", "published_metrics"),
        ("curated_sensors", "published_metrics"),
        ("curated_exceptions", "published_metrics"),
    ])
    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(6, 4))
    nx.draw(G, pos, with_labels=True, node_size=1200, font_size=8, ax=ax)
    st.pyplot(fig)

with tabs[2]:
    st.subheader("Data quality and contract checks")

    def check_required(df, req_cols):
        return [c for c in req_cols if c not in df.columns]

    def check_types(df, type_rules):
        problems = []
        for c, t in type_rules.items():
            if c not in df.columns:
                problems.append((c, "missing"))
                continue
            try:
                if t == "datetime":
                    pd.to_datetime(df[c])
                elif t == "float":
                    # robust numeric validation even if the column has mixed dtypes
                    pd.to_numeric(df[c], errors="raise")
                # treat "str" as pass-through
            except Exception:
                problems.append((c, f"type {t} cast failed"))
        return problems

    def check_unique(df, unique_cols):
        return {c: int(df[c].duplicated().sum()) if c in df.columns else None for c in unique_cols}

    def check_allowed_values(df, allowed):
        issues = []
        for c, vals in allowed.items():
            if c in df.columns:
                bad = ~df[c].isin(vals)
                count_bad = int(bad.sum())
                if count_bad > 0:
                    issues.append((c, count_bad))
        return issues

    def check_ranges(df, ranges):
        issues = []
        for c, rr in ranges.items():
            if c in df.columns:
                # coerce to numeric so comparisons never break on strings from the bad batch
                series = pd.to_numeric(df[c], errors="coerce")
                mn, mx = rr.get("min"), rr.get("max")
                bad = (series < mn) | (series > mx)
                count_bad = int(bad.fillna(False).sum())
                if count_bad > 0:
                    issues.append((c, count_bad))
        return issues

    datasets = {
        "shipments": shipments,
        "events": events,
        "sensors": pd.read_csv("data/sensors.csv"),  # use raw to test contract; cast separately for analytics
        "exceptions": exceptions
    }

    for name, df in datasets.items():
        st.markdown(f"### {name.capitalize()}")
        spec = contracts[name]
        missing = check_required(df, spec.get("required", []))
        types_bad = check_types(df, spec.get("types", {}))
        unique_bad = check_unique(df, spec.get("unique", []))
        allowed_bad = check_allowed_values(df, spec.get("allowed_values", {}))
        range_bad = check_ranges(df, spec.get("ranges", {}))

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Missing fields", len(missing))
        col2.metric("Type issues", len(types_bad))
        col3.metric("Duplicate key rows", sum(v for v in unique_bad.values() if v is not None))
        col4.metric("Bad categorical values", len(allowed_bad))
        col5.metric("Out-of-range values", len(range_bad))

        with st.expander("Details"):
            st.write({"missing_fields": missing})
            st.write({"type_issues": types_bad})
            st.write({"duplicate_counts": unique_bad})
            st.write({"disallowed_values": allowed_bad})
            st.write({"range_violations": range_bad})

with tabs[3]:
    st.subheader("Analytics")

    # Prepare flags safely using already cast numeric columns
    sensors["in_range"] = (sensors["temp_c"] >= 2.0) & (sensors["temp_c"] <= 8.0)

    excursion_by_shipment = sensors.groupby("shipment_id")["in_range"].apply(lambda x: (~x).any()).reset_index()
    excursion_rate = excursion_by_shipment["in_range"].mean()

    def mttd_for_shipment(sid):
        s = sensors[sensors["shipment_id"] == sid].sort_values("timestamp")
        origin = shipments.loc[shipments["shipment_id"] == sid, "origin_ts"]
        if origin.empty:
            return np.nan
        origin = pd.to_datetime(origin.values[0])
        out = s[~s["in_range"]]
        if out.empty:
            return np.nan
        return (out["timestamp"].iloc[0] - origin).total_seconds() / 3600.0

    def mtr_for_shipment(sid):
        s = sensors[sensors["shipment_id"] == sid].sort_values("timestamp")
        out = s[~s["in_range"]]
        if out.empty:
            return np.nan
        t0 = out["timestamp"].iloc[0]
        after = s[s["timestamp"] > t0]
        back_in = after[after["in_range"]]
        if back_in.empty:
            return np.nan
        return (back_in["timestamp"].iloc[0] - t0).total_seconds() / 3600.0

    mttd_vals = [mttd_for_shipment(sid) for sid in shipments["shipment_id"]]
    mtr_vals = [mtr_for_shipment(sid) for sid in shipments["shipment_id"]]
    mttd = np.nanmean(mttd_vals)
    mtr = np.nanmean(mtr_vals)

    shipments["duration_h"] = (shipments["delivered_ts"] - shipments["origin_ts"]).dt.total_seconds() / 3600.0
    on_time = (shipments["duration_h"] <= 96).mean()

    st.markdown("**Global KPIs**")
    st.caption("These reflect the entire lane in this synthetic dataset.")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Excursion rate (per shipment)", f"{excursion_rate:.2%}")
    col2.metric("Mean time to detect (hrs)", f"{mttd:.1f}" if not np.isnan(mttd) else "N/A")
    col3.metric("Mean time to resolve (hrs)", f"{mtr:.1f}" if not np.isnan(mtr) else "N/A")
    col4.metric("On-time arrival rate", f"{on_time:.2%}")

    st.markdown("**Control view for a sample shipment**")
    sid = st.selectbox("Choose shipment", shipments["shipment_id"].tolist())
    s_sel = sensors[sensors["shipment_id"] == sid].sort_values("timestamp")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(s_sel["timestamp"], s_sel["temp_c"], marker="o")
    ax.axhline(2.0)
    ax.axhline(8.0)
    ax.set_ylabel("temp_c")
    ax.set_xlabel("timestamp")
    st.pyplot(fig)

    # Selected shipment KPIs
    st.markdown("**Selected shipment KPIs**")
    in_range = (s_sel["temp_c"] >= 2.0) & (s_sel["temp_c"] <= 8.0)
    had_exc = (~in_range).any()

    def first_out_in(df_local):
        inr = (df_local["temp_c"] >= 2.0) & (df_local["temp_c"] <= 8.0)
        out = df_local[~inr]
        if len(out) == 0:
            return None, None
        t_out = out["timestamp"].iloc[0]
        after = df_local[df_local["timestamp"] > t_out]
        back_in = after[(after["temp_c"] >= 2.0) & (after["temp_c"] <= 8.0)]
        t_in = back_in["timestamp"].iloc[0] if len(back_in) else None
        return t_out, t_in

    t_out, t_in = first_out_in(s_sel)
    row = shipments[shipments["shipment_id"] == sid].iloc[0]
    origin = row["origin_ts"]
    duration_h = (row["delivered_ts"] - row["origin_ts"]).total_seconds() / 3600.0
    is_on_time = duration_h <= 96

    mttd_sel = (t_out - origin).total_seconds() / 3600.0 if t_out is not None else None
    mtr_sel = (t_in - t_out).total_seconds() / 3600.0 if (t_out is not None and t_in is not None) else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Had excursion", "Yes" if had_exc else "No")
    c2.metric("MTTD (hrs, selected)", f"{mttd_sel:.1f}" if mttd_sel is not None else "N/A")
    c3.metric("MTR (hrs, selected)", f"{mtr_sel:.1f}" if mtr_sel is not None else "N/A")
    c4.metric("On-time (selected)", "Yes" if is_on_time else "No")

with tabs[4]:
    st.subheader("Exception drill-through")
    if exceptions.empty:
        st.info("No exceptions recorded in this sample.")
    else:
        code = st.selectbox("Exception code", sorted(exceptions["exception_code"].unique()))
        subset = exceptions[exceptions["exception_code"] == code]
        st.write(subset[["exception_id", "shipment_id", "timestamp", "severity", "notes"]])
        affected = subset["shipment_id"].unique().tolist()
        st.write(f"Affected shipments: {len(affected)}")
        st.dataframe(
            shipments[shipments["shipment_id"].isin(affected)][
                ["shipment_id", "carrier", "origin_ts", "delivered_ts", "status"]
            ]
        )
