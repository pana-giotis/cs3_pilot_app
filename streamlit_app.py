
import pandas as pd
import numpy as np
import streamlit as st
import json
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime

st.set_page_config(page_title="Cold-Chain KM Pilot", layout="wide")

st.title("Cold-Chain Knowledge Management Pilot")

# Load data
@st.cache_data
def load_data():
    shipments = pd.read_csv("data/shipments.csv", parse_dates=["origin_ts","delivered_ts"])
    events = pd.read_csv("data/events.csv", parse_dates=["event_ts"])
    sensors = pd.read_csv("data/sensors.csv", parse_dates=["timestamp"])
    exceptions = pd.read_csv("data/exceptions.csv", parse_dates=["timestamp"])
    catalog = pd.read_csv("data/catalog_entities.csv")
    with open("contracts/contracts.json","r") as f:
        contracts = json.load(f)
    return shipments, events, sensors, exceptions, catalog, contracts

shipments, events, sensors, exceptions, catalog, contracts = load_data()

tabs = st.tabs(["Catalog", "Lineage", "Quality", "Analytics", "Exceptions"])

# ---------------- Catalog ----------------
with tabs[0]:
    st.subheader("Catalog")
    st.write("Definitions, units, ranges, and sampling cadence for core entities.")
    st.dataframe(catalog, use_container_width=True)
    st.markdown("**Glossary examples**")
    st.write({
        "shipment_id": "Unique identifier for a shipment",
        "event_type": "State change such as departed, arrived, handoff, accepted",
        "temp_c": "Temperature in Celsius for sensor readings"
    })

# ---------------- Lineage ----------------
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
    fig, ax = plt.subplots(figsize=(6,4))
    nx.draw(G, pos, with_labels=True, node_size=1200, font_size=8, ax=ax)
    st.pyplot(fig)

# ---------------- Quality ----------------
with tabs[2]:
    st.subheader("Data quality and contract checks")
    def check_required(df, req_cols):
        missing = [c for c in req_cols if c not in df.columns]
        return missing

    def check_types(df, type_rules):
        # simple type checks by column presence and cast attempt
        problems = []
        for c, t in type_rules.items():
            if c not in df.columns:
                problems.append((c, "missing"))
                continue
            if t == "datetime":
                try:
                    pd.to_datetime(df[c])
                except Exception:
                    problems.append((c, "not datetime parseable"))
            elif t == "float":
                if not np.issubdtype(df[c].dtype, np.number):
                    try:
                        df[c].astype(float)
                    except Exception:
                        problems.append((c, "not float"))
            elif t == "str":
                # accept anything that can be cast to string
                pass
        return problems

    def check_unique(df, unique_cols):
        dups = {}
        for c in unique_cols:
            dups[c] = int(df[c].duplicated().sum()) if c in df.columns else None
        return dups

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
                mn, mx = rr.get("min", None), rr.get("max", None)
                bad = ((df[c] < mn) | (df[c] > mx))
                count_bad = int(bad.sum())
                if count_bad > 0:
                    issues.append((c, count_bad))
        return issues

    datasets = {
        "shipments": shipments,
        "events": events,
        "sensors": sensors,
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
        col3.metric("Duplicate key rows", sum([v for v in unique_bad.values() if v is not None]))
        col4.metric("Bad categorical values", len(allowed_bad))
        col5.metric("Out-of-range values", len(range_bad))

        with st.expander("Details"):
            st.write({"missing_fields": missing})
            st.write({"type_issues": types_bad})
            st.write({"duplicate_counts": unique_bad})
            st.write({"disallowed_values": allowed_bad})
            st.write({"range_violations": range_bad})

# ---------------- Analytics ----------------
with tabs[3]:
    st.subheader("Analytics")
    # Excursion: temp outside 2-8C range
    sensors["in_range"] = (sensors["temp_c"] >= 2.0) & (sensors["temp_c"] <= 8.0)
    excursion_by_shipment = sensors.groupby("shipment_id")["in_range"].apply(lambda x: (~x).any()).reset_index()
    excursion_rate = excursion_by_shipment["in_range"].mean()

    # Mean time to detect: first out-of-range minus origin
    # Mean time to resolve: first return-to-range after the first out-of-range event
    def mttd_for_shipment(sid):
        s = sensors[sensors["shipment_id"] == sid].sort_values("timestamp")
        o = shipments.loc[shipments["shipment_id"] == sid, "origin_ts"].values
        if len(o) == 0:
            return np.nan
        origin = pd.to_datetime(o[0])
        out = s[~s["in_range"]]
        if len(out) == 0:
            return np.nan
        return (out["timestamp"].iloc[0] - origin).total_seconds()/3600.0

    mttd_vals = [mttd_for_shipment(sid) for sid in shipments["shipment_id"]]
    mttd = np.nanmean(mttd_vals)
    # MTR: time from first out-of-range to first return-in-range
    def mtr_for_shipment(sid):
        s = sensors[sensors["shipment_id"] == sid].sort_values("timestamp")
        out = s[~s["in_range"]]
        if len(out) == 0:
            return np.nan
        t0 = out["timestamp"].iloc[0]
        after = s[s["timestamp"] > t0]
        back_in = after[after["in_range"]]
        if len(back_in) == 0:
            return np.nan
        return (back_in["timestamp"].iloc[0] - t0).total_seconds()/3600.0

    mtr_vals = [mtr_for_shipment(sid) for sid in shipments["shipment_id"]]
    mtr = np.nanmean(mtr_vals)


    # On-time arrival: delivered within 4 days of origin
    shipments["duration_h"] = (shipments["delivered_ts"] - shipments["origin_ts"]).dt.total_seconds() / 3600.0
    on_time = (shipments["duration_h"] <= 96).mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Excursion rate (per shipment)", f"{excursion_rate:.2%}")
    col2.metric("Mean time to detect (hrs)", f"{mttd:.1f}")
    col3.metric("On-time arrival rate", f"{on_time:.2%}")

    # Simple control chart of temperature for a random shipment
    st.markdown("**Control view for a sample shipment**")
    sid = st.selectbox("Choose shipment", shipments["shipment_id"].tolist())
    s = sensors[sensors["shipment_id"] == sid].sort_values("timestamp")
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(s["timestamp"], s["temp_c"], marker="o")
    ax.axhline(2.0)
    ax.axhline(8.0)
    ax.set_ylabel("temp_c")
    ax.set_xlabel("timestamp")
    st.pyplot(fig)

# ---------------- Exceptions ----------------
with tabs[4]:
    st.subheader("Exception drill-through")
    if exceptions.empty:
        st.info("No exceptions recorded in this sample.")
    else:
        code = st.selectbox("Exception code", sorted(exceptions["exception_code"].unique()))
        subset = exceptions[exceptions["exception_code"] == code]
        st.write(subset[["exception_id","shipment_id","timestamp","severity","notes"]])
        # Show shipments list
        affected = subset["shipment_id"].unique().tolist()
        st.write(f"Affected shipments: {len(affected)}")
        st.dataframe(shipments[shipments['shipment_id'].isin(affected)][["shipment_id","carrier","origin_ts","delivered_ts","status"]])
