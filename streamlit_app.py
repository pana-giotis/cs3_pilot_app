import pandas as pd
import numpy as np
import streamlit as st
import json
import matplotlib.pyplot as plt
import networkx as nx
from io import BytesIO

st.set_page_config(page_title="Cold-Chain KM Pilot", layout="wide")
st.title("Cold-Chain Knowledge Management Pilot")

@st.cache_data
def load_data():
    # RAW for Quality
    shipments_raw = pd.read_csv("data/shipments.csv", dtype=str)
    events_raw    = pd.read_csv("data/events.csv", dtype=str)
    sensors_raw   = pd.read_csv("data/sensors.csv", dtype=str)
    exceptions_raw= pd.read_csv("data/exceptions.csv", dtype=str)
    catalog       = pd.read_csv("data/catalog_entities.csv")
    with open("contracts/contracts.json", "r") as f:
        contracts = json.load(f)

    # Casted for Analytics
    shipments = shipments_raw.copy()
    shipments["origin_ts"]   = pd.to_datetime(shipments["origin_ts"],   errors="coerce")
    shipments["delivered_ts"]= pd.to_datetime(shipments["delivered_ts"],errors="coerce")

    sensors = sensors_raw.copy()
    sensors["timestamp"]   = pd.to_datetime(sensors["timestamp"], errors="coerce")
    sensors["temp_c"]      = pd.to_numeric(sensors["temp_c"], errors="coerce")
    sensors["battery_pct"] = pd.to_numeric(sensors["battery_pct"], errors="coerce")

    exceptions = exceptions_raw.copy()
    if "timestamp" in exceptions.columns:
        exceptions["timestamp"] = pd.to_datetime(exceptions["timestamp"], errors="coerce")

    return (shipments_raw, events_raw, sensors_raw, exceptions_raw,
            shipments, sensors, exceptions, catalog, contracts)

(shipments_raw, events_raw, sensors_raw, exceptions_raw,
 shipments, sensors, exceptions, catalog, contracts) = load_data()

tabs = st.tabs(["Catalog", "Lineage", "Quality", "Analytics", "Exceptions"])

# Catalog
with tabs[0]:
    st.subheader("Catalog")
    st.dataframe(catalog, use_container_width=True)

    st.download_button(
        "Download catalog.csv",
        data=catalog.to_csv(index=False),
        file_name="catalog_entities.csv",
        mime="text/csv",
        use_container_width=True
    )

# Lineage
with tabs[1]:
    st.subheader("Lineage (raw → curated → metrics)")
    st.write("Illustrative graph of raw files → curated tables → published metrics.")

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
    fig, ax = plt.subplots(figsize=(8.5, 6))   # larger canvas
    nx.draw(G, pos, with_labels=True, node_size=1400, font_size=9, ax=ax)
    plt.margins(0.20)                          # add margins to avoid clipping
    plt.tight_layout(pad=1.0)
    st.pyplot(fig, use_container_width=True)

    # Optional: download the lineage figure as PNG
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    st.download_button("Download lineage.png", data=buf.getvalue(),
                       file_name="lineage.png", mime="image/png",
                       use_container_width=True)

# Quality
with tabs[2]:
    st.subheader("Data quality and contract checks")

    def check_required(df, req_cols):
        return [c for c in req_cols if c not in df.columns]

    def check_types(df, type_rules):
        problems = []
        bad_rows_per_col = {}
        for c, t in type_rules.items():
            if c not in df.columns:
                problems.append((c, "missing"))
                bad_rows_per_col[c] = df.head(0)
                continue
            try:
                if t == "datetime":
                    cast = pd.to_datetime(df[c], errors="coerce")
                    bad_mask = cast.isna() & df[c].notna()
                    bad_rows_per_col[c] = df[bad_mask].head(10)
                    if bad_mask.any():
                        raise ValueError("datetime cast failed")
                elif t == "float":
                    cast = pd.to_numeric(df[c], errors="coerce")
                    bad_mask = cast.isna() & df[c].notna()
                    bad_rows_per_col[c] = df[bad_mask].head(10)
                    if bad_mask.any():
                        raise ValueError("float cast failed")
            except Exception:
                problems.append((c, f"type {t} cast failed"))
        return problems, bad_rows_per_col

    def check_unique(df, unique_cols):
        dup_counts = {}
        dup_rows_per_col = {}
        for c in unique_cols:
            if c in df.columns:
                dmask = df[c].duplicated(keep=False)
                dup_counts[c] = int(dmask.sum())
                dup_rows_per_col[c] = df[dmask].head(10)
            else:
                dup_counts[c] = None
                dup_rows_per_col[c] = df.head(0)
        return dup_counts, dup_rows_per_col

    def check_allowed_values(df, allowed):
        issues = []
        bad_rows_per_col = {}
        for c, vals in allowed.items():
            if c in df.columns:
                bad_mask = ~df[c].isin(vals)
                cnt = int(bad_mask.sum())
                if cnt > 0:
                    issues.append((c, cnt))
                bad_rows_per_col[c] = df[bad_mask].head(10)
            else:
                bad_rows_per_col[c] = df.head(0)
        return issues, bad_rows_per_col

    def check_ranges(df, ranges):
        issues = []
        bad_rows_per_col = {}
        for c, rr in ranges.items():
            if c in df.columns:
                ser = pd.to_numeric(df[c], errors="coerce")
                mn, mx = rr.get("min"), rr.get("max")
                bad_mask = (ser < mn) | (ser > mx)
                cnt = int(bad_mask.fillna(False).sum())
                if cnt > 0:
                    issues.append((c, cnt))
                bad_rows_per_col[c] = df[bad_mask.fillna(False)].head(10)
            else:
                bad_rows_per_col[c] = df.head(0)
        return issues, bad_rows_per_col

    datasets = {
        "shipments": shipments_raw,
        "events": events_raw,
        "sensors": sensors_raw,
        "exceptions": exceptions_raw
    }

    for name, df in datasets.items():
        st.markdown(f"### {name.capitalize()}")
        spec = contracts[name]

        missing = check_required(df, spec.get("required", []))
        types_bad, type_rows = check_types(df, spec.get("types", {}))
        unique_bad, dup_rows  = check_unique(df, spec.get("unique", []))
        allowed_bad, allowed_rows = check_allowed_values(df, spec.get("allowed_values", {}))
        range_bad, range_rows = check_ranges(df, spec.get("ranges", {}))

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Missing fields", len(missing))
        c2.metric("Type issues", len(types_bad))
        c3.metric("Duplicate key rows", sum(v for v in unique_bad.values() if v is not None))
        c4.metric("Bad categorical values", len(allowed_bad))
        c5.metric("Out-of-range values", len(range_bad))

        # CSV download of the raw dataset shown to the validator
        st.download_button(
            f"Download {name}.csv",
            data=df.to_csv(index=False),
            file_name=f"{name}.csv",
            mime="text/csv",
            use_container_width=True
        )

        with st.expander("Details (including sample offending rows)"):
            st.write({"missing_fields": missing})
            st.write({"type_issues": types_bad})
            st.write({"duplicate_counts": unique_bad})
            st.write({"disallowed_values": allowed_bad})
            st.write({"range_violations": range_bad})

            # Show small samples for clarity
            if types_bad:
                st.markdown("**Type issue samples**")
                for col, sample in type_rows.items():
                    if not sample.empty:
                        st.caption(f"Column: `{col}`")
                        st.dataframe(sample, use_container_width=True)
            if unique_bad:
                st.markdown("**Duplicate key samples**")
                for col, sample in dup_rows.items():
                    if sample is not None and not sample.empty:
                        st.caption(f"Unique key: `{col}`")
                        st.dataframe(sample, use_container_width=True)
            if allowed_bad:
                st.markdown("**Disallowed value samples**")
                for col, sample in allowed_rows.items():
                    if not sample.empty:
                        st.caption(f"Column: `{col}`")
                        st.dataframe(sample, use_container_width=True)
            if range_bad:
                st.markdown("**Out-of-range samples**")
                for col, sample in range_rows.items():
                    if not sample.empty:
                        st.caption(f"Column: `{col}`")
                        st.dataframe(sample, use_container_width=True)

# Analytics
with tabs[3]:
    st.subheader("Analytics")

    sensors_valid   = sensors.dropna(subset=["timestamp"]).copy()
    shipments_valid = shipments.dropna(subset=["origin_ts", "delivered_ts"]).copy()

    if sensors_valid.empty or shipments_valid.empty:
        st.warning("Insufficient valid data for analytics due to invalid timestamps in the bad batch.")
    else:
        sensors_valid["in_range"] = (sensors_valid["temp_c"] >= 2.0) & (sensors_valid["temp_c"] <= 8.0)

        excursion_by_ship = sensors_valid.groupby("shipment_id")["in_range"].apply(lambda x: (~x).any()).reset_index()
        excursion_rate = excursion_by_ship["in_range"].mean() if not excursion_by_ship.empty else np.nan

        def mttd_for_shipment(sid):
            s = sensors_valid[sensors_valid["shipment_id"] == sid].sort_values("timestamp")
            if s.empty: return np.nan
            origin = shipments_valid.loc[shipments_valid["shipment_id"] == sid, "origin_ts"]
            if origin.empty: return np.nan
            origin = origin.iloc[0]
            out = s[~s["in_range"]]
            if out.empty: return np.nan
            return (out["timestamp"].iloc[0] - origin).total_seconds() / 3600.0

        def mtr_for_shipment(sid):
            s = sensors_valid[sensors_valid["shipment_id"] == sid].sort_values("timestamp")
            if s.empty: return np.nan
            out = s[~s["in_range"]]
            if out.empty: return np.nan
            t0 = out["timestamp"].iloc[0]
            after = s[s["timestamp"] > t0]
            back_in = after[after["in_range"]]
            if back_in.empty: return np.nan
            return (back_in["timestamp"].iloc[0] - t0).total_seconds() / 3600.0

        mttd = np.nanmean([mttd_for_shipment(sid) for sid in shipments_valid["shipment_id"]])
        mtr  = np.nanmean([mtr_for_shipment(sid) for sid in shipments_valid["shipment_id"]])

        shipments_valid["duration_h"] = (shipments_valid["delivered_ts"] - shipments_valid["origin_ts"]).dt.total_seconds() / 3600.0
        on_time = (shipments_valid["duration_h"] <= 96).mean()

        st.markdown("**Global KPIs**")
        st.caption("These reflect the entire lane in this synthetic dataset.")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Excursion rate (per shipment)", f"{excursion_rate:.2%}" if not np.isnan(excursion_rate) else "N/A")
        col2.metric("Mean time to detect (hrs)", f"{mttd:.1f}" if not np.isnan(mttd) else "N/A")
        col3.metric("Mean time to resolve (hrs)", f"{mtr:.1f}" if not np.isnan(mtr) else "N/A")
        col4.metric("On-time arrival rate", f"{on_time:.2%}")

        # Allow to download KPI snapshot
        kpi_df = pd.DataFrame([{
            "excursion_rate": excursion_rate,
            "mttd_hours": mttd,
            "mtr_hours": mtr,
            "on_time_rate": on_time
        }])
        st.download_button(
            "Download KPI summary (csv)",
            data=kpi_df.to_csv(index=False),
            file_name="kpi_summary.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.markdown("**Control view for a sample shipment**")
        sid = st.selectbox("Choose shipment", shipments_valid["shipment_id"].tolist())
        s_sel = sensors_valid[sensors_valid["shipment_id"] == sid].sort_values("timestamp")
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(s_sel["timestamp"], s_sel["temp_c"], marker="o")
        ax.axhline(2.0); ax.axhline(8.0)
        ax.set_ylabel("temp_c"); ax.set_xlabel("timestamp")
        st.pyplot(fig)

        # Download selected shipment time series
        st.download_button(
            "Download selected shipment series (csv)",
            data=s_sel[["shipment_id","timestamp","temp_c","battery_pct"]].to_csv(index=False),
            file_name=f"{sid}_timeseries.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Selected shipment KPIs
        st.markdown("**Selected shipment KPIs**")
        in_range = (s_sel["temp_c"] >= 2.0) & (s_sel["temp_c"] <= 8.0)
        had_exc = (~in_range).any()

        def first_out_in(df_local):
            inr = (df_local["temp_c"] >= 2.0) & (df_local["temp_c"] <= 8.0)
            out = df_local[~inr]
            if len(out) == 0: return None, None
            t_out = out["timestamp"].iloc[0]
            after = df_local[df_local["timestamp"] > t_out]
            back_in = after[(after["temp_c"] >= 2.0) & (after["temp_c"] <= 8.0)]
            t_in = back_in["timestamp"].iloc[0] if len(back_in) else None
            return t_out, t_in

        t_out, t_in = first_out_in(s_sel)
        row = shipments_valid[shipments_valid["shipment_id"] == sid].iloc[0]
        origin = row["origin_ts"]
        duration_h = (row["delivered_ts"] - row["origin_ts"]).total_seconds() / 3600.0
        is_on_time = duration_h <= 96

        mttd_sel = (t_out - origin).total_seconds() / 3600.0 if t_out is not None else None
        mtr_sel  = (t_in - t_out).total_seconds() / 3600.0 if (t_out is not None and t_in is not None) else None

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Had excursion", "Yes" if had_exc else "No")
        c2.metric("MTTD (hrs, selected)", f"{mttd_sel:.1f}" if mttd_sel is not None else "N/A")
        c3.metric("MTR (hrs, selected)", f"{mtr_sel:.1f}" if mtr_sel is not None else "N/A")
        c4.metric("On-time (selected)", "Yes" if is_on_time else "No")

# Exceptions
with tabs[4]:
    st.subheader("Exception drill-through")
    if exceptions.empty:
        st.info("No exceptions recorded in this sample.")
    else:
        code = st.selectbox("Exception code", sorted(exceptions["exception_code"].dropna().unique()))
        subset = exceptions[exceptions["exception_code"] == code].copy()
        st.dataframe(subset[["exception_id","shipment_id","timestamp","severity","notes"]],
                     use_container_width=True)

        affected = subset["shipment_id"].dropna().unique().tolist()
        st.write(f"Affected shipments: {len(affected)}")
        affected_df = shipments.loc[
            shipments["shipment_id"].isin(affected),
            ["shipment_id","carrier","origin_ts","delivered_ts","status"]
        ]
        st.dataframe(affected_df, use_container_width=True)

        # Downloads
        st.download_button(
            "Download exception subset (csv)",
            data=subset.to_csv(index=False),
            file_name=f"exceptions_{code}.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.download_button(
            "Download affected shipments (csv)",
            data=affected_df.to_csv(index=False),
            file_name=f"affected_shipments_{code}.csv",
            mime="text/csv",
            use_container_width=True
        )
