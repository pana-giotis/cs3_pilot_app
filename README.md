# Cold-Chain Knowledge Management Pilot — Use Case

This repository contains a lightweight pilot that demonstrates a **knowledge management design** for a cold-chain lane with a single biologic and one route. It focuses on cataloged definitions, governed lineage, simple contract-based validation, and a small analytics slice that turns raw events and sensor readings into explainable metrics.

## Purpose and Scope
- **Approach**: Use Case. We create and govern the data needed for a narrow pilot rather than relying on a public dataset.
- **Scope**: One biologic, one lane, approximately ninety days of synthetic operations.
- **Objectives**: Build an explainable shipment narrative per order and prove that catalog, lineage, and policy-aware checks support reproducible analytics.

## KM Design
- **Catalog**: Entities, definitions, units, acceptable ranges, and sampling cadence. The catalog serves as the contract of record for shared vocabulary.
- **Data Contracts**: JSON rules that specify required fields, basic types, allowed values, ranges, and key uniqueness.
- **Lineage**: A compact graph from raw CSVs to curated tables and published metrics. Lineage is used for reproducibility and auditability.
- **Access Model**: The app presents **scoped views** via Streamlit tabs. Operators see KPIs while investigators and quality leads can trace events end to end.
- **Exception Taxonomy**: A governed dictionary of exception codes that stabilizes trend analysis and accountability. The app includes drill-through to affected shipments and CSV export.

## Metrics and Analytics
- **Excursion rate (per shipment)**: Whether any temperature reading fell outside the **2–8 °C** range.
- **Mean time to detect (MTTD)**: Hours from shipment origin to the first out-of-range reading.
- **Mean time to resolve (MTR)**: Hours from first out-of-range reading to the first return-to-range reading.
- **On-time arrival rate**: Percent of shipments delivered within **four days** of origin.
- **Control view**: Time series of temperature for a selected shipment with 2–8 °C bounds, plus **per-shipment** KPIs (Had excursion, MTTD, MTR, On-time).

## Artifacts that Map to CS3 Deliverables
- **Architecture Blueprint**: `docs/architecture.png`
- **Tool Link**: Deploy `streamlit_app.py` on Streamlit Community Cloud to produce a public URL.
- **Screenshots PDF**: A printable proof-of-work showing Catalog, Lineage, Quality (with sample offending rows), Analytics, and Exceptions.
- **Synthetic Data & Contracts**: `data/` CSVs and `contracts/contracts.json` (includes a tiny **bad batch** to demonstrate validator hits)

## How to Use
1. Explore the **Catalog** tab to review entities and definitions.
2. Open **Lineage** to see how raw inputs become curated tables and published metrics.
3. Check **Quality** for contract validations (missing fields, type issues, disallowed values, out-of-range). Expand **Details** to see sample offending rows or download the CSV.
4. Review **Analytics** for excursion rate, MTTD, MTR, and on-time arrival. Use the control view to inspect a selected shipment. Download the KPI summary or time series.
5. Use **Exceptions** to filter by exception code, drill through to rows, view affected shipments, and export CSVs.

## Run Locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
A small dependency set keeps the pilot easy to inspect and reproduce.

## Notes
- Data is synthetic and safe to share.
- The lineage graph is illustrative and can be replaced later by exports from an enterprise catalog or lineage tool.
- The contracts demonstrate policy-aware checks. They can be extended with additional business rules.
