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
- **Access Model**: The app shows role-based views conceptually. Operators would see KPIs while investigators and quality leads would use end-to-end trace.
- **Exception Taxonomy**: A governed dictionary of exception codes that stabilizes trend analysis and accountability.

## Metrics and Analytics
- **Excursion rate (per shipment)**: Whether any temperature reading fell outside the 2–8 °C range.
- **Mean time to detect (MTTD)**: Hours from shipment origin to first out-of-range reading.
- **Mean time to resolve (MTR)**: Hours from first out-of-range reading to first return-to-range reading.
- **On-time arrival rate**: Percent of shipments delivered within four days of origin.
- **Control view**: Time series of temperature for a selected shipment with 2–8 °C bounds.

## Artifacts that Map to CS3 Deliverables
- **Architecture Blueprint**: `docs/architecture.png`
- **Tool Link**: Deploy `streamlit_app.py` on Streamlit Community Cloud to produce a public URL.
- **Screenshots PDF**: A printable proof-of-work showing Catalog, Lineage, Quality results, Analytics, and Exceptions.
- **Synthetic Data & Contracts**: `data/` CSVs and `contracts/contracts.json`

## How to Use
1. Explore the **Catalog** tab to review entities and definitions.
2. Open **Lineage** to see how raw inputs become curated tables and published metrics.
3. Check **Quality** for contract validations such as missing fields, type issues, and value ranges.
4. Review **Analytics** for excursion rate, MTTD, MTR, and on-time arrival with a control view.
5. Use **Exceptions** to drill through by exception code and inspect affected shipments.

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
