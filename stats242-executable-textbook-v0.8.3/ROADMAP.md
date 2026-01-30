# ROADMAP

This repository is an “executable textbook” for STATS 242, organized as:

- **Core lane**: fast, reproducible chapter runners that produce HTML + figures + machine-readable artifacts.
- **Principia lane**: deeper, thesis-style derivations and simulation studies that complement Core.

Each chapter run emits:
- `report.md`, `report.html`
- `figures/*.png`
- `artifacts/run_meta.json` (provenance: parameters, inputs + hashes, environment)

Book builds additionally emit:
- `manifest.json` (chapter list + outputs)
- `build_meta.json` (book-level provenance)

---

## Current state (v0.8.3)

### Core
- Chapter registry + `run_all` book builder.
- HTML export + charts.
- Chapter 01 primer + data contracts.
- Dataset validation CLI: `python -m stats242xtb.tools.validate`.
- Provenance: `artifacts/run_meta.json` per chapter + `build_meta.json` per book.
- Chapters **01–04** included in the full build.
- Chapters **05–07** upgraded to “research-grade” (standard metrics, turnover, cost sensitivity, portable summaries).

### Principia
- Principia Chapter 04 inference module (robust/HAC emphasis) integrated via registry.

---

## Near-term plan

### v0.8.4 — Microstructure “research-grade” (Ch 08–12)
**Goal:** Make the microstructure block empirically credible while remaining runnable on the toy suite.

Deliverables:
1. **Ch 08 (Durations / ACD)**
   - Seasonality adjustment (intraday bins) + residual diagnostics.
   - ACD(1,1) baseline with clean extensibility points.

2. **Ch 09 (Spread measures + signature plot)**
   - Quoted/effective/realized spread computations (when trades + quotes exist).
   - Realized variance vs sampling frequency “signature plot” scaffold.

3. **Ch 10 (Roll model)**
   - Roll spread estimation + comparison table vs quote-based spreads (if available).
   - Robustness toggles (sampling choices, winsorization/outlier handling).

4. **Ch 11 (Adverse selection / impact proxy)**
   - Kyle-lambda style regressions: Δmid on signed volume / order flow imbalance.
   - Bucketed response curves and stability checks.

5. **Ch 12 (LOB queueing baseline)**
   - L1 queue birth–death rate estimation from size deltas.
   - Depletion probability proxy → predicted next move probability report.

Definition of done:
- Each chapter emits plots + `artifacts/summary.json` + provenance.
- A book build containing Ch 08–12 completes successfully on the toy suite.

---

### v0.8.5 — Execution + workflow (Ch 13–14)
**Goal:** Turn execution into a real Almgren–Chriss lab, and make Ch 14 the canonical “how to extend this book” template.

Deliverables:
1. **Ch 13 (Almgren–Chriss)**
   - Closed-form schedule (baseline case) + parameter sweeps.
   - Schedule table + expected cost/variance decomposition.

2. **Ch 14 (Robust workflow)**
   - Experiment template: one config → one reproducible run folder.
   - Standardized “cost sweep + turnover + drawdown + stability checks” recipe.
   - Guidance for swapping toy data → real datasets via contracts/adapters.

---

### v0.8.6 — Data layer hardening (TAQ/LOB readiness)
**Goal:** Make canonicalization and validation “adult-grade” for large real exports.

Deliverables:
- Contract/validator expansion: locked/crossed markets, negative spreads, gap reports, size anomalies, session filters.
- Adapter upgrades: timezone rules, symbology, side conventions; optional Parquet; chunked reads.
- `tools.validate` gains optional “write canonicalized output” mode.

Definition of done:
- One documented real-ish import path runs end-to-end (even if anonymized).

---

## Principia expansion pack

### v0.9 — Principia becomes a full companion
Targets:
- Principia Ch 08 (ACD derivations + diagnostics).
- Principia Ch 12 (birth–death derivation + simulation comparison).
- Principia Ch 13 (AC derivation + calibration notes).
- Expand Principia Ch 04 (overlapping returns / inference distortion demos).

Definition of done:
- A coherent Principia HTML book build succeeds and is meaningfully deeper than Core.

---

## Stabilization and productization

### v1.0 — “Executable textbook” as a durable tool
Deliverables:
- Per-chapter unit tests reintroduced + deterministic seeds.
- CLI UX polish (book build / chapter run entry points).
- Reporting table consistency across chapters.
- Contributor docs: how to add a chapter, how to add a dataset adapter, how to interpret artifacts.
- Packaging polish (entry points, version stamping in outputs).

Definition of done:
- Fresh clone → install → toy data → full book build passes reliably.
