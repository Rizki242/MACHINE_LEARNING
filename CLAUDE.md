# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

FurnaceGuard AI — ML-based early warning system for furnace/boiler blocking at PLTU Jeranjang Unit 1 (25 MW CFB boiler). A decision-support system, not a control system: it only reads data and produces advisory risk scores, never writes to plant equipment. Full domain spec lives in `README.md` (23 sections, referenced throughout the code as "README §N") — read it before touching rules, features, or targets. `docs/STATUS.md` maps README spec to what's actually built; `docs/LIMITATIONS.md` states what must never be claimed from current results.

**The single most important constraint on this codebase: every model is currently trained on programmatically generated synthetic data.** Zero actual DCS timeseries exist in this repo. No performance number (PR-AUC, recall, false alarms/day, etc.) may ever be presented without stating it comes from synthetic data — see `SYNTHETIC_METRIC_WARNING` in `backend/app/core/constants.py`, which is force-attached to every model registry entry trained on synthetic data.

Code comments, docstrings, and docs in this repo are written in Indonesian — match that convention when editing existing modules.

## Commands

Run everything from the repository root (`D:\MACHINE_LEARNING`), not from `backend/`. Imports are rooted at `backend.app.*`, and pytest/config path resolution assume the repo root as cwd.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
```

Full pipeline, in dependency order:

```powershell
python -m backend.app.data.event_etl --cross-check       # parse XLSX journals -> event registry
python -m backend.app.rules.event_classifier               # classify events (keyword + text-model layers)
python -m backend.app.reports.event_analysis                # Pareto / trend analysis over the registry
python -m backend.app.data.synthetic --years 2020-2026      # generate synthetic DCS timeseries
python -m backend.app.models.train                          # train + calibrate + evaluate all models
python -m backend.app.reports.risk_demo                     # run hybrid risk engine over a demo slice
python -m backend.app.reports.final_report                  # assemble FURNACEGUARD_ANALYSIS_REPORT.md
```

Tests:

```powershell
pytest backend/tests -q                      # full suite (144 tests)
pytest backend/tests/test_features.py -q     # single file
pytest backend/tests/test_features.py::test_rolling_mean_does_not_see_future -q   # single test
```

Train a single horizon instead of all three:

```powershell
python -m backend.app.models.train --horizon blocking_next_60m
```

Run the API (from the repo root, same import-path reason as above):

```powershell
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

`ruff` is configured in `pyproject.toml` (`select = ["E", "F", "I", "UP", "B"]`)
but isn't listed as a dependency anywhere yet — install it separately
(`pip install ruff`) before relying on `ruff check .`. There is still no
frontend — don't invent commands for that.

## Architecture

### Config is centralized, never read ad hoc

`backend/app/core/config.py` loads every YAML file under `config/` exactly once via the cached `get_settings()`. No other module reads YAML directly. `Settings` is a frozen dataclass exposing typed helpers (`load_zone_for`, `risk_band_for`, `standard_names`, `dcs_tag_lookup`, etc.). Directory paths (`Paths`) are overridable via env vars (`FG_RAW_DATA_DIR`, `FG_DATASETS_DIR`, ...; see `.env.example`). `_validate()` runs at load time and fails fast on inconsistent config (load zones must tile 0–100% contiguously, risk bands must span 0–100, hybrid weights must sum to 1.0).

`backend/app/core/constants.py` holds only things that require a code change to alter (unit IDs, source sheet names, output schemas). Thresholds and model parameters are deliberately kept out of Python and live in `config/*.yaml` (`thresholds.yaml`, `units.yaml`, `tag_mapping.yaml`, `event_taxonomy.yaml`, `model_config.yaml`) so a boiler engineer can tune them without touching code.

### Pipeline stages (each stage is a module, run in order)

1. **`backend/app/data/event_etl.py`** — parses the two raw XLSX disturbance journals (`Derating`/`Outage` sheets, hard-coded sheet names with trailing spaces — do not `.strip()` them, `pd.read_excel` matches literally) into the unified Event Registry schema (`EVENT_REGISTRY_COLUMNS` in `constants.py`, per README §15).
2. **`backend/app/rules/event_classifier.py`** — two-layer classification of free-text Indonesian event descriptions into `event_type`/`event_location`/`severity`. Layer 1: deterministic keyword rules from `config/event_taxonomy.yaml`. Layer 2: char n-gram TF-IDF + Logistic Regression trained on layer-1 labels as weak supervision, used only to catch rows layer 1 missed (typos are common in source text: `funace`, `batuabra`). Disagreements are flagged `needs_review` and exported to `backend/reports/events_needing_review.csv` — nothing is silently dropped.
3. **`backend/app/data/synthetic.py`** — generates synthetic per-minute DCS timeseries from the boiler design spec, injecting degradation signatures before each real event's start time. This is a hypothesis about failure signatures, not an observation — see `docs/LIMITATIONS.md` §1.
4. **`backend/app/features/engineering.py` / `baseline.py`** — implements all 22 README §9 features (231 columns after rolling variants). **Hard rule: every rolling/derived feature is backward-looking only** — no feature at time `t` may see a sample after `t`. This is enforced by `model_config.yaml: features.backward_only` (raises if false) and tested directly in `backend/tests/test_features.py` using step signals. Baselines are per load-zone (README §10: startup/low/medium/high/near-rated) AND corrected for load *within* the zone via a fitted trend line — a single per-zone mean would make a unit at the low edge of a zone look deviant on almost everything just from its load position. Baseline fit excludes windows around known events, using MAD-based (outlier-robust) dispersion.
5. **`backend/app/models/train.py`** — trains XGBoost / Logistic Regression / Decision Tree / Random Forest per horizon (`blocking_next_30m/60m/180m`) plus an Isolation Forest anomaly detector trained only on normal (non-event) periods. Key correctness rules baked into this module:
   - Time-based train/validation/test split with an **embargo** gap at each boundary (not random split) — prevents rolling-feature leakage across the split.
   - Only the **training set** is subsampled for class balance; validation/test stay at true prevalence, because false-alarm-rate and threshold selection are meaningless if measured on an artificially balanced set.
   - Calibration (`CalibratedClassifierCV`) is fit on the *validation* set, never on training data the model has already seen.
   - Large validation/test feature sets are streamed to Parquet and read back in chunks (`Dataset.batches()`) rather than held in memory.
   - Decision thresholds are chosen on the validation set against a daily false-alarm budget (`choose_threshold` in `evaluate.py`).
6. **`backend/app/rules/risk_rules.py`** — the parallel rule engine (README §11's "Hybrid Risk Engine"). Rules read entirely from `thresholds.yaml`; rule scores take the **max** across triggered rules (not sum — one clear disturbance shouldn't be outvoted by several mild unrelated ones). `combine_scores()` blends rule/ML/anomaly scores by configured weights, redistributing weight if a source is unavailable. `probability_to_risk()` re-centers model probabilities around the decision threshold in log-odds space so rare-event probabilities (which sit near 0 even when correct) still populate the full 0–100 Risk Score range. Confidence Score is computed separately from Risk Score (README §13) and is penalized when Data Quality Score is low — a decisive prediction on bad data isn't confident.
7. **`backend/app/explainability/shap_explainer.py`**, **`backend/app/reports/*.py`** — SHAP explanations and report/figure generation (Pareto charts, event analysis, final markdown report, risk engine demo output).
8. **`backend/app/api/`** + **`backend/app/main.py`** — a read-only FastAPI layer over everything above (events, model registry, hybrid risk assessments). It does not add a database or historian connection; event/model registries load from the existing Parquet/JSON files once per process (`backend/app/api/deps.py`, `lru_cache`-backed) and require a process restart to see a re-run pipeline's output. The `/risk/*` routes reuse `risk_demo.py`'s `load_artifacts` → `prepare_period` → `score_period` building blocks and `risk_rules.assess_row()` directly rather than reimplementing scoring — don't duplicate that logic in a route handler. There's still no database (`backend/app/database/` doesn't exist).

Everything upstream of the API is still orchestrated through plain scripts/modules with `if __name__ == "__main__"` entry points, runnable via `python -m backend.app....`.

### Data source separation is load-bearing, not cosmetic

Every row and every model registry entry carries a `data_source` column/field (`synthetic` vs `actual`, see `constants.py`). `backend/app/models/registry.py`'s `ModelRegistry.add()` **raises** if an entry has no `data_source` label, and auto-appends the synthetic warning note when the source is synthetic. When adding new data ingestion or model-saving code, preserve this — it's the mechanism that prevents a synthetic-trained model from ever being mistaken for a field-validated one.

### Known permanent gaps (don't try to "fix" these — they're documented data limitations, not bugs)

- Slag pipe blocking targets (`slag_pipe_1..4_blocking_probability`) have **zero** training examples in ten years of journals — cannot be trained, only rule-covered.
- `return_system_disturbance_probability` has only 3 events in ten years — same problem, lesser degree.
- Location-specific probabilities (`main_bed_blocking_probability` etc.) aren't modeled separately; suspected location currently comes from the rule engine only (`suspected_area()` in `risk_rules.py`), because 116 blocking events split across six classes isn't enough per-location signal.
- Startup periods are excluded from training (`is_running == 1` filter), so blocking risk during startup — actually a higher-risk period for CFB boilers — isn't modeled.

Full list: `docs/LIMITATIONS.md`.
