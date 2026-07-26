# Status Implementasi — FurnaceGuard AI

Peta antara isi `README.md` (spesifikasi) dan apa yang benar-benar sudah
dibangun. Diperbarui 26 Juli 2026.

---

## Roadmap README §22

| Versi | Cakupan | Status |
|---|---|---|
| 0.1 — Data Foundation | Upload XLSX, tag mapping, validasi, Data Quality Score, data sintetis, Event Registry | **Selesai**, kecuali antarmuka unggah |
| 0.2 — Risk Engine | Rule-based monitoring, rekayasa fitur, baseline deviation, dashboard | **Sebagian** — mesin dan API read-only selesai, dashboard belum |
| 0.3 — Machine Learning | Logistic Regression, Random Forest, XGBoost, Isolation Forest, validasi berbasis waktu, backtesting | **Selesai** di atas data sintetis |
| 0.4 — Explainability | SHAP, indikator dominan, penjelasan aturan, umpan balik operator | **Sebagian** — umpan balik operator belum |
| 0.5 — Historian Integration | OPC-UA, PI Historian, PostgreSQL, ingestion terjadwal, shadow mode | Belum |
| 1.0 — Production Advisory | Multi-user, RBAC, audit log, alur persetujuan model, mode advisory, pelaporan otomatis | Belum |

---

## Struktur folder README §16

| Jalur | Status | Keterangan |
|---|---|---|
| `backend/app/core/` | ada | konfigurasi, konstanta, logging |
| `backend/app/data/` | ada | `event_etl.py`, `synthetic.py`, `quality.py`, `schemas.py` |
| `backend/app/features/` | ada | `engineering.py`, `baseline.py` |
| `backend/app/models/` | ada | `train.py`, `evaluate.py`, `registry.py` |
| `backend/app/rules/` | ada | `event_classifier.py`, `risk_rules.py` |
| `backend/app/explainability/` | ada | `shap_explainer.py` |
| `backend/app/reports/` | ada | `event_analysis.py`, `risk_demo.py`, `final_report.py`, `viz.py` |
| `backend/app/api/` | ada | `deps.py` + router `health`, `events`, `models`, `risk` (semua read-only) |
| `backend/app/main.py` | ada | titik masuk FastAPI read-only/advisory, CORS, seluruh router terpasang |
| `backend/app/database/` | **belum** | belum ada basis data; registri event dan model dibaca dari berkas dan di-cache per proses |
| `backend/app/schemas/` | ada | skema Pydantic `events.py`, `models.py`, `risk.py` |
| `backend/datasets/` | ada | `processed/`, `synthetic/` |
| `backend/models/` | ada | artefak `.joblib`, baseline, registri |
| `backend/reports/` | ada | laporan, tabel evaluasi, `figures/` |
| `backend/tests/` | ada | 144 pengujian pipeline + `test_api.py` |
| `backend/requirements.txt` | ada | |
| `frontend/` | **belum** | React, fase berikutnya |
| `config/` | ada | lima berkas YAML |
| `notebooks/` | ada | 01 sampai 05 |
| `docs/` | ada | audit data, metodologi, batasan, status |
| `docker-compose.yml`, `Dockerfile` | **belum** | menyusul bersama API |

Modul ditulis langsung ke struktur README §16, sehingga penambahan
lapisan API tidak memerlukan pemindahan berkas.

---

## Target README §5

| Target | Status |
|---|---|
| `blocking_next_30m` | dilatih, PR-AUC di atas data sintetis |
| `blocking_next_60m` | dilatih, horizon utama |
| `blocking_next_180m` | dilatih |
| `main_bed_blocking_probability` | belum — label per lokasi belum dipisah |
| `front_aux_bed_disturbance_probability` | belum |
| `rear_aux_bed_disturbance_probability` | belum |
| `slag_pipe_1..4_blocking_probability` | **tidak dapat dilatih** — nol contoh di jurnal gangguan |
| `coal_feeder_blocking_probability` | belum sebagai target terpisah; rule engine sudah menanganinya |
| `return_system_disturbance_probability` | belum — hanya 3 event dalam sepuluh tahun |

Lokasi terduga saat ini ditentukan rule engine, bukan model per lokasi.
Dengan 116 event blocking terbagi ke enam kelas, jumlah contoh per lokasi
belum memadai untuk melatih model terpisah.

---

## Fitur README §9

Seluruh 22 fitur pada daftar README §9 diimplementasikan di
`backend/app/features/engineering.py`, dengan formula persis seperti
tertulis. Total 231 kolom fitur terbentuk setelah seluruh varian bergulir
dan simpangan baseline diperhitungkan.

---

## Metrik README §12

Seluruh 12 metrik dihitung dan dilaporkan di
`backend/reports/model_evaluation.csv`: precision, recall, F1, PR-AUC,
ROC-AUC, false alarms per day, missed event rate, event detection rate,
average dan median warning horizon, Brier score, calibration error.

---

## Batasan README §20 dan §21

| Ketentuan | Penegakan |
|---|---|
| Tidak mengubah interlock atau proteksi | Tidak ada jalur tulis ke sistem apa pun |
| Koneksi read-only ke historian | Mode deployment terkunci `offline`, diuji otomatis |
| API key di `.env` | `.env.example` disediakan, `.env` di-gitignore |
| Data sintetis dan aktual dipisah | Kolom `data_source` wajib; registri model menolak entri tanpa label |
| Catat versi model aktif | `backend/models/model_registry.json` |
| Catat perubahan threshold | Kolom `version`, `reviewed_by`, `reviewed_date` di `thresholds.yaml` |

---

## Jejak penyimpanan

Menjalankan pipeline penuh menghasilkan sekitar **2,2 GB** keluaran, dan
seluruhnya sudah masuk `.gitignore`:

| Direktori | Ukuran | Isi |
|---|---:|---|
| `backend/datasets/synthetic/` | ~765 MB | timeseries sintetis 2020–2026, satu Parquet per tahun |
| `backend/datasets/processed/` | ~1,4 GB | Event Registry, serta cache fitur validasi dan uji yang dialirkan dari disk saat pelatihan |
| `backend/models/` | ~22 MB | artefak model, baseline, registri |
| `backend/reports/` | <1 MB | laporan, tabel evaluasi, grafik |

Cache fitur pada `processed/` dapat dihapus kapan saja; pelatihan akan
membangunnya ulang.

---

## Perintah

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt

python -m backend.app.data.event_etl --cross-check
python -m backend.app.rules.event_classifier
python -m backend.app.reports.event_analysis
python -m backend.app.data.synthetic --years 2020-2026
python -m backend.app.models.train
python -m backend.app.reports.risk_demo
python -m backend.app.reports.final_report

pytest backend/tests -q

uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Catatan: `README.md` §18 menyebut `git clone` dan `npm install` untuk
frontend. Keduanya belum berlaku pada fase ini — belum ada repositori
remote maupun frontend.
