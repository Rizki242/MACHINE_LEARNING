# FurnaceGuard AI — PLTU Jeranjang Unit 1

**Machine Learning-Based Furnace Blocking Early Warning System**

FurnaceGuard AI adalah sistem analisis data operasi dan machine learning yang dikembangkan khusus untuk mendukung deteksi dini risiko blocking pada furnace boiler **PLTU Jeranjang Unit 1 berkapasitas 25 MW**.

Sistem ini dirancang sebagai **decision-support system**, bukan sebagai sistem kontrol otomatis boiler.


[![CI](https://github.com/Rizki242/MACHINE_LEARNING/actions/workflows/ci.yml/badge.svg)](https://github.com/Rizki242/MACHINE_LEARNING/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-R%26D%20synthetic--data-orange)

> **Safety boundary:** FurnaceGuard AI adalah sistem **read-only decision-support**. Sistem ini tidak mengubah interlock/proteksi, tidak menulis perintah ke DCS/historian, dan seluruh metrik model pada fase ini wajib dibaca sebagai hasil **data sintetis**, bukan validasi lapangan.

## Quick Start

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
python -m pip install -e .[dev]
python -m pytest backend/tests -q
uvicorn backend.app.main:app --reload
```

Endpoint awal API:

```text
GET /api/v1/health
```

## Engineering Highlights

- Time-based train/validation/test split dengan embargo untuk mencegah leakage.
- Feature engineering backward-looking untuk rolling windows.
- Hybrid risk engine: rules + ML probability + anomaly score + confidence/data-quality score.
- Konfigurasi threshold, tag mapping, unit, dan model berada di `config/*.yaml` agar dapat diaudit engineer.
- CI GitHub Actions menjalankan test suite otomatis.
- Data aktual, `.env`, model private, dan artefak runtime besar dikecualikan dari Git.

## Repository Layout

```text
backend/app/core/              konfigurasi, konstanta, logging
backend/app/data/              ETL event, data quality, synthetic data generator
backend/app/features/          feature engineering dan baseline load-zone
backend/app/models/            training, evaluasi, model registry
backend/app/rules/             event classifier dan hybrid risk rules
backend/app/explainability/    SHAP/explainability helpers
backend/app/api/               FastAPI routers dan dependencies
backend/tests/                 regression tests untuk leakage, rules, config, API
config/                        YAML konfigurasi domain/model
notebooks/                     eksplorasi data dan eksperimen
docs/                          metodologi, status, limitasi, audit data
```

---

## 1. Informasi Proyek

| Item | Keterangan |
|---|---|
| Nama proyek | FurnaceGuard AI |
| Pembangkit | PLTU Jeranjang |
| Unit | Unit 1 |
| Kapasitas unit | 25 MW |
| Kapasitas boiler | 130 ton/jam |
| Jenis boiler | Circulating Fluidized Bed |
| Konfigurasi | Single drum, full membrane wall |
| Teknologi pembakaran | High-low differential-speed fluidized bed |
| Status proyek | Research and Development |
| Deployment awal | Offline analysis dan shadow mode |
| Zona waktu | Asia/Makassar |

---

## 2. Spesifikasi Desain Boiler

Berdasarkan manual desain boiler:

| Parameter | Nilai |
|---|---:|
| Rated evaporation capacity | 130 ton/jam |
| Superheated steam pressure | 9,81 MPa(g) |
| Superheated steam temperature | 540°C |
| Feedwater temperature | 215°C |
| Blowdown rate | 2% |
| Boiler efficiency | 90% |
| Furnace width | 8.235 mm |
| Furnace depth | 6.765 mm |
| Furnace cross-sectional area | ±55,7 m² |
| Boiler drum elevation | 32.000 mm |
| Highest boiler elevation | 40.776 mm |

Nilai di atas merupakan **design baseline**. Batas operasi, alarm, dan trip aktual harus diverifikasi terhadap SOP, logic DCS, cause and effect, commissioning report, serta data operasi aktual Unit 1.

---

## 3. Konfigurasi Furnace

Boiler menggunakan konfigurasi high-low differential-speed bed:

- Main bed berada di tengah furnace.
- Auxiliary bed berada di bagian depan dan belakang.
- Selisih ketinggian main bed dan auxiliary bed sekitar 500 mm.
- Batubara masuk dari kedua sisi furnace.
- Main bed dan auxiliary bed memiliki suplai udara independen.
- Material halus bersirkulasi dari main bed ke auxiliary bed dan kembali ke main bed.
- Material hasil pemisahan cyclone dikembalikan melalui return leg dan U-type return valve.
- Bottom ash dibuang melalui empat cold slag pipe berdiameter sekitar Φ219 mm.

Zona analisis minimum:

```text
main_bed
front_auxiliary_bed
rear_auxiliary_bed
left_side
right_side
front_side
rear_side
coal_feeder_1
coal_feeder_2
coal_feeder_3
coal_feeder_4
slag_pipe_1
slag_pipe_2
slag_pipe_3
slag_pipe_4
return_leg
u_type_return_valve
```

---

## 4. Design Fuel

Karakteristik bahan bakar desain:

| Parameter | Nilai |
|---|---:|
| Carbon, Car | 40,92% |
| Hydrogen, Har | 3,42% |
| Oxygen, Oar | 13,902% |
| Nitrogen, Nar | 0,678% |
| Sulfur, Sar | 1,08% |
| Ash, Aar | 5% |
| Moisture, Mar | 35% |
| Volatile matter, Vdaf | 58,3% |
| Net calorific value | 3.611 kcal/kg |
| Coal particle size | 0–8 mm |

Data batubara aktual yang perlu dikumpulkan:

```text
coal_source
coal_supplier
coal_shipment_id
coal_blending_ratio
gross_calorific_value
net_calorific_value
total_moisture
inherent_moisture
ash_content
volatile_matter
fixed_carbon
total_sulfur
coal_size_distribution
percentage_above_8mm
percentage_fines
ash_fusion_temperature
```

---

## 5. Tujuan Sistem

FurnaceGuard AI dikembangkan untuk:

1. Mendeteksi kondisi furnace yang mulai menyimpang dari baseline normal.
2. Mendeteksi gangguan fluidisasi main bed dan auxiliary bed.
3. Mendeteksi indikasi cold slag pipe blocking.
4. Mendeteksi ketidakseimbangan coal feeder.
5. Mendeteksi gangguan return material dan U-type return valve.
6. Memprediksi risiko blocking dalam 30, 60, dan 180 menit.
7. Menampilkan parameter dominan yang meningkatkan risiko.
8. Menampilkan suspected area atau potential location.
9. Mengukur kualitas data dan confidence model.
10. Memberikan rekomendasi pemeriksaan kepada operator dan engineer.

Target prediksi utama:

```text
blocking_next_30m
blocking_next_60m
blocking_next_180m
```

Target tambahan:

```text
main_bed_blocking_probability
front_aux_bed_disturbance_probability
rear_aux_bed_disturbance_probability
slag_pipe_1_blocking_probability
slag_pipe_2_blocking_probability
slag_pipe_3_blocking_probability
slag_pipe_4_blocking_probability
coal_feeder_blocking_probability
return_system_disturbance_probability
```

---

## 6. Jenis Gangguan yang Dianalisis

### Main bed agglomeration

Indikator potensial:

- Bed differential pressure meningkat.
- Primary air pressure meningkat.
- Primary air flow menurun.
- Pressure-to-flow ratio meningkat.
- Bed temperature spread membesar.
- Bottom ash discharge menurun.
- Furnace pressure berfluktuasi.
- Poking dilakukan lebih sering.

### Auxiliary bed fluidization disturbance

Indikator potensial:

- Temperatur auxiliary bed menyimpang.
- Selisih temperatur main bed dan auxiliary bed membesar.
- Auxiliary bed airflow menurun.
- Auxiliary bed pressure tidak stabil.
- Material circulation melemah.
- Pembakaran menjadi tidak merata.

### Cold slag pipe blocking

Indikator potensial:

- Slag valve terbuka tetapi ash tidak keluar.
- Ash discharge flow menurun.
- Slag pipe temperature berubah tidak normal.
- Ash cooler current meningkat.
- Bed pressure meningkat.
- Operator melakukan poking.

### Coal feeder atau coal chute blocking

Indikator potensial:

- Feeder command berbeda dengan actual flow.
- Feeder motor current meningkat.
- Coal flow menurun.
- Coal chute pressure berubah.
- Bed temperature satu sisi menurun.
- O₂ meningkat.
- CO berfluktuasi.
- Load response melambat.

### Return material disturbance

Indikator potensial:

- Return leg temperature berubah.
- U-valve air pressure menurun.
- Return leg pressure menyimpang.
- Rear auxiliary bed temperature berubah.
- Separator differential pressure berubah.
- Material circulation melemah.

### Air distribution disturbance

Indikator potensial:

- Main bed airflow tidak sesuai pressure.
- Auxiliary bed airflow tidak seimbang.
- Header pressure berubah.
- Air nozzle atau wind cap diduga tersumbat.
- Bed temperature spread meningkat.
- Furnace pressure tidak stabil.

---

## 7. Data yang Dibutuhkan

### Priority A — Wajib untuk MVP

```text
timestamp
unit_load_mw
main_steam_flow
main_steam_pressure
main_steam_temperature
coal_flow_total
coal_feeder_flow_per_feeder
coal_feeder_current_per_feeder
main_bed_temperature
front_aux_bed_temperature
rear_aux_bed_temperature
main_bed_pressure
bed_differential_pressure
furnace_pressure
main_bed_air_flow
auxiliary_bed_air_flow
primary_air_pressure
secondary_air_flow
oxygen_o2
carbon_monoxide_co
id_fan_current
bottom_ash_discharge_status
ash_cooler_motor_current
operator_event
```

### Priority B — Sangat disarankan

```text
separator_differential_pressure
return_leg_temperature
return_leg_pressure
u_valve_air_pressure
slag_pipe_status_1
slag_pipe_status_2
slag_pipe_status_3
slag_pipe_status_4
ash_cooler_inlet_temperature
ash_cooler_outlet_temperature
ash_hopper_level
furnace_outlet_temperature
coal_moisture
coal_ash_content
coal_calorific_value
coal_size_distribution
coal_source
coal_blending_ratio
```

### Priority C — Pengembangan lanjutan

```text
waterwall_metal_temperature
windbox_pressure
air_nozzle_differential_pressure
fly_ash_loi
bottom_ash_loi
ash_chemical_composition
ash_fusion_temperature
thermal_camera_data
furnace_camera_image
acoustic_sensor
ash_cooler_vibration
ash_system_motor_current_signature
```

---

## 8. Data Dictionary Awal

| Feature standar | Tag DCS aktual | Unit | Fungsi |
|---|---|---:|---|
| `unit_load_mw` | Belum dipetakan | MW | Beban generator |
| `main_steam_flow` | Belum dipetakan | t/h | Produksi steam |
| `main_steam_pressure` | Belum dipetakan | MPa | Tekanan steam |
| `main_steam_temperature` | Belum dipetakan | °C | Temperatur steam |
| `coal_flow_total` | Belum dipetakan | t/h | Total fuel input |
| `main_bed_temp` | Belum dipetakan | °C | Temperatur main bed |
| `front_aux_bed_temp` | Belum dipetakan | °C | Temperatur front auxiliary bed |
| `rear_aux_bed_temp` | Belum dipetakan | °C | Temperatur rear auxiliary bed |
| `bed_pressure` | Belum dipetakan | kPa | Indikator inventori bed |
| `bed_dp` | Belum dipetakan | kPa | Kondisi fluidisasi |
| `furnace_pressure` | Belum dipetakan | Pa | Stabilitas draft furnace |
| `main_bed_air_flow` | Belum dipetakan | Nm³/h | Fluidizing air main bed |
| `aux_bed_air_flow` | Belum dipetakan | Nm³/h | Fluidizing air auxiliary bed |
| `pa_header_pressure` | Belum dipetakan | kPa | Tekanan primary air |
| `o2` | Belum dipetakan | % | Excess air |
| `co` | Belum dipetakan | ppm | Combustion instability |
| `ash_cooler_current` | Belum dipetakan | A | Beban ash cooler |
| `bottom_ash_status` | Belum dipetakan | Boolean | Status ash discharge |
| `return_leg_temp` | Belum dipetakan | °C | Return material indicator |
| `u_valve_air_pressure` | Belum dipetakan | kPa | Kondisi return system |

Nama tag DCS harus dipetakan melalui file konfigurasi dan diverifikasi oleh engineer operasi atau instrumentation.

---

## 9. Feature Engineering

Fitur utama yang perlu dihitung:

```text
rolling_mean_5m
rolling_mean_15m
rolling_mean_30m
rolling_mean_60m
rolling_std_15m
slope_15m
slope_30m
rate_of_change
bed_temp_spread
main_aux_temp_spread
bed_pressure_imbalance
main_bed_air_resistance
aux_bed_air_resistance
coal_feeder_imbalance
air_to_fuel_ratio
combustion_instability_index
ash_cooler_current_slope
slag_discharge_failure
return_system_disturbance
steam_flow_per_mw
coal_flow_per_mw
deviation_from_baseline
```

Contoh formula:

```text
main_aux_temp_spread =
max(main_bed_temp, front_aux_bed_temp, rear_aux_bed_temp)
-
min(main_bed_temp, front_aux_bed_temp, rear_aux_bed_temp)
```

```text
main_bed_air_resistance =
main_bed_air_pressure / main_bed_air_flow
```

```text
coal_feeder_imbalance =
(max_feeder_flow - min_feeder_flow)
/
average_feeder_flow
```

```text
slag_discharge_failure =
slag_valve_open
AND
ash_flow_below_minimum
```

---

## 10. Baseline Operating Zones

Model tidak boleh menggunakan satu baseline untuk seluruh kondisi operasi.

Baseline awal dibagi berdasarkan beban:

| Zona | Rentang beban |
|---|---:|
| Startup | 0–5 MW |
| Low load | >5–10 MW |
| Medium load | >10–17 MW |
| High load | >17–22 MW |
| Near rated load | >22–25 MW |

Pembagian tersebut merupakan konfigurasi awal dan harus divalidasi menggunakan data operasi aktual.

Baseline juga dibedakan berdasarkan:

- Jumlah coal feeder aktif.
- Coal source.
- Coal blending.
- Konfigurasi fan.
- Startup atau normal operation.
- Ash discharge activity.
- Cleaning activity.
- Equipment abnormal condition.

---

## 11. Arsitektur Sistem

```text
DCS / Historian / CSV / XLSX
              │
              ▼
        Data Ingestion
              │
              ▼
      Tag Mapping & Validation
              │
              ▼
       Data Quality Pipeline
              │
              ▼
        Data Preprocessing
              │
              ▼
       Feature Engineering
              │
      ┌───────┼────────┐
      ▼       ▼        ▼
 Rule Engine ML Model Anomaly Model
      │       │        │
      └───────┼────────┘
              ▼
       Hybrid Risk Engine
              │
              ▼
 Risk Score, Confidence, Explanation
              │
      ┌───────┼────────┐
      ▼       ▼        ▼
 Dashboard   API     Reports
```

---

## 12. Model Machine Learning

Model MVP:

```text
Rule-Based Engineering
+
XGBoost Classifier
+
Isolation Forest
```

Model pembanding:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

Evaluasi model:

- Precision
- Recall
- F1-score
- PR-AUC
- ROC-AUC
- False alarms per day
- Missed event rate
- Event detection rate
- Average warning horizon
- Median warning horizon
- Brier score
- Calibration error

Data harus dibagi berdasarkan waktu, bukan random split biasa.

---

## 13. Risk Status

| Risk score | Status | Makna |
|---:|---|---|
| 0–24 | Normal | Belum ada pola blocking |
| 25–49 | Early Warning | Mulai terdapat penyimpangan |
| 50–74 | Warning | Kondisi abnormal perlu diverifikasi |
| 75–89 | High Risk | Pemeriksaan operator diperlukan |
| 90–100 | Critical | Indikasi kuat atau event berkembang |

Threshold tersebut bukan batas operasi resmi dan harus divalidasi menggunakan baseline Unit 1.

Pisahkan:

```text
Risk Score
Confidence Score
Data Quality Score
```

---

## 14. Contoh Output

```text
PLTU JERANJANG UNIT 1
FURNACE BLOCKING EARLY WARNING

Unit Load           : 23,4 MW
Boiler Steam Flow   : 122,8 t/h
Risk Score          : 81%
Status              : HIGH RISK
Prediction Horizon  : 35–70 menit
Confidence Score    : 79%
Data Quality Score  : 93%

Suspected Area:
Slag Pipe 2 / Main Bed Right Side

Dominant Indicators:
1. Main bed differential pressure meningkat
2. Air pressure-to-flow ratio sisi kanan meningkat
3. Main–auxiliary bed temperature spread membesar
4. Ash cooler current meningkat
5. Ash discharge response menurun

Recommended Verification:
- Verifikasi aliran bottom ash dari slag pipe 2
- Periksa ash cooler dan valve terkait
- Periksa distribusi udara main bed sisi kanan
- Periksa keseimbangan coal feeder
- Konfirmasi kondisi lokal sesuai SOP
```

---

## 15. Event Registry

Contoh struktur data event:

```csv
event_id,unit_id,start_time,end_time,event_type,event_location,severity,initial_symptom,operator_action,maintenance_action,derating_mw,trip_status,clinker_found,coal_source,coal_blending,notes
JRG-U1-BLK-001,UNIT_1,2026-01-01 08:00:00,2026-01-01 10:00:00,bottom_ash_blocking,slag_pipe_2,3,ash_not_discharging,poking,inspection,3,no,yes,SOURCE_A,60:40,example
```

Format kode event:

```text
JRG-U1-BLK-XXX
JRG-U1-CLINKER-XXX
JRG-U1-ASH-XXX
JRG-U1-FEEDER-XXX
JRG-U1-RETURN-XXX
JRG-U1-AIR-XXX
```

Severity:

```text
0 = Normal
1 = Initial indication
2 = Warning
3 = Blocking
4 = Critical blocking / derating / shutdown / trip
```

---

## 16. Struktur Folder

```text
furnaceguard-ai/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   ├── core/
│   │   ├── data/
│   │   ├── features/
│   │   ├── models/
│   │   ├── rules/
│   │   ├── explainability/
│   │   ├── reports/
│   │   ├── database/
│   │   └── schemas/
│   ├── datasets/
│   ├── models/
│   ├── reports/
│   ├── tests/
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── charts/
│   │   ├── services/
│   │   ├── hooks/
│   │   ├── types/
│   │   └── utils/
│   ├── package.json
│   └── vite.config.ts
│
├── config/
│   ├── tag_mapping.yaml
│   ├── thresholds.yaml
│   ├── units.yaml
│   └── model_config.yaml
│
├── notebooks/
├── docs/
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── LICENSE
└── README.md
```

---

## 17. Teknologi

### Backend

```text
Python
FastAPI
Pandas
NumPy
Scikit-learn
XGBoost
SHAP
Pydantic
SQLAlchemy
```

### Frontend

```text
React
TypeScript
Vite
Tailwind CSS
Recharts atau Plotly
```

### Database

```text
SQLite untuk MVP
PostgreSQL untuk production
```

### Deployment

```text
Docker
Docker Compose
Nginx opsional
```

---

## 18. Instalasi

### Clone repository

```bash
git clone https://github.com/USERNAME/furnaceguard-ai.git
cd furnaceguard-ai
```

### Backend

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux atau WSL:

```bash
source .venv/bin/activate
```

Instal dependency:

```bash
pip install --upgrade pip
pip install -r backend/requirements.txt
```

Jalankan backend (dari akar repositori — seluruh modul memakai impor
absolut `backend.app....`):

```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger UI:

```text
http://localhost:8000/docs
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend:

```text
http://localhost:5173
```

---

## 19. Deployment Mode

### Offline analysis

Digunakan untuk:

- Upload data historian.
- Data audit.
- Event labeling.
- Feature engineering.
- Training.
- Backtesting.

### Shadow mode

Model membaca data aktual dan menyimpan prediksi, tetapi belum menjadi alarm resmi.

### Advisory mode

Prediksi ditampilkan sebagai rekomendasi kepada operator dan engineer.

### Integrated mode

Integrasi historian atau OPC-UA hanya dilakukan setelah validasi, persetujuan teknis, dan pengujian keamanan.

---

## 20. Boiler Protection Boundaries

FurnaceGuard AI tidak boleh mengubah atau menggantikan:

- Low drum water level interlock.
- Steam overpressure alarm dan interlock.
- Pemutusan air dan coal feeding saat ID fan kehilangan daya.
- Pemutusan coal feeding saat seluruh FD fan kehilangan daya.
- Flame monitoring.
- Flame-out protection.
- MFT.
- Boiler trip logic.
- Safety valve.
- DCS protection.

Sistem hanya boleh membaca data yang diizinkan dan menghasilkan advisory.

---

## 21. Data Safety

- Jangan mengunggah data pembangkit ke layanan publik tanpa izin.
- Gunakan server internal jika memungkinkan.
- Simpan API key di `.env`.
- Jangan menyimpan password dalam source code.
- Gunakan role-based access.
- Aktifkan audit log.
- Gunakan koneksi read-only ke historian pada tahap awal.
- Catat setiap perubahan threshold.
- Catat model version yang aktif.
- Pisahkan data synthetic dan data aktual.

Contoh `.gitignore`:

```gitignore
.env
*.db
*.sqlite
backend/datasets/private/
backend/models/
backend/reports/private/
__pycache__/
.venv/
node_modules/
dist/
```

---

## 22. Roadmap

### Versi 0.1 — Data Foundation

- CSV dan XLSX upload.
- Tag mapping.
- Data validation.
- Data Quality Score.
- Synthetic demo data.
- Event registry.

### Versi 0.2 — Risk Engine

- Rule-based monitoring.
- Feature engineering.
- Baseline deviation.
- Basic dashboard.

### Versi 0.3 — Machine Learning

- Logistic Regression.
- Random Forest.
- XGBoost.
- Isolation Forest.
- Time-based validation.
- Backtesting.

### Versi 0.4 — Explainability

- SHAP.
- Dominant indicators.
- Rule trigger explanation.
- Operator feedback.

### Versi 0.5 — Historian Integration

- OPC-UA.
- PI Historian atau historian internal.
- PostgreSQL.
- Scheduled ingestion.
- Shadow mode.

### Versi 1.0 — Production Advisory

- Multi-user access.
- Role-based access.
- Audit logging.
- Model approval workflow.
- Validated advisory mode.
- Automated reporting.

---

## 23. Maintainer

```text
Name         : Rizki Firmansyah
Role         : Engineer / Project Developer
Plant        : PLTU Jeranjang
Unit         : Unit 1
Project      : FurnaceGuard AI
```

---

## 24. Disclaimer

FurnaceGuard AI merupakan alat analisis dan pendukung keputusan.

Hasil sistem:

- Bukan perintah operasi.
- Bukan pengganti SOP.
- Bukan pengganti operator.
- Bukan pengganti engineer boiler.
- Bukan pengganti interlock dan proteksi.
- Tidak menjamin blocking akan atau tidak akan terjadi.
- Harus diverifikasi dengan kondisi aktual.

Dilarang menggunakan aplikasi untuk:

- Bypass interlock.
- Menonaktifkan proteksi.
- Mengubah safety limit.
- Mengontrol fan secara otomatis.
- Mengontrol coal feeder secara otomatis.
- Mengoperasikan ash valve secara otomatis.
- Mengubah boiler set point tanpa otorisasi.
- Menggantikan prosedur operasi PLTU Jeranjang.

Keputusan akhir tetap berada pada operator dan engineer yang berwenang.
