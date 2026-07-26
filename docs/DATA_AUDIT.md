# Audit Data — FurnaceGuard AI Fase 0

**Unit:** PLTU Jeranjang Unit 1 · **Tanggal audit:** 26 Juli 2026 ·
**Sumber:** isi direktori `data/`

Dokumen ini mencatat apa yang ada, apa yang tidak ada, dan apa yang harus
dikumpulkan berikutnya. Bagian terakhir adalah keluaran paling berharga
dari fase ini: tanpa data tersebut, sistem tidak dapat berjalan di
kondisi nyata betapapun rapi kodenya.

---

## 1. Ringkasan

| Berkas | Isi | Kelayakan untuk ML |
|---|---|---|
| `05_PARETO Data Riwayat Gangguan Jeranjang 2014 sd Mei 2026.xlsx` | Riwayat outage dan derating seluruh unit, 31 sheet | **Tinggi** — sumber Event Registry |
| `04_PARETO Data Riwayat Gangguan Jeranjang 2014 sd April 2026.xlsx` | Snapshot bulan sebelumnya, struktur identik | Rendah — dipakai menguji konsistensi ETL |
| `04. LAPORAN EFISIENSI UNIT 1 UBP JERANJANG - APRIL 2026.pdf` | Laporan uji heat rate bulanan, 23 halaman | Rendah — dokumen referensi |
| `F13010设计说明书-Boiler Manual Book.doc` | Manual desain boiler | Referensi desain |

---

## 2. Riwayat gangguan Unit 1 — sumber utama

Dua sheet dipakai:

| Sheet | Baris data | Rentang |
|---|---:|---|
| `UNIT 1 Derating (Jurnal) ` | 302 | 2016-01-01 s/d 2026-05-10 |
| `UNIT 1 Outage (Jurnal) ` | 127 | 2016-01-25 s/d 2026-02-24 |
| **Total** | **429** | **10 tahun 5 bulan** |

> Nama kedua sheet mengandung **spasi di ujung**. Itu apa adanya di berkas
> sumber dan harus dipertahankan persis; merapikannya membuat pembacaan
> gagal total.

### Bentuk berkas

- Judul di baris 1, header bertingkat dengan sel gabungan di baris 2–3,
  data mulai baris 4.
- Sebelas kolom pertama identik pada kedua sheet. Kolom sesudahnya
  **berbeda**: sheet derating punya `MW DERATING` lalu `MWh HILANG`;
  sheet outage langsung `MWh HILANG`.
- Nama peralatan punya **95 varian** ejaan dan kapitalisasi untuk Unit 1
  saja: `FURNACE`/`Furnace`, `CONDENSER`/`CONDENSOR`/`Kondensor`,
  `COAL FEEDER`/`Coal Feeder`/`Coalfeeder 1A`/`COAL FEEDER 1B`.
- Sebagian sel mengandung mojibake dari karakter non-breaking yang rusak,
  contohnya `GENERATOR�OUTPUT�BREAKER�(X23G)�Unit 1`.
- Teks gangguan penuh salah ketik: `funace`, `slaging`, `pembebeban`,
  `batuabra`, `indikiasi`, `Cute Coal Feeder`, `Presssure`.

### Uji konsistensi antar-snapshot

Seluruh 425 event pada snapshot April 2026 muncul identik pada snapshot
Mei 2026, dengan 4 event baru bertambah. Tidak ada event yang hilang.
Sumber data konsisten.

### Kolom yang TIDAK ADA di sumber

Skema Event Registry README §15 meminta kolom berikut, dan tidak satu pun
tersedia:

```text
coal_source          coal_blending          clinker_found
```

Ketiganya dibiarkan kosong. Mengisinya dengan tebakan akan merusak
analisis pengaruh kualitas batubara — justru pertanyaan yang paling ingin
dijawab sistem ini.

Kolom `operator_action` juga kosong: sumber tidak membedakan tindakan
operator dari tindakan pemeliharaan, keduanya bercampur pada kolom
`PENYELESAIAN GANGGUAN`.

Kolom `trip_status` **diturunkan**, bukan dibaca: forced outage (`FO`,
`FOL`) dipetakan ke `yes`. Ini bukan bukti terjadinya trip proteksi dan
tidak boleh dipakai sebagai bukti.

### Hasil klasifikasi

Dari 429 event Unit 1:

| Jenis event | Jumlah | Lingkup |
|---|---:|---|
| `not_furnace_related` | 235 | di luar lingkup |
| `coal_feeder_blocking` | 43 | **blocking** |
| `furnace_agglomeration` | 33 | **blocking** |
| `tube_leak_bed_erosion` | 30 | **blocking** |
| `coal_quality_derating` | 19 | konteks |
| `fan_disturbance` | 14 | di luar lingkup |
| `unclassified` | 12 | perlu tinjauan |
| `boiler_auxiliary_other` | 11 | di luar lingkup |
| `startup_normalisation` | 11 | di luar lingkup |
| `coal_supply_constraint` | 8 | di luar lingkup |
| `air_distribution_disturbance` | 7 | **blocking** |
| `return_system_disturbance` | 3 | **blocking** |
| `boiler_tube_leak_other` | 3 | di luar lingkup |

**116 event masuk lingkup risiko blocking furnace.**

Dua belas event tersisa tidak terklasifikasi dan seluruhnya ditandai
`needs_review` di `backend/reports/events_needing_review.csv`. Isinya
memang ambigu di sumber: "peralatan banyak yang abnormal", "Beban ditahan
karena sistem". Tidak ada baris yang dibuang diam-diam.

---

## 3. Temuan yang harus diperhatikan

### 3.1 Tidak ada satu pun catatan blocking cold slag pipe

`bottom_ash_blocking` berjumlah **nol**. Sepanjang sepuluh tahun, jurnal
gangguan tidak pernah mencatat blocking bottom ash atau cold slag pipe
sebagai penyebab derating maupun outage Unit 1.

README §5 dan §6 menempatkan blocking slag pipe sebagai salah satu target
utama, lengkap dengan empat target probabilitas per pipa. Tanpa satu pun
contoh nyata, target itu **tidak dapat dilatih maupun divalidasi**.

Tiga kemungkinan, dan ketiganya perlu dikonfirmasi ke operasi:

1. Blocking slag pipe memang tidak pernah terjadi di Unit 1.
2. Terjadi tetapi selalu tertangani lewat poking rutin tanpa menyebabkan
   derating, sehingga tidak pernah masuk jurnal.
3. Tercatat dengan istilah lain yang belum masuk daftar kata kunci.

Dugaan paling mungkin adalah nomor 2 — poking adalah tindakan rutin yang
biasanya tidak dilaporkan sebagai gangguan. Bila benar, satu-satunya
sumber label untuk kelas ini adalah **log kegiatan poking**, yang belum
tersedia.

### 3.2 Jumlah coal feeder berbeda antara desain dan catatan

Manual desain menyebut **empat** coal feeder. Jurnal gangguan hanya
pernah menyebut **1A** dan **1B**. Konfigurasi feeder aktual Unit 1 perlu
dikonfirmasi sebelum tag DCS dipetakan, karena fitur
`coal_feeder_imbalance` bergantung pada jumlah feeder yang benar.

### 3.3 Pola gangguan berubah sejak 2022

Event blocking per tahun: 2016 = 5, 2017 = 7, 2018 = 4, 2019 = 7,
2020 = 16, 2021 = 8, 2022 = 10, 2023 = 13, **2024 = 24**, 2025 = 17,
2026 = 5 (sampai Mei).

Komposisinya juga bergeser: sebelum 2020 hampir semua event berseverity
3–4 (unit berhenti); sejak 2022 mayoritas berhenti di severity 2, yaitu
derating tanpa unit berhenti. Gangguan menjadi **lebih sering tetapi
lebih ringan**.

### 3.4 Tidak ada pola musiman

Hipotesis awal bahwa blocking coal feeder menumpuk pada musim hujan
karena batubara lembab **tidak terdukung data**:

| Kelas | Nov–Apr | Mei–Okt |
|---|---:|---:|
| Blocking coal feeder | 23 | 20 |
| Aglomerasi furnace | 16 | 17 |

Kelembaban batubara memang berulang kali disebut sebagai penyebab, tetapi
kejadiannya tersebar merata sepanjang tahun. Faktor pemicunya lebih
mungkin kiriman batubara tertentu, bukan musim.

### 3.5 Event blocking datang bergerombol

Selang antar-event blocking: median 10,2 hari, indeks dispersi **1,46**.
Proses acak tanpa memori bernilai sekitar 1,0; nilai di atasnya berarti
event menggerombol. Sebanyak 39 persen event menyusul event sebelumnya
dalam tujuh hari.

Pengelompokan sekuat ini menunjukkan **penyebab bersama** — kemungkinan
besar satu kiriman atau satu campuran batubara. Data `coal_source` dan
`coal_blending_ratio` akan menjawabnya, dan itulah alasan kedua kolom itu
diberi prioritas tinggi pada daftar di bawah.

---

## 4. Laporan efisiensi (PDF)

23 halaman, **hasil pemindaian tanpa lapisan teks** — nol karakter dapat
diekstraksi. Isinya uji heat rate bulanan pada satu titik beban 25,81 MW,
9 April 2026:

| Parameter | Baseline | Aktual |
|---|---:|---:|
| Gross Power Output, MW | 25,81 | 25,81 |
| Auxiliary Power Consumption, % | 12,77 | 12,67 |
| Net Plant Heat Rate (HHV), kCal/kWh | 4.022,15 | 4.725,66 |
| Boiler Efficiency (HHV), % | 81,26 | **80,39** |
| HP Turbine Efficiency, % | 71,00 | 71,17 |
| Condenser Cleanliness, % | 85,00 | 82,49 |

Nilai efisiensi boiler 80,39 % dipakai sebagai parameter neraca energi
pada generator sintetis. Selebihnya berupa satu titik uji bulanan, bukan
timeseries, sehingga tidak dapat dipakai melatih model.

Rekomendasi pada laporan menyebut **kalibrasi coal flow pada coal
feeder** — relevan langsung dengan keandalan fitur
`coal_feeder_command_deviation` bila sistem ini disambungkan ke data
nyata.

---

## 5. Yang tidak ada: seluruh timeseries DCS

**Tidak satu pun tag Priority A README §7 tersedia.** Seluruh entri
`dcs_tag` pada `config/tag_mapping.yaml` masih `null`.

Akibatnya seluruh isi README §9 (rekayasa fitur), §11 (risk engine), dan
§12 (model) tidak dapat dikerjakan di atas data nyata. Fase ini memakai
generator timeseries sintetis sebagai penggantinya —
lihat `LIMITATIONS.md` untuk batas pemakaiannya.

---

## 6. Data yang harus dikumpulkan berikutnya

Diurut menurut dampaknya terhadap kemampuan sistem.

### Prioritas 1 — tanpa ini sistem tidak dapat berjalan

Ekspor historian, resolusi **1 menit**, rentang minimum **12 bulan**,
mencakup periode di sekitar event pada Event Registry:

```text
timestamp                     unit_load_mw
main_steam_flow               main_steam_pressure
main_steam_temperature        coal_flow_total
main_bed_temperature          front_aux_bed_temperature
rear_aux_bed_temperature      main_bed_pressure
bed_differential_pressure     furnace_pressure
main_bed_air_flow             auxiliary_bed_air_flow
primary_air_pressure          secondary_air_flow
oxygen_o2                     carbon_monoxide_co
id_fan_current                ash_cooler_motor_current
bottom_ash_discharge_status
coal_feeder_<n>_flow          coal_feeder_<n>_command
coal_feeder_<n>_current
```

Bersamanya diperlukan **daftar nama tag DCS asli** beserta satuannya,
diverifikasi engineer operasi atau instrumentasi, untuk mengisi
`config/tag_mapping.yaml`.

### Prioritas 2 — menentukan mutu peringatan

```text
separator_differential_pressure    return_leg_temperature
return_leg_pressure                u_valve_air_pressure
slag_pipe_<n>_status               slag_pipe_<n>_temperature
bottom_ash_flow                    ash_hopper_level
furnace_outlet_temperature
```

Ditambah **log kegiatan poking**: waktu, lokasi, lama, dan alasannya.
Inilah satu-satunya jalan memperoleh label untuk kelas blocking bottom
ash, yang saat ini kosong sama sekali.

### Prioritas 3 — menjawab pertanyaan penyebab

```text
coal_source            coal_supplier          coal_shipment_id
coal_blending_ratio    gross_calorific_value  net_calorific_value
total_moisture         ash_content            volatile_matter
coal_size_distribution percentage_above_8mm   ash_fusion_temperature
```

Data kualitas batubara per kiriman, dengan tanggal, agar dapat
dihubungkan ke waktu event. Mengingat event blocking terbukti
menggerombol, inilah data yang paling mungkin menjelaskan **mengapa**
gerombolan itu terjadi.

### Prioritas 4 — verifikasi ambang

Dokumen yang diperlukan agar ambang pada `config/thresholds.yaml`
berhenti menjadi tebakan awal:

- SOP operasi Unit 1
- Logic DCS dan matriks cause and effect
- Commissioning report
- Daftar setting alarm dan trip aktual

README §2 menyatakan seluruh nilai desain harus diverifikasi terhadap
dokumen di atas. Sampai itu dilakukan, tidak satu pun ambang di berkas
konfigurasi boleh dianggap sebagai batas operasi.

---

## 7. Cara mengulang audit ini

```powershell
python -m backend.app.data.event_etl --cross-check
python -m backend.app.rules.event_classifier
python -m backend.app.reports.event_analysis
```

Keluaran:

- `backend/datasets/processed/event_registry_unit1.csv`
- `backend/reports/events_needing_review.csv`
- `backend/reports/source_cross_check.json`
- `backend/reports/event_analysis_summary.json`
- `backend/reports/figures/*.png`
