# Laporan Analisis — FurnaceGuard AI Fase 0

**Pembangkit:** PLTU Jeranjang · **Unit:** 1 · **Kapasitas:** 25 MW · **Boiler:** Circulating Fluidized Bed, 130 t/jam

**Lingkup fase ini:** fondasi data, analisis riwayat gangguan, dan pipeline machine learning yang teruji. Belum ada API, antarmuka pengguna, maupun sambungan ke historian.

---

## 1. Ringkasan eksekutif

Riwayat gangguan PLTU Jeranjang Unit 1 selama sepuluh tahun lima bulan (2016-01-01 sampai 2026-05-10) berisi **429 event**: 302 derating dan 127 outage. Sebanyak **116 event** masuk lingkup risiko blocking furnace.

Lima temuan yang menentukan arah pengembangan berikutnya:

1. **Tidak ada satu pun catatan blocking cold slag pipe.** Sepanjang sepuluh tahun, jurnal gangguan tidak pernah mencatatnya sebagai penyebab derating maupun outage. Empat target probabilitas per slag pipe pada README §5 karena itu tidak dapat dilatih maupun divalidasi.

2. **Blocking coal feeder dan aglomerasi furnace mendominasi.** Keduanya menyumbang 43 dan 33 event — 66 persen dari seluruh event blocking.

3. **Event blocking datang bergerombol.** Median selang antar-event 10.2 hari dengan indeks dispersi **1.46**; proses acak tanpa memori bernilai sekitar 1,0. Sebanyak 39 persen event menyusul event sebelumnya dalam tujuh hari. Pengelompokan sekuat ini menunjuk pada penyebab bersama, kemungkinan besar satu kiriman batubara.

4. **Tidak ada pola musiman.** Blocking coal feeder tercatat 23 kali pada musim hujan (Nov–Apr) dan 20 kali pada kemarau (Mei–Okt). Kelembaban batubara memang berulang kali disebut sebagai penyebab, tetapi kejadiannya tersebar merata sepanjang tahun.

5. **Tidak ada satu pun timeseries DCS.** Seluruh tag Priority A README §7 belum tersedia. Inilah kendala tunggal terbesar; tanpa data itu sistem tidak dapat berjalan di kondisi nyata betapapun rapi kodenya.

---

## 2. Analisis riwayat gangguan

Seluruh angka pada bagian ini berasal dari **data operasi nyata**.

### 2.1 Dampak per jenis event

| Jenis event | Event | Jam hilang | MWh hilang | Median jam |
|---|---:|---:|---:|---:|
| Blocking coal feeder | 43 | 1,715 | 13,718 | 8.7 |
| Aglomerasi / slagging furnace | 33 | 3,002 | 30,365 | 59.6 |
| Kebocoran tube erosi bed | 30 | 2,536 | 58,611 | 73.0 |
| Derating kualitas batubara | 19 | 290 | 765 | 3.0 |
| Gangguan distribusi udara | 7 | 1,061 | 12,322 | 173.9 |
| Gangguan return material | 3 | 379 | 8,970 | 51.7 |

### 2.2 Tren tahunan event blocking

| Tahun | Aglomerasi / slagging furnace | Blocking coal feeder | Blocking bottom ash | Gangguan return material | Gangguan distribusi udara | Kebocoran tube erosi bed | Total |
|---|---|---|---|---|---|---|---|
| 2016 | 0 | 2 | 0 | 0 | 0 | 3 | **5** |
| 2017 | 0 | 2 | 0 | 1 | 3 | 1 | **7** |
| 2018 | 1 | 0 | 0 | 0 | 1 | 2 | **4** |
| 2019 | 0 | 2 | 0 | 0 | 0 | 5 | **7** |
| 2020 | 7 | 4 | 0 | 2 | 3 | 0 | **16** |
| 2021 | 0 | 2 | 0 | 0 | 0 | 6 | **8** |
| 2022 | 4 | 6 | 0 | 0 | 0 | 0 | **10** |
| 2023 | 3 | 8 | 0 | 0 | 0 | 2 | **13** |
| 2024 | 8 | 8 | 0 | 0 | 0 | 8 | **24** |
| 2025 | 7 | 7 | 0 | 0 | 0 | 3 | **17** |
| 2026 | 3 | 2 | 0 | 0 | 0 | 0 | **5** |

Puncaknya pada 2024 dengan 24 event. Tahun terakhir pada tabel baru terisi sampai Mei.

Komposisi severity juga bergeser: sebelum 2020 hampir semua event berseverity 3–4 (unit berhenti); sejak 2022 mayoritas berhenti di severity 2, yaitu derating tanpa unit berhenti. Gangguan menjadi **lebih sering tetapi lebih ringan**.

### 2.3 Kekambuhan

| Kelas | Event | Median selang | Indeks dispersi | Pola |
|---|---:|---:|---:|---|
| Semua blocking | 116 | 10.2 hari | 1.46 | menggerombol |
| Aglomerasi furnace | 33 | 20.1 hari | 1.78 | menggerombol |
| Blocking coal feeder | 43 | 50.2 hari | 1.22 | menggerombol |

### 2.4 Durasi gangguan

| Jenis catatan | Jumlah | Min | Median | Rata-rata | P90 | Maks |
|---|---:|---:|---:|---:|---:|---:|
| derating | 302 | 0.37 | 9.83 | 56.19 | 159.59 | 744.00 |
| outage | 127 | 0.13 | 58.50 | 147.73 | 300.39 | 3,395.83 |

Satuan jam.

### 2.5 Grafik

- `01_pareto_equipment_mwh.png` — pareto equipment
- `02_pareto_event_type.png` — pareto event type
- `03_tren_tahunan.png` — yearly trend
- `04_profil_bulanan.png` — monthly profile
- `05_kekambuhan.png` — recurrence
- `06_severity_per_tahun.png` — severity by year

Seluruhnya di `backend/reports/figures/`.

---

## 3. Klasifikasi event otomatis

Teks gangguan berbahasa Indonesia diklasifikasikan menjadi `event_type`, `event_location`, dan `severity` sesuai README §15. Dua lapis dijalankan berurutan: aturan kata kunci yang dapat diaudit, lalu TF-IDF karakter n-gram yang menandai baris mencurigakan. Analyzer karakter dipilih karena sumbernya penuh salah ketik — `funace`, `slaging`, `pembebeban`, `batuabra`, `indikiasi`.

| Jenis event | Jumlah | Porsi | Lingkup |
|---|---:|---:|---|
| Di luar lingkup furnace | 235 | 54.8 % | di luar lingkup |
| Blocking coal feeder | 43 | 10.0 % | **blocking** |
| Aglomerasi / slagging furnace | 33 | 7.7 % | **blocking** |
| Kebocoran tube erosi bed | 30 | 7.0 % | **blocking** |
| Derating kualitas batubara | 19 | 4.4 % | konteks |
| Gangguan fan udara | 14 | 3.3 % | di luar lingkup |
| Belum terklasifikasi | 12 | 2.8 % | perlu tinjauan |
| Peralatan bantu boiler | 11 | 2.6 % | di luar lingkup |
| Penormalan pasca sinkron | 11 | 2.6 % | di luar lingkup |
| Pembatasan pasokan batubara | 8 | 1.9 % | di luar lingkup |
| Gangguan distribusi udara | 7 | 1.6 % | **blocking** |
| Gangguan return material | 3 | 0.7 % | **blocking** |
| Kebocoran tube lain | 3 | 0.7 % | di luar lingkup |

Sebanyak 12 event belum terklasifikasi dan seluruhnya ditandai untuk verifikasi di `backend/reports/events_needing_review.csv`. Isinya memang ambigu di sumber. Tidak ada baris yang dibuang diam-diam.

Kolom `coal_source`, `coal_blending`, dan `clinker_found` yang diminta README §15 tidak ada di berkas sumber dan dibiarkan kosong. Menebaknya akan merusak analisis pengaruh kualitas batubara — justru pertanyaan yang paling ingin dijawab sistem ini.

---

## 4. Unjuk kerja pipeline machine learning

> **Angka unjuk kerja berikut dihitung di atas DATA SINTETIS, bukan data operasi PLTU Jeranjang Unit 1. Angka ini TIDAK BOLEH dikutip sebagai unjuk kerja lapangan. Tujuannya hanya memvalidasi bahwa pipeline berjalan benar dari ujung ke ujung.**

Pembagian berbasis waktu: latih [2020, 2021, 2022, 2023], validasi [2024], uji [2025, 2026], dengan jeda embargo 6 jam di setiap perbatasan. Jumlah fitur 231.

### 4.1 Metrik per sampel

| Model | Horizon | PR-AUC | ROC-AUC | Precision | Recall | Brier | Calib. error |
|---|---|---:|---:|---:|---:|---:|---:|
| xgboost | blocking_next_180m | 0.2894 | 0.7660 | 0.0299 | 0.5225 | 0.0145 | 0.0587 |
| decision_tree | blocking_next_180m | 0.2005 | 0.7244 | 0.0057 | 0.9976 | 0.0148 | 0.0585 |
| logistic_regression | blocking_next_180m | 0.1227 | 0.7742 | 0.0205 | 0.5997 | 0.0238 | 0.0755 |
| random_forest | blocking_next_180m | 0.0934 | 0.7633 | 0.0268 | 0.5323 | 0.0213 | 0.0690 |
| xgboost | blocking_next_30m | 0.3768 | 0.9911 | 0.0009 | 1.0000 | 0.0059 | 0.0139 |
| decision_tree | blocking_next_30m | 0.1274 | 0.8964 | 0.0009 | 1.0000 | 0.0031 | 0.0141 |
| random_forest | blocking_next_30m | 0.0825 | 0.9844 | 0.0009 | 1.0000 | 0.0073 | 0.0223 |
| logistic_regression | blocking_next_30m | 0.0501 | 0.9660 | 0.0068 | 0.9270 | 0.0049 | 0.0177 |
| xgboost | blocking_next_60m | 0.4219 | 0.9731 | 0.0145 | 0.9508 | 0.0073 | 0.0206 |
| decision_tree | blocking_next_60m | 0.2052 | 0.9246 | 0.0019 | 1.0000 | 0.0066 | 0.0261 |
| random_forest | blocking_next_60m | 0.1086 | 0.9660 | 0.0100 | 0.9730 | 0.0101 | 0.0310 |
| logistic_regression | blocking_next_60m | 0.0676 | 0.9242 | 0.0131 | 0.8143 | 0.0092 | 0.0292 |

Perhatikan jarak antara ROC-AUC dan PR-AUC. Dengan prevalensi kelas positif di bawah satu persen, model yang selalu menjawab "tidak ada risiko" sudah benar hampir sepanjang waktu dan akan memperoleh ROC-AUC yang terlihat mengesankan. **PR-AUC adalah metrik utama**; ROC-AUC dicantumkan hanya karena README §12 memintanya.

### 4.2 Metrik per event

Inilah pertanyaan yang sebenarnya ditanyakan operator: berapa banyak event nyata yang tertangkap, berapa lama sebelumnya, dan berapa sering sistem berteriak tanpa sebab.

| Model | Horizon | Event | Terdeteksi | Tingkat deteksi | Alarm palsu/hari | Median warning |
|---|---|---:|---:|---:|---:|---:|
| xgboost | blocking_next_180m | — | — | 1.000 | 2.44 | 106 menit |
| decision_tree | blocking_next_180m | — | — | 1.000 | 0.74 | 180 menit |
| logistic_regression | blocking_next_180m | — | — | 1.000 | 2.72 | 125 menit |
| random_forest | blocking_next_180m | — | — | 1.000 | 1.81 | 100 menit |
| xgboost | blocking_next_30m | — | — | 1.000 | 0.36 | 180 menit |
| decision_tree | blocking_next_30m | — | — | 1.000 | 0.36 | 180 menit |
| random_forest | blocking_next_30m | — | — | 1.000 | 0.36 | 180 menit |
| logistic_regression | blocking_next_30m | — | — | 1.000 | 2.75 | 64 menit |
| xgboost | blocking_next_60m | — | — | 1.000 | 2.82 | 85 menit |
| decision_tree | blocking_next_60m | — | — | 1.000 | 0.36 | 180 menit |
| random_forest | blocking_next_60m | — | — | 1.000 | 2.10 | 86 menit |
| logistic_regression | blocking_next_60m | — | — | 1.000 | 2.43 | 87 menit |

Anggaran alarm palsu ditetapkan 2.0 per hari pada `config/thresholds.yaml`. Ambang dipilih sebagai nilai dengan tingkat deteksi tertinggi yang masih memenuhi anggaran itu, bukan nilai yang memaksimalkan F1.

Sekali lagi: angka-angka di atas mengukur **kebenaran pipeline**, bukan kemampuan deteksi di lapangan. Pola degradasi pada data sintetis adalah hipotesis engineering menurut README §6, bukan pengamatan. Bila tanda tangan gangguan sesungguhnya berbeda — dan besar kemungkinan berbeda — model ini tidak akan mengenalinya.

---

## 5. Data yang harus disediakan

Bagian ini adalah keluaran paling berharga dari fase ini.

### 5.1 Prioritas 1 — tanpa ini sistem tidak dapat berjalan

Ekspor historian resolusi **1 menit**, rentang minimum **12 bulan**, mencakup periode di sekitar event pada Event Registry. Seluruh 37 tag Priority A README §7:

```text
timestamp  unit_load_mw  main_steam_flow
main_steam_pressure  main_steam_temperature  coal_flow_total
coal_feeder_1_flow  coal_feeder_2_flow  coal_feeder_3_flow
coal_feeder_4_flow  coal_feeder_1_command  coal_feeder_2_command
coal_feeder_3_command  coal_feeder_4_command  coal_feeder_1_current
coal_feeder_2_current  coal_feeder_3_current  coal_feeder_4_current
main_bed_temperature  front_aux_bed_temperature  rear_aux_bed_temperature
main_bed_pressure  bed_differential_pressure  aux_bed_pressure
furnace_pressure  main_bed_air_flow  auxiliary_bed_air_flow
front_aux_bed_air_flow  rear_aux_bed_air_flow  primary_air_pressure
secondary_air_flow  oxygen_o2  carbon_monoxide_co
id_fan_current  bottom_ash_discharge_status  ash_cooler_motor_current
operator_event
```

Bersamanya diperlukan **daftar nama tag DCS asli beserta satuannya**, diverifikasi engineer operasi atau instrumentasi, untuk mengisi `config/tag_mapping.yaml`. Seluruh entri di berkas itu saat ini masih `null`.

### 5.2 Prioritas 2 — menentukan mutu peringatan

Tag Priority B README §7, ditambah satu yang tidak ada di README:

**Log kegiatan poking** — waktu, lokasi, lama, dan alasannya. Inilah satu-satunya jalan memperoleh label untuk kelas blocking bottom ash. Dugaan paling masuk akal atas ketiadaan catatan blocking slag pipe adalah bahwa gangguannya selalu tertangani lewat poking rutin tanpa menyebabkan derating, sehingga tidak pernah masuk jurnal gangguan.

### 5.3 Prioritas 3 — menjawab pertanyaan penyebab

Data kualitas batubara per kiriman, dengan tanggal:

```text
coal_source            coal_supplier          coal_shipment_id
coal_blending_ratio    gross_calorific_value  net_calorific_value
total_moisture         ash_content            volatile_matter
coal_size_distribution percentage_above_8mm   ash_fusion_temperature
```

Mengingat event blocking terbukti menggerombol, inilah data yang paling mungkin menjelaskan **mengapa** gerombolan itu terjadi.

### 5.4 Prioritas 4 — verifikasi ambang

SOP operasi Unit 1, logic DCS dan matriks cause and effect, commissioning report, serta daftar setting alarm dan trip aktual. README §2 menyatakan seluruh nilai desain harus diverifikasi terhadap dokumen tersebut. Sampai itu dilakukan, tidak satu pun ambang di berkas konfigurasi boleh dianggap sebagai batas operasi.

### 5.5 Konfirmasi yang diperlukan

1. **Jumlah coal feeder.** Manual desain menyebut empat; jurnal gangguan hanya pernah menyebut 1A dan 1B.
2. **Blocking slag pipe.** Apakah benar tidak pernah terjadi, atau tertangani rutin tanpa dicatat, atau tercatat dengan istilah lain.
3. **Dua belas event tak terklasifikasi** pada `backend/reports/events_needing_review.csv`.

---

## 6. Langkah berikutnya

Diurut menurut ketergantungan, bukan menurut kemudahan.

1. **Kumpulkan data Prioritas 1 dan petakan tag DCS.** Segala hal lain menunggu langkah ini.
2. **Hitung ulang baseline dan tinjau seluruh ambang** memakai data operasi nyata. Baseline saat ini berasal dari data sintetis.
3. **Latih ulang model pada data nyata** dan bandingkan hasilnya dengan angka pada laporan ini. Selisihnya akan menunjukkan seberapa jauh asumsi generator sintetis meleset.
4. **Verifikasi Event Registry** bersama engineer operasi, terutama event yang ditandai perlu tinjauan.
5. **Baru setelah itu** bangun lapisan API dan antarmuka pengguna (README §16, §17). Membangunnya lebih dulu berarti menampilkan angka yang belum layak ditampilkan.

Mode deployment tetap `offline` sampai langkah 1 sampai 4 selesai. README §19 menempatkan integrasi historian hanya setelah validasi, persetujuan teknis, dan pengujian keamanan.

---

## 7. Disclaimer

FurnaceGuard AI adalah alat analisis dan pendukung keputusan. Hasilnya bukan perintah operasi, bukan pengganti SOP, operator, engineer boiler, maupun interlock dan proteksi. Tidak menjamin blocking akan atau tidak akan terjadi. Wajib diverifikasi dengan kondisi aktual. Keputusan akhir berada pada operator dan engineer yang berwenang.

FurnaceGuard AI tidak boleh dipakai untuk bypass interlock, menonaktifkan proteksi, mengubah safety limit, mengontrol fan, coal feeder, atau ash valve secara otomatis, mengubah set point boiler tanpa otorisasi, maupun menggantikan prosedur operasi PLTU Jeranjang (README §20, §24).

Dokumen pendukung: `docs/DATA_AUDIT.md`, `docs/METHODOLOGY.md`, `docs/LIMITATIONS.md`.
