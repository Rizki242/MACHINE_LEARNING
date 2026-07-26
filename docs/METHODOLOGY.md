# Metodologi — FurnaceGuard AI Fase 0

Dokumen ini menjelaskan **mengapa** setiap keputusan teknis diambil.
Untuk cara menjalankannya, lihat `README.md`. Untuk apa yang belum bisa
dilakukan, lihat `LIMITATIONS.md`.

---

## 1. Bentuk masalah

Deteksi dini blocking furnace adalah masalah **prediksi peristiwa jarang
pada deret waktu**. Ciri-ciri yang menentukan seluruh rancangan:

| Ciri | Akibat pada rancangan |
|---|---|
| Kelas positif sangat jarang (0,09 % menit) | PR-AUC sebagai metrik utama, bukan ROC-AUC |
| Urutan waktu bermakna | Pembagian data berbasis waktu, bukan acak |
| Biaya alarm palsu tinggi | Ambang dipilih dari anggaran alarm harian |
| Operator perlu alasan, bukan angka | Rule engine berdampingan dengan model |
| Kondisi normal berbeda per beban | Baseline per zona beban, bukan global |

---

## 2. Event Registry sebagai fondasi

Sumber tunggal kebenaran adalah jurnal gangguan Unit 1: 429 event selama
sepuluh tahun. Seluruh label, seluruh timeline, dan seluruh analisis
bertumpu padanya.

### Klasifikasi dua lapis

**Lapis 1 — kata kunci.** Aturan berbahasa Indonesia pada
`config/event_taxonomy.yaml` memetakan teks gangguan, penyebab, dan
peralatan ke jenis event, lokasi, dan severity.

Skor kecocokan mengalikan bobot jenis event dengan panjang frasa, supaya
frasa spesifik mengalahkan frasa umum: "blocking pada coal feeder"
mengalahkan "blocking" saja. Peralatan yang tidak pernah relevan (CWP,
condenser, turbin, generator) langsung dikeluarkan dari lingkup, kecuali
teksnya secara eksplisit menyebut furnace, bed, atau feeder.

Lapis ini **deterministik dan dapat diaudit**. Engineer boiler dapat
membaca YAML-nya, tidak setuju, dan mengubahnya tanpa menyentuh Python
maupun melatih ulang apa pun. Itulah alasan lapis ini yang dipakai hilir.

**Lapis 2 — TF-IDF karakter n-gram.** Dilatih pada label lapis 1 sebagai
*weak supervision*. Analyzer karakter dipilih karena sumbernya penuh
salah ketik: `funace`, `slaging`, `pembebeban`, `batuabra`, `indikiasi`.
Analyzer kata akan memperlakukan `funace` dan `furnace` sebagai dua hal
yang sama sekali berbeda; analyzer karakter melihat keduanya berbagi
sebagian besar n-gram.

Tugas lapis 2 bukan menggantikan lapis 1, melainkan **menyalakan bendera**:
baris yang polanya mirip suatu kelas tetapi tidak tertangkap kata kunci
diadopsi dengan tanda `needs_review`; baris yang kedua lapis berbeda
pendapat juga ditandai. Tidak ada baris yang dibuang diam-diam.

### Severity

Diturunkan dari jenis catatan, kode status, durasi, dan besar derating,
mengikuti README §15. Catatan pada sheet outage berarti unit berhenti,
apa pun kode statusnya — sebagian baris di sana berkode `FD` atau `OMC`,
dan menilainya dari kode status saja akan menurunkan severity-nya secara
keliru.

---

## 3. Timeseries sintetis

### Mengapa dibangkitkan, bukan ditunggu

Tanpa timeseries, tidak ada satu pun bagian README §9 sampai §14 yang
dapat ditulis, apalagi diuji. Menunggu data DCS berarti menunda seluruh
pekerjaan sekaligus menunda pertanyaan "data apa saja yang sebenarnya
dibutuhkan" — padahal jawaban atas pertanyaan itulah yang paling berguna
bagi pembangkit saat ini.

Dengan membangun pipeline lebih dulu, daftar tag yang dibutuhkan menjadi
konkret dan teruji, bukan sekadar salinan daftar keinginan.

### Cara membangkitkannya

Sinyal disusun dari neraca energi dan spesifikasi desain, bukan dari
angka acak:

- Beban mengikuti profil harian dengan hanyutan antar-hari, ditambah
  hari-hari beban rendah agar kelima zona beban benar-benar terisi.
- Aliran steam mengikuti beban terhadap kapasitas rancangan 130 t/jam.
- Aliran batubara dihitung dari entalpi steam dan air pengisi, nilai
  kalor 3.611 kcal/kg, dan efisiensi boiler 80,39 % hasil uji April 2026.
- Temperatur bed, tekanan, aliran udara, O₂, dan CO saling terkait lewat
  hubungan proses, bukan dibangkitkan sendiri-sendiri.
- Derau memakai proses AR(1), bukan derau putih. Derau putih membuat
  fitur `rolling_std` kehilangan makna karena tidak ada gerakan yang
  menempel antar-langkah.

### Keterikatan pada kejadian nyata

Timeline mengikuti Event Registry: unit benar-benar berhenti selama
outage, beban benar-benar turun selama derating, dan setiap event
blocking didahului pola degradasi sesuai tanda tangan gangguannya menurut
README §6 — aglomerasi menaikkan bed differential pressure dan tekanan
primary air sekaligus menurunkan aliran udara; blocking feeder melebarkan
selisih perintah terhadap aliran nyata dan menaikkan O₂.

Panjang ramp diacak 30–240 menit sebelum event, lalu mereda selama event
berlangsung. Degradasi yang berhenti mendadak begitu event dimulai akan
memberi model petunjuk yang tidak akan pernah ada di lapangan.

### Cacat data yang disengaja

Data hilang berumpun dan sensor macet disisipkan dengan sengaja. Tanpa
keduanya, modul kualitas data tidak pernah teruji dan skornya selalu
sempurna secara palsu. Sensor macet lebih berbahaya daripada data hilang:
nilainya tampak sah tetapi sudah basi.

---

## 4. Rekayasa fitur

### Aturan yang tidak boleh dilanggar

**Setiap jendela rolling hanya melihat ke belakang.**

Kebocoran masa depan menghasilkan model yang tampak sangat akurat saat
pengujian dan gagal total saat dipasang, karena saat berjalan sungguhan
masa depan itu belum ada. Kegagalannya senyap: tidak ada pesan galat,
hanya metrik bagus yang berbohong.

Dua penjaga dipasang:

1. `model_config.yaml` punya bendera `backward_only` yang wajib bernilai
   benar; `build_features` menolak berjalan bila dimatikan.
2. `backend/tests/test_features.py` menguji sifat itu langsung: sebuah
   dataframe yang seluruh sinyalnya datar lalu melonjak di tengah harus
   menghasilkan fitur yang **konstan pada seluruh baris sebelum lonjakan**.
   Kolom mana pun yang berubah lebih awal berarti melihat ke depan.

### Kemiringan tertutup

Kemiringan regresi dihitung dari dua jumlah bergulir, bukan lewat
`rolling().apply()`. Untuk jendela selebar `W` dengan jarak sampel
seragam, kemiringan kuadrat terkecil dapat disusun sebagai:

```text
slope = [Σ(t·x) − (awal_jendela + t̄)·Σx] / [W(W²−1)/12]
```

Biayanya linear terhadap panjang deret, bukan perkalian panjang deret
dengan lebar jendela. Pada 3,3 juta baris dengan 22 kolom dasar dan dua
lebar jendela, selisihnya menentukan apakah pipeline selesai dalam menit
atau jam. Pengujian membandingkan hasilnya dengan `numpy.polyfit` untuk
memastikan keduanya identik.

### Baseline per zona beban

README §10 melarang satu baseline untuk seluruh kondisi operasi, dan
alasannya nyata: bed differential pressure pada 8 MW dan pada 24 MW
berbeda jauh. Satu ambang tunggal akan membanjiri operator dengan alarm
palsu di beban rendah sekaligus buta di beban tinggi.

Baseline memakai **median dan median absolute deviation**, bukan rata-rata
dan simpangan baku. Periode pelatihan sendiri mengandung event gangguan;
statistik yang tahan pencilan mencegah gangguan itu ikut mendefinisikan
"normal".

Jendela enam jam sebelum dan sesudah setiap event dikeluarkan dari
perhitungan baseline, dengan alasan yang sama.

#### Koreksi beban di dalam zona

Zona saja ternyata belum cukup, dan cacatnya baru terlihat setelah rule
engine dijalankan pada satu event nyata: sistem menyalakan aturan
**gangguan return material** pada sebuah event yang sebenarnya blocking
coal feeder, sementara aturan coal feeder sendiri diam.

Sebabnya bukan derau dan bukan baseline yang tipis — zona medium punya
lebih dari tiga ratus ribu sampel bersih. Sebabnya struktural. Zona medium
membentang 10 sampai 17 MW, sementara temperatur bed, tekanan, dan aliran
udara bergerak terus mengikuti beban di sepanjang rentang itu. Unit yang
berjalan pada 10,7 MW berada di tepi bawah zonanya, sehingga hampir setiap
sinyal terbaca "rendah" sekaligus terhadap nilai tengah zona. Aturan pun
menyala karena posisi beban, bukan karena ada yang tidak beres — kegagalan
yang menipu, karena alarmnya menunjuk peralatan yang sehat.

Perbaikannya: di dalam setiap zona, setiap kolom dimodelkan sebagai garis
terhadap beban, dan simpangan diukur dari garis itu. Titik potong garis
diambil dari median residu agar tetap tahan pencilan. Bila beban di dalam
zona nyaris tidak bervariasi, tren tidak dipasang dan baseline kembali
menjadi satu nilai tengah.

Pada sinyal uji, simpangan semu di tepi zona turun dari −2,99 menjadi
−1,33 — dari jauh di atas ambang aturan menjadi jauh di bawahnya. Dua
pengujian mengunci sifat ini: satu memastikan sinyal normal di tepi zona
tidak lagi terlihat menyimpang, satu lagi memastikan koreksi itu tidak
menutupi penyimpangan yang sungguhan.

---

## 5. Pemodelan

### Pembagian berbasis waktu

Latih 2020–2023, validasi 2024, uji 2025–2026. Jeda embargo enam jam
dipotong di kedua sisi setiap perbatasan, karena fitur bergulir selebar
60 menit di awal himpunan uji masih memuat jejak sampel terakhir himpunan
validasi.

Pembagian acak akan menempatkan menit ke-10 dan menit ke-11 dari kejadian
yang sama di dua sisi pembatas. Hasilnya nyaris sempurna dan sepenuhnya
tidak berarti.

### Hanya himpunan latih yang disubsampel

Ini keputusan yang paling menentukan kebenaran hasil.

Himpunan latih disubsampel: seluruh jendela positif dipertahankan, jendela
negatif diambil 2 %, dan menit-menit di perbatasan label dibuang karena
labelnya ambigu.

Himpunan validasi dan uji dibiarkan **utuh**. Alasannya dua:

1. Ambang yang dipilih pada data dengan prevalensi 10 % akan runtuh
   begitu dipakai pada aliran nyata dengan prevalensi 0,1 %.
2. Metrik alarm palsu per hari hanya bermakna di atas deret waktu yang
   tidak berlubang. Data yang disubsampel tidak lagi punya sumbu waktu
   yang utuh.

Konsekuensinya himpunan validasi dan uji berukuran ratusan megabyte.
Keduanya ditulis ke Parquet dan dibaca per potongan saat prediksi,
sehingga puncak pemakaian memori ditentukan oleh besar potongan, bukan
besar himpunan.

### Kalibrasi

Probabilitas dari model yang mengenal datanya sendiri selalu terlalu
percaya diri. Kalibrator dipasang pada data validasi dengan model dasar
dibekukan, memakai contoh berimbang: seluruh baris positif dipertahankan
karena jumlahnya sedikit dan setiap satunya berharga.

Kalibrasi penting karena README §13 memisahkan Risk Score dari Confidence
Score. Skor keyakinan yang dihitung dari probabilitas tak terkalibrasi
tidak berarti apa-apa.

### Pemilihan ambang

Ambang **tidak** dipilih dari nilai bulat yang kelihatan rapi, dan tidak
pula dari titik yang memaksimalkan F1. Ambang dipilih dari sasaran
operasional: cari ambang dengan tingkat deteksi event tertinggi yang
memenuhi **dua** syarat pada `config/thresholds.yaml` sekaligus —
anggaran alarm palsu harian, dan batas porsi waktu sistem boleh berada
dalam kondisi alarm.

Syarat kedua tampak berlebihan sampai celahnya terlihat. Alarm palsu
dihitung per rentetan, bukan per menit, karena operator melihat satu
alarm yang menyala terus. Tetapi hitungan itu sendiri dapat dieksploitasi:
ambang mendekati nol membuat alarm menyala nyaris sepanjang waktu, seluruh
periode menjadi **satu** rentetan raksasa, dan hasilnya tercatat sebagai
"kurang dari satu alarm palsu per hari" sambil mendeteksi setiap event.
Angkanya sempurna; sistemnya tidak berguna.

Dua penutup dipasang. Rentetan yang lebih panjang dari satu jendela
deteksi dihitung sebanyak jendela yang dilaluinya — alarm tiga jam tanpa
sebab bukan satu gangguan tunggal. Dan duty cycle dibatasi secara
eksplisit, sehingga ambang yang membuat sistem berteriak sepanjang hari
langsung gugur berapa pun angka lainnya.

Bila tidak ada satu pun ambang yang memenuhi keduanya, dipilih yang alarm
palsunya paling sedikit di antara yang masih lolos batas duty cycle, dan
**dicatat bahwa sasaran tidak tercapai**. Melonggarkan sasaran diam-diam
agar hasilnya terlihat baik adalah cara tercepat membuat sistem
peringatan dini kehilangan kepercayaan operator.

### Isolation Forest

Dilatih **hanya pada periode normal**. Detektor anomali harus belajar
seperti apa kondisi normal; melatihnya pada data yang berisi event justru
mengajarkan bahwa gangguan itu normal.

---

## 6. Evaluasi

Dua kelompok metrik, karena keduanya menjawab pertanyaan berbeda.

**Per sampel** — precision, recall, F1, PR-AUC, ROC-AUC, Brier score,
calibration error. Menjawab: seberapa baik model memisahkan menit
berisiko dari menit normal.

**Per event** — false alarms per hari, missed event rate, event detection
rate, warning horizon rata-rata dan median. Menjawab pertanyaan yang
sebenarnya ditanyakan operator: berapa banyak event nyata yang
tertangkap, berapa lama sebelumnya, dan berapa sering sistem berteriak
tanpa sebab.

Sebuah event dinyatakan terdeteksi bila ada alarm dalam 180 menit sebelum
waktu mulainya. Warning horizon dihitung dari alarm **pertama** dalam
jendela itu, karena itulah saat operator sebenarnya menerima peringatan.

Alarm berturut-turut dihitung **satu kali**. Operator melihat satu alarm
yang menyala terus, bukan ratusan alarm terpisah; menghitungnya per menit
akan melipatgandakan angka alarm palsu tanpa alasan.

**PR-AUC dan false alarms per hari adalah metrik utama.** Dengan
prevalensi positif 0,09 %, sebuah model yang selalu menjawab "tidak ada
risiko" sudah benar 99,91 % waktu dan akan memperoleh ROC-AUC yang
terlihat mengesankan. PR-AUC tidak dapat dibodohi dengan cara itu.

---

## 7. Mesin risiko hibrida

Tiga sumber digabung dengan bobot pada `config/thresholds.yaml`: rule
engine 0,35, model 0,50, detektor anomali 0,15.

Rule engine mengambil nilai aturan **tertinggi**, bukan penjumlahan. Satu
gangguan yang jelas sudah cukup untuk menaikkan risiko; menjumlahkan
beberapa aturan akan membuat gangguan tunggal yang parah kalah oleh
beberapa gangguan ringan yang tidak berhubungan.

Skor akhir dihaluskan dengan syarat ketahanan sepuluh menit: nilai yang
dilaporkan adalah minimum selama jendela itu, sehingga lonjakan satu
menit akibat derau sensor tidak memicu alarm.

### Tiga skor yang berdiri sendiri

README §13 mewajibkan Risk Score, Confidence Score, dan Data Quality Score
dilaporkan terpisah, dan alasannya penting.

Prediksi risiko yang dihitung dari data setengah hilang atau dari sensor
yang macet tetap menghasilkan angka yang terlihat meyakinkan. Operator
berhak tahu bahwa angka itu berdiri di atas data yang buruk **sebelum**
menindaklanjutinya. Karena itu keyakinan diturunkan bila kualitas data
di bawah 70.

---

## 8. Penjelasan

Skor tanpa alasan tidak dapat ditindaklanjuti: operator tidak bisa
memeriksa "81 persen", ia memeriksa slag pipe 2 karena aliran abunya
berhenti.

Dua lapis penjelasan disajikan bersama:

- **Rule engine** menyebut gejala teknik yang menyala, dalam kalimat yang
  langsung dapat diperiksa di lapangan.
- **SHAP** menyebut fitur yang paling memengaruhi prediksi model.

Aturan didahulukan pada daftar indikator dominan, karena kalimatnya sudah
berbentuk gejala peralatan. SHAP melengkapi bila aturan yang menyala
belum cukup banyak. Nama kolom fitur diterjemahkan ke bahasa peralatan
sebelum ditampilkan — operator tidak seharusnya membaca
`bed_differential_pressure_dev`.

---

## 9. Yang sengaja tidak dilakukan

- **Tidak ada pengisian nilai hilang yang canggih.** Nilai hilang diisi
  nol setelah penskalaan dan ditandai lewat Data Quality Score. Menebak
  nilai sensor yang hilang pada sistem keselamatan berarti mengarang data
  yang akan dipakai mengambil keputusan.
- **Tidak ada penyeimbangan kelas sintetis (SMOTE dan sejenisnya).**
  Menciptakan menit-menit blocking palsu di atas data yang sudah
  sintetis berarti menumpuk karangan di atas karangan. Bobot kelas dan
  subsampel negatif sudah memadai.
- **Tidak ada pencarian hiperparameter besar-besaran.** Menyetel model di
  atas data sintetis berarti menyetelnya terhadap asumsi generator, bukan
  terhadap kenyataan.
- **Tidak ada penggabungan data Unit 2 dan Unit 3.** Ketiganya berbeda
  konfigurasi dan riwayat; menggabungkannya akan mencampur baseline yang
  tidak sebanding.
