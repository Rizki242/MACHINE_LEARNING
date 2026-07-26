# Batasan — FurnaceGuard AI Fase 0

Dokumen ini menyatakan apa yang **belum** dapat dilakukan sistem ini dan
angka mana yang **tidak boleh** dikutip. Bacalah sebelum menggunakan
keluaran apa pun dari repositori ini.

---

## 1. Batasan terbesar: model dilatih di atas data sintetis

Tidak ada satu pun timeseries DCS pada data yang tersedia. Seluruh model,
seluruh fitur, dan seluruh metrik pada `backend/reports/model_evaluation.csv`
dihitung dari data yang **dibangkitkan program**, bukan diukur di
pembangkit.

**Yang dibuktikan angka-angka itu:** pipeline berjalan benar dari ujung ke
ujung — data masuk, fitur terbentuk tanpa kebocoran masa depan, model
terlatih, ambang terpilih, metrik terhitung.

**Yang TIDAK dibuktikan angka-angka itu:** kemampuan mendeteksi blocking
di PLTU Jeranjang Unit 1. Nilainya bisa jauh lebih baik atau jauh lebih
buruk. Belum ada dasar untuk menduga ke arah mana.

> Dilarang mengutip PR-AUC, recall, event detection rate, warning horizon,
> atau false alarms per day dari fase ini sebagai unjuk kerja sistem.
> Setiap penyajian angka tersebut wajib menyertakan keterangan bahwa
> sumbernya data sintetis.

Generator sintetis membangun sinyal dari spesifikasi desain boiler dan
menyuntikkan pola degradasi sesuai README §6 sebelum setiap event nyata.
Pola itu adalah **hipotesis engineering**, bukan pengamatan. Bila
tanda tangan gangguan sesungguhnya berbeda — dan besar kemungkinan
berbeda — model yang dilatih di sini tidak akan mengenalinya.

---

## 2. Kelas yang tidak dapat dilatih sama sekali

**Blocking bottom ash / cold slag pipe: nol contoh.** Sepanjang sepuluh
tahun jurnal gangguan Unit 1, tidak ada satu pun catatan blocking slag
pipe.

Akibatnya seluruh target berikut pada README §5 tidak punya label:

```text
slag_pipe_1_blocking_probability    slag_pipe_2_blocking_probability
slag_pipe_3_blocking_probability    slag_pipe_4_blocking_probability
```

Aturan rule engine untuk kelas ini tetap ditulis di
`config/thresholds.yaml` dan akan menyala bila gejalanya muncul, tetapi
**ambangnya belum pernah divalidasi terhadap satu kejadian pun**.

Hal serupa berlaku, dengan tingkat lebih ringan, untuk
`return_system_disturbance`: hanya 3 event dalam sepuluh tahun. Terlalu
sedikit untuk melatih maupun mengukur.

---

## 3. Batasan label

Label dibentuk dari **waktu mulai event pada jurnal gangguan**, bukan dari
saat gangguan sebenarnya mulai berkembang. Jurnal mencatat kapan derating
atau outage dinyatakan — biasanya jauh setelah proses fisiknya bermula.

Konsekuensinya:

- Warning horizon yang terukur cenderung **lebih panjang** daripada
  kemampuan sebenarnya, karena titik acuannya sudah bergeser ke belakang.
- Menit-menit awal perkembangan gangguan mungkin terlabeli negatif
  padahal kondisinya sudah menyimpang.

Perbaikan hanya mungkin bila tersedia catatan waktu yang lebih dekat ke
kejadian fisik: log poking, catatan shift operator, atau penandaan
manual oleh engineer terhadap data historian.

---

## 4. Batasan klasifikasi teks

Klasifikasi event bertumpu pada kata kunci berbahasa Indonesia yang
dikumpulkan dari teks yang benar-benar muncul di sumber. Kelemahannya:

- Istilah baru yang belum masuk daftar akan jatuh ke `unclassified`.
- 12 event memang belum terklasifikasi dan menunggu tinjauan engineer.
- Lapis kedua (TF-IDF karakter n-gram) dilatih pada label lapis pertama.
  Ia hanya dapat menemukan pola yang mirip label yang sudah ada; ia
  **tidak dapat menemukan kelas yang belum pernah dikenali**.
- 19 event `coal_quality_derating` diberi label dari frasa seperti
  "pressure main steam turun karena kondisi batubara". Hubungannya ke
  risiko blocking bersifat dugaan, bukan terukur.

Seluruh event bertanda `needs_review` di
`backend/reports/events_needing_review.csv` **harus** diverifikasi
engineer sebelum registry ini dipakai sebagai dasar keputusan.

---

## 5. Batasan ambang dan baseline

Seluruh angka pada `config/thresholds.yaml` adalah **konfigurasi awal
berbasis pemahaman teknik CFB umum**, bukan hasil pengukuran Unit 1.
Demikian pula pembagian lima zona beban pada `config/units.yaml` — README
§10 sendiri menyatakan pembagian itu harus divalidasi dengan data operasi
aktual.

Baseline per zona beban saat ini dihitung dari data sintetis. Begitu data
nyata tersedia, seluruh baseline harus dihitung ulang dan seluruh ambang
ditinjau ulang.

Batas rentang fisik pada `units.yaml` (`plausible_ranges`) dipakai modul
kualitas data untuk mendeteksi pembacaan mustahil. **Batas itu bukan
batas alarm dan bukan batas operasi.**

---

## 6. Batasan cakupan fase ini

Belum ada:

- **API.** Tidak ada FastAPI, tidak ada endpoint, tidak ada `main.py`.
  Modul dipakai lewat skrip dan notebook.
- **Antarmuka pengguna.** Tidak ada React, tidak ada dashboard.
- **Basis data.** Tidak ada SQLite maupun PostgreSQL; keluaran berupa
  berkas.
- **Sambungan historian.** Tidak ada OPC-UA, tidak ada koneksi ke DCS.
  Mode deployment terkunci pada `offline` dan pengujian memastikannya.
- **Multi-unit.** Hanya Unit 1. Data Unit 2 dan Unit 3 ada di berkas
  sumber tetapi tidak diproses.
- **Autentikasi, peran pengguna, audit log.** Seluruhnya bagian dari
  README §22 versi 1.0.

---

## 7. Batasan teknis

- Fitur dibangun per potongan tahun. Sekitar **60 menit pertama setiap
  tahun** punya riwayat tidak lengkap, sehingga sebagian fitur bergulir
  kosong di sana.
- Data sintetis dibangkitkan untuk **2020–2026 saja**. Event 2016–2019 ada
  di Event Registry dan ikut dianalisis, tetapi tidak punya timeseries
  pendamping.
- Baris saat unit berhenti dikeluarkan dari pelatihan. Risiko blocking
  saat start-up karena itu **tidak dimodelkan**, padahal zona startup
  justru periode rawan pada CFB.
- Isolation Forest dilatih pada maksimum 120.000 baris dan kalibrator pada
  maksimum 60.000 baris, demi menekan pemakaian memori.

---

## 8. Batasan yang bersifat permanen

Berikut ini bukan kekurangan yang akan diperbaiki, melainkan garis yang
memang tidak boleh dilewati (README §20 dan §24).

FurnaceGuard AI **tidak boleh** dan tidak akan:

- mengubah atau menggantikan interlock, proteksi, MFT, atau boiler trip
  logic;
- mengendalikan fan, coal feeder, atau ash valve secara otomatis;
- mengubah set point boiler;
- menggantikan SOP, operator, atau engineer boiler;
- menjamin blocking akan atau tidak akan terjadi.

Sistem ini hanya membaca data dan menghasilkan saran. Keputusan akhir
berada pada operator dan engineer yang berwenang.
