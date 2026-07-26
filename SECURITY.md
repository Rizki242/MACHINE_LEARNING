# Security Policy

FurnaceGuard AI adalah perangkat lunak analisis/advisory untuk lingkungan pembangkit. Repository publik ini tidak boleh memuat:

- data DCS/historian aktual yang bersifat internal,
- file `.env`, token, API key, password, atau private key,
- database lokal, dump historian, atau laporan private,
- artefak model yang dilatih pada data sensitif tanpa proses rilis terpisah.

## Melaporkan isu keamanan

Buka GitHub issue dengan informasi non-sensitif, atau hubungi maintainer secara privat jika isu melibatkan kredensial/data internal.

## Catatan operasional

Software ini tidak boleh dipakai untuk mengubah interlock, proteksi boiler, atau perintah kontrol. Semua rekomendasi wajib diverifikasi oleh engineer/operator sesuai SOP pembangkit.
