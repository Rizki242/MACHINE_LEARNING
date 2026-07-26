# Contributing to FurnaceGuard AI

Terima kasih sudah tertarik berkontribusi. Proyek ini berada pada domain industrial safety, sehingga perubahan harus mengutamakan kejelasan data, auditability, dan batasan operasional.

## Prinsip kontribusi

1. Sistem ini adalah decision-support, bukan sistem kontrol boiler.
2. Jangan menghapus disclaimer data sintetis dari metrik model.
3. Jangan commit data pembangkit aktual, file `.env`, kredensial, model private, atau output runtime besar.
4. Konfigurasi threshold/model berada di `config/*.yaml`; hindari hard-code di Python kecuali benar-benar konstan.
5. Fitur rolling harus backward-looking untuk mencegah leakage.

## Setup lokal

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
python -m pip install -e .[dev]
```

## Verifikasi sebelum pull request

```bash
python -m pytest backend/tests -q
```

Jika menjalankan pipeline penuh, pastikan hasil besar tetap ter-ignore oleh `.gitignore`.
