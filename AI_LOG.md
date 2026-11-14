# AI_LOG — Tim 14 (spectro-fan-squad)

Satu baris per pemakaian AI yang bermakna (ChatGPT, Claude, Copilot, Bravais, dan sejenisnya).
Ringkas dan jujur; transkrip penuh tidak diminta. Yang dinilai bukan "pakai AI atau tidak",
melainkan "mengerti atau tidak" — isi kolom terakhir dengan apa yang Anda ubah dari keluarannya.

| Tanggal | Siapa (NIM) | Alat | Untuk apa | Apa yang diubah dari keluarannya |
| --- | --- | --- | --- | --- |
| 2026-11-12 | 123450096 | Antigravity (Swarm) | Audit pemotong jendela (`src/protokol/data.py`), `PROTOKOL.md`, dan `rencana/t14-anomali-suara.yaml` | tolak ubah `sprint/sprint0/123450027.md` & `models/123450013/run.py`; rapikan header `PROTOKOL.md`; kerjakan `git push` & PR sendiri |
| 2026-11-12 | 123450096 | Antigravity (Swarm) | Kerangka Transformer (`models/123450096/run.py` dan `catatan.md`) | tolak fallback sintetis AI (`_buat_data_simulasi`, `--sim-*`, monkeypatch `evaluate`) dan arahkan penghapusannya; pertahankan `PatchTransformer` + `data.muat()` & terima pembaruan `catatan.md` |
