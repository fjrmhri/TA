# Frontend Static (Vercel)

Frontend ini adalah static site untuk menampilkan:
- highlight kalimat per paragraf (`red`/`green`/`orange`)
- panel confidence berurutan sesuai urutan kalimat
- panel NER
- ringkasan count hoaks/fakta/low-confidence

## Fitur UI

- Textarea input besar + tombol `Periksa`
- Toggle `Tampilkan NER`
- Input `Backend URL` (override cepat saat deploy)
- Input threshold confidence oranye
- Loader + error box
- Tombol `Reset`
- Tombol `Copy hasil`

## Konfigurasi Backend URL

Default backend URL ada di `app.js`:
```js
const BACKEND_URL = "http://127.0.0.1:7860";
```

Saat runtime, URL bisa diubah lewat field `Backend URL` di UI.

## Skema Data yang Diharapkan

Frontend mengasumsikan backend mengembalikan schema:
- `model`
- `summary`
- `paragraphs[]` dengan `sentences[]`
- `ner`

Tidak ada parsing bebas format; rendering langsung mengikuti schema deterministik API.

## Jalankan Lokal

Cara sederhana:
- buka `frontend/index.html` langsung di browser
- atau serve statis:
  ```bash
  cd frontend
  python -m http.server 5500
  ```

Lalu akses `http://127.0.0.1:5500`.

## Deploy ke Vercel

1. Import repository ke Vercel.
2. Set root directory ke `frontend/`.
3. Framework preset: `Other`.
4. Deploy tanpa build command khusus (static files).
