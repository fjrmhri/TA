# Frontend Static (Vercel)

Frontend ini adalah static site untuk menampilkan:
- highlight kalimat per paragraf (`red`/`green`/`orange`)
- panel confidence berurutan sesuai urutan kalimat
- panel NER
- ringkasan count hoaks/fakta/low-confidence

## Fitur UI

- Textarea input besar + tombol `Periksa`
- Toggle `Tampilkan NER`
- Loader + error box
- Tombol `Reset`
- Tombol `Copy hasil`

## Konfigurasi Endpoint Backend (non-UI)

Endpoint diatur langsung di `app.js`:
```js
const BACKEND_ANALYZE_URL_DEV = "http://127.0.0.1:7860/analyze";
const BACKEND_ANALYZE_URL_PROD = "https://fjrmhri-space-deteksi-hoax-ta.hf.space/analyze";
const IS_LOCAL = ["localhost", "127.0.0.1", "::1"].includes(window.location.hostname);
const ANALYZE_ENDPOINT = IS_LOCAL ? BACKEND_ANALYZE_URL_DEV : BACKEND_ANALYZE_URL_PROD;
```

Sebelum deploy, ubah `BACKEND_ANALYZE_URL_PROD` bila endpoint produksi Anda berubah.

Catatan threshold warna:
- Threshold oranye default berasal dari backend (`ORANGE_THRESHOLD=0.65`).
- Frontend tidak menampilkan input threshold dan tidak mengirim field threshold pada request.

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
