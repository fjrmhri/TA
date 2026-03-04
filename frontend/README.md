# Frontend Static (Vercel)

Frontend static untuk analisis hoaks per kalimat dengan hasil:
- highlight warna (`red` / `green` / `orange`)
- ringkasan count
- panel `Confidence` (collapsible)
- panel `NER` (collapsible)

## UI

- textarea input
- toggle `Tampilkan NER`
- tombol `Periksa`, `Reset`, `Copy hasil`
- loading state + error card + legend warna

UI tidak memiliki:
- input URL backend
- input threshold warna

## Konfigurasi endpoint

Konfigurasi ada di `app.js`:

```js
const API_BASE_URL = "https://fjrmhri-space-deteksi-hoax-ta.hf.space";
```

Override opsional via query param:
- `?api=https://your-backend-domain`

Contoh:
- `http://localhost:5500/?api=http://127.0.0.1:7860`

## Jalankan lokal

```bash
cd frontend
python -m http.server 5500
```

Lalu buka `http://127.0.0.1:5500`.

## Deploy ke Vercel

1. Import repo ke Vercel.
2. Set root directory ke `frontend/`.
3. Framework preset: `Other`.
4. Deploy sebagai static site (tanpa build command khusus).
