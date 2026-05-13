# 🥭 MangoScan — Ripeness Detector

Real-time mango ripeness classifier using EfficientNetB0 + Flask.

## Project Structure

```
mango_app/
├── app.py                              ← Flask backend (loads .h5 model)
├── requirements.txt
├── EfficientNetB0_mango_ripeness.h5   ← YOUR MODEL (place here)
└── templates/
    └── index.html                     ← Frontend UI
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your model
Copy `EfficientNetB0_mango_ripeness.h5` into the `mango_app/` folder
(same directory as `app.py`).

### 3. Run the server
```bash
cd mango_app
python app.py
```

### 4. Open in browser
```
http://localhost:5000
```

## Classes
The model predicts one of 4 classes:
- `ripe`
- `partially_ripe`
- `unripe`
- `rotten`

> If your model's class order is different, update `CLASS_LABELS` in `app.py`.

## API

### `POST /predict`
Send a multipart form with an `image` field.

**Response:**
```json
{
  "prediction": "ripe",
  "confidence": 97.43,
  "scores": {
    "partially_ripe": 0.21,
    "ripe": 97.43,
    "rotten": 1.12,
    "unripe": 1.24
  }
}
```

### `GET /health`
Returns `{"status": "ok", "model": "EfficientNetB0 ready ✓"}` when the model is loaded.
