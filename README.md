# ReTracer AI — Spatial Reasoning Geolocation Engine

> **Upload a memory. We retrace the map.**
> A privacy-first AI detective that reverse-geolocates any photo — no GPS tags required.

Built for the **Meta AI Hackathon**. ReTracer AI dissects a photo with SAM 3.1 segmentation, reasons like Sherlock Holmes with Muse Wonder, and returns a city, coordinates, confidence score, and a human-readable deduction trail — all behind strict privacy gates.

---
Live Demo: https://retracerai.streamlit.app/

## Inspiration

We capture thousands of photographs to freeze moments in time, but as years pass, the context of those memories often fades. For individuals experiencing old age, cognitive decline, or conditions like dementia, a cherished old photograph can become an isolated, frustrating artifact. The faces might be familiar, but the fundamental question *"Where was this?"* is lost.

Current reverse-image search and standard AI models fail to solve this because they only look at macro-structures, hallucinating answers when they lack obvious landmarks. They cannot deduce meaning from the micro-details of a forgotten past.

ReTracer AI was built to anchor lost memories. I wanted to engineer an AI that doesn't just blindly guess, but *reasons* like a detective. By combining SAM 3.1's micro-detail extraction with a rigorous multi-agent debate loop, this engine identifies the specific utility poles, foliage, and architectural nuances in a faded photograph to mathematically deduce its coordinates. 

It is designed to help individuals, families, and caretakers piece together the geography of their history—turning lost landscapes back into relived experiences.

We were inspired by:

- How much location leaks from pixels alone, even after EXIF/GPS metadata is stripped.
- The detective work of open-source geolocation — cross-referencing architecture, biomes, infrastructure, and text.
- Meta's new Model API lineup: **SAM 3.1** for pixel-perfect segmentation and **Muse Wonder** for deep spatial reasoning.

So we built ReTracer AI: forensic-grade geolocation with mathematical privacy guarantees.

## What It Does

Upload any street, landscape, or travel photo (plus an optional regional hint) and ReTracer AI:

1. **Sanitizes locally** — strips all EXIF/metadata, rebuilds a clean RGB image (never resized), rejects non-images and files over 10 MB.
2. **Screens faces** — fully offline Haar-cascade detection. Faces covering > 5% of the frame are **rejected**; smaller faces are **heavily blurred** before anything else runs.
3. **Gates visual quality** — Pillow-only variance + entropy check halts solid/blank/blurry images with no API call wasted.
4. **Segments with SAM 3.1** — full frame plus a 2×2 quadrant grid (5 tiles max, 5 concurrent requests), one concept phrase per request across architecture, flora/soil, infrastructure, signage/scripts, and transport. Only masks scoring **> 0.90 confidence** survive.
5. **Reasons with Muse Wonder** — weighs 3 competing location hypotheses against every clue (solar geometry, pole architecture, driving side, plate colors, vegetation, scripts/OCR, coastal and transit markers), eliminates the weak ones, and outputs strict JSON: `latitude`, `longitude`, `city`, `country`, `reasoning_steps`, `confidence_score`, `precision_level`.
6. **Protects the exact fix** — coordinates are rounded to **3 decimals (~100 m grid)** before they ever reach the browser.
7. **Renders the verdict** — a glassmorphism dashboard with location, precision badge, confidence, a "Detective's Deduction" timeline, and an interactive Folium map.

Two modes:

| Mode | What happens | Needs API key? |
|---|---|---|
| **Demo (Developer Mode)** | Instant hardcoded Utrecht walkthrough, no network | No |
| **Live** | Real SAM 3.1 + Muse Wonder pipeline | Yes (`META_API_KEY`) |

## How We Built It

### Pipeline

```text
Upload → Validate → Sanitize → Quality Gate → Face Screen
  → SAM 3.1 (full + 4 quadrants, streamed) → Summarize Masks
  → Muse Wonder (3-hypothesis triangulation + OCR)
  → Round Coords (3 decimals) → Dashboard + Folium Map
```

- `app.py` — Streamlit UI, glassmorphism theme, Lottie radar loader, animated milestone progress, result dashboard.
- `sam_segmentation.py` — async SAM 3.1 client: tiled segmentation, streamed output-text parsing via `meta_sam_parser`, confidence filtering, box remapping to full-frame pixels.
- `spatial_reasoning.py` — strict Muse Wonder prompter (triangulation heuristics, coastal/infrastructure/architectural/transit/cultural taxonomies, OCR + Tier-1 text anchoring) with rigorous JSON validation and halt conventions (`null` coords only for pitch-black/synthetic images).
- `backend.py` — FastAPI `/analyze` service: decodes the screened upload, runs quality → SAM → Wonder, applies the privacy rounding.
- `analysis.py` — frontend orchestrator: mock result or HTTP POST to the backend.
- `face_screening.py` — offline OpenCV Haar-cascade face gate + Pillow Gaussian blur.
- `file_sanitizer.py` — extension/MIME/size/magic-byte validation, metadata-stripping RGB rebuild, clean JPEG encoding.
- `image_quality.py` — grayscale variance + histogram entropy gate.

### Tech Stack (exact, from `requirements.txt`)

| Layer | Technology |
|---|---|
| Language | Python 3 (tested on 3.14.6) |
| Frontend | `streamlit>=1.45.0`, `streamlit-option-menu>=0.4.0`, `streamlit-extras>=0.7.0`, `streamlit-shadcn-ui` (git: `ObservedObserver/streamlit-shadcn-ui`), `streamlit-antd-components>=0.3.2`, `streamlit-lottie>=0.0.5`, `streamlit-folium>=0.15.0`, `folium`, `pandas` |
| Backend | `fastapi>=0.110.0`, `uvicorn>=0.29.0` |
| Meta AI | Model API `sam-3.1` + `muse-wonder-1.0` via `aiohttp>=3.9`, `meta-sam-parser>=0.0.5`, `tenacity>=9.0` (2 retries, 2s/4s backoff on 429/503/504) |
| Vision | `opencv-python-headless>=4.8,<5` (Haar cascade, offline), `pillow>=10.0.0` |
| Config | `.streamlit/secrets.toml` (`META_API_KEY`), `META_API_KEY` env var for backend, `SPATIAL_BACKEND_URL` (default `http://localhost:8000`) |

Meta API contract references (from code comments): [Segmenting](https://dev.meta.ai/docs/sam/segmenting), [Reading segmentation](https://dev.meta.ai/docs/sam/reading-segmentation), [Client libraries](https://dev.meta.ai/docs/sam/client-libraries), [Models](https://dev.meta.ai/docs/models).

## Installation & Local Setup

### Prerequisites

- **Python 3.10+** ([python.org](https://www.python.org/downloads/)) and `pip`
- **Git**
- A **Meta Model API key** with access to `sam-3.1` and `muse-wonder-1.0` (only needed for Live mode; Demo mode works without it)

### Step-by-step

**1. Clone the repository**

```bash
git clone <your-repo-url>
cd SpatialGeoguesser
```

**2. Create and activate a virtual environment**

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Add your Meta API key (Live mode only)**

Create `.streamlit/secrets.toml` (this file is git-ignored):

```toml
META_API_KEY = "your-meta-model-api-key-here"
```

The FastAPI backend reads the key from the environment instead of Streamlit secrets, so also set it in each backend terminal:

Windows (PowerShell):

```powershell
$env:META_API_KEY = "your-meta-model-api-key-here"
```

macOS / Linux:

```bash
export META_API_KEY="your-meta-model-api-key-here"
```

**5. Start the backend (Terminal 1)**

```bash
uvicorn backend:app --host 0.0.0.0 --port 8000
```

Leave this running. The frontend defaults to `http://localhost:8000`; override with `SPATIAL_BACKEND_URL` if needed:

```powershell
# Windows (PowerShell), before launching Streamlit
$env:SPATIAL_BACKEND_URL = "http://localhost:8000"
```

**6. Start the frontend (Terminal 2, venv activated)**

```bash
streamlit run app.py
```

**7. Open the app**

Go to **http://localhost:8501** in your browser.

- Keep **Developer Mode ON** for the instant Utrecht demo (no key, no backend needed).
- Turn it **OFF**, upload a JPG/PNG/WebP (≤ 10 MB), add an optional hint (e.g. `Kerala backwaters`), and hit **Analyze** for the live pipeline.

### Troubleshooting

| Symptom | Fix |
|---|---|
| `Missing Meta Model API key` | Check `.streamlit/secrets.toml` (frontend) and `$env:META_API_KEY` (backend terminal). |
| `Backend unavailable at ...` | Start step 5 first; confirm port 8000 is free and `SPATIAL_BACKEND_URL` matches. |
| `File exceeds 10 MB limit` | Compress/resize the image; the cap is intentional. |
| `Privacy warning: face covers ...` | Crop or blur faces and re-upload; large faces are never processed. |
| `Image lacks sufficient visual data` | Upload a sharper, more detailed photo. |
| `API congested, retrying...` | Transient 429/503/504 — the app retries automatically. |

## Privacy & Safety

- Metadata (EXIF/GPS) is destroyed on upload — geolocation uses **pixels only**.
- Faces are screened **before** any AI call; oversized faces halt the run.
- Exact Wonder coordinates are rounded to ~100 m server-side; the browser never sees the raw fix.
- Blank/synthetic images halt with a `null`-coordinate convention instead of a fake `0.0, 0.0`.

## Project Structure

```text
SpatialGeoguesser/
├── app.py                 # Streamlit frontend + dashboard + maps
├── backend.py             # FastAPI /analyze service + privacy rounding
├── analysis.py            # Mock vs. backend orchestrator
├── sam_segmentation.py    # SAM 3.1 tiled segmentation client
├── spatial_reasoning.py   # Muse Wonder prompting + validation
├── face_screening.py      # Offline face gate + blur
├── file_sanitizer.py      # Upload validation + metadata stripping
├── image_quality.py       # Variance/entropy visual-data gate
├── requirements.txt       # Exact dependency pins
├── assets/                # Demo image (demo-utrecht-street.jpg)
└── tests/                 # pytest suite (19 test files)
```

## What's Next

- Multi-image trip stitching (one route from a photo album).
- Confidence-calibrated search radius overlay on the map.
- Regional model packs (finer taxonomies per continent).
