"""Basic ReTracer AI upload app."""

import asyncio
import html
import json
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import streamlit as st

from analysis import execute_analysis
from backend import build_frontend_result, round_wonder_coordinates
from face_screening import screen_image_faces
from file_sanitizer import is_valid_image, sanitize_uploaded_image
from image_quality import has_sufficient_visual_data
from sam_segmentation import (
    DENSE_IMAGE_MESSAGE,
    RETRY_STATUS_TEXT,
    SamApiError,
    SamDenseImageError,
    SamSegmenter,
    draw_mask_boxes,
    get_meta_api_key,
    summarize_masks,
)
from spatial_reasoning import (
    LocationUnknownError,
    SpatialReasoner,
    SpatialReasoningError,
)

try:
    import streamlit_shadcn_ui as ui
except ImportError:  # pragma: no cover
    ui = None

try:
    from streamlit_lottie import st_lottie
except ImportError:  # pragma: no cover
    st_lottie = None

try:
    import streamlit_antd_components as sac
except Exception:  # pragma: no cover - missing OR broken install degrades
    sac = None

try:
    from streamlit_extras.metric_cards import style_metric_cards
except ImportError:  # pragma: no cover
    style_metric_cards = None

try:
    import folium
except ImportError:  # pragma: no cover
    folium = None

try:
    from streamlit_folium import st_folium
except ImportError:  # pragma: no cover
    st_folium = None

NEON_ACCENT = "#00FFAA"
METRIC_CARD_BACKGROUND = "#1E1E1E"

LOTTIE_RADAR_URL = (
    "https://raw.githubusercontent.com/"
    "soheltanvir0699/radar-scan-notification-logger/main/"
    "app/src/main/assets/lottie_radar.json"
)
LOTTIE_SEARCH_EMPTY_URL = (
    "https://raw.githubusercontent.com/ramani888/biblestudy/master/"
    "frontend/src/assets/animations/empty.json"
)
LOTTIE_FETCH_TIMEOUT = 10.0
LOTTIE_TINT_RGB = (0.0, 1.0, 170 / 255)
# Neon accent (#00FFAA) as Lottie 0-1 RGB floats.


def _is_lottie_payload(data) -> bool:
    """True when fetched JSON has the markers of a Lottie animation."""
    return (
        isinstance(data, dict)
        and isinstance(data.get("v"), str)
        and isinstance(data.get("layers"), list)
    )

def _tint_lottie_static_color(node: dict) -> None:
    """Remap an exact-white static color node to the neon tint in place."""
    key = node.get("k")
    if (
        isinstance(key, list)
        and len(key) == 4
        and all(channel == 1 for channel in key[:3])
    ):
        node["k"] = [*LOTTIE_TINT_RGB, key[3]]


def _tint_lottie_keyframes(node: dict) -> None:
    """Remap exact-white animated color keyframes to the neon tint."""
    frames = node.get("k")
    if not isinstance(frames, list):
        return
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        for edge in ("s", "e"):
            stop = frame.get(edge)
            if (
                isinstance(stop, list)
                and len(stop) == 4
                and all(channel == 1 for channel in stop[:3])
            ):
                frame[edge] = [*LOTTIE_TINT_RGB, stop[3]]


def tint_lottie_white_fills(payload: dict) -> dict:
    """Recolor white vector artwork in a Lottie payload to neon green.

    The remote radar asset ships all-white fills meant to be tinted;
    untinted it renders white against the dark theme. Only exact-white
    RGB is remapped (alpha preserved); every other color passes through
    untouched. Mutates and returns the payload.
    """
    def walk(node) -> None:
        if isinstance(node, dict):
            color = node.get("c")
            if isinstance(color, dict):
                if color.get("a"):
                    _tint_lottie_keyframes(color)
                else:
                    _tint_lottie_static_color(color)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(payload)
    return payload


@st.cache_data(show_spinner=False, ttl=3600)
def load_lottie_animation(url: str = LOTTIE_RADAR_URL) -> dict | None:
    """Fetch a Lottie JSON animation; None when unavailable.

    Any failure (network, timeout, invalid JSON, non-Lottie payload)
    returns None so callers fall back to the local scan sweep.
    White vector artwork in a valid payload is tinted to the neon
    accent so the asset matches the dark theme.
    """
    try:
        with urllib.request.urlopen(url, timeout=LOTTIE_FETCH_TIMEOUT) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    if not _is_lottie_payload(data):
        return None
    return tint_lottie_white_fills(data)


def show_centered_lottie(
    slot, animation: dict, key: str = "analyzing_lottie"
) -> None:
    """Render a Lottie animation centered inside a placeholder slot."""
    with slot.container():
        _, center, _ = st.columns([1, 2, 1])
        with center:
            st_lottie(animation, height=240, key=key)


SCAN_FALLBACK_HTML = '<div class="scan-radar" aria-hidden="true"></div>'


def show_scan_fallback(slot) -> None:
    """Render the black CSS radar sweep while analysis runs."""
    with slot.container():
        _, center, _ = st.columns([1, 2, 1])
        with center:
            st.markdown(SCAN_FALLBACK_HTML, unsafe_allow_html=True)


MILESTONE_INIT = (0, "Initializing spatial analysis...")
# SAM starts near zero so the bar climbs instead of jumping from 0%.
MILESTONE_SAM = (5, "Segmenting image and extracting infrastructural clues...")
MILESTONE_WONDER = (
    45,
    "Cross-referencing biomes, architecture, and topography...",
)
MILESTONE_DONE = (100, "Analysis complete.")
COMPLETION_HOLD_SECONDS = 0.5
SAM_PROGRESS_TICK_SECONDS = 2.0
SAM_PROGRESS_TICK_MAX_STEP = 3
SAM_PROGRESS_CAP = 44
WONDER_PROGRESS_TICK_SECONDS = 2.0
WONDER_PROGRESS_TICK_MAX_STEP = 3
WONDER_PROGRESS_CAP = 95
MILESTONE_WONDER_DONE = (WONDER_PROGRESS_CAP, MILESTONE_WONDER[1])
SAM_TICK_MESSAGES = (
    "SAM 3.1 is dissecting the image...",
    "SAM 3.1 is acting like a detective...",
    "Scanning the scene for infrastructural clues...",
    "Segmenting the scene into a thousand puzzle pieces...",
    "Searching for hidden regional artifacts and text...",
    "Evaluating biome, vegetation, and topography...",
    "Mapping shapes, edges, and spatial structure...",
    "Reading light, color, and atmosphere...",
    "Cataloguing every visible clue and pattern...",
)
WONDER_TICK_MESSAGES = (
    "ReTracer AI is now on the case...",
    "Cross-referencing global architectural patterns...",
    "Triangulating position from every available cue...",
    "Weighing natural versus built-environment signals...",
    "Hunting for text, symbols, and regional markers...",
    "Running multi-hypothesis spatial triangulation...",
    "Eliminating false leads and impossible biomes...",
    "Connecting the dots on a digital corkboard...",
    "Piecing together the final coordinates...",
    "Narrowing down the regional radius...",
    "Zeroing in on the exact coordinates...",
    "Locking onto the target grid...",
    "Preparing the final spatial verdict...",
)


def report_analysis_milestone(
    status_slot, progress_bar, percent: int, message: str
) -> None:
    """Update the status text and the progress bar together, in place.

    Called directly from the main execution flow as the pipeline hits
    real backend milestones; no threads, timers, or countdowns.
    """
    status_slot.markdown(
        f'<div class="loading-text">{percent}% - {message}</div>',
        unsafe_allow_html=True,
    )
    progress_bar.progress(percent)


BACKGROUND_IMAGE_URL = (
    "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b"
    "?auto=format&fit=crop&w=2000&q=80"
)
MOCK_DEMO_IMAGE_PATH = "assets/demo-utrecht-street.jpg"

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500;600&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

.stApp {
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.55) 0%, rgba(15, 23, 42, 0.75) 100%),
                url('""" + BACKGROUND_IMAGE_URL + """') no-repeat center center fixed;
    background-size: cover;
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #f8fafc;
}

h1, h2, .main-title {
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 300 !important;
    letter-spacing: -0.02em !important;
    color: #ffffff !important;
    text-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}

/* Dark encapsulation layer: unified opaque container for the upload section. */
div[data-testid="stVerticalBlockBorderWrapper"] > div {
    background: rgba(10, 14, 26, 0.88) !important;
    backdrop-filter: blur(18px) saturate(140%);
    -webkit-backdrop-filter: blur(18px) saturate(140%);
    border: 1px solid rgba(255, 255, 255, 0.16) !important;
    border-radius: 20px;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.45) !important;
    padding: 1.75rem !important;
}

section[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.65) !important;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
}

.glass-card, div[data-testid="stExpander"], div[data-testid="stSidebar"] > div {
    background: rgba(255, 255, 255, 0.07) !important;
    backdrop-filter: blur(20px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25) !important;
    padding: 1.5rem !important;
}

/* Upload a memory: outer wrapper stays invisible so only one seamless dropzone shows. */
div[data-testid="stFileUploader"] {
    background: transparent !important;
    background-image: none !important;
    border: none !important;
    box-shadow: none !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
    padding: 0 !important;
}

/* Remove the rogue white help/tooltip circle in the uploader header. */
div[data-testid="stFileUploader"] [data-testid="stTooltipIcon"],
div[data-testid="stFileUploader"] .stTooltipIcon,
div[data-testid="stFileUploader"] label[data-testid="stWidgetLabel"] button,
div[data-testid="stFileUploader"] div[data-testid="stWidgetLabel"] button,
div[data-testid="stFileUploader"] label button {
    display: none !important;
}

/* High-contrast labels for toggles and widgets */
div[data-testid="stWidgetLabel"] p,
.stToggle label p,
label[data-testid="stWidgetLabel"] {
    color: #ffffff !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    text-shadow: 0 2px 8px rgba(0, 0, 0, 0.7) !important;
}

/* Centered frosted glass pill for toggle */
div[data-testid="stCheckbox"] {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    background: rgba(255, 255, 255, 0.08) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
    border-radius: 50px !important;
    padding: 0.6rem 1.5rem !important;
    margin: 1rem auto !important;
    width: fit-content !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25) !important;
}

/* Ensure label inside pill is crisp */
div[data-testid="stCheckbox"] label span p {
    color: #ffffff !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
}

/* Horizontal capsule navbar with translucent active pill */
div[data-testid="stHorizontalBlock"] {
    justify-content: center !important;
}

div[data-testid="stRadio"] {
    display: flex !important;
    justify-content: center !important;
    width: 100% !important;
}

div[data-testid="stRadio"] > div[role="radiogroup"] {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    white-space: nowrap !important;
    align-items: center !important;
    justify-content: center !important;
    background: #0d0e12 !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 50px !important;
    padding: 6px 10px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5) !important;
    gap: 8px !important;
    width: fit-content !important;
    margin: 0 auto 2rem auto !important;
}

/* Hide default radio circles completely */
div[data-testid="stRadio"] div[role="radiogroup"] label > div:first-child {
    display: none !important;
}

/* Pill tab labels locked to a single row */
div[data-testid="stRadio"] div[role="radiogroup"] label {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 30px !important;
    padding: 8px 16px !important;
    color: rgba(255, 255, 255, 0.7) !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    cursor: pointer !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    white-space: nowrap !important;
    flex-shrink: 0 !important;
}

/* Crisp white inline-SVG vector icons per tab */
div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-of-type(1)::before {
    content: "" !important;
    width: 16px !important;
    height: 16px !important;
    flex-shrink: 0 !important;
    background: url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%23ffffff' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z'/><circle cx='12' cy='13' r='4'/></svg>") no-repeat center / contain !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-of-type(2)::before {
    content: "" !important;
    width: 16px !important;
    height: 16px !important;
    flex-shrink: 0 !important;
    background: url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%23ffffff' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M1 6v16l7-4 8 4 7-4V2l-7 4-8-4-7 4z'/><line x1='8' y1='2' x2='8' y2='18'/><line x1='16' y1='6' x2='16' y2='22'/></svg>") no-repeat center / contain !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] > label:nth-of-type(3)::before {
    content: "" !important;
    width: 16px !important;
    height: 16px !important;
    flex-shrink: 0 !important;
    background: url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%23ffffff' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z'/></svg>") no-repeat center / contain !important;
}

/* Selected active tab: translucent frosted-glass pill */
div[data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"],
div[data-testid="stRadio"] div[role="radiogroup"] input:checked + div {
    background: rgba(255, 255, 255, 0.18) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    color: #ffffff !important;
    padding: 8px 22px !important;
    border-radius: 30px !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 15px rgba(255, 255, 255, 0.15) !important;
}

/* Hover effect for unselected tabs */
div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
    color: #ffffff !important;
    background: rgba(255, 255, 255, 0.08) !important;
}

/* Hide Streamlit default top header bar completely */
header[data-testid="stHeader"],
div[data-testid="stHeader"],
div[data-testid="stToolbar"],
.stAppHeader {
    background-color: transparent !important;
    background: transparent !important;
    display: none !important;
    visibility: hidden !important;
    height: 0px !important;
}

/* Remove the extra top white padding forced by Streamlit */
.main .block-container,
div[data-testid="stAppViewBlockContainer"] {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    margin-top: 0px !important;
}

/* Ensure the app canvas touches the absolute top edge */
.stApp {
    top: 0 !important;
    margin-top: 0 !important;
}

/* Centered Single Glass Container Card */
.upload-card {
    background: rgba(15, 23, 42, 0.6) !important;
    backdrop-filter: blur(20px) saturate(160%) !important;
    -webkit-backdrop-filter: blur(20px) saturate(160%) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 24px !important;
    padding: 2rem !important;
    max-width: 620px !important;
    margin: 1rem auto 2rem auto !important;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4) !important;
}

/* Center the bordered upload form card at readable width */
div[data-testid="stVerticalBlockBorderWrapper"] {
    max-width: 620px !important;
    margin: 1rem auto 2rem auto !important;
}

/* High contrast white labels */
.upload-card label,
div[data-testid="stWidgetLabel"] p,
label[data-testid="stWidgetLabel"] {
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
    margin-bottom: 0.5rem !important;
}

/* Center "Upload a memory" / "Regional hint" titles above their inputs */
div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stWidgetLabel"],
div[data-testid="stVerticalBlockBorderWrapper"] label[data-testid="stWidgetLabel"] {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    text-align: center !important;
    width: 100% !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stWidgetLabel"] p,
div[data-testid="stVerticalBlockBorderWrapper"] label[data-testid="stWidgetLabel"] p,
div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stFileUploader"] label p,
div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stTextInput"] label p {
    text-align: center !important;
    width: 100% !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

/* Seamless single-surface file dropzone: one dashed border, no nested card. */
div[data-testid="stFileUploader"] {
    margin-bottom: 1.5rem !important;
}

div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"],
div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] {
    background: rgba(0, 0, 0, 0.2) !important;
    background-image: none !important;
    border: 1px dashed rgba(255, 255, 255, 0.22) !important;
    border-radius: 14px !important;
    box-shadow: none !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
    padding: 1.75rem 1.25rem !important;
    transition: border-color 0.25s ease, background-color 0.25s ease !important;
}

div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"]:hover,
div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"]:hover,
div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"]:focus-within,
div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"]:focus-within {
    background: rgba(0, 0, 0, 0.28) !important;
    border-color: rgba(255, 255, 255, 0.45) !important;
}

/* Primary dropzone prompt stays bright; inner layout divs stay unboxed. */
div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] > div,
div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] span,
div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] span {
    color: rgba(255, 255, 255, 0.85) !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
}

/* Limit/format subtext: legible soft translucent white. */
div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] small,
div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] small {
    color: rgba(255, 255, 255, 0.6) !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.01em !important;
}

/* Browse-files button: dark pill matching the text input, both testid spellings. */
div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] button,
div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] button {
    background: rgba(18, 18, 22, 0.85) !important;
    border: 1px solid rgba(255, 255, 255, 0.22) !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    font-weight: 500 !important;
    padding: 0.55rem 1.3rem !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3) !important;
    transition: background-color 0.25s ease, border-color 0.25s ease, transform 0.25s ease !important;
}

div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] button:hover,
div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] button:hover {
    background-color: rgba(30, 30, 38, 0.95) !important;
    border-color: rgba(255, 255, 255, 0.35) !important;
    transform: translateY(-1px);
}

div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] button:focus-visible,
div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] button:focus-visible {
    outline: 2px solid rgba(255, 255, 255, 0.5) !important;
    outline-offset: 2px !important;
}

div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] button svg,
div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] button svg {
    fill: #ffffff !important;
    stroke: #ffffff !important;
    color: #ffffff !important;
}

/* Staged file rows stay dark and quiet (no light blocks after selection). */
div[data-testid="stFileUploader"] li,
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
    background: rgba(0, 0, 0, 0.25) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 12px !important;
    color: #ffffff !important;
}

div[data-testid="stFileUploader"] li small,
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] small {
    color: rgba(255, 255, 255, 0.6) !important;
}

/* File-row delete control is a ghost icon button, not a second dark pill. */
div[data-testid="stFileUploader"] li button,
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] button {
    background: transparent !important;
    background-image: none !important;
    border: 1px solid transparent !important;
    border-radius: 8px !important;
    box-shadow: none !important;
    color: rgba(255, 255, 255, 0.7) !important;
    padding: 0.25rem !important;
}

div[data-testid="stFileUploader"] li button:hover,
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] button:hover {
    background: rgba(255, 255, 255, 0.1) !important;
    border-color: rgba(255, 255, 255, 0.2) !important;
    color: #ffffff !important;
}

/* Dark Glass Regional Hint Input Box */
div[data-testid="stTextInput"] input {
    background: rgba(0, 0, 0, 0.35) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    padding: 0.75rem 1rem !important;
    font-size: 0.92rem !important;
    transition: all 0.3s ease !important;
}

div[data-testid="stTextInput"] input:focus {
    border-color: rgba(255, 255, 255, 0.5) !important;
    box-shadow: 0 0 12px rgba(255, 255, 255, 0.15) !important;
}

/* Legible greyed-out placeholder */
div[data-testid="stTextInput"] input::placeholder {
    color: rgba(255, 255, 255, 0.5) !important;
}

/* Frosted-glass action buttons that sit on the dark backdrop */
div[data-testid="stButton"] {
    display: flex !important;
    justify-content: center !important;
    margin-top: 1.5rem !important;
}
div[data-testid="stButton"] button {
    background: rgba(15, 23, 42, 0.6) !important;
    color: #f8fafc !important;
    font-weight: 600 !important;
    border-radius: 30px !important;
    padding: 0.5rem 1rem !important;
    white-space: nowrap !important;
    font-size: 0.875rem !important;
    transition: all 0.3s ease !important;
    border: 1px solid rgba(255, 255, 255, 0.25) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35) !important;
}
div[data-testid="stButton"] button:hover {
    background: rgba(30, 41, 59, 0.75) !important;
    border-color: rgba(255, 255, 255, 0.45) !important;
    color: #ffffff !important;
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.45) !important;
}
div[data-testid="stButton"] button p {
    color: inherit !important;
}
div[data-testid="stButton"] button:disabled {
    opacity: 0.45 !important;
    transform: none !important;
}

/* Status banners (success/info/warning/error): dark frosted glass
   with guaranteed light-text contrast on the photographic backdrop.
   Severity stays recognizable through Streamlit's native icon. */
div[data-testid="stAlert"] > div {
    background: rgba(15, 23, 42, 0.72) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.22) !important;
    border-radius: 14px !important;
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.4) !important;
    padding: 0.9rem 1.1rem !important;
}
div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span,
div[data-testid="stAlert"] li {
    color: #f8fafc !important;
}
div[data-testid="stAlert"] a {
    color: #7dd3fc !important;
}

/* The background track of the progress bar */
div[data-testid="stProgressBar"] > div {
    background-color: rgba(255, 255, 255, 0.1) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 50px !important;
}

/* The actual colored fill of the progress bar indicating percentage */
div[data-testid="stProgressBar"] > div > div {
    background-image: linear-gradient(90deg, #ff5bd5 0%, #a25df5 50%, #5d8af5 100%) !important;
    background-color: transparent !important;
    border-radius: 50px !important;
    height: 100% !important;
}
/* Center the loading text above the progress bar */
.loading-text {
    text-align: center;
    color: #ffffff;
    font-weight: 600;
    letter-spacing: 1px;
    margin-bottom: 10px;
    font-size: 1rem;
}
/* Black radar sweep shown while analysis runs */
.scan-radar {
    width: 200px;
    height: 200px;
    margin: 0.5rem auto;
    border-radius: 50%;
    position: relative;
    border: 1px solid rgba(0, 255, 170, 0.5);
    background:
        radial-gradient(circle, rgba(0, 255, 170, 0.12) 0%, rgba(0, 0, 0, 0) 55%),
        repeating-radial-gradient(circle, rgba(0, 255, 170, 0.22) 0 1px, rgba(0, 0, 0, 0) 1px 34px),
        #000;
    overflow: hidden;
}
.scan-radar::before {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: 50%;
    background: conic-gradient(from 0deg, rgba(0, 255, 170, 0.95), rgba(0, 255, 170, 0.12) 18%, transparent 32%);
    animation: scan-spin 2.4s linear infinite;
}
.scan-radar::after {
    content: "";
    position: absolute;
    inset: 0;
    margin: auto;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #00ffaa;
    box-shadow: 0 0 8px rgba(0, 255, 170, 0.9);
}
@keyframes scan-spin {
    to {
        transform: rotate(360deg);
    }
}
@media (prefers-reduced-motion: reduce) {
    .scan-radar::before {
        animation: none;
    }
}
/* Luxury result dashboard: unified dark glass wrapper */
.result-glass {
    background: rgba(10, 14, 26, 0.88);
    backdrop-filter: blur(18px) saturate(140%);
    -webkit-backdrop-filter: blur(18px) saturate(140%);
    border: 1px solid rgba(255, 255, 255, 0.16);
    border-radius: 24px;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.45);
    padding: 2.5rem 2rem;
    max-width: 720px;
    margin: 1.5rem auto;
    text-align: center;
}
.result-eyebrow {
    text-transform: uppercase;
    letter-spacing: 0.3em;
    font-size: 0.72rem;
    color: rgba(255, 255, 255, 0.6);
    margin-bottom: 0.75rem;
}
.result-location {
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 300;
    font-size: 3rem;
    line-height: 1.1;
    color: #ffffff !important;
    margin: 0 0 0.25rem 0;
    text-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}
.result-coords {
    font-size: 0.9rem;
    letter-spacing: 0.08em;
    color: rgba(255, 255, 255, 0.65);
    margin-bottom: 1.5rem;
}
.result-metrics {
    display: flex;
    justify-content: center;
    gap: 1rem;
    flex-wrap: wrap;
    margin-bottom: 1.75rem;
}
.metric-mini-card {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 16px;
    padding: 1rem 1.75rem;
    min-width: 180px;
}
.metric-mini-card .metric-label {
    text-transform: uppercase;
    letter-spacing: 0.2em;
    font-size: 0.68rem;
    color: rgba(255, 255, 255, 0.6);
    margin-bottom: 0.35rem;
}
.metric-mini-card .metric-value {
    font-size: 1.4rem;
    font-weight: 600;
    color: #ffffff;
}
.result-divider {
    border: none;
    border-top: 1px solid rgba(255, 255, 255, 0.12);
    margin: 0 auto 1.5rem auto;
    max-width: 480px;
}
.deduction-title {
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 400;
    font-size: 1.5rem;
    color: #ffffff !important;
    margin-bottom: 1.25rem;
}
.deduction-timeline {
    list-style: none;
    counter-reset: deduction;
    margin: 0 auto;
    padding: 0;
    max-width: 560px;
    text-align: left;
}
.deduction-timeline li {
    counter-increment: deduction;
    position: relative;
    padding: 0 0 1.25rem 3rem;
    color: rgba(255, 255, 255, 0.85);
    font-size: 0.95rem;
    line-height: 1.55;
}
.deduction-timeline li:last-child {
    padding-bottom: 0;
}
.deduction-timeline li::before {
    content: counter(deduction);
    position: absolute;
    left: 0;
    top: 0;
    width: 2rem;
    height: 2rem;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.25);
    color: #ffffff;
    font-size: 0.8rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    justify-content: center;
}
.deduction-timeline li::after {
    content: "";
    position: absolute;
    left: 0.95rem;
    top: 2.2rem;
    bottom: 0.1rem;
    width: 1px;
    background: rgba(255, 255, 255, 0.15);
}
.deduction-timeline li:last-child::after {
    display: none;
}
.deduction-empty {
    color: rgba(255, 255, 255, 0.5);
    font-style: italic;
}
</style>
"""


def inject_custom_css() -> None:
    """Inject Google Fonts, the misty-mountain backdrop, and header styles."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def inject_heading_styles() -> None:
    inject_custom_css()


def render_result_map(result: dict) -> None:
    """Render a map for result coordinates when present."""
    coords = (result or {}).get("coordinates") or {}
    lat, lon = coords.get("latitude"), coords.get("longitude")
    if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
        st.map(pd.DataFrame([{"lat": lat, "lon": lon}]), zoom=13)


PRECISION_RADII_METERS = {
    "Public Landmark": 50,
    "Street Level": 100,
    "Neighborhood Radius": 500,
    "City Level": 10000,
    "State / Region Radius": 50000,
}
DEFAULT_RADIUS_METERS = 50000
FOLIUM_DARK_TILES = "CartoDB dark_matter"
CIRCLE_FILL_OPACITY = 0.3
WONDER_MAP_HEIGHT = 400


def precision_to_radius(precision_level: str) -> int:
    """Map a Wonder precision_level to a map circle radius in meters.

    Unknown levels fail toward the widest, most private radius.
    """
    return PRECISION_RADII_METERS.get(precision_level, DEFAULT_RADIUS_METERS)


def build_wonder_folium_map(result: dict):
    """Build a dark-theme folium map with a precision circle.

    Returns None when folium is unavailable or coordinates are missing.
    """
    if folium is None:
        return None
    result = result or {}
    lat, lon = result.get("latitude"), result.get("longitude")
    if not (
        isinstance(lat, (int, float)) and isinstance(lon, (int, float))
    ):
        return None
    radius = precision_to_radius(result.get("precision_level"))
    fmap = folium.Map(
        location=[lat, lon],
        zoom_start=14 if radius >= 500 else 15,
        tiles=FOLIUM_DARK_TILES,
    )
    folium.Circle(
        location=[lat, lon],
        radius=radius,
        color=NEON_ACCENT,
        weight=3,
        opacity=1.0,
        fill=True,
        fill_color=NEON_ACCENT,
        fill_opacity=CIRCLE_FILL_OPACITY,
    ).add_to(fmap)
    return fmap


def render_wonder_folium_map(result: dict) -> None:
    """Render the Wonder precision map, falling back to st.map if needed."""
    fmap = build_wonder_folium_map(result)
    if fmap is None or st_folium is None:
        render_result_map(
            {
                "coordinates": {
                    "latitude": (result or {}).get("latitude"),
                    "longitude": (result or {}).get("longitude"),
                }
            }
        )
        return
    st_folium(fmap, height=WONDER_MAP_HEIGHT, use_container_width=True)


def format_confidence(value) -> str:
    """Format a 0-1 confidence score as a percentage string."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "Unknown"
    return f"{value:.0%}"


def format_coordinates(coordinates) -> str | None:
    """Format a coordinates mapping as a compass-point string."""
    if not isinstance(coordinates, dict):
        return None
    lat = coordinates.get("latitude")
    lon = coordinates.get("longitude")
    for value in (lat, lon):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
    lat_hemi = "N" if lat >= 0 else "S"
    lon_hemi = "E" if lon >= 0 else "W"
    return f"{abs(lat)}° {lat_hemi}, {abs(lon)}° {lon_hemi}"


def render_result_dashboard(result: dict) -> None:
    """Render the luxury glass dashboard for an analysis result."""
    result = result or {}
    city = (result.get("city") or "").strip()
    country = (result.get("country") or "").strip()
    location = ", ".join(part for part in (city, country) if part)
    coords_text = format_coordinates(result.get("coordinates"))
    if not location:
        location = coords_text or "Location Identified"
    coords_line = ""
    if coords_text and coords_text != location:
        coords_line = (
            f'<p class="result-coords">{html.escape(coords_text)}</p>'
        )
    precision = result.get("precision") or result.get("precision_level")
    if not precision:
        precision = "Unknown"
    confidence_value = result.get("confidence")
    if confidence_value is None:
        confidence_value = result.get("confidence_score")
    confidence_text = html.escape(format_confidence(confidence_value))
    steps = []
    for step in result.get("reasoning_steps") or []:
        if step is None:
            continue
        text = str(step).strip()
        if text:
            steps.append(text)
    if steps:
        items = "".join(
            f"<li>{html.escape(step)}</li>" for step in steps
        )
        timeline = f'<ol class="deduction-timeline">{items}</ol>'
    else:
        timeline = (
            '<p class="deduction-empty">No reasoning steps recorded.</p>'
        )
    st.markdown(
        f"""
        <div class="result-glass">
            <p class="result-eyebrow">Spatial Verdict</p>
            <h2 class="result-location">{html.escape(location)}</h2>
            {coords_line}
            <div class="result-metrics">
                <div class="metric-mini-card">
                    <p class="metric-label">Precision</p>
                    <p class="metric-value">{html.escape(str(precision))}</p>
                </div>
                <div class="metric-mini-card">
                    <p class="metric-label">Confidence</p>
                    <p class="metric-value">{confidence_text}</p>
                </div>
            </div>
            <hr class="result-divider" />
            <h3 class="deduction-title">Detective&rsquo;s Deduction</h3>
            {timeline}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_analysis_outcome(outcome: dict | None) -> None:
    """Render a stored analysis outcome (dashboard + map, or halt/error)."""
    if not outcome:
        return
    kind = outcome.get("kind")
    if kind == "ok":
        st.success("Analysis complete.")
        render_result_dashboard(outcome.get("result", {}))
        render_result_map(outcome.get("result", {}))
    elif kind == "halt":
        st.warning(outcome.get("message", "Analysis halted."))
    else:
        st.error(outcome.get("message", "Analysis failed."))


async def _run_wonder_reasoning(
    sam_summary, regional_hint, on_retry, image=None, api_key=None
):
    async with SpatialReasoner(api_key=api_key, on_retry=on_retry) as reasoner:
        return await reasoner.reason(sam_summary, regional_hint, image=image)


def _reason_wonder_sync(sam_summary, regional_hint, on_retry, image, api_key):
    """Blocking Wonder entry point submitted to the executor worker."""
    return asyncio.run(
        _run_wonder_reasoning(
            sam_summary, regional_hint, on_retry, image=image, api_key=api_key
        )
    )


async def _run_sam_segmentation(safe_image, api_key, on_retry=None):
    """Segment one image; runs on the executor worker's own event loop."""
    async with SamSegmenter(api_key=api_key, on_retry=on_retry) as segmenter:
        return await segmenter.segment_image(safe_image)


def _segment_image_sync(safe_image, api_key, on_retry=None):
    """Blocking SAM entry point submitted to the executor worker."""
    return asyncio.run(_run_sam_segmentation(safe_image, api_key, on_retry))


def _await_future_with_ticks(
    future,
    start_percent,
    messages,
    cap,
    max_step,
    interval_seconds,
    on_progress=None,
):
    """Wait for a worker future, ticking progress on an interval.

    Paints upward from start_percent every interval_seconds while the
    future runs, advancing each tick by a random 1..max_step so the
    climb never looks mechanical, and rotating the status text through
    messages (wrapping with a msg_index). Holds silently at cap.
    Returns future.result(), re-raising worker errors on the main
    thread. With no observer, waits without polling so a fast finish
    costs no extra sleep.
    """
    if on_progress is None:
        return future.result()
    percent = start_percent
    msg_index = 0
    while not future.done():
        time.sleep(interval_seconds)
        if future.done():
            break
        if percent < cap:
            percent = min(percent + random.randint(1, max_step), cap)
            on_progress(percent, messages[msg_index % len(messages)])
            msg_index += 1
    return future.result()


def _await_sam_future(future, on_progress=None):
    """Wait for the SAM worker, ticking 6..44 through SAM messages."""
    return _await_future_with_ticks(
        future,
        start_percent=MILESTONE_SAM[0],
        messages=SAM_TICK_MESSAGES,
        cap=SAM_PROGRESS_CAP,
        max_step=SAM_PROGRESS_TICK_MAX_STEP,
        interval_seconds=SAM_PROGRESS_TICK_SECONDS,
        on_progress=on_progress,
    )


def _await_wonder_future(future, on_progress=None):
    """Wait for the Wonder worker, ticking 46..95 through Wonder messages."""
    return _await_future_with_ticks(
        future,
        start_percent=MILESTONE_WONDER[0],
        messages=WONDER_TICK_MESSAGES,
        cap=WONDER_PROGRESS_CAP,
        max_step=WONDER_PROGRESS_TICK_MAX_STEP,
        interval_seconds=WONDER_PROGRESS_TICK_SECONDS,
        on_progress=on_progress,
    )


def _run_live_pipeline_sync(
    safe_image, regional_hint, on_retry=None, on_progress=None
):
    """Run SAM then Wonder on worker threads with animated waits.

    The Meta key resolves once on the main thread so neither worker
    touches st.secrets. Test seam: patch app.ThreadPoolExecutor.
    """
    if on_progress is not None:
        on_progress(*MILESTONE_SAM)
    api_key = get_meta_api_key()
    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="sam-segment"
    ) as sam_pool:
        sam_future = sam_pool.submit(
            _segment_image_sync, safe_image, api_key, on_retry
        )
        sam_result = _await_sam_future(sam_future, on_progress)
    if on_progress is not None:
        on_progress(*MILESTONE_WONDER)
    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="wonder-reason"
    ) as wonder_pool:
        wonder_future = wonder_pool.submit(
            _reason_wonder_sync,
            summarize_masks(sam_result),
            regional_hint,
            on_retry,
            safe_image,
            api_key,
        )
        wonder_result = _await_wonder_future(wonder_future, on_progress)
    return wonder_result


def execute_live_analysis_direct(
    uploaded_file,
    regional_hint: str,
    screener=screen_image_faces,
    on_progress=None,
) -> dict:
    """Validate gates then run SAM+Wonder directly (no localhost backend).

    Returns the same outcome shape as analysis.execute_analysis:
    {"kind": "ok", "mock": False, "result": dict} on success, or
    {"kind": "halt"|"error", "message": str}.
    """
    valid, message = is_valid_image(uploaded_file)
    if not valid:
        return {"kind": "halt", "message": f"Cannot analyze yet: {message}"}
    clean_ok, clean_message, rgb_image = sanitize_uploaded_image(uploaded_file)
    if not clean_ok:
        return {"kind": "halt", "message": f"Cannot analyze yet: {clean_message}"}
    data_ok, data_message, _ = has_sufficient_visual_data(rgb_image)
    if not data_ok:
        return {"kind": "halt", "message": data_message}
    status, face_message, safe_image = screener(rgb_image)
    if status in ("rejected", "error"):
        return {"kind": "halt", "message": f"Cannot analyze: {face_message}"}
    try:
        wonder_result = _run_live_pipeline_sync(
            safe_image, regional_hint, on_progress=on_progress
        )
    except (SamApiError, SpatialReasoningError) as exc:
        if isinstance(exc, SamApiError) and (
            isinstance(exc, SamDenseImageError)
            or "HTTP 500" in str(exc)
        ):
            return {"kind": "error", "message": DENSE_IMAGE_MESSAGE}
        return {"kind": "error", "message": str(exc)}
    try:
        rounded = round_wonder_coordinates(wonder_result)
    except ValueError as exc:
        return {"kind": "error", "message": f"Invalid location result: {exc}"}
    result = build_frontend_result(rounded, uploaded_file.name, regional_hint)
    result["mode"] = "live"
    return {"kind": "ok", "mock": False, "result": result}


LOW_CONFIDENCE_THRESHOLD = 0.2


def is_uncertain_result(result: dict) -> bool:
    """Null coordinates or confidence below 20%: location not determined."""
    result = result or {}
    if result.get("latitude") is None or result.get("longitude") is None:
        return True
    confidence = result.get("confidence_score", 0)
    if isinstance(confidence, bool) or not isinstance(
        confidence, (int, float)
    ):
        return True
    return confidence < LOW_CONFIDENCE_THRESHOLD


def render_uncertain_result(result: dict) -> None:
    """Empty-search animation plus the AI's reasoning for the failure."""
    steps = list((result or {}).get("reasoning_steps") or ["Analysis halted."])
    if st_lottie is not None:
        animation = load_lottie_animation(LOTTIE_SEARCH_EMPTY_URL)
        if animation is not None:
            show_centered_lottie(st.empty(), animation, key="uncertain_lottie")
    st.info(
        "**Why the location couldn't be determined:**\n"
        + "\n".join(f"- {step}" for step in steps)
    )


def render_wonder_result(result: dict) -> None:
    """Render a Wonder result: reasoning as nodes, scores as metric cards."""
    steps_list = list(result.get("reasoning_steps") or [])
    if sac is not None and steps_list:
        sac.steps(
            items=steps_list,
            index=len(steps_list) - 1,
            direction="vertical",
            dot=True,
            color=NEON_ACCENT,
        )
    else:
        for step in steps_list:
            st.markdown(f"- {step}")
    confidence_col, precision_col = st.columns(2)
    with confidence_col:
        st.metric(
            "Confidence Score",
            f"{result.get('confidence_score', 0):.0%}",
        )
    with precision_col:
        st.metric(
            "Precision Level",
            result.get("precision_level", "Unknown"),
        )
    if style_metric_cards is not None:
        style_metric_cards(
            background_color=METRIC_CARD_BACKGROUND,
            border_color=NEON_ACCENT,
            border_left_color=NEON_ACCENT,
            border_size_px=1,
            border_radius_px=8,
        )


def render_wonder_section(
    sam_summary, regional_hint: str, is_processing: bool = False
) -> None:
    """Muse Wonder geolocation over SAM masks plus the regional hint."""
    if not sam_summary:
        return
    st.space("small")
    st.subheader("Muse Wonder geolocation")
    st.caption(
        "Multi-step spatial reasoning over the SAM masks "
        "(requires META_API_KEY in secrets)."
    )
    if st.button(
        "Geolocate with Wonder",
        key="wonder_btn",
        disabled=is_processing,
        help="Sends SAM masks and the hint to Muse Wonder.",
    ):
        retry_status = st.empty()

        def _show_retry(attempt: int, wait_seconds: float) -> None:
            retry_status.info(RETRY_STATUS_TEXT)

        try:
            with st.spinner("Reasoning about location..."):
                wonder_result = asyncio.run(
                    _run_wonder_reasoning(
                        sam_summary, regional_hint, _show_retry
                    )
                )
        except LocationUnknownError as exc:
            retry_status.empty()
            st.session_state["wonder_result"] = {"location_unknown": str(exc)}
        except (SamApiError, SpatialReasoningError) as exc:
            retry_status.empty()
            st.session_state["wonder_result"] = {"error": str(exc)}
        else:
            retry_status.empty()
            st.session_state["wonder_result"] = {"result": wonder_result}
    stored = st.session_state.get("wonder_result")
    if not stored:
        return
    if "location_unknown" in stored:
        st.error(f"Location Unknown: {stored['location_unknown']}")
    elif "error" in stored:
        st.error(stored["error"])
    else:
        result = stored["result"]
        if is_uncertain_result(result):
            if result.get("latitude") is None:
                steps = result.get("reasoning_steps") or ["Analysis halted."]
                st.warning(f"Analysis halted: {steps[0]}")
            render_uncertain_result(result)
        else:
            st.success(
                f"Located: {result['city']}, {result['country']} "
                f"({result['confidence_score']:.0%} confidence)."
            )
            result_col, map_col = st.columns([3, 2])
            with result_col:
                render_wonder_result(result)
            with map_col:
                render_wonder_folium_map(result)


def render_validation_badge(text: str) -> None:
    """Centered frosted-glass success pill above the image preview."""
    st.markdown(
        f"""
        <div style="
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            background: rgba(34, 197, 94, 0.15);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(74, 222, 128, 0.4);
            border-radius: 30px;
            padding: 10px 20px;
            margin: 1.5rem auto;
            width: fit-content;
            box-shadow: 0 4px 20px rgba(34, 197, 94, 0.15);
        ">
            <span style="color: #4ade80; font-weight: 600; font-size: 0.9rem; letter-spacing: 0.02em;">
                {text}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def start_analysis() -> None:
    """Lock the UI and clear stale results before the script body runs.

    Registered as the Analyze button's on_click callback. Widget
    callbacks fire at the start of the triggering run, so render_home
    observes the armed lock immediately and runs the pipeline inline
    with no manual rerun under the button.
    """
    st.session_state["is_analyzing"] = True
    st.session_state["analysis_results"] = None


def reset_home_for_new_upload() -> None:
    """Clear persisted Home state so the user can upload another picture.

    Registered as the Refresh button's on_click callback. Widget
    callbacks fire at the start of the triggering run, so render_home
    observes the cleared state immediately. Bumping the upload epoch
    remounts the uploader with a fresh identity (guaranteed empty),
    and deleting the hint widget key clears its text.
    """
    st.session_state["analysis_results"] = None
    st.session_state["last_upload_key"] = None
    st.session_state["screened_image"] = None
    st.session_state["sam_result"] = None
    st.session_state["wonder_result"] = None
    st.session_state["upload_epoch"] = (
        st.session_state.get("upload_epoch", 0) + 1
    )
    if "regional_hint_input" in st.session_state:
        del st.session_state["regional_hint_input"]


def render_home(developer_mode: bool = False) -> None:
    st.session_state.setdefault("is_analyzing", False)
    st.session_state.setdefault("analysis_results", None)
    st.session_state.setdefault("last_upload_key", None)
    st.session_state.setdefault("screened_image", None)
    st.session_state.setdefault("sam_result", None)
    st.session_state.setdefault("wonder_result", None)
    st.session_state.setdefault("upload_epoch", 0)
    if "dev_mode" not in st.session_state:
        if "developer_mode" in st.session_state:
            st.session_state["dev_mode"] = st.session_state["developer_mode"]
        else:
            st.session_state["dev_mode"] = developer_mode
    if "developer_mode" not in st.session_state:
        st.session_state["developer_mode"] = st.session_state["dev_mode"]
    developer_mode = bool(st.session_state["dev_mode"])
    is_analyzing = st.session_state["is_analyzing"]

    _, _, top_right = st.columns([4, 1, 1])
    with top_right:
        st.button(
            ":material/refresh: Refresh",
            key="reset_home_btn",
            disabled=is_analyzing,
            on_click=reset_home_for_new_upload,
            help="Clear the current result and upload another picture.",
        )

    st.markdown(
        """
        <div style="text-align: center; padding: 2.5rem 0 1.5rem 0;">
            <p style="text-transform: uppercase; letter-spacing: 0.25em; font-size: 0.8rem; color: rgba(255,255,255,0.7); margin-bottom: 0.5rem;">Spatial Reasoning Engine</p>
            <h1 style="font-size: 3.5rem; margin-bottom: 0.5rem;">ReTracer AI</h1>
            <p style="font-size: 1.05rem; color: rgba(255,255,255,0.8); max-width: 600px; margin: 0 auto; font-weight: 300;">
                Privacy-first multi-hypothesis spatial triangulation powered by SAM 3.1 and Muse Wonder.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.space("small")

    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        dev_mode = st.toggle("Developer Mode (Use Mock Data)", value=False, key="dev_mode")
    if isinstance(dev_mode, bool):
        developer_mode = dev_mode
    else:
        # Test doubles return a mock; session_state stays the source of truth.
        developer_mode = bool(st.session_state["dev_mode"])
    st.session_state["developer_mode"] = developer_mode
    st.space("small")

    if ui is None:
        st.error(
            "Missing dependency `streamlit-shadcn-ui`. "
            "Install it with: "
            "`pip install git+https://github.com/ObservedObserver/streamlit-shadcn-ui.git`"
        )
        return

    st.space("small")
    with st.container(border=True):
        uploaded_file = st.file_uploader(
            "Upload a memory",
            key=f"memory_upload_{st.session_state["upload_epoch"]}",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=False,
            help="Sanitized locally with metadata removal and face blurring before spatial analysis.",
            disabled=is_analyzing or developer_mode,
        )
        st.space("small")
        regional_hint = st.text_input(
            "Regional hint (Optional)",
            key="regional_hint_input",
            placeholder="e.g., Kerala backwaters, Western Ghats, Old Delhi, Tokyo alley, Scottish Highlands",
            help="Providing a state, landmark, or landscape context helps narrow spatial confidence faster.",
            disabled=is_analyzing or developer_mode,
        )
        if developer_mode:
            st.space("small")
            st.image(
                MOCK_DEMO_IMAGE_PATH,
                caption="Demo photo (mock data)",
                width="stretch",
            )
    if developer_mode:
        # Demo mode ignores any retained live inputs: uploads and hints
        # are locked, and the mock pipeline runs on the pre-made demo.
        uploaded_file = None
        regional_hint = ""
    st.space("small")
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        # The on_click callback (start_analysis) is the source of truth
        # for arming; the trigger return value is intentionally ignored
        # so a settled or re-read trigger can never re-arm or loop.
        ui.button(
            "Analyze Memory",
            key="analyze_btn",
            size="lg",
            width="stretch",
            disabled=is_analyzing,
            on_click=start_analysis,
        )

    # Armed by start_analysis() before this run. Placeholders and API
    # calls live inside this gate so the bar exists only while analysis
    # runs; the finished output persists in session state and renders
    # below on every run, surviving the unlock.
    if st.session_state["is_analyzing"]:
        status_slot = st.empty()
        progress_slot = st.empty()
        anim_slot = None
        try:

            def report(percent, message):
                report_analysis_milestone(
                    status_slot, progress_slot, percent, message
                )

            report(*MILESTONE_INIT)
            anim_slot = st.empty()
            # The black radar sweep always renders while analysis runs,
            # so the scanning visual never depends on the network.
            show_scan_fallback(anim_slot)
            if developer_mode:
                report(*MILESTONE_SAM)
                outcome = execute_analysis(
                    uploaded_file, regional_hint, True
                )
                report(*MILESTONE_WONDER)
                report(*MILESTONE_WONDER_DONE)
            else:
                outcome = execute_live_analysis_direct(
                    uploaded_file, regional_hint, on_progress=report
                )
            st.session_state["analysis_results"] = outcome
            report(*MILESTONE_DONE)
            time.sleep(COMPLETION_HOLD_SECONDS)
        except Exception as exc:
            st.session_state["analysis_results"] = {
                "kind": "error",
                "message": f"Analysis failed unexpectedly: {exc}",
            }
        finally:
            if anim_slot is not None:
                anim_slot.empty()
            st.session_state["is_analyzing"] = False
        st.rerun()

    if uploaded_file is not None:
        valid, message = is_valid_image(uploaded_file)
        if not valid:
            st.error(message)
        else:
            with st.spinner("Processing & Sanitizing Image..."):
                clean_ok, clean_message, rgb_image = sanitize_uploaded_image(
                    uploaded_file
                )
                if not clean_ok:
                    st.error(clean_message)
                else:
                    data_ok, data_message, _ = has_sufficient_visual_data(
                        rgb_image
                    )
                    if not data_ok:
                        st.warning(data_message)
                    else:
                        status, face_message, safe_image = screen_image_faces(
                            rgb_image
                        )
                        if status == "rejected":
                            st.error(face_message)
                        elif status == "error":
                            st.error(face_message)
                        else:
                            if status == "blurred":
                                st.warning(face_message)
                            else:
                                render_validation_badge(
                                    f"{message} {clean_message} {face_message}"
                                )
                            st.image(
                                safe_image, caption="Privacy-safe preview",
                                width="stretch",
                            )
                            st.session_state["screened_image"] = safe_image

    upload_key = (
        f"{uploaded_file.name}:{uploaded_file.size}"
        if uploaded_file is not None
        else None
    )
    # Only a newly arrived upload clears persisted results. Returning
    # to Home from another view remounts an empty uploader (upload_key
    # None), which must leave the finished result untouched; the
    # Refresh button is the explicit way to clear it.
    if (
        upload_key is not None
        and st.session_state["last_upload_key"] != upload_key
    ):
        st.session_state["last_upload_key"] = upload_key
        st.session_state["analysis_results"] = None
        st.session_state["screened_image"] = None
        st.session_state["sam_result"] = None
        st.session_state["wonder_result"] = None

    render_analysis_outcome(st.session_state["analysis_results"])


PIPELINE_DIAGRAM = """```mermaid
flowchart LR
    A(["Upload photo"]) --> B["Validate + strip metadata"]
    B --> C["Visual-data gate"]
    C --> D{"Face screening"}
    D -->|"Face > 5%: reject"| X(["Halt with message"])
    D -->|"Clean or blurred"| E["SAM 3.1 segmentation"]
    E --> F["Wonder reasoning + hint"]
    F --> G["Round + build result"]
    G --> H(["Metrics + map"])
    B -.->|"Developer Mode"| M["Mock result"]
    M --> H

    classDef endpoint fill:#00FFAA,stroke:#00FFAA,color:#0B1220,font-weight:bold;
    classDef stage fill:#1E1E1E,stroke:#64748B,stroke-width:1.5px,color:#F8FAFC;
    classDef ai fill:#10201A,stroke:#00FFAA,stroke-width:2px,color:#A7F3D0;
    classDef halt fill:#2A1215,stroke:#F87171,stroke-width:2px,color:#FECACA;
    classDef dev fill:#1E1E1E,stroke:#FBBF24,stroke-width:1.5px,stroke-dasharray:6 3,color:#FDE68A;
    class A,H endpoint;
    class B,C,D,G stage;
    class E,F ai;
    class X halt;
    class M dev;
    linkStyle default stroke:#94A3B8,stroke-width:2px;
```"""


def render_how_it_works() -> None:
    st.title("How it Works")
    st.caption("From photo to place.")
    st.subheader("Pipeline architecture")
    st.markdown(PIPELINE_DIAGRAM)
    st.subheader("How each stage works")
    st.markdown(
        "1. **Upload** a JPEG, PNG, or WebP photo (max 10 MB). Files are "
        "stripped of metadata and normalized to RGB at full resolution.\n"
        "2. **Pre-processing gates** run automatically: uploads lacking "
        "visual data are halted, any face over 5% of the image area is "
        "rejected, smaller faces are blurred.\n"
        "3. **Add a regional hint** such as nearby landmarks or architecture.\n"
        "4. **Analyze** to get location hypotheses based on visual clues.\n"
        "5. **Live mode**: with Developer Mode off, Analyze runs SAM 3.1 "
        "segmentation and Muse Wonder geolocation directly via the Meta "
        "Model API.\n"
        "6. **Developer Mode**: with the toggle on, Analyze skips the API "
        "and returns instant mock data for UI testing. Uploads and hints "
        "are locked, and a pre-made demo photo is shown instead."
    )


def render_privacy_safety() -> None:
    st.title("Privacy & Safety")
    st.warning(
        "No images containing unblurred faces are processed. "
        "Please blur or crop out faces before uploading."
    )
    st.subheader("Local face screening")
    st.markdown(
        "- Every upload is screened for faces before anything else runs.\n"
        "- Any face covering more than 5% of the image area is "
        "**rejected**.\n"
        "- Smaller faces are **blurred automatically**, and only the "
        "privacy-safe preview is shown and analyzed."
    )
    st.subheader("Zero-data logging")
    st.markdown(
        "- Uploads are validated in memory: metadata (EXIF) is stripped and "
        "images are normalized to RGB.\n"
        "- No image is written to disk or kept after your session: analysis "
        "state lives only in Streamlit session state.\n"
        "- Uploads are previewed locally in your browser and never stored "
        "with identifying facial information."
    )
    st.subheader("Security & compliance")
    st.markdown(
        "- Only screened, privacy-safe images leave the browser, and only "
        "to run segmentation and geolocation.\n"
        "- Returned coordinates are rounded before display, and precision "
        "is shown as a radius so exact locations stay private.\n"
        "- Every gate fails closed: uploads that are invalid, lack visual "
        "data, or fail face screening halt with a message instead of "
        "guessing."
    )


def main() -> None:
    st.set_page_config(page_title="ReTracer AI", layout="centered")
    inject_heading_styles()

    _, nav_col, _ = st.columns([1, 3, 1])
    with nav_col:
        nav_selection = st.radio(
            label="",
            options=["Home", "How it Works", "Privacy & Safety"],
            horizontal=True,
            label_visibility="collapsed",
            key="main_nav",
        )

    views = {
        "Home": render_home,
        "How it Works": render_how_it_works,
        "Privacy & Safety": render_privacy_safety,
    }
    views.get(nav_selection, render_home)()


if __name__ == "__main__":
    main()
