"""Face screening for uploads.

Uses OpenCV's bundled Haar cascade (frontal-face weights ship with
opencv-python, so detection is fully offline with no downloads).
Strict cascade parameters suppress background noise, and boxes under
MIN_FACE_PIXELS are dropped as pixel artifacts. Each surviving
candidate is then verified as an actual face: the bundled eye
cascades must find an eye in the upper region, or the region must
show strong skin tone plus face-like symmetry (covers sunglasses
and missed eye detections). Unverified boxes are dropped before any
blur or reject decision, so brickwork, windows, and foliage are no
longer blurred. Any single verified face covering more than 5% of
the image area rejects the image; smaller verified faces are heavily
blurred with Pillow (soft-feathered edges) before the image is
passed on. Only clean or successfully blurred images move forward.
"""

from __future__ import annotations

import os
from typing import BinaryIO, Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

FACE_AREA_THRESHOLD = 0.05
CASCADE_SCALE_FACTOR = 1.1
CASCADE_MIN_NEIGHBORS = 10
CASCADE_MIN_SIZE = (50, 50)
MIN_FACE_PIXELS = 50
BLUR_PADDING_RATIO = 0.15
BLUR_FEATHER_DIVISOR = 12
BLUR_FEATHER_MIN = 4
EYE_CASCADE_FILES = (
    "haarcascade_eye_tree_eyeglasses.xml",
    "haarcascade_eye.xml",
)
EYE_SCALE_FACTOR = 1.1
EYE_MIN_NEIGHBORS = 4
EYE_SEARCH_HEIGHT_RATIO = 0.65
SKIN_FRACTION_MIN = 0.5
SYMMETRY_SCORE_MAX = 0.9

Box = tuple[int, int, int, int]
Detector = Callable[[np.ndarray], list[Box]]

_EXTENDED_PATH_PREFIX = "\\\\?\\"
_EXTENDED_UNC_PREFIX = "\\\\?\\UNC\\"


def _cascade_path(cascade_dir: str, filename: str) -> str:
    """Join a cascade file to a path OpenCV's FileStorage can open.

    ``cv2.data.haarcascades`` inherits any prefix on the interpreter's
    own path. Under extended-length (``\\\\?\\``) path virtualization
    the joined path names a file Python can see but OpenCV cannot
    open, so the cascade loads empty and every upload halts at face
    screening. Strip the prefix to the plain equivalent when that
    file exists; otherwise return the joined path unchanged.
    """
    path = cascade_dir + filename
    plain = None
    if path.startswith(_EXTENDED_UNC_PREFIX):
        plain = "\\\\" + path[len(_EXTENDED_UNC_PREFIX):]
    elif path.startswith(_EXTENDED_PATH_PREFIX):
        plain = path[len(_EXTENDED_PATH_PREFIX):]
    if plain is not None and os.path.exists(plain):
        return plain
    return path


def face_area_ratio(box: Box, image_width: int, image_height: int) -> float:
    """Fraction of total image area covered by one bounding box."""
    _, _, w, h = box
    return (w * h) / (image_width * image_height)


def find_oversized_face(
    boxes: list[Box], image_width: int, image_height: int
) -> Box | None:
    """Return the first face above the area threshold, else None."""
    for box in boxes:
        if face_area_ratio(box, image_width, image_height) > FACE_AREA_THRESHOLD:
            return box
    return None


def skin_fraction(roi_bgr: np.ndarray) -> float:
    """Fraction of skin-toned pixels via the YCrCb chrominance rule.

    Pure numpy (BT.601, matching cv2.COLOR_BGR2YCrCb), so callers and
    tests need no OpenCV for this signal.
    """
    rgb = roi_bgr[:, :, ::-1].astype(np.float64)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    luma = 0.299 * red + 0.587 * green + 0.114 * blue
    cr = (red - luma) * 0.713 + 128.0
    cb = (blue - luma) * 0.564 + 128.0
    skin = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    return float(skin.mean())


def symmetry_score(roi_gray: np.ndarray) -> float:
    """Mirror difference between halves over local contrast.

    Lower means more face-like vertical symmetry. Flat regions score
    infinite (no structure, never a face).
    """
    gray = roi_gray.astype(np.float64)
    width = gray.shape[1]
    left = gray[:, : width // 2]
    right = np.fliplr(gray[:, width - width // 2 :])
    contrast = np.abs(gray - gray.mean()).mean()
    if contrast < 1e-6:
        return float("inf")
    return float(np.abs(left - right).mean() / contrast)


def is_verified_face(
    roi_bgr: np.ndarray, roi_gray: np.ndarray, eye_hits: int
) -> bool:
    """True when a Haar candidate looks like an actual face.

    An eye hit confirms immediately. Otherwise the region must show
    both strong skin tone and face-like symmetry (sunglasses and
    missed eye detections still pass; brickwork and windows do not).
    """
    if eye_hits >= 1:
        return True
    return (
        skin_fraction(roi_bgr) >= SKIN_FRACTION_MIN
        and symmetry_score(roi_gray) <= SYMMETRY_SCORE_MAX
    )


def _load_eye_cascades() -> list:
    """Load the bundled eye cascades, skipping any that fail.

    Fail-soft: a missing auxiliary file only disables the eye signal
    (fallback signals still decide) instead of halting screening.
    """
    cascades = []
    for filename in EYE_CASCADE_FILES:
        try:
            cascade = cv2.CascadeClassifier(
                _cascade_path(cv2.data.haarcascades, filename)
            )
            empty = cascade.empty()
        except Exception:
            continue
        if not empty:
            cascades.append(cascade)
    return cascades


def _count_eye_hits(upper_gray: np.ndarray, eye_cascades: list) -> int:
    """Total eye hits across the loaded eye cascades."""
    total = 0
    for cascade in eye_cascades:
        hits = cascade.detectMultiScale(
            upper_gray,
            scaleFactor=EYE_SCALE_FACTOR,
            minNeighbors=EYE_MIN_NEIGHBORS,
        )
        total += len(hits)
    return total


def detect_face_boxes(bgr: np.ndarray) -> list[Box]:
    """Detect verified faces with the offline Haar cascade.

    Fully local: every cascade XML ships with OpenCV, so no weights
    are ever downloaded. Boxes under MIN_FACE_PIXELS are dropped as
    pixel artifacts, then each candidate must pass is_verified_face
    (eye evidence, or skin tone plus symmetry) before it can blur
    anything or reject the image.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed; face detection unavailable.")
    cascade = cv2.CascadeClassifier(
        _cascade_path(
            cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
        )
    )
    if cascade.empty():
        raise RuntimeError(
            "Haar cascade weights missing from this OpenCV install; "
            "face detection unavailable."
        )
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=CASCADE_SCALE_FACTOR,
        minNeighbors=CASCADE_MIN_NEIGHBORS,
        minSize=CASCADE_MIN_SIZE,
    )
    if len(faces) == 0:
        return []
    candidates = [
        (int(x), int(y), int(w), int(h))
        for x, y, w, h in faces
        if int(w) >= MIN_FACE_PIXELS and int(h) >= MIN_FACE_PIXELS
    ]
    if not candidates:
        return []
    eye_cascades = _load_eye_cascades()
    verified = []
    for x, y, w, h in candidates:
        roi_gray = gray[y : y + h, x : x + w]
        roi_bgr = bgr[y : y + h, x : x + w]
        if roi_gray.size == 0:
            continue
        cutoff = max(1, int(h * EYE_SEARCH_HEIGHT_RATIO))
        eye_hits = (
            _count_eye_hits(roi_gray[:cutoff, :], eye_cascades)
            if eye_cascades
            else 0
        )
        if is_verified_face(roi_bgr, roi_gray, eye_hits):
            verified.append((x, y, w, h))
    return verified


def to_bgr(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to BGR for the detector."""
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed; face detection unavailable.")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def _padded_box(box: Box, image_width: int, image_height: int) -> Box:
    x, y, w, h = box
    pad_w = int(w * BLUR_PADDING_RATIO)
    pad_h = int(h * BLUR_PADDING_RATIO)
    x0 = max(0, x - pad_w)
    y0 = max(0, y - pad_h)
    x1 = min(image_width, x + w + pad_w)
    y1 = min(image_height, y + h + pad_h)
    return (x0, y0, x1 - x0, y1 - y0)


def blur_face_regions(image: Image.Image, boxes: list[Box]) -> Image.Image:
    """Return a copy with a heavy Gaussian blur applied inside each box.

    The blurred patch blends in through a soft-feathered mask, so the
    transition melts into the photo instead of cutting a hard box.
    Nothing outside the padded region ever changes.
    """
    width, height = image.size
    blurred = image.copy()
    for box in boxes:
        x, y, w, h = _padded_box(box, width, height)
        if w <= 0 or h <= 0:
            continue
        region = blurred.crop((x, y, x + w, y + h))
        radius = max(10, max(w, h) // 2)
        soft = region.filter(ImageFilter.GaussianBlur(radius))
        feather = max(BLUR_FEATHER_MIN, min(w, h) // BLUR_FEATHER_DIVISOR)
        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).rectangle(
            [feather, feather, w - feather - 1, h - feather - 1],
            fill=255,
        )
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
        blurred.paste(soft, (x, y), mask)
    return blurred


def screen_image_faces(
    image: Image.Image, detector: Detector | None = None
) -> tuple[str, str, Image.Image | None]:
    """Screen one image: reject large faces, blur small ones.

    Returns (status, message, safe_image) with status in
    {"clean", "blurred", "rejected", "error"}. Only "clean" and "blurred"
    carry an image forward; callers must not use the original otherwise.
    """
    width, height = image.size
    try:
        if detector is not None:
            boxes = detector(np.array(image.convert("L")))
        else:
            boxes = detect_face_boxes(to_bgr(image))
    except Exception as exc:
        return ("error", f"Face detector failed: {exc}", None)

    if not boxes:
        return ("clean", "No faces detected.", image)

    oversized = find_oversized_face(boxes, width, height)
    if oversized is not None:
        pct = face_area_ratio(oversized, width, height) * 100
        return (
            "rejected",
            f"Privacy warning: a detected face covers {pct:.1f}% of the image "
            f"(limit {FACE_AREA_THRESHOLD * 100:.0f}%). "
            "No images containing unblurred faces are processed; "
            "please blur or crop out faces and re-upload.",
            None,
        )

    return (
        "blurred",
        f"{len(boxes)} small face(s) detected and blurred before processing.",
        blur_face_regions(image, boxes),
    )


def preprocess_uploaded_image(
    uploaded_file: BinaryIO, detector: Detector | None = None
) -> tuple[str, str, Image.Image | None]:
    """Open an uploaded file and screen it for faces (fail closed)."""
    try:
        uploaded_file.seek(0)
        with Image.open(uploaded_file) as img:
            image = img.convert("RGB")
    except Exception:
        return ("error", "Could not decode the uploaded image.", None)
    finally:
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
    return screen_image_faces(image, detector=detector)
