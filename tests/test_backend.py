"""Tests for the FastAPI backend's privacy rounding and payload mapping.

Run: python -m unittest discover -s tests -v
Requires Pillow only: the rounding, payload, and decode helpers are
pure and importable without fastapi/uvicorn installed. The endpoint
status tests below skip themselves when fastapi is not installed.
"""

import base64
import io
import os
import unittest
from unittest.mock import patch

from PIL import Image

from analysis import mock_analysis_result

import backend
from backend import (
    build_frontend_result,
    decode_upload,
    round_wonder_coordinates,
)

try:
    from fastapi import HTTPException

    BACKEND_ENDPOINT_AVAILABLE = backend.app is not None
except ImportError:  # pragma: no cover - server extras not installed
    HTTPException = None
    BACKEND_ENDPOINT_AVAILABLE = False


def wonder_payload(latitude=48.8584, longitude=2.2945):
    return {
        "latitude": latitude,
        "longitude": longitude,
        "city": "Paris",
        "country": "France",
        "reasoning_steps": ["Iron lattice tower", "French signage"],
        "confidence_score": 0.95,
        "precision_level": "Public Landmark",
    }


def halt_payload():
    return {
        "latitude": None,
        "longitude": None,
        "city": "",
        "country": "",
        "reasoning_steps": ["Not enough distinct features"],
        "confidence_score": 0.0,
        "precision_level": "Unknown",
    }


class RoundWonderCoordinatesTests(unittest.TestCase):
    def test_rounds_to_three_decimals(self):
        rounded = round_wonder_coordinates(wonder_payload())
        self.assertEqual(rounded["latitude"], 48.858)
        self.assertEqual(rounded["longitude"], 2.295)
        self.assertIsInstance(rounded["latitude"], float)
        self.assertIsInstance(rounded["longitude"], float)

    def test_rounds_negative_coordinates(self):
        rounded = round_wonder_coordinates(wonder_payload(-33.8568, -151.2153))
        self.assertEqual(rounded["latitude"], -33.857)
        self.assertEqual(rounded["longitude"], -151.215)

    def test_short_and_whole_values_become_floats(self):
        rounded = round_wonder_coordinates(wonder_payload(48.8, 48))
        self.assertEqual(rounded["latitude"], 48.8)
        self.assertEqual(rounded["longitude"], 48.0)
        self.assertIsInstance(rounded["longitude"], float)

    def test_tiny_values_round_to_zero_float(self):
        rounded = round_wonder_coordinates(wonder_payload(0.00049, -0.00049))
        self.assertEqual(rounded["latitude"], 0.0)
        self.assertEqual(rounded["longitude"], -0.0)
        self.assertIsInstance(rounded["latitude"], float)

    def test_null_halt_coordinates_pass_through(self):
        self.assertEqual(round_wonder_coordinates(halt_payload()), halt_payload())

    def test_custom_decimals_honored(self):
        rounded = round_wonder_coordinates(wonder_payload(), decimals=2)
        self.assertEqual(rounded["latitude"], 48.86)
        self.assertEqual(rounded["longitude"], 2.29)

    def test_input_never_mutated_and_other_keys_kept(self):
        original = wonder_payload()
        snapshot = dict(original)
        rounded = round_wonder_coordinates(original)
        self.assertEqual(original, snapshot)
        self.assertIsNot(rounded, original)
        for key in (
            "city",
            "country",
            "reasoning_steps",
            "confidence_score",
            "precision_level",
        ):
            self.assertEqual(rounded[key], snapshot[key])

    def test_missing_coordinate_key_fails_closed(self):
        for key in ("latitude", "longitude"):
            with self.subTest(key=key):
                payload = wonder_payload()
                del payload[key]
                with self.assertRaises(ValueError) as ctx:
                    round_wonder_coordinates(payload)
                self.assertIn(key, str(ctx.exception))

    def test_non_numeric_coordinates_fail_closed(self):
        for bad in (True, "48.85", [48.85], {"lat": 48.85}):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    round_wonder_coordinates(wonder_payload(bad, 2.29))
                with self.assertRaises(ValueError):
                    round_wonder_coordinates(wonder_payload(48.85, bad))


class BuildFrontendResultTests(unittest.TestCase):
    def test_maps_rounded_wonder_result(self):
        result = build_frontend_result(
            round_wonder_coordinates(wonder_payload()),
            "memory.jpg",
            "riverside cafes",
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["mode"], "backend")
        self.assertEqual(result["filename"], "memory.jpg")
        self.assertEqual(result["regional_hint"], "riverside cafes")
        self.assertEqual(result["city"], "Paris")
        self.assertEqual(result["country"], "France")
        self.assertEqual(result["precision"], "Public Landmark")
        self.assertEqual(result["confidence"], 0.95)
        self.assertEqual(
            result["coordinates"], {"latitude": 48.858, "longitude": 2.295}
        )
        self.assertEqual(
            result["reasoning_steps"],
            ["Iron lattice tower", "French signage"],
        )

    def test_blank_hint_gets_default(self):
        result = build_frontend_result(
            round_wonder_coordinates(wonder_payload()), "m.jpg", "   "
        )
        self.assertEqual(result["regional_hint"], "(no hint provided)")

    def test_halt_coordinates_stay_null(self):
        result = build_frontend_result(halt_payload(), "m.jpg", "")
        self.assertEqual(
            result["coordinates"], {"latitude": None, "longitude": None}
        )

    def test_mock_result_shares_frontend_shape(self):
        live = build_frontend_result(
            round_wonder_coordinates(wonder_payload()),
            "memory.jpg",
            "riverside cafes",
        )
        mock = mock_analysis_result("memory.jpg", "riverside cafes")
        self.assertEqual(set(mock), set(live))


class DecodeUploadTests(unittest.TestCase):
    def test_valid_jpeg_base64_round_trip(self):
        img = Image.new("RGB", (16, 12), (10, 200, 90))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        decoded = decode_upload(
            base64.b64encode(buf.getvalue()).decode("ascii")
        )
        self.assertEqual(decoded.size, (16, 12))
        self.assertEqual(decoded.mode, "RGB")

    def test_invalid_base64_rejected(self):
        with self.assertRaises(ValueError):
            decode_upload("!!! not base64 !!!")

    def test_empty_payload_rejected(self):
        with self.assertRaises(ValueError):
            decode_upload("")

    def test_oversize_payload_rejected(self):
        with patch("backend.MAX_UPLOAD_BYTES", 10):
            with self.assertRaises(ValueError) as ctx:
                decode_upload(base64.b64encode(b"x" * 11).decode("ascii"))
        message = str(ctx.exception)
        self.assertIn("exceeds", message)
        self.assertIn("limit", message)

    def test_non_image_bytes_rejected(self):
        with self.assertRaises(ValueError):
            decode_upload(base64.b64encode(b"x" * 64).decode("ascii"))


def drive_endpoint(coro):
    """Run an endpoint coroutine without an event loop.

    Upload validation raises before the first await, and the mocked-AI
    paths below only await pure-Python coroutines, so no real loop is
    ever needed. Returns the endpoint result, lets HTTPException
    propagate, and fails loudly if the coroutine ever suspends on
    real I/O.
    """
    send_value = None
    while True:
        try:
            yielded = coro.send(send_value)
        except StopIteration as done:
            return done.value
        send_value = None
        coro.close()
        raise AssertionError(
            f"endpoint suspended on {yielded!r}; mocks must stay pure"
        )


class _FakeSegmenter:
    """Stand-in for SamSegmenter returning a canned segmentation."""

    sam_result = {"masks": []}

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def segment_image(self, image):
        return dict(self.sam_result)


class _FakeReasoner:
    """Stand-in for SpatialReasoner returning a canned Wonder result."""

    wonder_result = {}

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def reason(self, *args, **kwargs):
        return dict(self.wonder_result)


def _jpeg_base64(image):
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _textured_image():
    img = Image.new("RGB", (64, 48))
    pixels = img.load()
    for x in range(64):
        for y in range(48):
            pixels[x, y] = (
                (x * 4) % 256,
                (y * 5) % 256,
                ((x + y) * 3) % 256,
            )
    return img


@unittest.skipIf(
    not BACKEND_ENDPOINT_AVAILABLE, "fastapi/uvicorn not installed"
)
class AnalyzeEndpointStatusTests(unittest.TestCase):
    """Upload validation must fail as 422 even without an API key.

    Regression tests: the /analyze endpoint used to check META_API_KEY
    before decoding the upload, so every invalid upload surfaced as a
    500 Internal Server Error instead of a 422. A malformed upstream
    Wonder result also escaped as an unhandled 500 instead of a 502.
    """

    def _payload(self, image_jpeg_base64):
        return backend.AnalyzeRequest(
            filename="memory.jpg",
            regional_hint="riverside cafes",
            image_jpeg_base64=image_jpeg_base64,
        )

    def _assert_status(self, payload, expected_status, api_key=""):
        with patch.dict(os.environ, {"META_API_KEY": api_key}):
            with self.assertRaises(HTTPException) as ctx:
                drive_endpoint(backend.analyze(payload))
        self.assertEqual(ctx.exception.status_code, expected_status)
        return str(ctx.exception.detail)

    def _assert_mocked_ai_status(
        self, wonder_result, expected_status
    ):
        _FakeReasoner.wonder_result = wonder_result
        payload = self._payload(_jpeg_base64(_textured_image()))
        with (
            patch.dict(os.environ, {"META_API_KEY": "test-key"}),
            patch.object(backend, "SamSegmenter", _FakeSegmenter),
            patch.object(backend, "SpatialReasoner", _FakeReasoner),
        ):
            with self.assertRaises(HTTPException) as ctx:
                drive_endpoint(backend.analyze(payload))
        self.assertEqual(ctx.exception.status_code, expected_status)
        return str(ctx.exception.detail)

    def test_invalid_base64_is_422_without_api_key(self):
        detail = self._assert_status(self._payload("!!! not base64 !!!"), 422)
        self.assertIn("base64", detail)

    def test_empty_upload_is_422_without_api_key(self):
        detail = self._assert_status(self._payload(""), 422)
        self.assertIn("empty", detail)

    def test_oversize_upload_is_422_without_api_key(self):
        payload = self._payload(
            base64.b64encode(b"x" * 11).decode("ascii")
        )
        with patch.object(backend, "MAX_UPLOAD_BYTES", 10):
            detail = self._assert_status(payload, 422)
        self.assertIn("limit", detail)

    def test_non_image_bytes_is_422_without_api_key(self):
        payload = self._payload(
            base64.b64encode(b"x" * 64).decode("ascii")
        )
        detail = self._assert_status(payload, 422)
        self.assertIn("decode", detail)

    def test_blank_image_is_422_without_api_key(self):
        solid = Image.new("RGB", (32, 32), (128, 128, 128))
        detail = self._assert_status(self._payload(_jpeg_base64(solid)), 422)
        self.assertIn("sufficient visual data", detail)

    def test_valid_upload_without_api_key_is_500(self):
        payload = self._payload(_jpeg_base64(_textured_image()))
        detail = self._assert_status(payload, 500)
        self.assertIn("META_API_KEY", detail)

    def test_malformed_wonder_result_is_502(self):
        detail = self._assert_mocked_ai_status({"latitude": 48.85}, 502)
        self.assertIn("invalid", detail)

    def test_non_numeric_wonder_coordinates_is_502(self):
        detail = self._assert_mocked_ai_status(
            wonder_payload(True, 2.29), 502
        )
        self.assertIn("invalid", detail)

    def test_valid_upload_returns_rounded_result(self):
        _FakeReasoner.wonder_result = wonder_payload()
        payload = self._payload(_jpeg_base64(_textured_image()))
        with (
            patch.dict(os.environ, {"META_API_KEY": "test-key"}),
            patch.object(backend, "SamSegmenter", _FakeSegmenter),
            patch.object(backend, "SpatialReasoner", _FakeReasoner),
        ):
            result = drive_endpoint(backend.analyze(payload))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["mode"], "backend")
        self.assertEqual(result["filename"], "memory.jpg")
        self.assertEqual(
            result["coordinates"], {"latitude": 48.858, "longitude": 2.295}
        )


if __name__ == "__main__":
    unittest.main()
