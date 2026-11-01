"""Tests for the async SAM 3.1 client: request fan-out, mask parsing,
full-resolution uploads, and error mapping.

Run: python -m unittest discover -s tests -v
Requires aiohttp, meta-sam-parser, numpy, Pillow. No network: HTTP is
fully mocked. Fixture mask payloads are generated with the official
parser's own encoder so they are valid by construction.
"""

import asyncio
import base64
import io
import json
import unittest
import unittest.mock
from unittest.mock import patch

import aiohttp
import numpy as np
from meta_sam_parser import (
    ResponsesStreamSourceError,
    SegmentationMaskRecord,
)
from meta_sam_parser._mask_codec import _encode_raster
from PIL import Image
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_none,
)

from sam_segmentation import (
    DENSE_IMAGE_MESSAGE,
    FALLBACK_MAX_LONG_EDGE,
    FOCUS_CONCEPTS,
    MAX_CONCURRENT_SAM_REQUESTS,
    MAX_TILE_PAYLOADS,
    MIN_MASK_CONFIDENCE,
    RETRY_STATUS_TEXT,
    SAM_RETRY_WAIT,
    SamApiError,
    SamDenseImageError,
    SamRetryableError,
    SamSegmenter,
    assemble_tile_payloads,
    downscale_to_long_edge,
    draw_mask_boxes,
    has_transient_cause,
    offset_box_to_full_frame,
    passes_confidence_filter,
    select_latest_masks,
    slice_grid_tiles,
    sniff_image_mime,
    summarize_masks,
)


def fixture_line(confidence="0.95"):
    # High-confidence default: the >0.90 aggregation filter keeps it.
    raster = [1, 0, 0, 1]
    payload = _encode_raster(raster, 2, 2)
    if payload[:1] not in "!~":
        payload = "!" + payload
    return (
        f"<0f>0<|box;x1=1;y1=2;x2=10;y2=12;w=64;h=48;c={confidence}|>"
        f"<|mask;x=0;y=0;data=2,2,{payload}|>"
    )


def sse_bytes(events):
    lines = "".join(f"data: {json.dumps(e)}\n\n" for e in events)
    return lines.encode("utf-8")


def success_events():
    line = fixture_line()
    lane = {"item_id": "msg_1", "output_index": 0, "content_index": 0}
    return [
        {"type": "response.output_text.delta", **lane, "delta": line},
        {"type": "response.output_text.done", **lane, "text": line},
        {"type": "response.completed"},
    ]


def low_confidence_events():
    line = fixture_line(confidence="0.50")
    lane = {"item_id": "msg_1", "output_index": 0, "content_index": 0}
    return [
        {"type": "response.output_text.delta", **lane, "delta": line},
        {"type": "response.output_text.done", **lane, "text": line},
        {"type": "response.completed"},
    ]


class FakeContent:
    def __init__(self, data: bytes):
        self._data = data

    def __aiter__(self):
        async def gen():
            for line in self._data.splitlines(keepends=True):
                yield line

        return gen()


class FakeResponse:
    def __init__(self, status=200, events=(), body=""):
        self.status = status
        self._events = events
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def text(self):
        return self._body

    @property
    def content(self):
        return FakeContent(sse_bytes(self._events))


class FakeSession:
    def __init__(self, response):
        self.calls = []
        self._responses = (
            response if isinstance(response, list) else [response]
        )

    def post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        index = min(len(self.calls) - 1, len(self._responses) - 1)
        return self._responses[index]


class RaisingSession:
    """Session whose POST always fails at the connection level."""

    def __init__(self, error):
        self.calls = []
        self._error = error

    def post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        raise self._error


class ExplodingContent:
    """Stream content that drops like a severed connection."""

    def __init__(self, error):
        self._error = error

    def __aiter__(self):
        async def gen():
            raise self._error
            yield b""  # pragma: no cover - unreachable

        return gen()


class ExplodingResponse(FakeResponse):
    def __init__(self, error):
        super().__init__(status=200)
        self._error = error

    @property
    def content(self):
        return ExplodingContent(self._error)


class BadJsonResponse(FakeResponse):
    @property
    def content(self):
        return FakeContent(b"data: this is not json\n\n")


def fast_retrying():
    return AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_none(),
        retry=retry_if_exception_type(SamRetryableError),
        reraise=True,
    )


def error_response(status):
    return FakeResponse(status=status, body=f'{{"error": {{"code": {status}}}}}')


def success_response():
    return FakeResponse(events=success_events())


def gradient_png(width=32, height=24):
    x = np.tile(np.arange(width, dtype=np.uint8), (height, 1))
    rgb = np.stack([x, x, x], axis=-1)
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


class FanOutTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_request_per_focus_concept(self):
        session = FakeSession(FakeResponse(events=success_events()))
        seg = SamSegmenter(api_key="test-key", session=session)
        result = await seg.segment_image(gradient_png())
        self.assertEqual(len(session.calls), 5 * len(FOCUS_CONCEPTS))
        self.assertEqual(result["concepts"], list(FOCUS_CONCEPTS))
        self.assertEqual(result["model"], "sam-3.1")
        seen = set()
        payloads_per_concept = {}
        for call in session.calls:
            self.assertEqual(call["url"], "https://api.meta.ai/v1/responses")
            self.assertEqual(
                call["headers"]["Authorization"], "Bearer test-key"
            )
            body = call["json"]
            self.assertEqual(body["model"], "sam-3.1")
            self.assertTrue(body["stream"])
            self.assertEqual(body["metadata"], {"mask_encoding": "one_bit"})
            parts = body["input"][0]["content"]
            texts = [p["text"] for p in parts if p["type"] == "input_text"]
            images = [p["image_url"] for p in parts if p["type"] == "input_image"]
            self.assertEqual(len(texts), 1)
            self.assertEqual(len(images), 1)
            self.assertTrue(images[0].startswith("data:image/png;base64,"))
            seen.add(texts[0])
            payloads_per_concept.setdefault(texts[0], []).append(images[0])
        self.assertEqual(seen, set(FOCUS_CONCEPTS))
        for concept, urls in payloads_per_concept.items():
            self.assertEqual(len(urls), 5)
            self.assertEqual(len(set(urls)), 5)

    async def test_single_string_concept(self):
        session = FakeSession(FakeResponse(events=success_events()))
        seg = SamSegmenter(api_key="k", session=session)
        result = await seg.segment_image(gradient_png(), concepts="text")
        self.assertEqual(len(session.calls), 5)
        self.assertEqual(result["concepts"], ["text"])

    async def test_empty_concepts_rejected(self):
        seg = SamSegmenter(api_key="k", session=FakeSession(FakeResponse()))
        with self.assertRaises(ValueError):
            await seg.segment_image(gradient_png(), concepts=[])

    def test_missing_secret_rejected(self):
        with patch("sam_segmentation.st") as mock_st:
            mock_st.secrets = {}
            with self.assertRaises(SamApiError) as ctx:
                SamSegmenter(api_key=None)
        self.assertIn("META_API_KEY", str(ctx.exception))

    def test_empty_secret_rejected(self):
        with patch("sam_segmentation.st") as mock_st:
            mock_st.secrets = {"META_API_KEY": ""}
            with self.assertRaises(SamApiError):
                SamSegmenter(api_key=None)

    async def test_key_read_from_streamlit_secrets(self):
        session = FakeSession(FakeResponse(events=success_events()))
        with patch("sam_segmentation.st") as mock_st:
            mock_st.secrets = {"META_API_KEY": "secret-from-store"}
            seg = SamSegmenter(api_key=None, session=session)
            await seg.segment_image(gradient_png(), concepts="text")
        self.assertEqual(
            session.calls[0]["headers"]["Authorization"],
            "Bearer secret-from-store",
        )


class MaskParsingTests(unittest.IsolatedAsyncioTestCase):
    async def test_masks_decoded_with_raster(self):
        session = FakeSession(FakeResponse(events=success_events()))
        seg = SamSegmenter(api_key="k", session=session)
        result = await seg.segment_image(gradient_png(), concepts=["text"])
        self.assertEqual(result["mask_count"], 5)
        mask = result["masks"][0]
        self.assertEqual(mask["concept"], "text")
        self.assertEqual(mask["object_id"], "0")
        self.assertEqual(mask["confidence"], 0.95)
        # The official parser reports exclusive right/bottom edges.
        self.assertEqual(
            mask["box"], {"left": 1, "top": 2, "right": 11, "bottom": 13}
        )
        self.assertEqual(mask["mask"]["width"], 2)
        self.assertEqual(mask["mask"]["height"], 2)
        self.assertEqual(len(mask["mask"]["raster"]), 4)
        self.assertEqual(mask["mask"]["pixel_count"], 2)
        self.assertEqual(bytes(mask["mask"]["raster"]), bytes([1, 0, 0, 1]))

    def test_latest_revision_wins_per_identity(self):
        def record(identity, revision):
            mock = unittest.mock.create_autospec(
                SegmentationMaskRecord, instance=True
            )
            mock.identity = identity
            mock.revision = revision
            return mock

        old = record("a", 1)
        new = record("a", 3)
        other = record("b", 1)
        chosen = select_latest_masks([old, other, new])
        self.assertEqual(len(chosen), 2)
        self.assertIn(new, chosen)
        self.assertIn(other, chosen)
        self.assertNotIn(old, chosen)


class FidelityTests(unittest.IsolatedAsyncioTestCase):
    async def test_pil_sent_full_resolution_lossless(self):
        session = FakeSession(FakeResponse(events=success_events()))
        seg = SamSegmenter(api_key="k", session=session)
        arr = np.arange(32 * 24 * 3, dtype=np.uint8).reshape(24, 32, 3)
        await seg.segment_image(
            Image.fromarray(arr, mode="RGB"), concepts="text"
        )
        body = session.calls[0]["json"]
        url = body["input"][0]["content"][1]["image_url"]
        self.assertTrue(url.startswith("data:image/png;base64,"))
        sent = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
        self.assertEqual(sent.size, (32, 24))
        np.testing.assert_array_equal(np.asarray(sent.convert("RGB")), arr)

    async def test_raw_bytes_sent_unmodified(self):
        data = gradient_png()
        session = FakeSession(FakeResponse(events=success_events()))
        seg = SamSegmenter(api_key="k", session=session)
        await seg.segment_image(data, concepts="text")
        url = session.calls[0]["json"]["input"][0]["content"][1]["image_url"]
        self.assertEqual(base64.b64decode(url.split(",", 1)[1]), data)

    def test_unknown_bytes_rejected(self):
        with self.assertRaises(ValueError):
            sniff_image_mime(b"definitely not an image")


class ErrorMappingTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_error_maps_to_sam_api_error(self):
        session = FakeSession(
            FakeResponse(status=401, body='{"error": {"code": "bad_key"}}')
        )
        seg = SamSegmenter(api_key="bad", session=session)
        with self.assertRaises(SamApiError) as ctx:
            await seg.segment_image(gradient_png(), concepts="text")
        self.assertIn("401", str(ctx.exception))
        self.assertEqual(len(session.calls), 5)


class RetryPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_congestion_statuses_retry_then_succeed(self):
        for status in (429, 503, 504):
            with self.subTest(status=status):
                session = FakeSession(
                    [error_response(status), success_response()]
                )
                seg = SamSegmenter(
                    api_key="k", session=session, retrying=fast_retrying()
                )
                result = await seg.segment_image(
                    gradient_png(), concepts="text"
                )
                self.assertEqual(len(session.calls), 6)
                self.assertEqual(result["mask_count"], 5)

    async def test_exhaustion_reraises_after_three_attempts(self):
        session = FakeSession(error_response(503))
        seg = SamSegmenter(
            api_key="k", session=session, retrying=fast_retrying()
        )
        with self.assertRaises(SamRetryableError) as ctx:
            await seg.segment_image(gradient_png(), concepts="text")
        self.assertIn("503", str(ctx.exception))
        self.assertEqual(len(session.calls), 15)

    async def test_client_errors_never_retry(self):
        for status in (400, 401, 404):
            with self.subTest(status=status):
                session = FakeSession(error_response(status))
                seg = SamSegmenter(
                    api_key="k", session=session, retrying=fast_retrying()
                )
                with self.assertRaises(SamApiError) as ctx:
                    await seg.segment_image(gradient_png(), concepts="text")
                self.assertNotIsInstance(ctx.exception, SamRetryableError)
                self.assertEqual(len(session.calls), 5)

    def test_backoff_schedule_is_two_then_four_seconds(self):
        for attempt, expected in ((1, 2), (2, 4)):
            state = RetryCallState(Retrying(), None, (), {})
            state.attempt_number = attempt
            self.assertEqual(SAM_RETRY_WAIT(state), expected)

    def test_retryable_error_is_sam_api_error(self):
        self.assertTrue(issubclass(SamRetryableError, SamApiError))

    def test_retry_status_text_is_exact(self):
        self.assertEqual(RETRY_STATUS_TEXT, "API congested, retrying...")

    async def test_hook_reports_real_backoff_waits(self):
        seen = []
        session = FakeSession(
            [error_response(503), error_response(429), success_response()]
        )
        seg = SamSegmenter(
            api_key="k",
            session=session,
            on_retry=lambda attempt, wait: seen.append((attempt, wait)),
        )
        result = await seg.segment_image(gradient_png(), concepts="text")
        self.assertEqual(result["mask_count"], 5)
        self.assertCountEqual(seen, [(1, 2), (1, 2)])

    async def test_connection_drop_retries_three_times(self):
        session = RaisingSession(
            aiohttp.ClientConnectionError("connection reset by peer")
        )
        seg = SamSegmenter(
            api_key="k", session=session, retrying=fast_retrying()
        )
        with self.assertRaises(SamRetryableError):
            await seg.segment_image(gradient_png(), concepts="text")
        self.assertEqual(len(session.calls), 15)

    async def test_timeout_retries_three_times(self):
        session = RaisingSession(asyncio.TimeoutError())
        seg = SamSegmenter(
            api_key="k", session=session, retrying=fast_retrying()
        )
        with self.assertRaises(SamRetryableError):
            await seg.segment_image(gradient_png(), concepts="text")
        self.assertEqual(len(session.calls), 15)

    async def test_winerror_socket_fault_retries_and_reports(self):
        session = RaisingSession(
            OSError(
                "[WinError 121] The semaphore timeout period has expired"
            )
        )
        seg = SamSegmenter(
            api_key="k", session=session, retrying=fast_retrying()
        )
        with self.assertRaises(SamRetryableError) as ctx:
            await seg.segment_image(gradient_png(), concepts="text")
        self.assertIn("WinError 121", str(ctx.exception))
        self.assertEqual(len(session.calls), 15)

    async def test_midstream_drop_retries(self):
        session = FakeSession(
            ExplodingResponse(
                aiohttp.ClientPayloadError("connection lost mid-stream")
            )
        )
        seg = SamSegmenter(
            api_key="k", session=session, retrying=fast_retrying()
        )
        with self.assertRaises(SamRetryableError):
            await seg.segment_image(gradient_png(), concepts="text")
        self.assertEqual(len(session.calls), 15)

    async def test_malformed_stream_never_retries(self):
        session = FakeSession(BadJsonResponse())
        seg = SamSegmenter(
            api_key="k", session=session, retrying=fast_retrying()
        )
        with self.assertRaises(SamApiError) as ctx:
            await seg.segment_image(gradient_png(), concepts="text")
        self.assertNotIsInstance(ctx.exception, SamRetryableError)
        self.assertEqual(len(session.calls), 5)


class TransientCauseTests(unittest.TestCase):
    def test_direct_transport_errors_are_transient(self):
        self.assertTrue(
            has_transient_cause(
                aiohttp.ClientConnectionError("connection reset")
            )
        )
        self.assertTrue(has_transient_cause(asyncio.TimeoutError()))
        self.assertTrue(
            has_transient_cause(
                OSError(
                    "[WinError 121] The semaphore timeout period "
                    "has expired"
                )
            )
        )

    def test_wrapped_transport_errors_are_transient(self):
        cause = OSError(
            "[WinError 121] The semaphore timeout period has expired"
        )
        wrapped = ResponsesStreamSourceError("iterator", cause)
        self.assertTrue(has_transient_cause(wrapped))

    def test_non_transport_errors_are_not_transient(self):
        self.assertFalse(has_transient_cause(ValueError("bad payload")))
        wrapped = ResponsesStreamSourceError(
            "iterator", ValueError("bad json")
        )
        self.assertFalse(has_transient_cause(wrapped))


class MaskDisplayTests(unittest.TestCase):
    def _mask(self, concept="text", box="default", raster=b"\x01\x00\x00\x01"):
        if box == "default":
            box = {"left": 2, "top": 3, "right": 8, "bottom": 9}
        return {
            "concept": concept,
            "object_id": "0",
            "confidence": 0.9,
            "box": box,
            "mask": {"width": 2, "height": 2, "pixel_count": 2, "raster": raster},
        }

    def test_summarize_strips_raster_keeps_fields(self):
        summary = summarize_masks({"masks": [self._mask()]})
        self.assertEqual(len(summary), 1)
        entry = summary[0]
        self.assertNotIn("raster", entry)
        self.assertEqual(entry["concept"], "text")
        self.assertEqual(entry["pixel_count"], 2)
        self.assertEqual(entry["box"]["left"], 2)
        json.dumps(summary)

    def test_summarize_empty_result(self):
        self.assertEqual(summarize_masks(None), [])
        self.assertEqual(summarize_masks({}), [])

    def test_draw_boxes_marks_concept_color(self):
        img = Image.new("RGB", (20, 20), (255, 255, 255))
        out = draw_mask_boxes(img, [self._mask()])
        self.assertEqual(out.size, (20, 20))
        self.assertEqual(out.getpixel((2, 3)), (255, 80, 80))
        self.assertEqual(img.getpixel((2, 3)), (255, 255, 255))

    def test_draw_boxes_skips_missing_box(self):
        img = Image.new("RGB", (20, 20), (255, 255, 255))
        out = draw_mask_boxes(img, [self._mask(box=None)])
        self.assertEqual(list(out.getdata()), list(img.getdata()))

    def test_draw_boxes_unknown_concept_uses_default(self):
        img = Image.new("RGB", (20, 20), (255, 255, 255))
        out = draw_mask_boxes(img, [self._mask(concept="car")])
        self.assertEqual(out.getpixel((2, 3)), (200, 200, 200))

    async def test_failed_stream_maps_to_sam_api_error(self):
        session = FakeSession(
            FakeResponse(events=[{"type": "response.failed"}])
        )
        seg = SamSegmenter(api_key="k", session=session)
        with self.assertRaises(SamApiError):
            await seg.segment_image(gradient_png(), concepts="text")


def large_rgb_image(width=2048, height=1024):
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :, 0] = np.tile(
        np.linspace(0, 255, width, dtype=np.uint8), (height, 1)
    )
    return Image.fromarray(arr, mode="RGB")


def sent_image_size(call):
    url = call["json"]["input"][0]["content"][1]["image_url"]
    return Image.open(
        io.BytesIO(base64.b64decode(url.split(",", 1)[1]))
    ).size


class GracefulDegradationTests(unittest.IsolatedAsyncioTestCase):
    def test_fallback_long_edge_is_1024(self):
        self.assertEqual(FALLBACK_MAX_LONG_EDGE, 1024)

    def test_downscale_caps_long_edge_keeps_aspect_ratio(self):
        wide = downscale_to_long_edge(large_rgb_image(2048, 1024))
        self.assertEqual(wide.size, (1024, 512))
        tall = downscale_to_long_edge(large_rgb_image(1024, 2048))
        self.assertEqual(tall.size, (512, 1024))
        small = downscale_to_long_edge(large_rgb_image(512, 256))
        self.assertEqual(small.size, (512, 256))

    def test_dense_error_is_sam_api_error(self):
        self.assertTrue(issubclass(SamDenseImageError, SamApiError))

    def test_dense_message_is_exact(self):
        self.assertEqual(
            DENSE_IMAGE_MESSAGE,
            "The image is too dense for spatial processing. "
            "Please try a different memory.",
        )

    async def test_500_retries_once_with_resized_image(self):
        session = FakeSession(
            [error_response(500), success_response()]
        )
        seg = SamSegmenter(
            api_key="k", session=session, retrying=fast_retrying()
        )
        result = await seg.segment_image(
            large_rgb_image(), concepts="text"
        )
        self.assertEqual(result["mask_count"], 5)
        self.assertEqual(len(session.calls), 10)
        self.assertEqual(sent_image_size(session.calls[0]), (2048, 1024))
        initial_sizes = sorted(
            sent_image_size(call) for call in session.calls[:5]
        )
        self.assertEqual(
            initial_sizes, [(1024, 512)] * 4 + [(2048, 1024)]
        )
        self.assertEqual(sent_image_size(session.calls[5]), (1024, 512))
        fallback_sizes = sorted(
            sent_image_size(call) for call in session.calls[5:])
        self.assertEqual(fallback_sizes, [(512, 256)] * 4 + [(1024, 512)])

    async def test_500_fallback_bytes_retries_resized(self):
        buf = io.BytesIO()
        large_rgb_image().save(buf, format="PNG")
        session = FakeSession(
            [error_response(500), success_response()]
        )
        seg = SamSegmenter(
            api_key="k", session=session, retrying=fast_retrying()
        )
        result = await seg.segment_image(buf.getvalue(), concepts="text")
        self.assertEqual(result["mask_count"], 5)
        self.assertEqual(len(session.calls), 10)
        self.assertEqual(sent_image_size(session.calls[0]), (2048, 1024))
        initial_sizes = sorted(
            sent_image_size(call) for call in session.calls[:5]
        )
        self.assertEqual(
            initial_sizes, [(1024, 512)] * 4 + [(2048, 1024)]
        )
        self.assertEqual(sent_image_size(session.calls[5]), (1024, 512))
        fallback_sizes = sorted(
            sent_image_size(call) for call in session.calls[5:])
        self.assertEqual(fallback_sizes, [(512, 256)] * 4 + [(1024, 512)])

    async def test_500_fallback_failure_raises_dense_message(self):
        session = FakeSession(error_response(500))
        seg = SamSegmenter(
            api_key="k", session=session, retrying=fast_retrying()
        )
        with self.assertRaises(SamApiError) as ctx:
            await seg.segment_image(
                large_rgb_image(), concepts="text"
            )
        self.assertIn("too dense for spatial processing", str(ctx.exception))
        self.assertEqual(len(session.calls), 10)

    async def test_non_500_errors_never_trigger_fallback(self):
        session = FakeSession(error_response(400))
        seg = SamSegmenter(
            api_key="k", session=session, retrying=fast_retrying()
        )
        with self.assertRaises(SamApiError):
            await seg.segment_image(
                large_rgb_image(), concepts="text"
            )
        self.assertEqual(len(session.calls), 5)


class ConfidenceFilterTests(unittest.TestCase):
    def test_bounds_constants_are_exact(self):
        self.assertEqual(MIN_MASK_CONFIDENCE, 0.90)
        self.assertEqual(MAX_TILE_PAYLOADS, 5)
        self.assertEqual(MAX_CONCURRENT_SAM_REQUESTS, 5)

    def test_keeps_strictly_above_threshold(self):
        self.assertTrue(passes_confidence_filter({"confidence": 0.901}))
        self.assertTrue(passes_confidence_filter({"confidence": 1.0}))

    def test_drops_at_or_below_threshold(self):
        for bad in (0.90, 0.50, 0.0, -0.1):
            with self.subTest(bad=bad):
                self.assertFalse(passes_confidence_filter({"confidence": bad}))

    def test_drops_missing_or_non_numeric_confidence(self):
        for bad in (None, "0.99", True, float("nan")):
            with self.subTest(bad=bad):
                self.assertFalse(passes_confidence_filter({"confidence": bad}))
        self.assertFalse(passes_confidence_filter({}))


class TileBudgetTests(unittest.TestCase):
    def test_assembles_full_plus_quadrants(self):
        quads = [(f"q{i}", f"url{i}", (i, i)) for i in range(4)]
        payloads = assemble_tile_payloads("full-url", quads)
        self.assertEqual(len(payloads), 5)
        self.assertEqual(payloads[0], ("full", "full-url", None))

    def test_rejects_sixth_payload(self):
        quads = [(f"q{i}", f"url{i}", (i, i)) for i in range(5)]
        with self.assertRaises(SamApiError):
            assemble_tile_payloads("full-url", quads)

    def test_concurrency_defaults_and_clamps_to_five(self):
        default = SamSegmenter(api_key="k")
        self.assertEqual(default._max_concurrency, 5)
        clamped = SamSegmenter(api_key="k", max_concurrency=10)
        self.assertEqual(clamped._max_concurrency, 5)
        low = SamSegmenter(api_key="k", max_concurrency=0)
        self.assertEqual(low._max_concurrency, 1)


class GridCropGeometryTests(unittest.TestCase):
    """Pure slicing/offset geometry: no event loop needed."""

    def test_slice_even_grid(self):
        image = Image.new("RGB", (64, 48))
        tiles = slice_grid_tiles(image)
        self.assertEqual(
            [(name, crop.size, origin) for name, crop, origin in tiles],
            [
                ("top-left", (32, 24), (0, 0)),
                ("top-right", (32, 24), (32, 0)),
                ("bottom-left", (32, 24), (0, 24)),
                ("bottom-right", (32, 24), (32, 24)),
            ],
        )

    def test_slice_odd_dimensions_cover_frame_exactly(self):
        image = Image.new("RGB", (33, 25))
        tiles = slice_grid_tiles(image)
        self.assertEqual(
            [(name, crop.size, origin) for name, crop, origin in tiles],
            [
                ("top-left", (16, 12), (0, 0)),
                ("top-right", (17, 12), (16, 0)),
                ("bottom-left", (16, 13), (0, 12)),
                ("bottom-right", (17, 13), (16, 12)),
            ],
        )
        area = sum(crop.size[0] * crop.size[1] for _, crop, _ in tiles)
        self.assertEqual(area, 33 * 25)

    def test_slice_skips_degenerate_tiles(self):
        tiles = slice_grid_tiles(Image.new("RGB", (1, 24)))
        self.assertEqual(
            [(name, crop.size) for name, crop, _ in tiles],
            [("top-right", (1, 12)), ("bottom-right", (1, 12))],
        )

    def test_offset_box_translates_to_full_frame(self):
        self.assertEqual(
            offset_box_to_full_frame(
                {"left": 1, "top": 2, "right": 11, "bottom": 13}, (16, 12)
            ),
            {"left": 17, "top": 14, "right": 27, "bottom": 25},
        )

    def test_offset_box_passes_through_bad_boxes(self):
        self.assertIsNone(offset_box_to_full_frame(None, (16, 12)))
        partial = {"left": 1}
        self.assertIs(offset_box_to_full_frame(partial, (16, 12)), partial)
        non_numeric = {"left": "x", "top": 2, "right": 3, "bottom": 4}
        self.assertIs(
            offset_box_to_full_frame(non_numeric, (16, 12)), non_numeric
        )

    def test_summarize_prefers_global_frame_geometry(self):
        summaries = summarize_masks(
            {
                "masks": [
                    {
                        "concept": "text",
                        "tile": "bottom-right",
                        "object_id": "0",
                        "confidence": 0.9,
                        "box": {"left": 17, "top": 14, "right": 27, "bottom": 25},
                        "frame_width": 32,
                        "frame_height": 24,
                        "mask": {"width": 16, "height": 12, "pixel_count": 5},
                    },
                    {
                        "concept": "text",
                        "object_id": "1",
                        "confidence": 0.5,
                        "box": {"left": 1, "top": 2, "right": 3, "bottom": 4},
                        "mask": {"width": 32, "height": 24, "pixel_count": 6},
                    },
                ]
            }
        )
        self.assertEqual(summaries[0]["tile"], "bottom-right")
        self.assertEqual(summaries[0]["width"], 32)
        self.assertEqual(summaries[0]["height"], 24)
        self.assertEqual(summaries[1]["tile"], "full")
        self.assertEqual(summaries[1]["width"], 32)
        self.assertEqual(summaries[1]["height"], 24)


class GridCropFanOutTests(unittest.IsolatedAsyncioTestCase):
    """Tile fan-out, remap, and aggregation over segment_image."""

    async def test_five_tiles_aggregate_with_remapped_boxes(self):
        session = FakeSession(FakeResponse(events=success_events()))
        seg = SamSegmenter(api_key="k", session=session)
        result = await seg.segment_image(gradient_png(), concepts="text")
        self.assertEqual(len(session.calls), 5)
        self.assertEqual(result["mask_count"], 5)
        self.assertEqual(
            [mask["tile"] for mask in result["masks"]],
            ["full", "top-left", "top-right", "bottom-left", "bottom-right"],
        )
        boxes = [mask["box"] for mask in result["masks"]]
        # gradient_png() is 32x24: origins (0,0), (16,0), (0,12), (16,12).
        self.assertEqual(
            boxes[0], {"left": 1, "top": 2, "right": 11, "bottom": 13}
        )
        self.assertEqual(
            boxes[1], {"left": 1, "top": 2, "right": 11, "bottom": 13}
        )
        self.assertEqual(
            boxes[2], {"left": 17, "top": 2, "right": 27, "bottom": 13}
        )
        self.assertEqual(
            boxes[3], {"left": 1, "top": 14, "right": 11, "bottom": 25}
        )
        self.assertEqual(
            boxes[4], {"left": 17, "top": 14, "right": 27, "bottom": 25}
        )
        for mask in result["masks"][1:]:
            self.assertEqual(mask["frame_width"], 32)
            self.assertEqual(mask["frame_height"], 24)
        self.assertNotIn("frame_width", result["masks"][0])
        # Decoded rasters keep their own tile dimensions.
        for mask in result["masks"]:
            self.assertEqual(mask["mask"]["width"], 2)
            self.assertEqual(mask["mask"]["height"], 2)

    async def test_low_confidence_masks_filtered_from_aggregation(self):
        session = FakeSession(FakeResponse(events=low_confidence_events()))
        seg = SamSegmenter(api_key="k", session=session)
        result = await seg.segment_image(gradient_png(), concepts="text")
        self.assertEqual(len(session.calls), 5)
        self.assertEqual(result["mask_count"], 0)
        self.assertEqual(result["masks"], [])

if __name__ == "__main__":
    unittest.main()
