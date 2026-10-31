"""Tests for the Analyze Memory Lottie loading animation.

Run: python -m unittest discover -s tests -v
Requires streamlit and pandas (no browser, no network: HTTP is mocked).
Each test uses a distinct URL because load_lottie_animation() caches
per URL.
"""

import json
import unittest
import urllib.error
from unittest.mock import patch

import app
from app import LOTTIE_RADAR_URL, load_lottie_animation


def lottie_bytes():
    return json.dumps(
        {
            "v": "5.5.9",
            "fr": 60,
            "ip": 0,
            "op": 241,
            "w": 766,
            "h": 576,
            "layers": [],
        }
    ).encode("utf-8")


class FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self):
        return self._payload


class LoadLottieAnimationTests(unittest.TestCase):
    def test_valid_lottie_json_returned_as_dict(self):
        with patch(
            "urllib.request.urlopen",
            return_value=FakeResponse(lottie_bytes()),
        ) as mock_open:
            animation = load_lottie_animation("https://example.com/a.json")
        self.assertIsInstance(animation, dict)
        self.assertEqual(animation["v"], "5.5.9")
        self.assertEqual(animation["fr"], 60)
        mock_open.assert_called_once_with(
            "https://example.com/a.json", timeout=app.LOTTIE_FETCH_TIMEOUT
        )

    def test_network_failure_returns_none(self):
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("down"),
        ):
            self.assertIsNone(
                load_lottie_animation("https://example.com/b.json")
            )

    def test_invalid_json_returns_none(self):
        with patch(
            "urllib.request.urlopen",
            return_value=FakeResponse(b"<html>not json</html>"),
        ):
            self.assertIsNone(
                load_lottie_animation("https://example.com/c.json")
            )

    def test_non_object_json_returns_none(self):
        with patch(
            "urllib.request.urlopen",
            return_value=FakeResponse(b"[1, 2, 3]"),
        ):
            self.assertIsNone(
                load_lottie_animation("https://example.com/d.json")
            )

    def test_error_json_returns_none(self):
        with patch(
            "urllib.request.urlopen",
            return_value=FakeResponse(b'{"error": "gone"}'),
        ):
            self.assertIsNone(
                load_lottie_animation("https://example.com/e.json")
            )


def white_fill_payload():
    return {
        "v": "5.5.9",
        "fr": 60,
        "ip": 0,
        "op": 10,
        "w": 100,
        "h": 100,
        "layers": [
            {
                "nm": "radar",
                "shapes": [
                    {
                        "ty": "gr",
                        "it": [
                            {"ty": "sh", "ks": {}},
                            {
                                "ty": "fl",
                                "c": {"a": 0, "k": [1, 1, 1, 1]},
                                "o": {"a": 0, "k": 100},
                            },
                            {
                                "ty": "fl",
                                "c": {"a": 0, "k": [1, 1, 1, 0.5]},
                                "o": {"a": 0, "k": 100},
                            },
                            {
                                "ty": "fl",
                                "c": {"a": 0, "k": [1, 0, 0, 1]},
                                "o": {"a": 0, "k": 100},
                            },
                        ],
                    }
                ],
            }
        ],
    }


class TintLottieTests(unittest.TestCase):
    def test_white_fills_tinted_neon_alpha_preserved(self):
        payload = white_fill_payload()
        tinted = app.tint_lottie_white_fills(payload)
        self.assertIs(tinted, payload)
        fills = payload["layers"][0]["shapes"][0]["it"][1:]
        neon = [0.0, 1.0, 170 / 255]
        self.assertEqual(fills[0]["c"]["k"], [*neon, 1])
        self.assertEqual(fills[1]["c"]["k"], [*neon, 0.5])
        self.assertEqual(fills[2]["c"]["k"], [1, 0, 0, 1])

    def test_animated_white_keyframes_tinted(self):
        payload = {
            "v": "5.5.9",
            "layers": [
                {
                    "shapes": [
                        {
                            "ty": "fl",
                            "c": {
                                "a": 1,
                                "k": [
                                    {
                                        "t": 0,
                                        "s": [1, 1, 1, 1],
                                        "e": [1, 1, 1, 0.2],
                                    },
                                    {"t": 10, "s": [0, 0, 1, 1]},
                                ],
                            },
                        }
                    ]
                }
            ],
        }
        app.tint_lottie_white_fills(payload)
        frames = payload["layers"][0]["shapes"][0]["c"]["k"]
        neon = [0.0, 1.0, 170 / 255]
        self.assertEqual(frames[0]["s"], [*neon, 1])
        self.assertEqual(frames[0]["e"], [*neon, 0.2])
        self.assertEqual(frames[1]["s"], [0, 0, 1, 1])

    def test_closed_path_flags_untouched(self):
        payload = {"v": "5.5.9", "layers": [{"c": True}]}
        app.tint_lottie_white_fills(payload)
        self.assertTrue(payload["layers"][0]["c"])

    def test_loaded_animation_returns_tinted(self):
        raw = json.dumps(white_fill_payload()).encode("utf-8")
        with patch(
            "urllib.request.urlopen",
            return_value=FakeResponse(raw),
        ):
            animation = load_lottie_animation("https://example.com/f.json")
        fills = animation["layers"][0]["shapes"][0]["it"][1:]
        neon = [0.0, 1.0, 170 / 255]
        self.assertEqual(fills[0]["c"]["k"], [*neon, 1])
        self.assertEqual(fills[2]["c"]["k"], [1, 0, 0, 1])


class LottiePayloadTests(unittest.TestCase):
    def test_valid_payload_accepted(self):
        self.assertTrue(
            app._is_lottie_payload(
                {"v": "5.5.9", "fr": 30, "layers": [], "w": 1, "h": 1}
            )
        )

    def test_rejects_non_lottie_payloads(self):
        for bad in (
            None,
            [],
            "lottie",
            {},
            {"v": "5.5.9"},
            {"layers": []},
            {"v": 5, "layers": []},
            {"v": "5.5.9", "layers": {}},
            {"error": "gone"},
        ):
            self.assertFalse(app._is_lottie_payload(bad), bad)

    def test_default_url_points_at_radar_json(self):
        self.assertTrue(LOTTIE_RADAR_URL.startswith("https://"))
        self.assertTrue(LOTTIE_RADAR_URL.endswith("lottie_radar.json"))

    def test_lottie_import_degrades_gracefully(self):
        # Guarded import: st_lottie is either the component or None,
        # never an ImportError at module load.
        self.assertTrue(
            app.st_lottie is None or callable(app.st_lottie)
        )


if __name__ == "__main__":
    unittest.main()
