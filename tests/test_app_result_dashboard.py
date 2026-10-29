"""Tests for the luxury result dashboard (no raw JSON dump).

Run: python -m unittest discover -s tests -v
Requires streamlit and pandas. Streamlit is mocked: no browser.
"""

import unittest
from unittest.mock import MagicMock, patch

import app


def live_result(**overrides):
    result = {
        "status": "ok",
        "mode": "live",
        "filename": "memory.jpg",
        "regional_hint": "riverside cafes",
        "city": "Paris",
        "country": "France",
        "reasoning_steps": [
            "Detected wrought-iron lattice tower structure in frame.",
            "Matched silhouette against public landmark embeddings.",
        ],
        "precision": "Public Landmark",
        "confidence": 0.95,
        "coordinates": {"latitude": 48.858, "longitude": 2.295},
    }
    result.update(overrides)
    return result


def dashboard_html(mock_st):
    return mock_st.markdown.call_args[0][0]


class FormatHelpersTests(unittest.TestCase):
    def test_confidence_formats_as_percent(self):
        self.assertEqual(app.format_confidence(0.95), "95%")
        self.assertEqual(app.format_confidence(0), "0%")

    def test_confidence_rejects_non_numbers(self):
        self.assertEqual(app.format_confidence(None), "Unknown")
        self.assertEqual(app.format_confidence(True), "Unknown")
        self.assertEqual(app.format_confidence("high"), "Unknown")

    def test_coordinates_format_compass_points(self):
        self.assertEqual(
            app.format_coordinates(
                {"latitude": 48.8584, "longitude": 2.2945}
            ),
            "48.8584° N, 2.2945° E",
        )
        self.assertEqual(
            app.format_coordinates(
                {"latitude": -33.86, "longitude": -118.24}
            ),
            "33.86° S, 118.24° W",
        )

    def test_coordinates_reject_bad_shapes(self):
        self.assertIsNone(app.format_coordinates(None))
        self.assertIsNone(app.format_coordinates("48.8, 2.2"))
        self.assertIsNone(
            app.format_coordinates({"latitude": None, "longitude": 2.0})
        )
        self.assertIsNone(
            app.format_coordinates({"latitude": True, "longitude": 2.0})
        )


class ResultDashboardTests(unittest.TestCase):
    def render(self, result):
        mock_st = MagicMock()
        with patch.object(app, "st", mock_st):
            app.render_result_dashboard(result)
        return mock_st

    def test_location_header_and_metrics(self):
        html = dashboard_html(self.render(live_result()))
        self.assertIn("Paris, France", html)
        self.assertIn("result-location", html)
        self.assertIn("Public Landmark", html)
        self.assertIn("95%", html)
        self.assertIn("result-glass", html)

    def test_reasoning_timeline_has_no_json_syntax(self):
        html = dashboard_html(self.render(live_result()))
        self.assertIn("deduction-timeline", html)
        self.assertIn(
            "Detected wrought-iron lattice tower structure in frame.",
            html,
        )
        self.assertIn(
            "Matched silhouette against public landmark embeddings.", html
        )
        self.assertNotIn('"reasoning_steps"', html)
        self.assertNotIn('{"status"', html)
        self.assertNotIn("[", html.split("deduction-timeline")[1])

    def test_coordinates_caption_shown(self):
        html = dashboard_html(self.render(live_result()))
        self.assertIn("48.858° N, 2.295° E", html)

    def test_mock_without_city_falls_back_to_coordinates(self):
        result = live_result()
        del result["city"]
        del result["country"]
        html = dashboard_html(self.render(result))
        self.assertIn("48.858° N, 2.295° E", html)
        self.assertNotIn("Location Identified", html)

    def test_missing_everything_stays_graceful(self):
        html = dashboard_html(self.render({}))
        self.assertIn("Location Identified", html)
        self.assertIn("Unknown", html)
        self.assertIn("No reasoning steps recorded.", html)

    def test_dynamic_values_are_escaped(self):
        html = dashboard_html(
            self.render(live_result(city="<script>alert(1)</script>"))
        )
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)


class AnalysisOutcomeDashboardTests(unittest.TestCase):
    def test_ok_renders_dashboard_and_map_without_json(self):
        mock_st = MagicMock()
        outcome = {"kind": "ok", "mock": False, "result": live_result()}
        with patch.object(app, "st", mock_st):
            app.render_analysis_outcome(outcome)
        mock_st.json.assert_not_called()
        mock_st.markdown.assert_called_once()
        mock_st.map.assert_called_once()

    def test_ok_without_coordinates_skips_map(self):
        mock_st = MagicMock()
        result = live_result()
        del result["coordinates"]
        with patch.object(app, "st", mock_st):
            app.render_analysis_outcome(
                {"kind": "ok", "mock": True, "result": result}
            )
        mock_st.json.assert_not_called()
        mock_st.map.assert_not_called()

    def test_halt_and_error_paths_unchanged(self):
        mock_st = MagicMock()
        with patch.object(app, "st", mock_st):
            app.render_analysis_outcome(
                {"kind": "halt", "message": "Face too large."}
            )
        mock_st.warning.assert_called_once_with("Face too large.")
        mock_st.markdown.assert_not_called()
        mock_st.json.assert_not_called()
        mock_st = MagicMock()
        with patch.object(app, "st", mock_st):
            app.render_analysis_outcome(
                {"kind": "error", "message": "Backend down."}
            )
        mock_st.error.assert_called_once_with("Backend down.")
        mock_st.markdown.assert_not_called()
        mock_st.json.assert_not_called()


if __name__ == "__main__":
    unittest.main()
