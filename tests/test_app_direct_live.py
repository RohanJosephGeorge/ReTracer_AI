"""Direct live pipeline: no localhost backend, toggle defaults off.

Run: python -m unittest discover -s tests -v
Requires streamlit and pandas. Streamlit is mocked: no browser, no network.
"""

import inspect
import io
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

import app


class StateDict(dict):
    """Dict with attribute access, mirroring Streamlit session_state."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class FakeUpload(io.BytesIO):
    def __init__(self, data, name="photo.png", mime="image/png"):
        super().__init__(data)
        self.name = name
        self.type = mime
        self.size = len(data)


def png_upload(width=64, height=48, name="photo.png"):
    x = np.tile(np.arange(width, dtype=np.uint8), (height, 1))
    y = np.tile(np.arange(height, dtype=np.uint8)[:, None], (1, width))
    img = Image.fromarray(np.stack([x, y, x], axis=-1), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return FakeUpload(buf.getvalue(), name=name)


def clean_screener(image):
    return ("clean", "No faces detected.", image)


def wonder_payload():
    return {
        "latitude": 48.8584,
        "longitude": 2.2945,
        "city": "Paris",
        "country": "France",
        "reasoning_steps": ["Iron lattice tower"],
        "confidence_score": 0.95,
        "precision_level": "Public Landmark",
    }


class ToggleDefaultTests(unittest.TestCase):
    def test_render_home_defaults_to_live(self):
        sig = inspect.signature(app.render_home)
        self.assertFalse(sig.parameters["developer_mode"].default)

    def test_fresh_session_defaults_developer_mode_off(self):
        state = StateDict(
            {
                "is_analyzing": False,
                "analysis_results": None,
                "last_upload_key": None,
                "screened_image": None,
                "sam_result": None,
                "wonder_result": None,
            }
        )
        mock_st = MagicMock()
        mock_st.session_state = state
        mock_st.file_uploader.return_value = None
        mock_st.text_input.return_value = ""
        mock_st.columns.return_value = (MagicMock(), MagicMock(), MagicMock())
        mock_ui = MagicMock()
        mock_ui.button.return_value = False
        with (
            patch.object(app, "st", mock_st),
            patch.object(app, "ui", mock_ui),
        ):
            app.render_home()
        self.assertFalse(state["developer_mode"])

    def test_main_page_toggle_defaults_off(self):
        state = StateDict(
            {
                "is_analyzing": False,
                "analysis_results": None,
                "last_upload_key": None,
                "screened_image": None,
                "sam_result": None,
                "wonder_result": None,
            }
        )
        mock_st = MagicMock()
        mock_st.session_state = state
        mock_st.file_uploader.return_value = None
        mock_st.text_input.return_value = ""
        mock_st.columns.return_value = (MagicMock(), MagicMock(), MagicMock())
        mock_ui = MagicMock()
        mock_ui.button.return_value = False
        with (
            patch.object(app, "st", mock_st),
            patch.object(app, "ui", mock_ui),
        ):
            app.render_home()
        mock_st.toggle.assert_called_once_with(
            "Developer Mode (Use Mock Data)", value=False, key="dev_mode"
        )
        self.assertEqual(mock_st.columns.call_count, 3)
        self.assertEqual(
            mock_st.columns.call_args_list[0].args, ([4, 1, 1],)
        )
        for call in mock_st.columns.call_args_list[1:]:
            self.assertEqual(call.args, ([1, 2, 1],))
        mock_st.sidebar.toggle.assert_not_called()

    def test_main_routes_nav_radio_selection(self):
        mock_st = MagicMock()
        mock_st.columns.return_value = (MagicMock(), MagicMock(), MagicMock())
        mock_st.radio.return_value = "How it Works"
        with (
            patch.object(app, "st", mock_st),
            patch.object(app, "render_home") as mock_home,
            patch.object(app, "render_how_it_works") as mock_how,
            patch.object(app, "render_privacy_safety") as mock_privacy,
        ):
            app.main()
        mock_st.radio.assert_called_once_with(
            label="",
            options=["Home", "How it Works", "Privacy & Safety"],
            horizontal=True,
            label_visibility="collapsed",
            key="main_nav",
        )
        mock_st.columns.assert_called_once_with([1, 3, 1])
        mock_how.assert_called_once_with()
        mock_home.assert_not_called()
        mock_privacy.assert_not_called()
        mock_st.sidebar.toggle.assert_not_called()

    def test_main_home_nav_renders_home(self):
        mock_st = MagicMock()
        mock_st.columns.return_value = (MagicMock(), MagicMock(), MagicMock())
        mock_st.radio.return_value = "Home"
        with (
            patch.object(app, "st", mock_st),
            patch.object(app, "render_home") as mock_home,
            patch.object(app, "render_how_it_works") as mock_how,
            patch.object(app, "render_privacy_safety") as mock_privacy,
        ):
            app.main()
        mock_home.assert_called_once_with()
        mock_how.assert_not_called()
        mock_privacy.assert_not_called()

    def test_custom_css_has_high_contrast_widget_labels(self):
        self.assertIn('div[data-testid="stWidgetLabel"]', app.CUSTOM_CSS)
        self.assertIn(".stToggle label p", app.CUSTOM_CSS)
        self.assertIn("color: #ffffff", app.CUSTOM_CSS)

    def test_custom_css_has_toggle_glass_pill(self):
        self.assertIn('div[data-testid="stCheckbox"]', app.CUSTOM_CSS)
        self.assertIn("border-radius: 50px", app.CUSTOM_CSS)
        self.assertIn(
            'div[data-testid="stCheckbox"] label span p', app.CUSTOM_CSS
        )

    def test_custom_css_has_expanding_pill_navbar(self):
        self.assertIn(
            'div[data-testid="stRadio"] > div[role="radiogroup"]',
            app.CUSTOM_CSS,
        )
        self.assertIn("background: #0d0e12", app.CUSTOM_CSS)
        self.assertIn("flex-wrap: nowrap", app.CUSTOM_CSS)
        self.assertIn("white-space: nowrap", app.CUSTOM_CSS)
        self.assertIn(
            "transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
            app.CUSTOM_CSS,
        )
        self.assertIn(
            'div[role="radiogroup"] label > div:first-child',
            app.CUSTOM_CSS,
        )
        self.assertIn(
            "background: rgba(255, 255, 255, 0.18)", app.CUSTOM_CSS
        )
        self.assertIn("data:image/svg+xml", app.CUSTOM_CSS)
        self.assertNotIn("📷", app.CUSTOM_CSS)
        self.assertNotIn("🗺️", app.CUSTOM_CSS)
        self.assertNotIn("🛡️", app.CUSTOM_CSS)

    def test_custom_css_has_glass_form_card(self):
        self.assertIn(".upload-card {", app.CUSTOM_CSS)
        self.assertIn("max-width: 620px", app.CUSTOM_CSS)
        self.assertIn(
            "background: rgba(15, 23, 42, 0.6)", app.CUSTOM_CSS
        )
        self.assertIn(
            'div[data-testid="stFileUploader"] li', app.CUSTOM_CSS
        )
        self.assertIn(
            'div[data-testid="stVerticalBlockBorderWrapper"]',
            app.CUSTOM_CSS,
        )
        self.assertIn(
            'div[data-testid="stTextInput"] input', app.CUSTOM_CSS
        )
        self.assertIn("background: rgba(0, 0, 0, 0.35)", app.CUSTOM_CSS)
        self.assertIn("input::placeholder", app.CUSTOM_CSS)
        self.assertIn(
            "color: rgba(255, 255, 255, 0.5)", app.CUSTOM_CSS
        )
        self.assertIn(
            'section[data-testid="stFileUploadDropzone"]', app.CUSTOM_CSS
        )
        self.assertIn(
            'section[data-testid="stFileUploadDropzone"]:hover',
            app.CUSTOM_CSS,
        )
        self.assertIn(
            "background: rgba(15, 23, 42, 0.6)", app.CUSTOM_CSS
        )
        self.assertIn(
            'div[data-testid="stButton"] button {', app.CUSTOM_CSS
        )
        self.assertIn(
            "color: #f8fafc", app.CUSTOM_CSS
        )

    def test_custom_css_styles_inner_dropzone_button(self):
        self.assertIn(
            'section[data-testid="stFileUploadDropzone"] button {',
            app.CUSTOM_CSS,
        )
        self.assertIn(
            "background: rgba(18, 18, 22, 0.85)", app.CUSTOM_CSS
        )
        self.assertIn(
            'section[data-testid="stFileUploadDropzone"] button:hover',
            app.CUSTOM_CSS,
        )
        self.assertIn(
            'section[data-testid="stFileUploadDropzone"] button svg',
            app.CUSTOM_CSS,
        )
        self.assertIn(
            "background: rgba(0, 0, 0, 0.2)", app.CUSTOM_CSS
        )

    def test_custom_css_uploader_hides_tooltip_and_lightens_subtext(self):
        self.assertIn("stTooltipIcon", app.CUSTOM_CSS)
        self.assertIn(
            'div[data-testid="stFileUploader"] label button',
            app.CUSTOM_CSS,
        )
        self.assertIn(
            "color: rgba(255, 255, 255, 0.6)", app.CUSTOM_CSS
        )

    def test_custom_css_hides_header_and_trims_top_margin(self):
        self.assertIn('header[data-testid="stHeader"]', app.CUSTOM_CSS)
        self.assertIn(".stAppHeader", app.CUSTOM_CSS)
        self.assertIn("display: none", app.CUSTOM_CSS)
        self.assertIn(
            'div[data-testid="stAppViewBlockContainer"]', app.CUSTOM_CSS
        )
        self.assertIn("padding-top: 1.5rem", app.CUSTOM_CSS)

    def test_custom_css_styles_status_banners_for_contrast(self):
        self.assertIn(
            'div[data-testid="stAlert"] > div {', app.CUSTOM_CSS
        )
        self.assertIn(
            "background: rgba(15, 23, 42, 0.72)", app.CUSTOM_CSS
        )
        self.assertIn(
            'div[data-testid="stAlert"] p', app.CUSTOM_CSS
        )


class DirectLivePipelineTests(unittest.TestCase):
    def test_live_path_bypasses_localhost_backend(self):
        state = StateDict(
            {
                "developer_mode": False,
                "is_analyzing": True,
                "analysis_results": None,
                "last_upload_key": None,
                "screened_image": None,
                "sam_result": None,
                "wonder_result": None,
            }
        )
        mock_st = MagicMock()
        mock_st.session_state = state
        mock_st.file_uploader.return_value = None
        mock_st.text_input.return_value = ""
        mock_st.columns.return_value = (MagicMock(), MagicMock(), MagicMock())
        mock_ui = MagicMock()
        mock_ui.button.return_value = False
        mock_execute = MagicMock()
        mock_direct = MagicMock(
            return_value={"kind": "ok", "mock": False, "result": {}}
        )
        with (
            patch.object(app, "st", mock_st),
            patch.object(app, "ui", mock_ui),
            patch.object(app, "execute_analysis", mock_execute),
            patch.object(app, "execute_live_analysis_direct", mock_direct),
            patch.object(app, "load_lottie_animation", return_value=None),
            patch("time.sleep"),
        ):
            app.render_home(developer_mode=False)
        mock_execute.assert_not_called()
        mock_direct.assert_called_once()

    def test_direct_missing_file_halts(self):
        outcome = app.execute_live_analysis_direct(
            None, "", screener=clean_screener
        )
        self.assertEqual(outcome["kind"], "halt")
        self.assertIn("No file", outcome["message"])

    def test_direct_success_returns_live_result_without_localhost(self):
        with (
            patch.object(
                app, "_run_live_pipeline_sync", return_value=wonder_payload()
            ) as mock_run,
            patch("urllib.request.urlopen") as mock_urlopen,
        ):
            outcome = app.execute_live_analysis_direct(
                png_upload(), "riverside", screener=clean_screener
            )
        mock_run.assert_called_once()
        mock_urlopen.assert_not_called()
        self.assertEqual(outcome["kind"], "ok")
        self.assertFalse(outcome["mock"])
        self.assertEqual(outcome["result"]["mode"], "live")
        self.assertEqual(
            outcome["result"]["coordinates"],
            {"latitude": 48.858, "longitude": 2.295},
        )

    def test_live_pipeline_catches_full_dense_stream(self):
        from spatial_reasoning import build_spatial_prompt

        # 5 tiles x 3 concepts x 2 high-confidence masks.
        tiles = ("full", "top-left", "top-right", "bottom-left",
                 "bottom-right")
        masks = []
        for tile in tiles:
            for concept in ("text", "flora", "road"):
                for ordinal in range(2):
                    masks.append({
                        "concept": concept,
                        "tile": tile,
                        "object_id": f"{tile}-{concept}-{ordinal}",
                        "confidence": 0.95,
                        "box": {"left": 1, "top": 2, "right": 3, "bottom": 4},
                        "mask": {"width": 32, "height": 24, "pixel_count": 6},
                    })
        sam_result = {
            "model": "m",
            "concepts": ["text", "flora", "road"],
            "mask_count": len(masks),
            "masks": masks,
        }
        captured = {}

        def fake_reason(summary, hint, on_retry, image, api_key):
            captured["summary"] = summary
            return wonder_payload()

        with (
            patch.object(app, "_segment_image_sync", return_value=sam_result),
            patch.object(app, "_reason_wonder_sync", side_effect=fake_reason),
            patch.object(app, "get_meta_api_key", return_value="k"),
        ):
            outcome = app.execute_live_analysis_direct(
                png_upload(), "riverside", screener=clean_screener
            )
        self.assertEqual(outcome["kind"], "ok")
        summary = captured["summary"]
        self.assertEqual(len(summary), 30)
        self.assertEqual({entry["tile"] for entry in summary}, set(tiles))
        # The real prompt embeds every tile of the dense list.
        prompt = build_spatial_prompt(summary, "riverside")
        for tile in tiles:
            self.assertIn(tile, prompt)
        self.assertEqual(prompt.count('"tile"'), 30)
        self.assertIn("bottom-right-road-1", prompt)


    def test_direct_forwards_on_progress_to_pipeline(self):
        seen = []

        def on_progress(percent, message):
            seen.append((percent, message))

        with patch.object(
            app, "_run_live_pipeline_sync", return_value=wonder_payload()
        ) as mock_run:
            outcome = app.execute_live_analysis_direct(
                png_upload(),
                "riverside",
                screener=clean_screener,
                on_progress=on_progress,
            )
        self.assertEqual(outcome["kind"], "ok")
        self.assertIs(
            mock_run.call_args.kwargs["on_progress"], on_progress
        )
        self.assertEqual(seen, [])

    def test_direct_api_error_returns_error_outcome(self):
        from sam_segmentation import SamApiError

        with patch.object(
            app, "_run_live_pipeline_sync", side_effect=SamApiError("down")
        ):
            outcome = app.execute_live_analysis_direct(
                png_upload(), "", screener=clean_screener
            )
        self.assertEqual(outcome["kind"], "error")
        self.assertIn("down", outcome["message"])

    def test_direct_dense_image_returns_friendly_message(self):
        from sam_segmentation import (
            DENSE_IMAGE_MESSAGE,
            SamApiError,
            SamDenseImageError,
        )

        for error in (
            SamDenseImageError(DENSE_IMAGE_MESSAGE),
            SamApiError('SAM request failed (HTTP 500): {"error": {}}'),
        ):
            with (
                self.subTest(error=error),
                patch.object(
                    app, "_run_live_pipeline_sync", side_effect=error
                ),
            ):
                outcome = app.execute_live_analysis_direct(
                    png_upload(), "", screener=clean_screener
                )
            self.assertEqual(outcome["kind"], "error")
            self.assertEqual(outcome["message"], DENSE_IMAGE_MESSAGE)


if __name__ == "__main__":
    unittest.main()
