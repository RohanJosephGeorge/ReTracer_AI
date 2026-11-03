"""Mobile upload and regional-hint carryover regressions.

Run: python -m unittest discover -s tests -v
Streamlit is mocked: no browser.
"""

import io
import unittest
from unittest.mock import MagicMock, patch

import app
from file_sanitizer import HEIC_HELP_MESSAGE, is_valid_image


class StateDict(dict):
    """Dict with attribute access, mirroring Streamlit session_state."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def home_state(**overrides):
    state = StateDict(
        {
            "dev_mode": False,
            "developer_mode": False,
            "is_analyzing": False,
            "analysis_results": None,
            "last_upload_key": None,
            "screened_image": None,
            "sam_result": None,
            "wonder_result": None,
            "upload_epoch": 0,
        }
    )
    state.update(overrides)
    return state


def run_home(state, uploaded_file=None, hint=""):
    mock_st = MagicMock()
    mock_st.session_state = state
    mock_st.file_uploader.return_value = uploaded_file
    mock_st.text_input.return_value = hint
    mock_st.columns.return_value = (MagicMock(), MagicMock(), MagicMock())
    mock_ui = MagicMock()
    mock_ui.button.return_value = False
    with (
        patch.object(app, "st", mock_st),
        patch.object(app, "ui", mock_ui),
        patch.object(
            app, "is_valid_image", return_value=(False, "invalid upload")
        ),
    ):
        app.render_home()
    return mock_st


def fake_upload(name="photo.jpg", size=1234):
    upload = MagicMock()
    upload.name = name
    upload.size = size
    return upload


class HintKeyScopingTests(unittest.TestCase):
    def test_hint_key_follows_epoch(self):
        self.assertEqual(app.hint_key_for_epoch(0), "regional_hint_input_0")
        self.assertEqual(app.hint_key_for_epoch(4), "regional_hint_input_4")

    def test_clear_hint_state_removes_all_hint_keys_only(self):
        state = StateDict(
            {
                "regional_hint_input": "legacy",
                "regional_hint_input_0": "old",
                "regional_hint_input_5": "older",
                "last_upload_key": "a:1",
            }
        )
        mock_st = MagicMock()
        mock_st.session_state = state
        with patch.object(app, "st", mock_st):
            app.clear_hint_state()
        self.assertEqual(state, {"last_upload_key": "a:1"})

    def test_reset_bumps_epoch_so_next_hint_key_is_fresh(self):
        state = home_state(upload_epoch=3)
        state["regional_hint_input_3"] = "Kerala"
        mock_st = MagicMock()
        mock_st.session_state = state
        with patch.object(app, "st", mock_st):
            app.reset_home_for_new_upload()
        self.assertEqual(state["upload_epoch"], 4)
        self.assertNotIn("regional_hint_input_3", state)
        self.assertEqual(app.hint_key_for_epoch(4), "regional_hint_input_4")


class HintCarryoverTests(unittest.TestCase):
    def test_new_upload_clears_stale_hint_and_results(self):
        state = home_state(
            last_upload_key="old.jpg:1",
            analysis_results={"kind": "ok", "mock": False, "result": {}},
            regional_hint_input_0="Kerala backwaters",
        )
        run_home(state, uploaded_file=fake_upload("new.jpg", 2))
        self.assertIsNone(state["analysis_results"])
        self.assertEqual(state["last_upload_key"], "new.jpg:2")
        self.assertNotIn("regional_hint_input_0", state)

    def test_same_upload_keeps_hint_and_results(self):
        outcome = {"kind": "ok", "mock": False, "result": {}}
        state = home_state(
            last_upload_key="photo.jpg:1234",
            analysis_results=outcome,
            regional_hint_input_0="Kerala backwaters",
        )
        run_home(state, uploaded_file=fake_upload("photo.jpg", 1234))
        self.assertIs(state["analysis_results"], outcome)
        self.assertEqual(
            state["regional_hint_input_0"], "Kerala backwaters"
        )

    def test_armed_live_run_submits_current_hint(self):
        state = home_state(developer_mode=False, is_analyzing=True)
        upload = fake_upload()
        mock_st = MagicMock()
        mock_st.session_state = state
        mock_st.file_uploader.return_value = upload
        mock_st.text_input.return_value = "new hint"
        mock_st.columns.return_value = (MagicMock(), MagicMock(), MagicMock())
        mock_ui = MagicMock()
        mock_ui.button.return_value = False
        outcome = {"kind": "ok", "mock": False, "result": {}}
        mock_direct = MagicMock(return_value=outcome)
        with (
            patch.object(app, "st", mock_st),
            patch.object(app, "ui", mock_ui),
            patch.object(app, "execute_analysis", MagicMock()),
            patch.object(app, "execute_live_analysis_direct", mock_direct),
            patch.object(app, "load_lottie_animation", return_value=None),
            patch("time.sleep"),
        ):
            app.render_home(developer_mode=False)
        self.assertEqual(mock_direct.call_args.args, (upload, "new hint"))
        self.assertIs(state["analysis_results"], outcome)


class HintNoEnterCommitTests(unittest.TestCase):
    def test_hint_commits_while_typing_without_enter(self):
        state = home_state()
        mock_st = run_home(state)
        mock_st.text_input.assert_called_once()
        kwargs = mock_st.text_input.call_args.kwargs
        self.assertTrue(
            kwargs.get("live"),
            "hint must commit while typing (live) so Enter is not required",
        )
        self.assertEqual(kwargs.get("on_change"), "ignore")


class UploadKeyTests(unittest.TestCase):
    def test_none_returns_none(self):
        self.assertIsNone(app.upload_key_for_file(None))

    def test_key_combines_name_and_size(self):
        self.assertEqual(
            app.upload_key_for_file(fake_upload("a.jpg", 1)), "a.jpg:1"
        )


class MobileUploadGuidanceTests(unittest.TestCase):
    def test_upload_tip_rendered_under_uploader(self):
        state = home_state()
        mock_st = run_home(state)
        mock_st.caption.assert_called_once_with(app.MOBILE_UPLOAD_TIP)
        self.assertIn("Files or Gallery", app.MOBILE_UPLOAD_TIP)
        self.assertIn("Most Compatible", app.MOBILE_UPLOAD_TIP)


class HeicGuidanceTests(unittest.TestCase):
    def _upload(self, data, name, mime):
        f = io.BytesIO(data)
        f.name = name
        f.type = mime
        f.size = len(data)
        return f

    def test_heic_extension_points_to_jpeg_setting(self):
        valid, message = is_valid_image(
            self._upload(b"0" * 20, "IMG_1234.HEIC", "image/heic")
        )
        self.assertFalse(valid)
        self.assertEqual(message, HEIC_HELP_MESSAGE)
        self.assertIn("Most Compatible", message)

    def test_heif_mime_rejected_with_guidance(self):
        valid, message = is_valid_image(
            self._upload(b"0" * 20, "photo.jpg", "image/heif")
        )
        self.assertFalse(valid)
        self.assertEqual(message, HEIC_HELP_MESSAGE)


class MobileCssTests(unittest.TestCase):
    def test_touch_and_caption_styles_present(self):
        self.assertIn('div[data-testid="stCaption"]', app.CUSTOM_CSS)
        self.assertIn("@media (max-width: 640px)", app.CUSTOM_CSS)
        self.assertIn("font-size: 16px", app.CUSTOM_CSS)
        self.assertIn("min-height: 48px", app.CUSTOM_CSS)
        self.assertIn("overflow-x: auto", app.CUSTOM_CSS)
        self.assertNotIn("stCameraInput", app.CUSTOM_CSS)


if __name__ == "__main__":
    unittest.main()
