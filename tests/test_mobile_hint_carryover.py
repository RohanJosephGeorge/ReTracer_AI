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
    with (
        patch.object(app, "st", mock_st),
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


class HintKeyTests(unittest.TestCase):
    def test_hint_widget_uses_specified_persistence_key(self):
        mock_st = run_home(home_state())
        self.assertEqual(
            mock_st.text_input.call_args.kwargs["key"], "regional_hint_input"
        )

    def test_clear_hint_state_removes_hint_and_snapshot_only(self):
        state = StateDict(
            {
                "regional_hint_input": "Kerala backwaters",
                "active_hint": "Kerala backwaters",
                "last_upload_key": "a:1",
            }
        )
        mock_st = MagicMock()
        mock_st.session_state = state
        with patch.object(app, "st", mock_st):
            app.clear_hint_state()
        self.assertEqual(state, {"last_upload_key": "a:1"})

    def test_reset_clears_hint_text_and_bumps_upload_epoch(self):
        state = home_state(upload_epoch=3)
        state["regional_hint_input"] = "Kerala"
        state["active_hint"] = "Kerala"
        mock_st = MagicMock()
        mock_st.session_state = state
        with patch.object(app, "st", mock_st):
            app.reset_home_for_new_upload()
        self.assertEqual(state["upload_epoch"], 4)
        self.assertNotIn("regional_hint_input", state)
        self.assertNotIn("active_hint", state)


class StartAnalysisSnapshotTests(unittest.TestCase):
    def run_callback(self, state):
        mock_st = MagicMock()
        mock_st.session_state = state
        with patch.object(app, "st", mock_st):
            app.start_analysis()
        return state

    def test_click_snapshots_stripped_hint_and_arms(self):
        state = home_state(
            regional_hint_input="  Turkey  ",
            analysis_results={"kind": "ok"},
        )
        self.run_callback(state)
        self.assertEqual(state["active_hint"], "Turkey")
        self.assertTrue(state["is_analyzing"])
        self.assertIsNone(state["analysis_results"])

    def test_click_without_hint_snapshots_empty(self):
        state = self.run_callback(home_state())
        self.assertEqual(state["active_hint"], "")


class HintCarryoverTests(unittest.TestCase):
    def test_new_upload_clears_stale_hint_and_results(self):
        state = home_state(
            last_upload_key="old.jpg:1",
            analysis_results={"kind": "ok", "mock": False, "result": {}},
            regional_hint_input="Kerala backwaters",
        )
        run_home(state, uploaded_file=fake_upload("new.jpg", 2))
        self.assertIsNone(state["analysis_results"])
        self.assertEqual(state["last_upload_key"], "new.jpg:2")
        self.assertNotIn("regional_hint_input", state)

    def test_same_upload_keeps_hint_and_results(self):
        outcome = {"kind": "ok", "mock": False, "result": {}}
        state = home_state(
            last_upload_key="photo.jpg:1234",
            analysis_results=outcome,
            regional_hint_input="Kerala backwaters",
        )
        run_home(state, uploaded_file=fake_upload("photo.jpg", 1234))
        self.assertIs(state["analysis_results"], outcome)
        self.assertEqual(
            state["regional_hint_input"], "Kerala backwaters"
        )

    def run_armed_live(
        self, state_hint, widget_return="", active_hint_override=None
    ):
        """Arm via the real click callback, then run live analysis."""
        state = home_state(
            developer_mode=False,
            last_upload_key="photo.jpg:1234",
            regional_hint_input=state_hint,
        )
        upload = fake_upload()
        mock_st = MagicMock()
        mock_st.session_state = state
        with patch.object(app, "st", mock_st):
            app.start_analysis()
        if active_hint_override is not None:
            state["active_hint"] = active_hint_override
        self.assertTrue(state["is_analyzing"])
        mock_st.file_uploader.return_value = upload
        mock_st.text_input.return_value = widget_return
        mock_st.columns.return_value = (MagicMock(), MagicMock(), MagicMock())
        outcome = {"kind": "ok", "mock": False, "result": {}}
        mock_direct = MagicMock(return_value=outcome)
        with (
            patch.object(app, "st", mock_st),
            patch.object(app, "execute_analysis", MagicMock()),
            patch.object(app, "execute_live_analysis_direct", mock_direct),
            patch.object(app, "load_lottie_animation", return_value=None),
            patch("time.sleep"),
        ):
            app.render_home(developer_mode=False)
        self.assertIs(state["analysis_results"], outcome)
        return upload, mock_direct, mock_st

    def test_armed_live_run_submits_state_hint(self):
        upload, mock_direct, _ = self.run_armed_live(
            state_hint="new hint", widget_return="stale widget return"
        )
        self.assertEqual(mock_direct.call_args.args, (upload, "new hint"))

    def test_armed_live_run_strips_hint_padding(self):
        upload, mock_direct, _ = self.run_armed_live(
            state_hint="  Kerala backwaters  "
        )
        self.assertEqual(
            mock_direct.call_args.args, (upload, "Kerala backwaters")
        )

    def test_armed_live_run_normalizes_blank_hint(self):
        upload, mock_direct, _ = self.run_armed_live(state_hint="   ")
        self.assertEqual(mock_direct.call_args.args, (upload, ""))

    def test_armed_run_locks_hint_input(self):
        # The hint cannot be edited while analysis runs; a pre-widget
        # pin (not the disabled flag alone) keeps the textbox populated.
        _, _, mock_st = self.run_armed_live(state_hint="Turkey")
        self.assertTrue(mock_st.text_input.call_args.kwargs["disabled"])

    def test_armed_run_heals_stale_snapshot_from_widget_key(self):
        # Simulates a click callback that raced ahead of the submitted
        # form value: the pre-widget re-freeze corrects the snapshot.
        upload, mock_direct, _ = self.run_armed_live(
            state_hint="Turkey", active_hint_override="stale"
        )
        self.assertEqual(mock_direct.call_args.args, (upload, "Turkey"))


class HintFormSubmitTests(unittest.TestCase):
    def test_hint_and_analyze_submit_together_in_one_form(self):
        state = home_state()
        mock_st = run_home(state)
        mock_st.form.assert_called_once_with("hint_form", border=False)
        calls = [str(call) for call in mock_st.mock_calls]

        def first_index(predicate):
            return next(
                i for i, text in enumerate(calls) if predicate(text)
            )

        enter = first_index(
            lambda s: s.startswith("call.form(")
            and s.endswith(".__enter__()")
        )
        hint = first_index(lambda s: s.startswith("call.text_input("))
        submit = first_index(
            lambda s: s.startswith("call.form_submit_button(")
        )
        exit_ = first_index(
            lambda s: s.startswith("call.form(") and ".__exit__(" in s
        )
        self.assertLess(enter, hint)
        self.assertLess(hint, submit)
        self.assertLess(submit, exit_)

    def test_submit_button_wired_to_start_analysis(self):
        state = home_state()
        mock_st = run_home(state)
        mock_st.form_submit_button.assert_called_once_with(
            "Analyze Memory",
            key="analyze_btn",
            type="primary",
            width="stretch",
            disabled=False,
            on_click=app.start_analysis,
        )

    def test_hint_needs_no_staging_crutches_inside_form(self):
        state = home_state()
        mock_st = run_home(state)
        kwargs = mock_st.text_input.call_args.kwargs
        self.assertNotIn("live", kwargs)
        self.assertNotEqual(kwargs.get("on_change"), "ignore")

    def test_hint_widget_bound_to_snapshot_value(self):
        state = home_state(active_hint="Turkey")
        mock_st = run_home(state)
        self.assertEqual(
            mock_st.text_input.call_args.kwargs["value"], "Turkey"
        )


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
