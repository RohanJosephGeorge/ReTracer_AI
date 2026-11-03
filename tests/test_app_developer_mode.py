"""Regression: main-page toggle session_state is source of truth; single Analyze button.

Run: python -m unittest discover -s tests -v
Requires streamlit and pandas. Streamlit is mocked: no browser.
"""

import unittest
from unittest.mock import MagicMock, patch

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


def base_state(**overrides):
    state = StateDict(
        {
            "developer_mode": True,
            "is_analyzing": False,
            "analysis_results": None,
            "last_upload_key": None,
            "screened_image": None,
            "sam_result": None,
            "wonder_result": None,
        }
    )
    state.update(overrides)
    return state


class DeveloperModeToggleTests(unittest.TestCase):
    def run_armed_analysis(self, state, param):
        mock_st = MagicMock()
        mock_st.session_state = state
        mock_st.file_uploader.return_value = None
        mock_st.text_input.return_value = ""
        mock_st.columns.return_value = (MagicMock(), MagicMock(), MagicMock())
        execute = MagicMock(
            return_value={"kind": "ok", "mock": True, "result": {}}
        )
        direct = MagicMock(
            return_value={"kind": "ok", "mock": False, "result": {}}
        )
        with (
            patch.object(app, "st", mock_st),
            patch.object(app, "execute_analysis", execute),
            patch.object(app, "execute_live_analysis_direct", direct),
            patch.object(app, "load_lottie_animation", return_value=None),
            patch("time.sleep"),
        ):
            app.render_home(developer_mode=param)
        return execute, direct

    def test_session_state_false_overrides_stale_true_param(self):
        state = base_state(developer_mode=False, is_analyzing=True)
        execute, direct = self.run_armed_analysis(state, param=True)
        execute.assert_not_called()
        direct.assert_called_once()

    def test_session_state_true_overrides_stale_false_param(self):
        state = base_state(developer_mode=True, is_analyzing=True)
        execute, direct = self.run_armed_analysis(state, param=False)
        execute.assert_called_once()
        passed_flag = execute.call_args.args[2]
        self.assertTrue(passed_flag)
        direct.assert_not_called()


class SingleButtonFlowTests(unittest.TestCase):
    def test_no_sam_section_or_extra_buttons(self):
        state = base_state(
            developer_mode=False,
            screened_image=MagicMock(),
            sam_result={
                "summary": [{"concept": "text"}],
                "mask_count": 1,
                "annotated": MagicMock(),
            },
        )
        mock_st = MagicMock()
        mock_st.session_state = state
        mock_st.file_uploader.return_value = None
        mock_st.text_input.return_value = ""
        mock_st.columns.return_value = (MagicMock(), MagicMock(), MagicMock())
        mock_st.button.return_value = False
        # Keep the upload-key guard from resetting our staged state.
        mock_st.session_state["last_upload_key"] = None
        with (
            patch.object(app, "st", mock_st),
        ):
            app.render_home()
        for call in mock_st.subheader.call_args_list:
            self.assertNotIn("SAM 3.1", call.args[0])
        # The only st.button on Home is the Refresh reset; no legacy
        # per-section buttons may reappear.
        self.assertEqual(len(mock_st.button.call_args_list), 1)
        self.assertIn(
            "Refresh", mock_st.button.call_args_list[0].args[0]
        )
        mock_st.form_submit_button.assert_called_once_with(
            "Analyze Memory",
            key="analyze_btn",
            type="primary",
            width="stretch",
            disabled=False,
            on_click=app.start_analysis,
        )


class MockDemoModeTests(unittest.TestCase):
    def run_home(self, dev_mode):
        state = base_state(developer_mode=dev_mode, dev_mode=dev_mode)
        mock_st = MagicMock()
        mock_st.session_state = state
        mock_st.file_uploader.return_value = None
        mock_st.text_input.return_value = ""
        mock_st.columns.return_value = (MagicMock(), MagicMock(), MagicMock())
        with (
            patch.object(app, "st", mock_st),
        ):
            app.render_home()
        return mock_st

    def test_dev_mode_locks_inputs_and_shows_demo_picture(self):
        mock_st = self.run_home(True)
        self.assertTrue(mock_st.file_uploader.call_args.kwargs["disabled"])
        self.assertTrue(mock_st.text_input.call_args.kwargs["disabled"])
        mock_st.image.assert_called_once_with(
            app.MOCK_DEMO_IMAGE_PATH,
            caption="Demo photo (mock data)",
            width="stretch",
        )

    def test_live_mode_unlocks_inputs_and_hides_demo_picture(self):
        mock_st = self.run_home(False)
        self.assertFalse(mock_st.file_uploader.call_args.kwargs["disabled"])
        self.assertFalse(mock_st.text_input.call_args.kwargs["disabled"])
        mock_st.image.assert_not_called()


if __name__ == "__main__":
    unittest.main()
