"""Tests that the Analyze lock always unlocks the UI, even on errors.

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


def home_state(**overrides):
    state = StateDict(
        {
            "developer_mode": True,
            "is_analyzing": True,
            "analysis_results": None,
            "last_upload_key": None,
            "screened_image": None,
            "sam_result": None,
            "wonder_result": None,
        }
    )
    state.update(overrides)
    return state


def run_armed_home(state, execute):
    mock_st = MagicMock()
    mock_st.session_state = state
    mock_st.file_uploader.return_value = None
    mock_st.text_input.return_value = ""
    mock_st.columns.return_value = (MagicMock(), MagicMock(), MagicMock())
    with (
        patch.object(app, "st", mock_st),
        patch.object(app, "execute_analysis", execute),
        patch.object(app, "load_lottie_animation", return_value=None),
        patch("time.sleep"),
    ):
        app.render_home(developer_mode=True)
    return mock_st


class AnalysisUnlockTests(unittest.TestCase):
    def test_exception_unlocks_and_reports_error(self):
        def boom(*args, **kwargs):
            raise RuntimeError("mock backend exploded")

        state = home_state()
        mock_st = run_armed_home(state, boom)
        self.assertFalse(state["is_analyzing"])
        self.assertEqual(state["analysis_results"]["kind"], "error")
        self.assertIn(
            "mock backend exploded", state["analysis_results"]["message"]
        )
        mock_st.error.assert_called_once_with(
            state["analysis_results"]["message"]
        )

    def test_success_unlocks_and_stores_outcome(self):
        outcome = {"kind": "ok", "mock": True, "result": {"coordinates": {}}}
        state = home_state()
        run_armed_home(state, MagicMock(return_value=outcome))
        self.assertFalse(state["is_analyzing"])
        self.assertIs(state["analysis_results"], outcome)


if __name__ == "__main__":
    unittest.main()
