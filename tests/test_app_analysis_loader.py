"""Tests for the timer-driven analysis loader (bar, text, countdown).

Run: python -m unittest discover -s tests -v
Requires streamlit and pandas. Streamlit is mocked: no browser.
"""

import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import app


class LoaderCssTests(unittest.TestCase):
    def test_progress_bar_gradient_styles(self):
        self.assertIn(
            ".stProgress > div > div > div > div {", app.CUSTOM_CSS
        )
        self.assertIn(
            "linear-gradient(90deg, #ff5bd5 0%, #a25df5 50%, #5d8af5 100%)",
            app.CUSTOM_CSS,
        )
        self.assertIn("border-radius: 50px", app.CUSTOM_CSS)

    def test_loading_text_style(self):
        self.assertIn(".loading-text {", app.CUSTOM_CSS)
        self.assertIn("letter-spacing: 1px", app.CUSTOM_CSS)

    def test_eta_text_style(self):
        self.assertIn(".eta-text {", app.CUSTOM_CSS)
        self.assertIn("color: rgba(255, 255, 255, 0.55)", app.CUSTOM_CSS)
        self.assertIn("font-size: 0.8rem", app.CUSTOM_CSS)


class ProgressScriptTests(unittest.TestCase):
    def test_detective_script_has_23_unique_messages(self):
        self.assertEqual(len(app.SAM_WONDER_PROGRESS_MESSAGES), 23)
        self.assertEqual(len(set(app.SAM_WONDER_PROGRESS_MESSAGES)), 23)

    def test_universal_action_phrasing_present(self):
        for line in (
            "Scanning the scene for infrastructural clues...",
            "Evaluating biome, vegetation, and topography...",
            "Searching for hidden regional artifacts and text...",
            "Cross-referencing global architectural patterns...",
        ):
            self.assertIn(line, app.SAM_WONDER_PROGRESS_MESSAGES)

    def test_messages_assume_no_specific_features(self):
        banned = (
            "utility pole",
            "power line",
            "license plate",
            "street sign",
            "roofline",
            "soil composition",
            "traffic marking",
            "shadow vector",
            "solar angle",
        )
        for message in app.SAM_WONDER_PROGRESS_MESSAGES:
            lowered = message.lower()
            for phrase in banned:
                self.assertNotIn(phrase, lowered)

    def test_message_pace_is_five_to_seven_seconds(self):
        dwell = app.API_PROGRESS_ETA_SECONDS / len(
            app.SAM_WONDER_PROGRESS_MESSAGES
        )
        self.assertGreaterEqual(dwell, 5)
        self.assertLessEqual(dwell, 7)


def shown_texts(status_slot):
    return [
        call[0][0]
        .replace('<div class="loading-text">', "")
        .replace("</div>", "")
        for call in status_slot.markdown.call_args_list
    ]


def shown_message_runs(status_slot):
    """Message parts with consecutive duplicates collapsed."""
    messages = [text.split(" - ", 1)[1] for text in shown_texts(status_slot)]
    return [
        message
        for i, message in enumerate(messages)
        if i == 0 or message != messages[i - 1]
    ]


def shown_percents(status_slot):
    return [
        int(text.split("%", 1)[0]) for text in shown_texts(status_slot)
    ]


def shown_eta_texts(eta_slot):
    return [
        call[0][0]
        .replace('<div class="eta-text">', "")
        .replace("</div>", "")
        for call in eta_slot.markdown.call_args_list
    ]


class SimulateProgressTests(unittest.TestCase):
    def run_animator(
        self, messages, duration_seconds, until, timeout=8.0
    ):
        stop_event = threading.Event()
        status_slot = MagicMock()
        progress_slot = MagicMock()
        thread = threading.Thread(
            target=app.simulate_smooth_progress,
            args=(stop_event, status_slot, progress_slot, messages),
            kwargs={
                "tick_seconds": 0.005,
                "duration_seconds": duration_seconds,
            },
            daemon=True,
        )
        thread.start()
        deadline = time.time() + timeout
        while (
            not until(status_slot, progress_slot)
            and time.time() < deadline
        ):
            time.sleep(0.02)
        stop_event.set()
        thread.join(timeout=3.0)
        self.assertFalse(thread.is_alive())
        return status_slot, progress_slot

    def test_fills_zero_to_95_over_duration(self):
        _, progress_slot = self.run_animator(
            ("M1", "M2", "M3", "M4"),
            2.0,
            lambda status, progress: (
                progress.progress.call_args_list
                and progress.progress.call_args_list[-1][0][0] >= 95
            ),
        )
        progresses = [
            call[0][0] for call in progress_slot.progress.call_args_list
        ]
        self.assertEqual(progresses[0], 0)
        self.assertTrue(
            all(a <= b for a, b in zip(progresses, progresses[1:]))
        )
        self.assertEqual(max(progresses), 95)
        self.assertEqual(progresses[-1], 95)

    def test_messages_advance_proportionally_without_repeats(self):
        status_slot, _ = self.run_animator(
            ("M1", "M2", "M3", "M4"),
            1.0,
            lambda status, progress: "Wrapping up"
            in "".join(shown_texts(status)),
        )
        runs = shown_message_runs(status_slot)
        self.assertEqual(runs[:4], ["M1", "M2", "M3", "M4"])
        self.assertEqual(runs[4], app.API_PROGRESS_WRAP_UP_TEXT)

    def test_holds_95_with_wrap_up_past_deadline(self):
        stop_event = threading.Event()
        status_slot = MagicMock()
        progress_slot = MagicMock()
        thread = threading.Thread(
            target=app.simulate_smooth_progress,
            args=(stop_event, status_slot, progress_slot, ("M1", "M2")),
            kwargs={"tick_seconds": 0.005, "duration_seconds": 0.4},
            daemon=True,
        )
        thread.start()
        deadline = time.time() + 5.0
        while (
            "Wrapping up" not in "".join(shown_texts(status_slot))
            and time.time() < deadline
        ):
            time.sleep(0.02)
        time.sleep(0.3)
        stop_event.set()
        thread.join(timeout=3.0)
        self.assertFalse(thread.is_alive())
        progresses = [
            call[0][0] for call in progress_slot.progress.call_args_list
        ]
        self.assertEqual(max(progresses), 95)
        self.assertTrue(all(p == 95 for p in progresses[-5:]))
        runs = shown_message_runs(status_slot)
        first_hold = runs.index(app.API_PROGRESS_WRAP_UP_TEXT)
        self.assertTrue(
            all(
                text == app.API_PROGRESS_WRAP_UP_TEXT
                for text in runs[first_hold:]
            )
        )

    def test_displayed_percent_matches_bar_tick_for_tick(self):
        status_slot, progress_slot = self.run_animator(
            ("M1", "M2"),
            0.5,
            lambda status, progress: len(progress.progress.call_args_list)
            >= 20,
        )
        progresses = [
            call[0][0] for call in progress_slot.progress.call_args_list
        ]
        self.assertEqual(shown_percents(status_slot), progresses)

    def test_empty_messages_return_immediately(self):
        stop_event = threading.Event()
        status_slot = MagicMock()
        progress_slot = MagicMock()
        app.simulate_smooth_progress(
            stop_event,
            status_slot,
            progress_slot,
            (),
            tick_seconds=0.001,
            duration_seconds=0.001,
        )
        status_slot.markdown.assert_not_called()
        progress_slot.progress.assert_not_called()


class CountdownTimerTests(unittest.TestCase):
    def run_animator(self, duration_seconds, until, timeout=8.0):
        stop_event = threading.Event()
        status_slot = MagicMock()
        progress_slot = MagicMock()
        eta_slot = MagicMock()
        thread = threading.Thread(
            target=app.simulate_smooth_progress,
            args=(stop_event, status_slot, progress_slot, ("M",)),
            kwargs={
                "tick_seconds": 0.005,
                "eta_slot": eta_slot,
                "duration_seconds": duration_seconds,
            },
            daemon=True,
        )
        thread.start()
        deadline = time.time() + timeout
        while not until(eta_slot) and time.time() < deadline:
            time.sleep(0.02)
        stop_event.set()
        thread.join(timeout=3.0)
        self.assertFalse(thread.is_alive())
        return eta_slot

    def test_countdown_decreases_then_wraps_up(self):
        eta_slot = self.run_animator(
            2.0,
            lambda slot: "Wrapping up" in "".join(shown_eta_texts(slot)),
        )
        texts = shown_eta_texts(eta_slot)
        self.assertEqual(texts[-1], "Wrapping up final deductions...")
        countdowns = [
            text for text in texts if "Approximated time left" in text
        ]
        self.assertGreaterEqual(len(countdowns), 1)
        seconds = [
            int(text.rsplit(" ", 1)[1].rstrip("s")) for text in countdowns
        ]
        self.assertTrue(all(s > 0 for s in seconds))
        self.assertTrue(
            all(a > b for a, b in zip(seconds, seconds[1:]))
        )
        for text in texts:
            self.assertNotIn("-", text)

    def test_wrap_up_replaces_countdown_without_negatives(self):
        eta_slot = self.run_animator(
            0.3,
            lambda slot: "Wrapping up" in "".join(shown_eta_texts(slot)),
            timeout=5.0,
        )
        texts = shown_eta_texts(eta_slot)
        self.assertEqual(texts[-1], "Wrapping up final deductions...")
        for text in texts:
            self.assertNotIn("-", text)


class LoaderLifecycleTests(unittest.TestCase):
    def test_start_stop_round_trip_with_mocked_streamlit(self):
        mock_st = MagicMock()
        status_slot = MagicMock()
        progress_slot = MagicMock()
        eta_slot = MagicMock()
        mock_st.empty.side_effect = [status_slot, progress_slot, eta_slot]
        with patch.object(app, "st", mock_st):
            loader = app.start_analysis_loader("Analyzing memory...")
            _, animator, _, _, _ = loader
            self.assertTrue(animator.is_alive())
            app.stop_analysis_loader(loader)
        self.assertFalse(animator.is_alive())
        self.assertEqual(mock_st.empty.call_count, 3)
        mock_st.progress.assert_not_called()
        first_html = status_slot.markdown.call_args_list[0][0][0]
        self.assertIn("0% - Analyzing memory...", first_html)
        first_progress = progress_slot.progress.call_args_list[0][0][0]
        self.assertEqual(first_progress, 0)
        first_eta = eta_slot.markdown.call_args_list[0][0][0]
        self.assertIn("Approximated time left: 120s", first_eta)
        progress_slot.progress.assert_called_with(100)
        status_slot.empty.assert_called_once_with()
        progress_slot.empty.assert_called_once_with()
        eta_slot.empty.assert_called_once_with()


class StateDict(dict):
    """Dict with attribute access, mirroring Streamlit session_state."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class AnalysisPhaseLoaderTests(unittest.TestCase):
    def test_analysis_phase_uses_progress_loader_not_spinner(self):
        outcome = {"kind": "ok", "mock": True, "result": {"coordinates": {}}}
        state = StateDict(
            {
                "developer_mode": True,
                "is_processing": True,
                "analysis_pending": True,
                "analysis_outcome": None,
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
        mock_st.columns.return_value = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )
        mock_execute = MagicMock(return_value=outcome)
        show_centered_lottie_mock = MagicMock()
        with (
            patch.object(app, "st", mock_st),
            patch.object(app, "ui", MagicMock()),
            patch.object(app, "execute_analysis", mock_execute),
            patch.object(app, "load_lottie_animation", return_value=None),
            patch.object(
                app, "show_centered_lottie", show_centered_lottie_mock
            ),
        ):
            app.render_home(developer_mode=True)
        # No animation: only the three loader placeholders exist, no
        # empty animation slot above the progress text.
        self.assertEqual(mock_st.empty.call_count, 3)
        show_centered_lottie_mock.assert_not_called()
        mock_st.progress.assert_not_called()
        mock_st.spinner.assert_not_called()
        mock_execute.assert_called_once_with(None, "", True)
        self.assertIs(state["analysis_outcome"], outcome)

    def test_animation_slot_created_only_when_animation_renders(self):
        outcome = {"kind": "ok", "mock": True, "result": {"coordinates": {}}}
        state = StateDict(
            {
                "developer_mode": True,
                "is_processing": True,
                "analysis_pending": True,
                "analysis_outcome": None,
                "screened_image": None,
                "sam_result": None,
                "wonder_result": None,
            }
        )
        mock_st = MagicMock()
        mock_st.session_state = state
        mock_st.file_uploader.return_value = None
        mock_st.text_input.return_value = ""
        mock_st.columns.return_value = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )
        animation = {"v": "5.5.9"}
        mock_show = MagicMock()
        with (
            patch.object(app, "st", mock_st),
            patch.object(app, "ui", MagicMock()),
            patch.object(
                app,
                "execute_analysis",
                MagicMock(return_value=outcome),
            ),
            patch.object(app, "st_lottie", MagicMock()),
            patch.object(
                app, "load_lottie_animation", return_value=animation
            ),
            patch.object(app, "show_centered_lottie", mock_show),
        ):
            app.render_home(developer_mode=True)
        self.assertEqual(mock_st.empty.call_count, 4)
        mock_show.assert_called_once_with(
            mock_st.empty.return_value, animation
        )
        self.assertIs(state["analysis_outcome"], outcome)


if __name__ == "__main__":
    unittest.main()
