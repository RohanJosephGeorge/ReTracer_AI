"""Tests for the event-driven analysis progress (milestones, bar, text).

Run: python -m unittest discover -s tests -v
Requires streamlit and pandas. Streamlit is mocked: no browser.
"""

import unittest
from unittest.mock import MagicMock, call, patch

import app


class LoaderCssTests(unittest.TestCase):
    def test_progress_bar_gradient_styles(self):
        self.assertIn(
            'div[data-testid="stProgressBar"] > div {', app.CUSTOM_CSS
        )
        self.assertIn(
            'div[data-testid="stProgressBar"] > div > div {',
            app.CUSTOM_CSS,
        )
        self.assertIn(
            "linear-gradient(90deg, #ff5bd5 0%, #a25df5 50%, #5d8af5 100%)",
            app.CUSTOM_CSS,
        )
        self.assertIn("border-radius: 50px", app.CUSTOM_CSS)

    def test_loading_text_style(self):
        self.assertIn(".loading-text {", app.CUSTOM_CSS)
        self.assertIn("letter-spacing: 1px", app.CUSTOM_CSS)

    def test_scan_fallback_style(self):
        self.assertIn(".scan-radar", app.CUSTOM_CSS)
        self.assertIn("@keyframes scan-spin", app.CUSTOM_CSS)
        self.assertIn("conic-gradient", app.CUSTOM_CSS)
        self.assertIn("prefers-reduced-motion", app.CUSTOM_CSS)
        self.assertIn("#000", app.CUSTOM_CSS)
        self.assertIn("repeating-radial-gradient", app.CUSTOM_CSS)


class MilestoneConstantTests(unittest.TestCase):
    def test_milestone_values_match_spec(self):
        self.assertEqual(
            app.MILESTONE_INIT, (0, "Initializing spatial analysis...")
        )
        self.assertEqual(
            app.MILESTONE_SAM,
            (5, "Segmenting image and extracting infrastructural clues..."),
        )
        self.assertEqual(
            app.MILESTONE_WONDER,
            (
                45,
                "Cross-referencing biomes, architecture, and "
                "topography...",
            ),
        )
        self.assertEqual(app.MILESTONE_DONE, (100, "Analysis complete."))
        self.assertEqual(app.COMPLETION_HOLD_SECONDS, 0.5)
        self.assertEqual(app.SAM_PROGRESS_TICK_SECONDS, 2.0)
        self.assertEqual(app.SAM_PROGRESS_TICK_MAX_STEP, 3)
        self.assertEqual(app.SAM_PROGRESS_CAP, 44)
        self.assertEqual(app.WONDER_PROGRESS_TICK_SECONDS, 2.0)
        self.assertEqual(app.WONDER_PROGRESS_TICK_MAX_STEP, 3)
        self.assertEqual(app.WONDER_PROGRESS_CAP, 95)
        self.assertEqual(
            app.MILESTONE_WONDER_DONE,
            (
                95,
                "Cross-referencing biomes, architecture, and "
                "topography...",
            ),
        )
        self.assertEqual(len(app.SAM_TICK_MESSAGES), 9)
        self.assertEqual(
            app.SAM_TICK_MESSAGES[0], "SAM 3.1 is dissecting the image..."
        )
        self.assertEqual(
            app.SAM_TICK_MESSAGES[-1],
            "Cataloguing every visible clue and pattern...",
        )
        self.assertEqual(len(app.WONDER_TICK_MESSAGES), 13)
        self.assertEqual(
            app.WONDER_TICK_MESSAGES[0],
            "ReTracer AI is now on the case...",
        )
        self.assertEqual(
            app.WONDER_TICK_MESSAGES[-1],
            "Preparing the final spatial verdict...",
        )
        for message in (
            app.SAM_TICK_MESSAGES + app.WONDER_TICK_MESSAGES
        ):
            self.assertNotIn("geoguessr", message.lower())

    def test_timer_loader_is_gone(self):
        for name in (
            "simulate_smooth_progress",
            "start_analysis_loader",
            "stop_analysis_loader",
            "SAM_WONDER_PROGRESS_MESSAGES",
            "API_PROGRESS_TICK_SECONDS",
            "API_PROGRESS_CAP",
            "API_PROGRESS_ETA_SECONDS",
            "API_PROGRESS_WRAP_UP_TEXT",
        ):
            self.assertFalse(hasattr(app, name), name)
        self.assertNotIn("eta-text", app.CUSTOM_CSS)
        self.assertNotIn("Approximated time left", app.CUSTOM_CSS)


class ReportMilestoneTests(unittest.TestCase):
    def test_updates_text_and_bar_together(self):
        status_slot = MagicMock()
        progress_bar = MagicMock()
        app.report_analysis_milestone(
            status_slot,
            progress_bar,
            35,
            "Segmenting image and extracting infrastructural clues...",
        )
        html = status_slot.markdown.call_args_list[0][0][0]
        self.assertIn(
            "35% - Segmenting image and extracting infrastructural "
            "clues...",
            html,
        )
        self.assertIn('class="loading-text"', html)
        progress_bar.progress.assert_called_once_with(35)


class StubFuture:
    """concurrent.futures.Future stand-in with scripted done() flips."""

    def __init__(self, done_flags, result=None, error=None):
        self._done_flags = list(done_flags)
        self._result = result
        self._error = error

    def done(self):
        if len(self._done_flags) > 1:
            return self._done_flags.pop(0)
        return self._done_flags[0]

    def result(self):
        if self._error is not None:
            raise self._error
        return self._result


def make_executor_factory(futures, calls):
    """ThreadPoolExecutor stand-in dispensing one canned future per pool."""

    class FakeExecutor:
        def __init__(self, *args, **kwargs):
            calls.setdefault("inits", []).append((args, kwargs))
            self._future = futures[len(calls["inits"]) - 1]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            calls.setdefault("closed", []).append(True)
            return False

        def submit(self, fn, *args, **kwargs):
            calls.setdefault("submits", []).append((fn, args, kwargs))
            return self._future

    return FakeExecutor


class AwaitSamFutureTests(unittest.TestCase):
    def test_completed_future_paints_nothing(self):
        progress = MagicMock()
        with patch("time.sleep") as mock_sleep:
            result = app._await_sam_future(
                StubFuture([True], result=["mask"]), progress
            )
        self.assertEqual(result, ["mask"])
        mock_sleep.assert_not_called()
        progress.assert_not_called()

    def test_slow_future_ticks_random_amounts_on_interval(self):
        events = []

        def on_progress(percent, message):
            events.append((percent, message))

        with (
            patch("time.sleep") as mock_sleep,
            patch.object(
                app.random, "randint", side_effect=[2, 1]
            ) as mock_rand,
        ):
            result = app._await_sam_future(
                StubFuture(
                    [False, False, False, False, True], result=["mask"]
                ),
                on_progress,
            )
        self.assertEqual(result, ["mask"])
        self.assertEqual(mock_sleep.call_args_list, [call(2.0)] * 2)
        self.assertEqual(
            mock_rand.call_args_list, [call(1, 3), call(1, 3)]
        )
        self.assertEqual(
            events,
            [
                (7, "SAM 3.1 is dissecting the image..."),
                (8, "SAM 3.1 is acting like a detective..."),
            ],
        )

    def test_ticks_hold_below_wonder_milestone(self):
        events = []

        def on_progress(percent, message):
            events.append((percent, message))

        with (
            patch("time.sleep"),
            patch.object(app.random, "randint", return_value=1),
        ):
            app._await_sam_future(
                StubFuture([False] * 80 + [True], result=["mask"]),
                on_progress,
            )
        expected_messages = [
            app.SAM_TICK_MESSAGES[i % len(app.SAM_TICK_MESSAGES)]
            for i in range(39)
        ]
        self.assertEqual(
            events, list(zip(range(6, 45), expected_messages))
        )

    def test_large_steps_clamp_to_cap(self):
        events = []

        def on_progress(percent, message):
            events.append((percent, message))

        with (
            patch("time.sleep"),
            patch.object(app.random, "randint", return_value=2),
        ):
            app._await_sam_future(
                StubFuture([False] * 45 + [True], result=["mask"]),
                on_progress,
            )
        percents = list(range(7, 44, 2)) + [44]
        expected_messages = [
            app.SAM_TICK_MESSAGES[i % len(app.SAM_TICK_MESSAGES)]
            for i in range(len(percents))
        ]
        self.assertEqual(events, list(zip(percents, expected_messages)))

    def test_worker_error_propagates_without_paints(self):
        from sam_segmentation import SamApiError

        progress = MagicMock()
        with patch("time.sleep"):
            with self.assertRaises(SamApiError):
                app._await_sam_future(
                    StubFuture([True], error=SamApiError("down")),
                    progress,
                )
        progress.assert_not_called()

    def test_none_observer_waits_without_polling(self):
        with patch("time.sleep") as mock_sleep:
            result = app._await_sam_future(
                StubFuture([False, False, True], result=["mask"]), None
            )
        self.assertEqual(result, ["mask"])
        mock_sleep.assert_not_called()


class AwaitWonderFutureTests(unittest.TestCase):
    def test_completed_future_paints_nothing(self):
        progress = MagicMock()
        with patch("time.sleep") as mock_sleep:
            result = app._await_wonder_future(
                StubFuture([True], result={"ok": True}), progress
            )
        self.assertEqual(result, {"ok": True})
        mock_sleep.assert_not_called()
        progress.assert_not_called()

    def test_slow_future_ticks_random_amounts_on_interval(self):
        events = []

        def on_progress(percent, message):
            events.append((percent, message))

        with (
            patch("time.sleep") as mock_sleep,
            patch.object(
                app.random, "randint", side_effect=[2, 1]
            ) as mock_rand,
        ):
            result = app._await_wonder_future(
                StubFuture(
                    [False, False, False, False, True],
                    result={"ok": True},
                ),
                on_progress,
            )
        self.assertEqual(result, {"ok": True})
        self.assertEqual(mock_sleep.call_args_list, [call(2.0)] * 2)
        self.assertEqual(
            mock_rand.call_args_list, [call(1, 3), call(1, 3)]
        )
        self.assertEqual(
            events,
            [
                (47, "ReTracer AI is now on the case..."),
                (48, "Cross-referencing global architectural patterns..."),
            ],
        )

    def test_tick_messages_wrap_around_pool(self):
        events = []

        def on_progress(percent, message):
            events.append((percent, message))

        with (
            patch("time.sleep"),
            patch.object(app.random, "randint", return_value=1),
        ):
            app._await_wonder_future(
                StubFuture([False] * 32 + [True], result={"ok": True}),
                on_progress,
            )
        expected_messages = list(app.WONDER_TICK_MESSAGES[:13]) + list(
            app.WONDER_TICK_MESSAGES[:3]
        )
        self.assertEqual(
            events, list(zip(range(46, 62), expected_messages))
        )

    def test_ticks_hold_at_wonder_cap(self):
        events = []

        def on_progress(percent, message):
            events.append(percent)

        with (
            patch("time.sleep"),
            patch.object(app.random, "randint", return_value=1),
        ):
            app._await_wonder_future(
                StubFuture([False] * 120 + [True], result={"ok": True}),
                on_progress,
            )
        self.assertEqual(events, list(range(46, 96)))


class LivePipelineSyncTests(unittest.TestCase):
    def run_pipeline(self, sam_done_flags, wonder_done_flags):
        events = []

        def on_progress(percent, message):
            events.append((percent, message))

        calls = {}
        retry = object()
        safe_image = MagicMock()
        wonder_result = {"latitude": 1.0, "longitude": 2.0}
        with (
            patch.object(
                app,
                "ThreadPoolExecutor",
                make_executor_factory(
                    [
                        StubFuture(sam_done_flags, result=["mask"]),
                        StubFuture(
                            wonder_done_flags, result=wonder_result
                        ),
                    ],
                    calls,
                ),
            ),
            patch.object(
                app, "get_meta_api_key", return_value="key-123"
            ) as mock_key,
            patch.object(app, "summarize_masks", return_value="summary"),
            patch.object(app.random, "randint", return_value=1),
            patch("time.sleep"),
        ):
            result = app._run_live_pipeline_sync(
                safe_image, "riverside", retry, on_progress
            )
        return events, result, calls, mock_key, safe_image, retry

    def test_fast_futures_emit_plain_milestones(self):
        events, result, calls, mock_key, safe_image, retry = (
            self.run_pipeline([True], [True])
        )
        self.assertEqual(
            result, {"latitude": 1.0, "longitude": 2.0}
        )
        mock_key.assert_called_once_with()
        sam_fn, sam_args, _kwargs = calls["submits"][0]
        self.assertIs(sam_fn, app._segment_image_sync)
        self.assertEqual(sam_args, (safe_image, "key-123", retry))
        wonder_fn, wonder_args, _kwargs = calls["submits"][1]
        self.assertIs(wonder_fn, app._reason_wonder_sync)
        self.assertEqual(
            wonder_args, ("summary", "riverside", retry, safe_image, "key-123")
        )
        self.assertEqual(
            [init[1] for init in calls["inits"]],
            [
                {"max_workers": 1, "thread_name_prefix": "sam-segment"},
                {"max_workers": 1, "thread_name_prefix": "wonder-reason"},
            ],
        )
        self.assertEqual(calls["closed"], [True, True])
        self.assertEqual(
            events,
            [
                (
                    5,
                    "Segmenting image and extracting infrastructural "
                    "clues...",
                ),
                (
                    45,
                    "Cross-referencing biomes, architecture, and "
                    "topography...",
                ),
            ],
        )

    def test_slow_futures_interleave_ticks(self):
        events, _result, _calls, _key, _image, _retry = self.run_pipeline(
            [False, False, False, False, True],
            [False, False, False, False, True],
        )
        self.assertEqual(
            [percent for percent, _message in events],
            [5, 6, 7, 45, 46, 47],
        )


class StateDict(dict):
    """Dict with attribute access, mirroring Streamlit session_state."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def clicked_state(**overrides):
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


def shown_status_texts(status_slot):
    return [call[0][0] for call in status_slot.markdown.call_args_list]


def shown_bar_values(progress_bar):
    return [call[0][0] for call in progress_bar.progress.call_args_list]


class LockedProgressTests(unittest.TestCase):
    def test_start_analysis_locks_and_clears_results(self):
        state = clicked_state(
            analysis_results={"kind": "ok", "mock": True, "result": {}},
        )
        mock_st = MagicMock()
        mock_st.session_state = state
        with patch.object(app, "st", mock_st):
            app.start_analysis()
        self.assertTrue(state["is_analyzing"])
        self.assertIsNone(state["analysis_results"])
        mock_st.rerun.assert_not_called()

    def test_armed_run_executes_mock_milestones_0_to_100(self):
        outcome = {"kind": "ok", "mock": True, "result": {"coordinates": {}}}
        state = clicked_state(is_analyzing=True)
        mock_st = MagicMock()
        mock_st.session_state = state
        mock_st.file_uploader.return_value = None
        mock_st.text_input.return_value = ""
        mock_st.columns.return_value = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )
        mock_ui = MagicMock()
        mock_ui.button.return_value = False
        mock_execute = MagicMock(return_value=outcome)
        mock_show = MagicMock()
        with (
            patch.object(app, "st", mock_st),
            patch.object(app, "ui", mock_ui),
            patch.object(app, "execute_analysis", mock_execute),
            patch.object(app, "load_lottie_animation", return_value=None),
            patch.object(app, "show_centered_lottie", mock_show),
            patch("time.sleep") as mock_sleep,
        ):
            app.render_home(developer_mode=True)
        mock_ui.button.assert_called_once_with(
            "Analyze Memory",
            key="analyze_btn",
            size="lg",
            width="stretch",
            disabled=True,
            on_click=app.start_analysis,
        )
        mock_execute.assert_called_once_with(None, "", True)
        # No animation: progress slots plus the fallback sweep slot.
        self.assertEqual(mock_st.empty.call_count, 3)
        mock_show.assert_not_called()
        fallback_html = [c[0][0] for c in mock_st.markdown.call_args_list]
        self.assertIn(app.SCAN_FALLBACK_HTML, fallback_html)
        slot = mock_st.empty.return_value
        mock_st.progress.assert_not_called()
        self.assertEqual(
            shown_status_texts(slot),
            [
                '<div class="loading-text">0% - Initializing spatial '
                "analysis...</div>",
                '<div class="loading-text">5% - Segmenting image and '
                "extracting infrastructural clues...</div>",
                '<div class="loading-text">45% - Cross-referencing biomes, '
                "architecture, and topography...</div>",
                '<div class="loading-text">95% - Cross-referencing biomes, '
                "architecture, and topography...</div>",
                '<div class="loading-text">100% - Analysis '
                "complete.</div>",
            ],
        )
        self.assertEqual(shown_bar_values(slot), [0, 5, 45, 95, 100])
        mock_sleep.assert_called_once_with(0.5)
        # Only the animation slot is cleared; the completed bar persists.
        slot.empty.assert_called_once_with()
        mock_st.spinner.assert_not_called()
        mock_st.rerun.assert_called_once_with()
        self.assertFalse(state["is_analyzing"])
        self.assertIs(state["analysis_results"], outcome)

    def test_armed_run_routes_live_and_forwards_progress_callback(self):
        outcome = {"kind": "ok", "mock": False, "result": {}}
        state = clicked_state(developer_mode=False, is_analyzing=True)
        mock_st = MagicMock()
        mock_st.session_state = state
        mock_st.file_uploader.return_value = None
        mock_st.text_input.return_value = ""
        mock_st.columns.return_value = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )
        mock_ui = MagicMock()
        mock_ui.button.return_value = False
        mock_execute = MagicMock()
        mock_direct = MagicMock(return_value=outcome)
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
        # Mid-run milestones come from the real backend through this
        # callback; the mocked backend swallows them, so only the
        # start and completion milestones render here.
        slot = mock_st.empty.return_value
        self.assertEqual(
            shown_status_texts(slot),
            [
                '<div class="loading-text">0% - Initializing spatial '
                "analysis...</div>",
                '<div class="loading-text">100% - Analysis '
                "complete.</div>",
            ],
        )
        self.assertEqual(shown_bar_values(slot), [0, 100])
        report = mock_direct.call_args.kwargs["on_progress"]
        report(46, "ReTracer AI is now on the case...")
        self.assertIn(
            "46% - ReTracer AI is now on the case...",
            shown_status_texts(slot)[-1],
        )
        self.assertEqual(shown_bar_values(slot)[-1], 46)
        self.assertIs(state["analysis_results"], outcome)

    def test_armed_run_always_renders_black_radar_sweep(self):
        outcome = {"kind": "ok", "mock": True, "result": {"coordinates": {}}}
        state = clicked_state(is_analyzing=True)
        mock_st = MagicMock()
        mock_st.session_state = state
        mock_st.file_uploader.return_value = None
        mock_st.text_input.return_value = ""
        mock_st.columns.return_value = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )
        mock_ui = MagicMock()
        mock_ui.button.return_value = False
        animation = {"v": "5.5.9"}
        mock_show = MagicMock()
        with (
            patch.object(app, "st", mock_st),
            patch.object(app, "ui", mock_ui),
            patch.object(
                app,
                "execute_analysis",
                MagicMock(return_value=outcome),
            ),
            patch.object(app, "st_lottie", MagicMock()),
            patch.object(
                app, "load_lottie_animation", return_value=animation
            ) as mock_load,
            patch.object(app, "show_centered_lottie", mock_show),
            patch("time.sleep"),
        ):
            app.render_home(developer_mode=True)
        self.assertEqual(mock_st.empty.call_count, 3)
        # The analyzing slot never touches the remote Lottie: the black
        # radar sweep renders even when an animation would load.
        mock_load.assert_not_called()
        mock_show.assert_not_called()
        fallback_html = [c[0][0] for c in mock_st.markdown.call_args_list]
        self.assertIn(app.SCAN_FALLBACK_HTML, fallback_html)
        # Only the animation slot is cleared; the progress UI persists.
        mock_st.empty.return_value.empty.assert_called_once_with()
        self.assertIs(state["analysis_results"], outcome)

    def test_scan_fallback_renders_when_lottie_library_missing(self):
        outcome = {"kind": "ok", "mock": True, "result": {"coordinates": {}}}
        state = clicked_state(is_analyzing=True)
        mock_st = MagicMock()
        mock_st.session_state = state
        mock_st.file_uploader.return_value = None
        mock_st.text_input.return_value = ""
        mock_st.columns.return_value = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )
        mock_ui = MagicMock()
        mock_ui.button.return_value = False
        mock_show = MagicMock()
        with (
            patch.object(app, "st", mock_st),
            patch.object(app, "ui", mock_ui),
            patch.object(
                app, "execute_analysis", MagicMock(return_value=outcome)
            ),
            patch.object(app, "st_lottie", None),
            patch.object(app, "show_centered_lottie", mock_show),
            patch("time.sleep"),
        ):
            app.render_home(developer_mode=True)
        mock_show.assert_not_called()
        fallback_html = [c[0][0] for c in mock_st.markdown.call_args_list]
        self.assertIn(app.SCAN_FALLBACK_HTML, fallback_html)
        self.assertIs(state["analysis_results"], outcome)


class ResultsPersistenceTests(unittest.TestCase):
    def run_home(self, state, clicked, uploaded_file=None):
        mock_st = MagicMock()
        mock_st.session_state = state
        mock_st.file_uploader.return_value = uploaded_file
        mock_st.text_input.return_value = ""
        mock_st.columns.return_value = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )
        mock_ui = MagicMock()
        mock_ui.button.return_value = clicked
        mock_execute = MagicMock()
        with (
            patch.object(app, "st", mock_st),
            patch.object(app, "ui", mock_ui),
            patch.object(app, "execute_analysis", mock_execute),
            patch.object(app, "load_lottie_animation", return_value=None),
            patch("time.sleep"),
        ):
            app.render_home(developer_mode=True)
        return mock_st, mock_execute

    def test_fresh_run_paints_no_progress(self):
        state = clicked_state()
        mock_st, mock_execute = self.run_home(state, clicked=False)
        mock_execute.assert_not_called()
        mock_st.empty.assert_not_called()
        mock_st.progress.assert_not_called()

    def test_settled_run_renders_persisted_results_without_bar(self):
        outcome = {"kind": "ok", "mock": True, "result": {"coordinates": {}}}
        state = clicked_state(analysis_results=outcome)
        mock_st, mock_execute = self.run_home(state, clicked=False)
        mock_execute.assert_not_called()
        mock_st.empty.assert_not_called()
        mock_st.success.assert_called_once_with(
            "Analysis complete."
        )
        self.assertIs(state["analysis_results"], outcome)

    def test_new_upload_clears_persisted_results(self):
        # Live mode: uploads are only honored outside demo mode.
        state = clicked_state(
            developer_mode=False,
            dev_mode=False,
            last_upload_key="old:1",
            analysis_results={"kind": "ok", "mock": True, "result": {}},
        )
        new_file = MagicMock()
        new_file.name = "new.jpg"
        new_file.size = 2
        with patch.object(
            app, "is_valid_image", return_value=(False, "invalid upload")
        ):
            mock_st, _ = self.run_home(
                state, clicked=False, uploaded_file=new_file
            )
        self.assertIsNone(state["analysis_results"])
        self.assertEqual(state["last_upload_key"], "new.jpg:2")
        mock_st.empty.assert_not_called()

    def test_empty_uploader_keeps_persisted_results(self):
        # Returning to Home from another view remounts an empty
        # uploader; the finished result must survive the round trip.
        outcome = {"kind": "ok", "mock": True, "result": {}}
        state = clicked_state(
            last_upload_key="old:1", analysis_results=outcome
        )
        mock_st, _ = self.run_home(state, clicked=False)
        self.assertIs(state["analysis_results"], outcome)
        self.assertEqual(state["last_upload_key"], "old:1")
        mock_st.empty.assert_not_called()

    def test_refresh_button_wired_to_reset_callback(self):
        state = clicked_state()
        mock_st, _ = self.run_home(state, clicked=False)
        reset_calls = [
            call
            for call in mock_st.button.call_args_list
            if call.args and "Refresh" in call.args[0]
        ]
        self.assertEqual(len(reset_calls), 1)
        self.assertIs(
            reset_calls[0].kwargs["on_click"],
            app.reset_home_for_new_upload,
        )

    def test_reset_callback_clears_home_state(self):
        sentinel = object()
        state = clicked_state(
            last_upload_key="old:1",
            analysis_results={"kind": "ok"},
            screened_image=sentinel,
            sam_result=sentinel,
            wonder_result=sentinel,
            upload_epoch=0,
        )
        state["regional_hint_input"] = "old hint"
        mock_st = MagicMock()
        mock_st.session_state = state
        with patch.object(app, "st", mock_st):
            app.reset_home_for_new_upload()
        for key in (
            "analysis_results",
            "last_upload_key",
            "screened_image",
            "sam_result",
            "wonder_result",
        ):
            self.assertIsNone(state[key])
        self.assertEqual(state["upload_epoch"], 1)
        self.assertNotIn("regional_hint_input", state)


if __name__ == "__main__":
    unittest.main()
