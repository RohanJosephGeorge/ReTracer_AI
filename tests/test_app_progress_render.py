"""Headless rendering regression: lock, progress bar, and persistence.

Run: python -m unittest tests.test_app_progress_render -v
Requires a working event loop, so it skips itself in sandboxes that
block socket creation (AppTest cannot start its runner there). The
mocked unit tests cover call wiring; only this test observes real
element rendering across the click -> settle runs.
"""

import asyncio
import unittest

from streamlit.testing.v1 import AppTest

import app

SCRIPT = '''
from unittest.mock import MagicMock

import app

# First run only: fire start_analysis() exactly as the button's
# on_click callback would before the script body. Later runs see the
# settled session, like real post-click reruns. The mocked button's
# return value is ignored by design.
if not getattr(app, "_progress_probe_armed", False):
    app._progress_probe_armed = True
    fake_ui = MagicMock()
    app.ui = fake_ui
    app.st_lottie = None
    app.load_lottie_animation = lambda *args, **kwargs: None
    app.start_analysis()

app.render_home(developer_mode=True)
'''


def _event_loop_available():
    try:
        loop = asyncio.new_event_loop()
    except OSError:
        return False
    try:
        loop.close()
    except Exception:
        pass
    return True


@unittest.skipUnless(
    _event_loop_available(), "event loop unavailable in this sandbox"
)
class ProgressRenderTests(unittest.TestCase):
    def test_lock_bar_and_persistence_across_runs(self):
        orig_ui = app.ui
        orig_lottie = app.st_lottie
        orig_load = app.load_lottie_animation
        try:
            at = AppTest.from_string(SCRIPT)
            at.run()
            self.assertEqual(len(at.exception), 0)
            # Click run: callback armed the lock, the pipeline executed
            # inline, and the finally block unlocked.
            self.assertFalse(at.session_state["is_analyzing"])
            self.assertEqual(
                at.session_state["analysis_results"]["kind"], "halt"
            )
            self.assertEqual(len(at.progress), 1)
            self.assertEqual(at.progress[0].value, 100)
            self.assertTrue(
                any(
                    "100% - Analysis complete." in m.value
                    for m in at.markdown
                )
            )
            at.run()
            self.assertEqual(len(at.exception), 0)
            # Settled run: the transient bar is gone, but the finished
            # output persists in session state and still renders while
            # nothing re-executes.
            self.assertEqual(len(at.progress), 0)
            self.assertEqual(
                at.session_state["analysis_results"]["kind"], "halt"
            )
            self.assertTrue(
                any("Cannot analyze yet" in w.value for w in at.warning)
            )
        finally:
            app.ui = orig_ui
            app.st_lottie = orig_lottie
            app.load_lottie_animation = orig_load
            if hasattr(app, "_progress_probe_armed"):
                delattr(app, "_progress_probe_armed")


if __name__ == "__main__":
    unittest.main()
