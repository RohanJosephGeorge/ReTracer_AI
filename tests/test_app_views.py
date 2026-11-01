"""View content: pipeline diagram and Privacy & Safety sections.

Run: python -m unittest discover -s tests -v
Streamlit is mocked: no browser.
"""

import unittest
from unittest.mock import MagicMock, patch

import app


def render_with_mock_st(func):
    mock_st = MagicMock()
    with patch.object(app, "st", mock_st):
        func()
    return mock_st


def first_positional_texts(mock_method):
    return [
        call.args[0] for call in mock_method.call_args_list if call.args
    ]


class StateDict(dict):
    """Dict with attribute access, mirroring Streamlit session_state."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class UploadFeedbackTests(unittest.TestCase):
    def run_home_with_clean_upload(self):
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
            }
        )
        mock_st = MagicMock()
        mock_st.session_state = state
        mock_st.columns.return_value = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )
        fake_upload = MagicMock()
        fake_upload.name = "photo.png"
        fake_upload.size = 1234
        mock_st.file_uploader.return_value = fake_upload
        mock_st.text_input.return_value = ""
        mock_ui = MagicMock()
        mock_ui.button.return_value = False
        fake_image = MagicMock()
        with (
            patch.object(app, "st", mock_st),
            patch.object(app, "ui", mock_ui),
            patch.object(
                app, "is_valid_image", return_value=(True, "Valid image.")
            ),
            patch.object(
                app,
                "sanitize_uploaded_image",
                return_value=(
                    True,
                    "Sanitized to metadata-free RGB (64x48).",
                    fake_image,
                ),
            ),
            patch.object(
                app,
                "has_sufficient_visual_data",
                return_value=(True, "ok", None),
            ),
            patch.object(
                app,
                "screen_image_faces",
                return_value=("clean", "No faces detected.", fake_image),
            ),
        ):
            app.render_home()
        return mock_st

    def test_processing_spinner_shown_before_preview(self):
        mock_st = self.run_home_with_clean_upload()
        mock_st.spinner.assert_called_once_with(
            "Processing & Sanitizing Image..."
        )
        mock_st.image.assert_called_once()

    def test_validation_badge_is_centered_frosted_pill(self):
        mock_st = self.run_home_with_clean_upload()
        mock_st.success.assert_not_called()
        bodies = first_positional_texts(mock_st.markdown)
        pills = [
            body
            for body in bodies
            if "Valid image." in body and "border-radius: 30px" in body
        ]
        self.assertEqual(len(pills), 1)
        self.assertIn("rgba(34, 197, 94, 0.15)", pills[0])
        self.assertIn("margin: 1.5rem auto", pills[0])
        self.assertIn(
            "Sanitized to metadata-free RGB (64x48).", pills[0]
        )
        self.assertIn("No faces detected.", pills[0])

    def test_regional_hint_has_relevant_placeholder(self):
        mock_st = self.run_home_with_clean_upload()
        mock_st.text_input.assert_called_once_with(
            "Regional hint (Optional)",
            placeholder="e.g., Kerala backwaters, Western Ghats, Old Delhi, Tokyo alley, Scottish Highlands",
            help="Providing a state, landmark, or landscape context helps narrow spatial confidence faster.",
        )

    def test_single_uploader_without_duplicate_header_card(self):
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
            }
        )
        mock_st = MagicMock()
        mock_st.session_state = state
        mock_st.columns.return_value = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )
        mock_st.file_uploader.return_value = None
        mock_st.text_input.return_value = ""
        mock_ui = MagicMock()
        mock_ui.button.return_value = False
        with (
            patch.object(app, "st", mock_st),
            patch.object(app, "ui", mock_ui),
        ):
            app.render_home()
        mock_ui.card.assert_not_called()
        mock_st.file_uploader.assert_called_once()

    def test_uploader_uses_memory_label_and_sanitization_help(self):
        mock_st = self.run_home_with_clean_upload()
        mock_st.file_uploader.assert_called_once_with(
            "Upload a memory",
            key="memory_upload",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=False,
            help="Sanitized locally with metadata removal and face blurring before spatial analysis.",
            disabled=False,
        )


class HowItWorksViewTests(unittest.TestCase):
    def test_renders_pipeline_diagram(self):
        mock_st = render_with_mock_st(app.render_how_it_works)
        mock_st.title.assert_called_once_with("How it Works")
        bodies = first_positional_texts(mock_st.markdown)
        diagrams = [body for body in bodies if "```mermaid" in body]
        self.assertEqual(len(diagrams), 1)
        for stage in (
            "SAM 3.1",
            "Wonder",
            "Face screening",
            "Developer Mode",
        ):
            self.assertIn(stage, diagrams[0])
        for styling in ("classDef", "linkStyle"):
            self.assertIn(styling, diagrams[0])

    def test_explains_each_stage(self):
        mock_st = render_with_mock_st(app.render_how_it_works)
        bodies = " ".join(first_positional_texts(mock_st.markdown))
        for keyword in (
            "Upload",
            "gates",
            "regional hint",
            "Analyze",
            "Live mode",
        ):
            self.assertIn(keyword, bodies)


class PrivacySafetyViewTests(unittest.TestCase):
    def test_covers_screening_zero_data_and_compliance(self):
        mock_st = render_with_mock_st(app.render_privacy_safety)
        mock_st.title.assert_called_once_with("Privacy & Safety")
        headings = first_positional_texts(mock_st.subheader)
        self.assertEqual(len(headings), 3)
        bodies = " ".join(first_positional_texts(mock_st.markdown))
        self.assertIn("5%", bodies)
        for keyword in ("blur", "exif", "session", "secrets", "round"):
            self.assertIn(keyword, bodies.lower())


class ViewRoutingTests(unittest.TestCase):
    def test_unknown_selection_defaults_to_home(self):
        mock_st = MagicMock()
        mock_st.columns.return_value = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )
        mock_st.radio.return_value = "something unexpected"
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


if __name__ == "__main__":
    unittest.main()
