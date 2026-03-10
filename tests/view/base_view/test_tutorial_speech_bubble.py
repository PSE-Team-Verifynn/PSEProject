import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QHBoxLayout, QSpacerItem

from nn_verification_visualisation.view.base_view.tutorial_speech_bubble import TutorialSpeechBubble

HEADING = "My Heading"
TEXT = "My text content"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_card(bubble: TutorialSpeechBubble):
    """Return the QWidget with objectName 'card', or None."""
    return bubble.findChild(type(bubble).__mro__[1], "card") or bubble.findChildren(
        __import__("PySide6.QtWidgets", fromlist=["QWidget"]).QWidget, "card"
    )[0]


def _find_label(bubble: TutorialSpeechBubble, object_name: str):
    """Return the QLabel with the given objectName."""
    from PySide6.QtWidgets import QWidget
    labels = bubble.findChildren(QLabel)
    for label in labels:
        if label.objectName() == object_name:
            return label
    return None


def _find_labels(bubble: TutorialSpeechBubble):
    """Return (title_label, body_label) based on objectName."""
    title = _find_label(bubble, "title")
    body = next(
        (lbl for lbl in bubble.findChildren(QLabel) if lbl.objectName() != "title"),
        None,
    )
    return title, body


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def bubble(qtbot):
    widget = TutorialSpeechBubble(HEADING, TEXT)
    qtbot.addWidget(widget)
    return widget


# ---------------------------------------------------------------------------
# Widget structure
# ---------------------------------------------------------------------------

class TestWidgetStructure:
    def test_contains_a_card_container(self, bubble):
        from PySide6.QtWidgets import QWidget
        cards = bubble.findChildren(QWidget, "card")
        assert len(cards) == 1

    def test_card_has_correct_margins(self, bubble):
        from PySide6.QtWidgets import QWidget
        card = bubble.findChildren(QWidget, "card")[0]
        margins = card.layout().contentsMargins()
        assert (margins.left(), margins.top(), margins.right(), margins.bottom()) == (bubble._TutorialSpeechBubble__margin, bubble._TutorialSpeechBubble__margin, bubble._TutorialSpeechBubble__margin, bubble._TutorialSpeechBubble__margin)

    def test_card_has_correct_spacing(self, bubble):
        from PySide6.QtWidgets import QWidget
        card = bubble.findChildren(QWidget, "card")[0]
        assert card.layout().spacing() == bubble._TutorialSpeechBubble__spacing


# ---------------------------------------------------------------------------
# Title label
# ---------------------------------------------------------------------------

class TestTitleLabel:
    def test_title_label_exists(self, bubble):
        assert _find_label(bubble, "title") is not None

    def test_title_label_displays_heading(self, bubble):
        title = _find_label(bubble, "title")
        assert title.text() == HEADING

    def test_title_label_is_centered(self, bubble):
        title = _find_label(bubble, "title")
        assert title.alignment() == Qt.AlignmentFlag.AlignCenter

    def test_title_label_is_inside_card(self, bubble):
        from PySide6.QtWidgets import QWidget
        card = bubble.findChildren(QWidget, "card")[0]
        title = _find_label(bubble, "title")
        assert title.parent() == card


# ---------------------------------------------------------------------------
# Body label
# ---------------------------------------------------------------------------

class TestBodyLabel:
    def test_body_label_exists(self, bubble):
        _, body = _find_labels(bubble)
        assert body is not None

    def test_body_label_displays_text(self, bubble):
        _, body = _find_labels(bubble)
        assert body.text() == TEXT

    def test_body_label_is_inside_card(self, bubble):
        from PySide6.QtWidgets import QWidget
        card = bubble.findChildren(QWidget, "card")[0]
        _, body = _find_labels(bubble)
        assert body.parent() == card


# ---------------------------------------------------------------------------
# Layout & centering
# ---------------------------------------------------------------------------

class TestLayout:
    def test_title_appears_before_body_in_card_layout(self, bubble):
        from PySide6.QtWidgets import QWidget
        card = bubble.findChildren(QWidget, "card")[0]
        layout = card.layout()
        title = _find_label(bubble, "title")
        _, body = _find_labels(bubble)

        title_index = next(i for i in range(layout.count()) if layout.itemAt(i).widget() == title)
        body_index = next(i for i in range(layout.count()) if layout.itemAt(i).widget() == body)

        assert title_index < body_index

    def test_card_is_horizontally_centered_with_stretches(self, bubble):
        """The HBoxLayout holding the card must have stretch–card–stretch (3 items)."""
        outer_layout = bubble.layout()
        # Find the hbox among the outer layout's items
        hbox = next(
            outer_layout.itemAt(i).layout()
            for i in range(outer_layout.count())
            if isinstance(outer_layout.itemAt(i).layout(), QHBoxLayout)
        )
        assert hbox.count() == 3
        assert hbox.itemAt(0).spacerItem() is not None
        assert hbox.itemAt(2).spacerItem() is not None

    def test_card_is_vertically_centered_with_stretches(self, bubble):
        """The outer VBoxLayout must have stretch–hbox–stretch (3 items)."""
        outer_layout = bubble.layout()
        assert outer_layout.count() == 3
        assert outer_layout.itemAt(0).spacerItem() is not None
        assert outer_layout.itemAt(2).spacerItem() is not None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_heading(self, qtbot):
        widget = TutorialSpeechBubble("", TEXT)
        qtbot.addWidget(widget)
        title = _find_label(widget, "title")
        assert title.text() == ""

    def test_empty_text(self, qtbot):
        widget = TutorialSpeechBubble(HEADING, "")
        qtbot.addWidget(widget)
        _, body = _find_labels(widget)
        assert body.text() == ""

    def test_both_empty(self, qtbot):
        widget = TutorialSpeechBubble("", "")
        qtbot.addWidget(widget)
        title = _find_label(widget, "title")
        _, body = _find_labels(widget)
        assert title.text() == ""
        assert body.text() == ""

    def test_long_strings_displayed_verbatim(self, qtbot):
        long_heading = "A" * 500
        long_text = "B" * 2000
        widget = TutorialSpeechBubble(long_heading, long_text)
        qtbot.addWidget(widget)
        title = _find_label(widget, "title")
        _, body = _find_labels(widget)
        assert title.text() == long_heading
        assert body.text() == long_text

    def test_html_special_characters_not_interpreted_in_heading(self, qtbot):
        """QLabel may silently strip/transform HTML tags; the raw string must be preserved."""
        raw = "<b>bold</b> & <i>italic</i>"
        widget = TutorialSpeechBubble(raw, TEXT)
        qtbot.addWidget(widget)
        title = _find_label(widget, "title")
        assert title.text() == raw

    def test_html_special_characters_not_interpreted_in_body(self, qtbot):
        raw = "<script>alert('xss')</script>"
        widget = TutorialSpeechBubble(HEADING, raw)
        qtbot.addWidget(widget)
        _, body = _find_labels(widget)
        assert body.text() == raw

    def test_newlines_in_text_preserved(self, qtbot):
        multiline = "Line one\nLine two\nLine three"
        widget = TutorialSpeechBubble(HEADING, multiline)
        qtbot.addWidget(widget)
        _, body = _find_labels(widget)
        assert body.text() == multiline