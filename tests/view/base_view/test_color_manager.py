"""Tests for ColorManager."""
import pytest
from unittest.mock import MagicMock, patch
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QWidget

from nn_verification_visualisation.view.base_view.color_manager import ColorManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A minimal stylesheet that contains every @key used in the real style.qss,
# so we can verify complete replacement without loading from disk.
_SAMPLE_STYLESHEET = (
    "background: @bg0; color: @bg1; hover: @hbg1; border: @bg2; h2: @hbg2; "
    "fg: @fg0; fg1: @fg1; hfg: @hfg1; c0: @c0; "
    "success: @success; warning: @warning; error: @error; herror: @herror; "
    "text: @bgt; hint: @ht; fg-text: @fgt; comp: @component;"
)

_ALL_KEYS = list(ColorManager.NETWORK_COLORS.keys())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_app(qtbot):
    """A real QApplication is provided by qtbot; we wrap it in a Mock so we
    can assert calls, but keep enough real behaviour for QPalette to work."""
    app = QApplication.instance()
    spy = MagicMock(wraps=app)
    # We only care about setStyleSheet / setPalette being callable and
    # recordable – wrapping the real instance does that.
    return spy


@pytest.fixture
def color_manager(mock_app):
    """ColorManager with raw_stylesheet injected directly (no disk I/O)."""
    cm = ColorManager.__new__(ColorManager)
    cm.app = mock_app
    cm.raw_stylesheet = _SAMPLE_STYLESHEET

    mock_window = MagicMock(spec=QWidget)
    cm.main_window = mock_window
    return cm


@pytest.fixture
def color_manager_with_load_raw(mock_app, mocker):
    """ColorManager whose load_raw() is exercised via a mocked QFile."""
    mock_file = mocker.patch("nn_verification_visualisation.view.base_view.color_manager.QFile")
    instance = mock_file.return_value
    instance.open.return_value = True
    instance.readAll.return_value.data.return_value.decode.return_value = _SAMPLE_STYLESHEET

    cm = ColorManager(mock_app)
    cm.load_raw(":fake/path.qss")

    mock_window = MagicMock(spec=QWidget)
    cm.main_window = mock_window
    return cm, mock_file


# ---------------------------------------------------------------------------
# load_raw
# ---------------------------------------------------------------------------

class TestLoadRaw:

    def test_stores_raw_stylesheet_verbatim(self, color_manager_with_load_raw):
        """raw_stylesheet must equal the file content before any replacement."""
        cm, _ = color_manager_with_load_raw
        assert cm.raw_stylesheet == _SAMPLE_STYLESHEET

    def test_sets_stylesheet_on_app(self, color_manager_with_load_raw):
        """app.setStyleSheet must be called exactly once during load_raw."""
        cm, _ = color_manager_with_load_raw
        cm.app.setStyleSheet.assert_called_once()

    def test_sets_palette_on_app(self, color_manager_with_load_raw):
        """app.setPalette must be called exactly once during load_raw."""
        cm, _ = color_manager_with_load_raw
        cm.app.setPalette.assert_called_once()

    def test_initial_stylesheet_uses_network_colors(self, color_manager_with_load_raw):
        """The stylesheet passed to app on load must contain NETWORK_COLORS values."""
        cm, _ = color_manager_with_load_raw
        applied_stylesheet = cm.app.setStyleSheet.call_args[0][0]
        for value in ColorManager.NETWORK_COLORS.values():
            assert value in applied_stylesheet, (
                f"Expected NETWORK_COLORS value '{value}' in initial stylesheet"
            )

    def test_initial_stylesheet_has_no_placeholders(self, color_manager_with_load_raw):
        """No @key placeholder may survive in the stylesheet sent to app."""
        cm, _ = color_manager_with_load_raw
        applied_stylesheet = cm.app.setStyleSheet.call_args[0][0]
        for key in _ALL_KEYS:
            assert f"@{key}" not in applied_stylesheet


# ---------------------------------------------------------------------------
# set_colors – target object
# ---------------------------------------------------------------------------

class TestSetColorsTarget:

    def test_sets_stylesheet_on_main_window_not_app(self, color_manager):
        """set_colors must update main_window, not app."""
        color_manager.set_colors(ColorManager.NETWORK_COLORS)

        color_manager.main_window.setStyleSheet.assert_called_once()
        color_manager.app.setStyleSheet.assert_not_called()

    def test_sets_palette_on_main_window_not_app(self, color_manager):
        """set_colors must update the palette on main_window, not app."""
        color_manager.set_colors(ColorManager.NETWORK_COLORS)

        color_manager.main_window.setPalette.assert_called_once()
        color_manager.app.setPalette.assert_not_called()


# ---------------------------------------------------------------------------
# set_colors – placeholder replacement
# ---------------------------------------------------------------------------

class TestSetColorsReplacement:

    def test_no_placeholders_remain_with_network_colors(self, color_manager):
        """After applying NETWORK_COLORS every @key must be gone."""
        color_manager.set_colors(ColorManager.NETWORK_COLORS)
        result = color_manager.main_window.setStyleSheet.call_args[0][0]
        for key in _ALL_KEYS:
            assert f"@{key}" not in result, (
                f"Placeholder @{key} was not replaced with NETWORK_COLORS"
            )

    def test_no_placeholders_remain_with_diagram_colors(self, color_manager):
        """After applying DIAGRAM_COLORS every @key must be gone."""
        color_manager.set_colors(ColorManager.DIAGRAM_COLORS)
        result = color_manager.main_window.setStyleSheet.call_args[0][0]
        for key in _ALL_KEYS:
            assert f"@{key}" not in result, (
                f"Placeholder @{key} was not replaced with DIAGRAM_COLORS"
            )

    def test_network_color_values_present_in_stylesheet(self, color_manager):
        """Every NETWORK_COLORS hex value must appear in the resulting stylesheet."""
        color_manager.set_colors(ColorManager.NETWORK_COLORS)
        result = color_manager.main_window.setStyleSheet.call_args[0][0]
        for key, value in ColorManager.NETWORK_COLORS.items():
            assert value in result, (
                f"NETWORK_COLORS['{key}'] = '{value}' missing from stylesheet"
            )

    def test_diagram_color_values_present_in_stylesheet(self, color_manager):
        """Every DIAGRAM_COLORS hex value must appear in the resulting stylesheet."""
        color_manager.set_colors(ColorManager.DIAGRAM_COLORS)
        result = color_manager.main_window.setStyleSheet.call_args[0][0]
        for key, value in ColorManager.DIAGRAM_COLORS.items():
            assert value in result, (
                f"DIAGRAM_COLORS['{key}'] = '{value}' missing from stylesheet"
            )

    def test_unknown_placeholder_is_left_intact(self, color_manager):
        """A @placeholder not present in the colors dict must survive unchanged,
        so that programming errors surface clearly rather than silently vanishing."""
        color_manager.raw_stylesheet = "color: @unknown_key; bg: @bg0;"
        color_manager.set_colors(ColorManager.NETWORK_COLORS)
        result = color_manager.main_window.setStyleSheet.call_args[0][0]
        assert "@unknown_key" in result


# ---------------------------------------------------------------------------
# set_colors – raw_stylesheet immutability
# ---------------------------------------------------------------------------

class TestSetColorsDoesNotMutateRaw:

    def test_raw_stylesheet_unchanged_after_set_colors(self, color_manager):
        """Applying colors must never modify raw_stylesheet (used for future switches)."""
        original = color_manager.raw_stylesheet
        color_manager.set_colors(ColorManager.NETWORK_COLORS)
        assert color_manager.raw_stylesheet == original

    def test_raw_stylesheet_still_has_placeholders_after_set_colors(self, color_manager):
        """raw_stylesheet must still contain @key tokens after set_colors."""
        color_manager.set_colors(ColorManager.NETWORK_COLORS)
        for key in _ALL_KEYS:
            assert f"@{key}" in color_manager.raw_stylesheet, (
                f"raw_stylesheet was mutated: @{key} is missing after set_colors"
            )

    def test_switching_themes_twice_produces_same_output(self, color_manager):
        """Calling set_colors repeatedly with the same theme must yield identical
        stylesheets — cumulative replacement bugs would break this."""
        color_manager.set_colors(ColorManager.NETWORK_COLORS)
        first = color_manager.main_window.setStyleSheet.call_args[0][0]

        color_manager.set_colors(ColorManager.DIAGRAM_COLORS)
        color_manager.set_colors(ColorManager.NETWORK_COLORS)
        second = color_manager.main_window.setStyleSheet.call_args[0][0]

        assert first == second

    def test_switching_themes_applies_diagram_values(self, color_manager):
        """After switching to DIAGRAM_COLORS the stylesheet must not still
        contain NETWORK_COLORS values that differ between themes."""
        differing_keys = {
            k for k in ColorManager.NETWORK_COLORS
            if ColorManager.NETWORK_COLORS[k] != ColorManager.DIAGRAM_COLORS[k]
        }
        color_manager.set_colors(ColorManager.DIAGRAM_COLORS)
        result = color_manager.main_window.setStyleSheet.call_args[0][0]
        for key in differing_keys:
            network_val = ColorManager.NETWORK_COLORS[key]
            diagram_val = ColorManager.DIAGRAM_COLORS[key]
            assert diagram_val in result
            assert network_val not in result, (
                f"NETWORK_COLORS['{key}'] = '{network_val}' still present after "
                f"switching to DIAGRAM_COLORS"
            )


# ---------------------------------------------------------------------------
# Palette values
# ---------------------------------------------------------------------------

class TestPaletteValues:
    """
    We call the private helper indirectly through set_colors and capture
    the QPalette passed to main_window.setPalette.
    Because main_window is a Mock, we need to read the palette from the
    call args rather than from the widget itself.
    """

    def _get_palette(self, color_manager, colors) -> QPalette:
        color_manager.set_colors(colors)
        return color_manager.main_window.setPalette.call_args[0][0]

    def test_palette_window_color_matches_bg0(self, color_manager):
        palette = self._get_palette(color_manager, ColorManager.NETWORK_COLORS)
        expected = QColor(ColorManager.NETWORK_COLORS["bg0"])
        assert palette.color(QPalette.ColorRole.Window) == expected

    def test_palette_button_color_matches_fg0(self, color_manager):
        palette = self._get_palette(color_manager, ColorManager.NETWORK_COLORS)
        expected = QColor(ColorManager.NETWORK_COLORS["fg0"])
        assert palette.color(QPalette.ColorRole.Button) == expected

    def test_palette_highlight_is_always_fixed_blue(self, color_manager):
        """Highlight color is hardcoded to #0078d7 regardless of theme."""
        network_palette = self._get_palette(color_manager, ColorManager.NETWORK_COLORS)
        diagram_palette = self._get_palette(color_manager, ColorManager.DIAGRAM_COLORS)
        expected = QColor("#0078d7")
        assert network_palette.color(QPalette.ColorRole.Highlight) == expected
        assert diagram_palette.color(QPalette.ColorRole.Highlight) == expected

# ---------------------------------------------------------------------------
# Edge case – main_window not set
# ---------------------------------------------------------------------------

class TestMainWindowNotSet:

    def test_set_colors_raises_if_main_window_not_assigned(self, mock_app):
        """set_colors must raise AttributeError when main_window was never set,
        making the missing initialisation obvious rather than silently failing."""
        cm = ColorManager.__new__(ColorManager)
        cm.app = mock_app
        cm.raw_stylesheet = _SAMPLE_STYLESHEET
        # deliberately NOT setting cm.main_window

        with pytest.raises(AttributeError):
            cm.set_colors(ColorManager.NETWORK_COLORS)