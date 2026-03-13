import pytest
from unittest.mock import Mock
from PySide6.QtWidgets import QLabel, QScrollArea, QWidget

from nn_verification_visualisation.view.dialogs.settings_dialog import SettingsDialog
from nn_verification_visualisation.view.dialogs.settings_option import SettingsOption


def make_setting(name="My Setting", type="General", widget=None):
    """Helper to create a SettingsOption with a mock factory."""
    if widget is None:
        widget = QWidget()
    factory = Mock(return_value=widget)
    return SettingsOption(name=name, factory=factory, type=type)


@pytest.fixture(autouse=True)
def clear_settings():
    """Ensure SettingsDialog.settings is empty before and after each test."""
    SettingsDialog.settings.clear()
    yield
    SettingsDialog.settings.clear()


@pytest.fixture
def dialog(qtbot):
    on_close = Mock()
    d = SettingsDialog(on_close=on_close)
    qtbot.addWidget(d)
    return d


class TestSettingsDialogInit:
    def test_title_is_settings(self, qtbot):
        on_close = Mock()
        d = SettingsDialog(on_close=on_close)
        qtbot.addWidget(d)
        assert d.title == "Settings"

    def test_on_close_stored(self, qtbot):
        on_close = Mock()
        d = SettingsDialog(on_close=on_close)
        qtbot.addWidget(d)
        assert d.on_close is on_close


class TestGetContent:
    def test_content_contains_scroll_area(self, dialog):
        content = dialog.get_content()
        scroll_areas = content.findChildren(QScrollArea)
        assert len(scroll_areas) == 1

    def test_no_settings_renders_empty(self, dialog):
        content = dialog.get_content()
        labels = content.findChildren(QLabel)
        # No setting rows should be present — no heading or name labels
        heading_labels = [l for l in labels if l.objectName() == "heading"]
        assert heading_labels == []

    def test_single_setting_creates_name_label(self, qtbot):
        SettingsDialog.add_setting(make_setting(name="Alpha", type="GroupA"))
        d = SettingsDialog(on_close=Mock())
        qtbot.addWidget(d)
        content = d.get_content()
        name_labels = [l for l in content.findChildren(QLabel) if l.objectName() == "label"]
        assert any(l.text() == "Alpha" for l in name_labels)

    def test_single_setting_creates_group_heading(self, qtbot):
        SettingsDialog.add_setting(make_setting(name="Beta", type="MyGroup"))
        d = SettingsDialog(on_close=Mock())
        qtbot.addWidget(d)
        content = d.get_content()
        heading_labels = [l for l in content.findChildren(QLabel) if l.objectName() == "heading"]
        assert any(l.text() == "MyGroup" for l in heading_labels)

    def test_single_setting_calls_factory(self, qtbot):
        setting = make_setting()
        SettingsDialog.add_setting(setting)
        d = SettingsDialog(on_close=Mock())
        qtbot.addWidget(d)
        d.get_content()
        setting.factory.assert_called()

    def test_multiple_settings_same_group_one_heading(self, qtbot):
        SettingsDialog.add_setting(make_setting(name="S1", type="SharedGroup"))
        SettingsDialog.add_setting(make_setting(name="S2", type="SharedGroup"))
        d = SettingsDialog(on_close=Mock())
        qtbot.addWidget(d)
        content = d.get_content()
        heading_labels = [l for l in content.findChildren(QLabel) if l.objectName() == "heading"]
        shared_headings = [l for l in heading_labels if l.text() == "SharedGroup"]
        assert len(shared_headings) == 1

    def test_multiple_settings_different_groups_multiple_headings(self, qtbot):
        SettingsDialog.add_setting(make_setting(name="S1", type="GroupX"))
        SettingsDialog.add_setting(make_setting(name="S2", type="GroupY"))
        d = SettingsDialog(on_close=Mock())
        qtbot.addWidget(d)
        content = d.get_content()
        heading_labels = [l for l in content.findChildren(QLabel) if l.objectName() == "heading"]
        heading_texts = {l.text() for l in heading_labels}
        assert "GroupX" in heading_texts
        assert "GroupY" in heading_texts

    def test_factory_widget_is_embedded(self, qtbot):
        child_widget = QWidget()
        setting = make_setting(widget=child_widget)
        SettingsDialog.add_setting(setting)
        d = SettingsDialog(on_close=Mock())
        qtbot.addWidget(d)
        content = d.get_content()
        all_widgets = content.findChildren(QWidget)
        assert child_widget in all_widgets


class TestAddSetting:
    def test_add_setting_appends_to_class_list(self):
        setting = make_setting()
        SettingsDialog.add_setting(setting)
        assert setting in SettingsDialog.settings

    def test_add_setting_returns_callable(self):
        setting = make_setting()
        remover = SettingsDialog.add_setting(setting)
        assert callable(remover)

    def test_remover_removes_setting(self):
        setting = make_setting()
        remover = SettingsDialog.add_setting(setting)
        assert setting in SettingsDialog.settings
        remover()
        assert setting not in SettingsDialog.settings

    def test_add_multiple_settings(self):
        s1 = make_setting(name="One")
        s2 = make_setting(name="Two")
        SettingsDialog.add_setting(s1)
        SettingsDialog.add_setting(s2)
        assert s1 in SettingsDialog.settings
        assert s2 in SettingsDialog.settings

    def test_remover_only_removes_its_own_setting(self):
        s1 = make_setting(name="Keep")
        s2 = make_setting(name="Remove")
        SettingsDialog.add_setting(s1)
        remover = SettingsDialog.add_setting(s2)
        remover()
        assert s1 in SettingsDialog.settings
        assert s2 not in SettingsDialog.settings