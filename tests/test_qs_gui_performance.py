from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "QS"))

from qs_common import write_gui_performance_plot


class _FakeBar:
    def __init__(self, index: int, width: float = 0.62):
        self._index = float(index)
        self._width = width

    def get_x(self) -> float:
        return self._index

    def get_width(self) -> float:
        return self._width


class _FakeAxis:
    def __init__(self):
        self.bar_calls = []
        self.title = None
        self.ylabel = None

    def bar(self, labels, values, color=None, width=None):
        self.bar_calls.append(
            {
                "labels": list(labels),
                "values": list(values),
                "color": color,
                "width": width,
            }
        )
        return [_FakeBar(index, width or 0.62) for index, _ in enumerate(values)]

    def set_title(self, title):
        self.title = title

    def set_ylabel(self, ylabel):
        self.ylabel = ylabel

    def grid(self, axis=None, alpha=None):
        return None

    def tick_params(self, axis=None, rotation=None):
        return None

    def text(self, x, y, text, ha=None, va=None, fontsize=None):
        return None


class _FakeFigure:
    def tight_layout(self):
        return None

    def savefig(self, output_path, dpi=None):
        return None


def test_write_gui_performance_plot_compares_startup_with_network_loads(monkeypatch, tmp_path):
    captured_axes = [_FakeAxis(), _FakeAxis()]

    def fake_subplots(*args, **kwargs):
        return _FakeFigure(), captured_axes

    monkeypatch.setattr("qs_common.plt.subplots", fake_subplots)
    monkeypatch.setattr("qs_common.plt.close", lambda fig: None)

    payload = {
        "startup_time_ms": 530.111,
        "startup_memory_kb": 19732,
        "network_load_results": [
            {"case": "simple_3_layer_x10", "load_to_display_ms": 499.341, "load_memory_kb": 5120, "scene_items": 142},
            {"case": "NN1", "load_to_display_ms": 311.096, "load_memory_kb": 1536, "scene_items": 2017},
        ],
    }

    write_gui_performance_plot(payload, tmp_path / "gui_performance.png")

    assert captured_axes[0].title == "Startup vs Network Load-to-Display"
    assert captured_axes[0].bar_calls == [
        {
            "labels": ["gui_startup", "simple_3_layer_x10", "NN1"],
            "values": [530.111, 499.341, 311.096],
            "color": "#2563eb",
            "width": 0.62,
        }
    ]
    assert captured_axes[1].title == "Startup vs Network Load Memory Delta"
    assert captured_axes[1].bar_calls == [
        {
            "labels": ["gui_startup", "simple_3_layer_x10", "NN1"],
            "values": [19.27, 5.0, 1.5],
            "color": "#dc2626",
            "width": 0.62,
        }
    ]
