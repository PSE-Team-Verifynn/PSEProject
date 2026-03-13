import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "GUI"))
sys.path.insert(0, str(BASE_DIR / "QUALITY"))
sys.path.insert(0, str(BASE_DIR / "PROFILING"))

from qs_common import GUI_OUT_DIR, PROFILING_OUT_DIR, QS_DIR, QUALITY_OUT_DIR, ensure_output_dir, write_gui_outputs, write_json
from run_gui_qs import run_gui_smoke_tests
from run_profiling_qs import run_profiling_checks
from run_quality_qs import run_quality_checks


def main() -> None:
    ensure_output_dir(QS_DIR)
    ensure_output_dir(GUI_OUT_DIR)
    ensure_output_dir(QUALITY_OUT_DIR)
    ensure_output_dir(PROFILING_OUT_DIR)

    gui_rows = run_gui_smoke_tests()
    write_gui_outputs(gui_rows)
    quality_rows = run_quality_checks()
    profiling_rows = run_profiling_checks()

    summary = {
        "gui_checks_passed": all(row["passed"] for row in gui_rows),
        "quality_cases": len(quality_rows),
        "profiling_cases": len(profiling_rows),
    }
    write_json(QS_DIR / "summary.json", summary)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
