import py_compile
from pathlib import Path


def test_main_py_has_valid_syntax():
    project_root = Path(__file__).resolve().parents[1]
    main_py = project_root / "main.py"
    py_compile.compile(str(main_py), doraise=True)

