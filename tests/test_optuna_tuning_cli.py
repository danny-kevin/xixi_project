import sys


def test_main_parse_args_accepts_tune_mode(monkeypatch):
    import main

    monkeypatch.setattr(sys, "argv", ["main.py", "--mode", "tune"])
    args = main.parse_args()
    assert args.mode == "tune"


def test_main_parse_args_accepts_tune_flags(monkeypatch):
    import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--mode",
            "tune",
            "--tune-trials",
            "3",
            "--tune-study-name",
            "unit_test_study",
        ],
    )
    args = main.parse_args()
    assert args.tune_trials == 3
    assert args.tune_study_name == "unit_test_study"

