"""Tests for detect_language."""

from pathlib import Path

from src.analyze_conventions.language import detect_language


class TestDetectLanguage:
    def test_cpp_dominant(self, tmp_path: Path):
        files = [
            str(tmp_path / "a.cpp"),
            str(tmp_path / "b.h"),
            str(tmp_path / "c.c"),
        ]
        for f in files:
            p = Path(f)
            _ = p.write_text("x")
        assert detect_language(files) == "Cpp"

    def test_no_files(self):
        assert detect_language([]) is None

    def test_pure_c(self, tmp_path: Path):
        files = [str(tmp_path / "a.c"), str(tmp_path / "b.c")]
        for f in files:
            _ = Path(f).write_text("x")
        assert detect_language(files) == "C"
