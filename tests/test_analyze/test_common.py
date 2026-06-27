"""Tests for _source_files."""

import os
from pathlib import Path

from src.analyze_conventions.common import _source_files  # pyright: ignore[reportPrivateUsage]


class TestFindSourceFiles:
    def test_finds_c_and_cpp(self, tmp_path: Path):
        p = tmp_path / "a.c"
        _ = p.write_text("int x;")
        p = tmp_path / "b.cpp"
        _ = p.write_text("int y;")
        p = tmp_path / "c.txt"
        _ = p.write_text("not code")
        files = _source_files(str(tmp_path))
        names = {os.path.basename(f) for f in files}
        assert "a.c" in names
        assert "b.cpp" in names
        assert "c.txt" not in names

    def test_recursive(self, tmp_path: Path):
        sub = tmp_path / "sub"
        sub.mkdir()
        p = sub / "deep.h"
        _ = p.write_text("// header")
        files = _source_files(str(tmp_path))
        assert any("deep.h" in f for f in files)

    def test_empty_directory(self, tmp_path: Path):
        assert _source_files(str(tmp_path)) == []

    def test_objective_c_extensions(self, tmp_path: Path):
        p = tmp_path / "a.m"
        _ = p.write_text("@interface Foo")
        p = tmp_path / "b.mm"
        _ = p.write_text("#import <Foundation/Foundation.h>")
        files = _source_files(str(tmp_path))
        names = {os.path.basename(f) for f in files}
        assert "a.m" in names
        assert "b.mm" in names
