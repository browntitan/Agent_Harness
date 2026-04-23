"""Unit tests for SessionWorkspace.

All tests run entirely on the local filesystem using pytest's tmp_path fixture.
No Docker, no LLM, no database required.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentic_chatbot_next.runtime.context import filesystem_key
from agentic_chatbot_next.sandbox.workspace import (
    SessionWorkspace,
    WorkspacePathError,
    _safe_filename,
)


# ---------------------------------------------------------------------------
# _safe_filename
# ---------------------------------------------------------------------------


class TestSafeFilename:
    def test_valid_name_returned_stripped(self):
        assert _safe_filename("  results.csv  ") == "results.csv"

    def test_empty_raises(self):
        with pytest.raises(WorkspacePathError, match="empty"):
            _safe_filename("")

    def test_whitespace_only_raises(self):
        with pytest.raises(WorkspacePathError, match="empty"):
            _safe_filename("   ")

    def test_forward_slash_raises(self):
        with pytest.raises(WorkspacePathError, match="separator"):
            _safe_filename("sub/file.txt")

    def test_backslash_raises(self):
        with pytest.raises(WorkspacePathError, match="separator"):
            _safe_filename("sub\\file.txt")

    def test_null_byte_raises(self):
        with pytest.raises(WorkspacePathError, match="null"):
            _safe_filename("file\x00.txt")

    def test_double_dot_prefix_raises(self):
        # No slash so the path-separator check is bypassed;
        # the leading-".." check must fire.
        with pytest.raises(WorkspacePathError, match="'\\.\\.'"):
            _safe_filename("..secret")

    def test_too_long_raises(self):
        with pytest.raises(WorkspacePathError, match="too long"):
            _safe_filename("a" * 129)

    def test_exactly_max_length_ok(self):
        name = "a" * 128
        assert _safe_filename(name) == name

    def test_single_dot_allowed(self):
        # ".gitignore"-style names: single leading dot is fine
        assert _safe_filename(".notes") == ".notes"


# ---------------------------------------------------------------------------
# SessionWorkspace.for_session
# ---------------------------------------------------------------------------


class TestForSession:
    def test_root_is_workspace_dir_slash_session_id(self, tmp_path):
        ws = SessionWorkspace.for_session("abc123", tmp_path)
        assert ws.root == tmp_path / filesystem_key("abc123")

    def test_directory_not_created_before_open(self, tmp_path):
        ws = SessionWorkspace.for_session("abc123", tmp_path)
        assert not ws.root.exists()

    def test_session_id_stored(self, tmp_path):
        ws = SessionWorkspace.for_session("mySession", tmp_path)
        assert ws.session_id == "mySession"


# ---------------------------------------------------------------------------
# open / close / is_open
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_open_creates_directory(self, tmp_path):
        ws = SessionWorkspace.for_session("sess1", tmp_path)
        ws.open()
        assert ws.root.is_dir()

    def test_is_open_true_after_open(self, tmp_path):
        ws = SessionWorkspace.for_session("sess1", tmp_path)
        ws.open()
        assert ws.is_open is True

    def test_is_open_false_before_open(self, tmp_path):
        ws = SessionWorkspace.for_session("sess1", tmp_path)
        assert ws.is_open is False

    def test_open_writes_meta_file(self, tmp_path):
        ws = SessionWorkspace.for_session("sess1", tmp_path)
        ws.open()
        meta_path = ws.root / ".meta"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["session_id"] == "sess1"
        assert "opened_at" in meta

    def test_open_idempotent(self, tmp_path):
        ws = SessionWorkspace.for_session("sess1", tmp_path)
        ws.open()
        ws.open()  # should not raise
        assert ws.root.is_dir()

    def test_close_removes_directory(self, tmp_path):
        ws = SessionWorkspace.for_session("sess1", tmp_path)
        ws.open()
        ws.close()
        assert not ws.root.exists()

    def test_close_noop_when_not_open(self, tmp_path):
        ws = SessionWorkspace.for_session("sess1", tmp_path)
        ws.close()  # should not raise even if dir never created

    def test_is_open_false_after_close(self, tmp_path):
        ws = SessionWorkspace.for_session("sess1", tmp_path)
        ws.open()
        ws.close()
        assert ws.is_open is False


# ---------------------------------------------------------------------------
# copy_file
# ---------------------------------------------------------------------------


class TestCopyFile:
    def test_copies_file_into_workspace(self, tmp_path):
        src = tmp_path / "data.csv"
        src.write_text("a,b\n1,2\n")

        ws = SessionWorkspace.for_session("s1", tmp_path / "ws")
        ws.open()
        dest = ws.copy_file(src)

        assert dest.exists()
        assert dest.read_text() == "a,b\n1,2\n"
        assert dest.name == "data.csv"

    def test_custom_filename(self, tmp_path):
        src = tmp_path / "orig.xlsx"
        src.write_bytes(b"\x00" * 16)

        ws = SessionWorkspace.for_session("s1", tmp_path / "ws")
        ws.open()
        dest = ws.copy_file(src, "renamed.xlsx")

        assert dest.name == "renamed.xlsx"

    def test_invalid_filename_raises(self, tmp_path):
        src = tmp_path / "file.csv"
        src.write_text("x")

        ws = SessionWorkspace.for_session("s1", tmp_path / "ws")
        ws.open()
        with pytest.raises(WorkspacePathError):
            ws.copy_file(src, "../outside.csv")


# ---------------------------------------------------------------------------
# write_text / read_text
# ---------------------------------------------------------------------------


class TestTextIO:
    def _ws(self, tmp_path: Path) -> SessionWorkspace:
        ws = SessionWorkspace.for_session("s1", tmp_path / "ws")
        ws.open()
        return ws

    def test_roundtrip(self, tmp_path):
        ws = self._ws(tmp_path)
        ws.write_text("notes.txt", "hello world")
        assert ws.read_text("notes.txt") == "hello world"

    def test_overwrite(self, tmp_path):
        ws = self._ws(tmp_path)
        ws.write_text("notes.txt", "v1")
        ws.write_text("notes.txt", "v2")
        assert ws.read_text("notes.txt") == "v2"

    def test_read_missing_raises_file_not_found(self, tmp_path):
        ws = self._ws(tmp_path)
        with pytest.raises(FileNotFoundError):
            ws.read_text("nonexistent.txt")

    def test_write_invalid_filename_raises(self, tmp_path):
        ws = self._ws(tmp_path)
        with pytest.raises(WorkspacePathError):
            ws.write_text("sub/dir.txt", "bad")

    def test_read_invalid_filename_raises(self, tmp_path):
        ws = self._ws(tmp_path)
        with pytest.raises(WorkspacePathError):
            ws.read_text("../escape.txt")

    def test_large_file_truncated(self, tmp_path):
        ws = self._ws(tmp_path)
        # Write >5 MiB
        big = "x" * (5 * 1024 * 1024 + 1)
        ws.write_text("big.txt", big)
        result = ws.read_text("big.txt")
        assert result.endswith("...[truncated]")

    def test_returns_path_object(self, tmp_path):
        ws = self._ws(tmp_path)
        dest = ws.write_text("out.txt", "data")
        assert isinstance(dest, Path)
        assert dest.name == "out.txt"


# ---------------------------------------------------------------------------
# exists / list_files / delete_file
# ---------------------------------------------------------------------------


class TestFileManagement:
    def _ws(self, tmp_path: Path) -> SessionWorkspace:
        ws = SessionWorkspace.for_session("s1", tmp_path / "ws")
        ws.open()
        return ws

    def test_exists_true_for_written_file(self, tmp_path):
        ws = self._ws(tmp_path)
        ws.write_text("a.txt", "data")
        assert ws.exists("a.txt") is True

    def test_exists_false_for_missing(self, tmp_path):
        ws = self._ws(tmp_path)
        assert ws.exists("missing.txt") is False

    def test_exists_false_for_invalid_filename(self, tmp_path):
        ws = self._ws(tmp_path)
        assert ws.exists("../bad") is False  # does not raise

    def test_list_files_empty_after_open(self, tmp_path):
        ws = self._ws(tmp_path)
        assert ws.list_files() == []

    def test_list_files_returns_written_files(self, tmp_path):
        ws = self._ws(tmp_path)
        ws.write_text("b.txt", "b")
        ws.write_text("a.txt", "a")
        files = ws.list_files()
        assert "a.txt" in files
        assert "b.txt" in files

    def test_list_files_sorted(self, tmp_path):
        ws = self._ws(tmp_path)
        ws.write_text("c.txt", "c")
        ws.write_text("a.txt", "a")
        ws.write_text("b.txt", "b")
        assert ws.list_files() == ["a.txt", "b.txt", "c.txt"]

    def test_list_files_excludes_meta(self, tmp_path):
        ws = self._ws(tmp_path)
        files = ws.list_files()
        assert ".meta" not in files

    def test_list_files_empty_when_not_open(self, tmp_path):
        ws = SessionWorkspace.for_session("s1", tmp_path / "ws")
        assert ws.list_files() == []

    def test_delete_file_removes_file(self, tmp_path):
        ws = self._ws(tmp_path)
        ws.write_text("del.txt", "bye")
        result = ws.delete_file("del.txt")
        assert result is True
        assert not ws.exists("del.txt")

    def test_delete_file_returns_false_when_missing(self, tmp_path):
        ws = self._ws(tmp_path)
        result = ws.delete_file("ghost.txt")
        assert result is False

    def test_delete_invalid_filename_raises(self, tmp_path):
        ws = self._ws(tmp_path)
        with pytest.raises(WorkspacePathError):
            ws.delete_file("../etc/passwd")


# ---------------------------------------------------------------------------
# read_bytes
# ---------------------------------------------------------------------------


class TestReadBytes:
    def test_reads_binary_content(self, tmp_path):
        ws = SessionWorkspace.for_session("s1", tmp_path / "ws")
        ws.open()
        data = b"\x00\x01\x02\x03"
        (ws.root / "bin.dat").write_bytes(data)
        assert ws.read_bytes("bin.dat") == data

    def test_raises_for_missing_file(self, tmp_path):
        ws = SessionWorkspace.for_session("s1", tmp_path / "ws")
        ws.open()
        with pytest.raises(FileNotFoundError):
            ws.read_bytes("nope.bin")


# ---------------------------------------------------------------------------
# Isolation between sessions
# ---------------------------------------------------------------------------


class TestIsolation:
    def test_separate_sessions_have_separate_directories(self, tmp_path):
        ws1 = SessionWorkspace.for_session("sess-a", tmp_path)
        ws2 = SessionWorkspace.for_session("sess-b", tmp_path)
        ws1.open()
        ws2.open()

        ws1.write_text("secret.txt", "session A data")
        assert not ws2.exists("secret.txt")

    def test_close_one_session_leaves_other_intact(self, tmp_path):
        ws1 = SessionWorkspace.for_session("sess-a", tmp_path)
        ws2 = SessionWorkspace.for_session("sess-b", tmp_path)
        ws1.open()
        ws2.open()

        ws1.write_text("file.txt", "data")
        ws2.write_text("other.txt", "other")

        ws1.close()
        assert not ws1.is_open
        assert ws2.is_open
        assert ws2.exists("other.txt")
