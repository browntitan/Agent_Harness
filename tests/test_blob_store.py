from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_chatbot_next.persistence.postgres.documents import DocumentRecord
from agentic_chatbot_next.storage import BlobRef, BlobStore, blob_ref_from_record


class _FakeFsspecFS:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.open_kwargs: list[dict[str, object]] = []

    def _path(self, remote_path: str) -> Path:
        return self.root / remote_path

    def makedirs(self, remote_path: str, exist_ok: bool = False) -> None:
        del exist_ok
        self._path(remote_path).mkdir(parents=True, exist_ok=True)

    def open(self, remote_path: str, mode: str, **kwargs):
        self.open_kwargs.append(dict(kwargs))
        path = self._path(remote_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.open(mode)

    def info(self, remote_path: str) -> dict[str, str]:
        if not self._path(remote_path).exists():
            raise FileNotFoundError(remote_path)
        return {"ETag": "fake-etag"}

    def exists(self, remote_path: str) -> bool:
        return self._path(remote_path).exists()

    def rm(self, remote_path: str) -> None:
        self._path(remote_path).unlink()


def test_local_blob_store_round_trips_file(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("hello blob store", encoding="utf-8")
    store = BlobStore(
        backend="local",
        local_root=tmp_path / "objects",
        cache_dir=tmp_path / "cache",
    )

    ref = store.put_file(source, key="tenant/docs/source.txt", content_type="text/plain")

    assert ref.backend == "local"
    assert ref.content_type == "text/plain"
    assert store.exists(ref)
    assert store.open_read(ref).read() == b"hello blob store"
    copy_path = store.materialize_to_path(ref, tmp_path / "copy.txt")
    assert copy_path.read_text(encoding="utf-8") == "hello blob store"
    assert blob_ref_from_record(DocumentRecord(
        doc_id="doc",
        title="Doc",
        source_type="upload",
        content_hash="hash",
        source_metadata={"blob_ref": ref.to_dict()},
    )) == ref


@pytest.mark.parametrize(
    ("backend", "expected_protocol", "uri_prefix"),
    [
        ("s3", "s3", "s3://bucket/uploads/tenant/docs/source.txt"),
        ("azure_blob", "az", "az://bucket/uploads/tenant/docs/source.txt"),
    ],
)
def test_remote_blob_store_uses_fsspec_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    expected_protocol: str,
    uri_prefix: str,
) -> None:
    source = tmp_path / "source.txt"
    source.write_text("remote bytes", encoding="utf-8")
    fake_fs = _FakeFsspecFS(tmp_path / "remote")
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_filesystem(protocol: str, **kwargs):
        calls.append((protocol, dict(kwargs)))
        return fake_fs

    monkeypatch.setitem(__import__("sys").modules, "fsspec", SimpleNamespace(filesystem=fake_filesystem))
    store = BlobStore(
        backend=backend,
        local_root=tmp_path / "objects",
        cache_dir=tmp_path / "cache",
        bucket="bucket",
        prefix="uploads",
        endpoint_url="http://object-store:8333",
        region="us-east-1",
        access_key="access",
        secret_key="secret",
        azure_account_name="acct",
        azure_account_key="key",
    )

    ref = store.put_file(source, key="tenant/docs/source.txt", content_type="text/plain")

    assert calls[0][0] == expected_protocol
    assert ref.backend == backend
    assert ref.uri == uri_prefix
    assert ref.etag == "fake-etag"
    assert store.exists(ref)
    assert store.materialize_to_path(ref).read_text(encoding="utf-8") == "remote bytes"
    assert fake_fs.open_kwargs[0] == {"content_type": "text/plain"}


def test_blob_ref_from_new_document_columns() -> None:
    record = DocumentRecord(
        doc_id="doc",
        title="Doc",
        source_type="upload",
        content_hash="hash",
        source_uri="s3://bucket/uploads/doc.pdf",
        source_storage_backend="s3",
        source_object_bucket="bucket",
        source_object_key="uploads/doc.pdf",
        source_etag="etag",
        source_size_bytes=12,
        source_content_type="application/pdf",
    )

    assert blob_ref_from_record(record) == BlobRef(
        backend="s3",
        uri="s3://bucket/uploads/doc.pdf",
        bucket="bucket",
        key="uploads/doc.pdf",
        etag="etag",
        size=12,
        content_type="application/pdf",
    )
