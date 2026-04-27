from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Dict, Iterator
from urllib.parse import quote, urlparse

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class BlobRef:
    backend: str
    uri: str
    bucket: str = ""
    key: str = ""
    etag: str = ""
    size: int = 0
    content_type: str = ""
    sha1: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "uri": self.uri,
            "bucket": self.bucket,
            "key": self.key,
            "etag": self.etag,
            "size": self.size,
            "content_type": self.content_type,
            "sha1": self.sha1,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "BlobRef":
        return cls(
            backend=str(raw.get("backend") or ""),
            uri=str(raw.get("uri") or ""),
            bucket=str(raw.get("bucket") or raw.get("container") or ""),
            key=str(raw.get("key") or ""),
            etag=str(raw.get("etag") or ""),
            size=int(raw.get("size") or 0),
            content_type=str(raw.get("content_type") or raw.get("mime_type") or ""),
            sha1=str(raw.get("sha1") or ""),
        )


StoredObject = BlobRef


def _sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_key_part(value: str) -> str:
    cleaned = str(value or "").replace("\\", "/").strip("/")
    parts = [part for part in PurePosixPath(cleaned).parts if part not in {"", ".", ".."}]
    return "/".join(parts)


def _metadata_blob_ref(metadata: Dict[str, Any]) -> BlobRef | None:
    raw_ref = metadata.get("blob_ref") or metadata.get("object_ref")
    if isinstance(raw_ref, BlobRef):
        return raw_ref
    if isinstance(raw_ref, dict):
        ref = BlobRef.from_dict(raw_ref)
        if ref.backend and ref.uri:
            return ref
    raw_json = metadata.get("blob_ref_json")
    if isinstance(raw_json, str) and raw_json.strip():
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict):
                ref = BlobRef.from_dict(parsed)
                if ref.backend and ref.uri:
                    return ref
        except Exception:
            return None
    return None


def blob_ref_from_record(record: Any) -> BlobRef | None:
    metadata = dict(getattr(record, "source_metadata", {}) or {})
    ref = _metadata_blob_ref(metadata)
    if ref is not None:
        return ref
    backend = str(getattr(record, "source_storage_backend", "") or "").strip()
    uri = str(getattr(record, "source_uri", "") or "").strip()
    bucket = str(getattr(record, "source_object_bucket", "") or "").strip()
    key = str(getattr(record, "source_object_key", "") or "").strip()
    if backend and uri:
        return BlobRef(
            backend=backend,
            uri=uri,
            bucket=bucket,
            key=key,
            etag=str(getattr(record, "source_etag", "") or ""),
            size=int(getattr(record, "source_size_bytes", 0) or 0),
            content_type=str(getattr(record, "source_content_type", "") or ""),
        )
    source_path = str(getattr(record, "source_path", "") or "").strip()
    if source_path.startswith("s3://"):
        parsed = urlparse(source_path)
        return BlobRef(
            backend="s3",
            uri=source_path,
            bucket=parsed.netloc,
            key=parsed.path.lstrip("/"),
            content_type=str(metadata.get("mime_type") or ""),
        )
    if source_path.startswith(("az://", "abfs://")):
        parsed = urlparse(source_path)
        return BlobRef(
            backend="azure_blob",
            uri=source_path,
            bucket=parsed.netloc,
            key=parsed.path.lstrip("/"),
            content_type=str(metadata.get("mime_type") or ""),
        )
    if source_path.startswith("file://"):
        path = Path(source_path.removeprefix("file://"))
        return BlobRef(
            backend="local",
            uri=source_path,
            bucket="",
            key=str(path),
            size=path.stat().st_size if path.exists() else 0,
            content_type=str(metadata.get("mime_type") or ""),
            sha1=_sha1_file(path) if path.exists() and path.is_file() else "",
        )
    return None


class BlobStore:
    def __init__(
        self,
        *,
        backend: str,
        local_root: Path,
        cache_dir: Path,
        bucket: str = "",
        prefix: str = "",
        endpoint_url: str = "",
        region: str = "",
        access_key: str = "",
        secret_key: str = "",
        session_token: str = "",
        azure_connection_string: str = "",
        azure_account_name: str = "",
        azure_account_key: str = "",
    ) -> None:
        self.backend = str(backend or "local").strip().lower()
        self.local_root = Path(local_root)
        self.cache_dir = Path(cache_dir)
        self.bucket = str(bucket or "").strip()
        self.prefix = _safe_key_part(prefix)
        self.endpoint_url = str(endpoint_url or "").strip()
        self.region = str(region or "").strip()
        self.access_key = str(access_key or "").strip()
        self.secret_key = str(secret_key or "").strip()
        self.session_token = str(session_token or "").strip()
        self.azure_connection_string = str(azure_connection_string or "").strip()
        self.azure_account_name = str(azure_account_name or "").strip()
        self.azure_account_key = str(azure_account_key or "").strip()
        self.local_root.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._fs: Any | None = None

    @property
    def remote(self) -> bool:
        return self.backend in {"s3", "azure_blob"}

    def _full_key(self, key: str) -> str:
        clean = _safe_key_part(key)
        if not clean:
            raise ValueError("blob key must not be empty")
        return "/".join(part for part in (self.prefix, clean) if part)

    def _filesystem(self) -> Any:
        if self._fs is not None:
            return self._fs
        try:
            import fsspec
        except Exception as exc:
            raise RuntimeError(
                "Remote object storage requires fsspec plus the backend extra "
                "(s3fs for S3, adlfs for Azure Blob)."
            ) from exc

        if self.backend == "s3":
            kwargs: Dict[str, Any] = {}
            if self.access_key:
                kwargs["key"] = self.access_key
            if self.secret_key:
                kwargs["secret"] = self.secret_key
            if self.session_token:
                kwargs["token"] = self.session_token
            client_kwargs: Dict[str, Any] = {}
            if self.endpoint_url:
                client_kwargs["endpoint_url"] = self.endpoint_url
            if self.region:
                client_kwargs["region_name"] = self.region
            if client_kwargs:
                kwargs["client_kwargs"] = client_kwargs
            self._fs = fsspec.filesystem("s3", **kwargs)
            return self._fs

        if self.backend == "azure_blob":
            kwargs = {}
            if self.azure_connection_string:
                kwargs["connection_string"] = self.azure_connection_string
            if self.azure_account_name:
                kwargs["account_name"] = self.azure_account_name
            if self.azure_account_key:
                kwargs["account_key"] = self.azure_account_key
            self._fs = fsspec.filesystem("az", **kwargs)
            return self._fs

        raise ValueError(f"Unsupported blob store backend: {self.backend!r}")

    def _uri(self, key: str) -> str:
        if self.backend == "local":
            return (self.local_root / key).resolve().as_uri()
        if self.backend == "s3":
            return f"s3://{self.bucket}/{key}"
        if self.backend == "azure_blob":
            return f"az://{self.bucket}/{key}"
        raise ValueError(f"Unsupported blob store backend: {self.backend!r}")

    def put_file(self, source: Path, *, key: str, content_type: str = "") -> StoredObject:
        source = Path(source)
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(f"Blob source file does not exist: {source}")
        full_key = self._full_key(key)
        size = source.stat().st_size
        sha1 = _sha1_file(source)

        if self.backend == "local":
            destination = self.local_root / full_key
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            return StoredObject(
                backend="local",
                uri=destination.resolve().as_uri(),
                bucket="",
                key=str(destination),
                etag=sha1,
                size=size,
                content_type=content_type,
                sha1=sha1,
            )

        remote_path = f"{self.bucket}/{full_key}"
        fs = self._filesystem()
        parent = str(PurePosixPath(remote_path).parent)
        if parent and parent != ".":
            try:
                fs.makedirs(parent, exist_ok=True)
            except Exception:
                logger.debug("BlobStore: backend did not require explicit directory creation for %s", parent)
        open_kwargs: Dict[str, Any] = {}
        if content_type:
            open_kwargs["content_type"] = content_type

        def _copy_to_remote(extra_kwargs: Dict[str, Any]) -> None:
            with source.open("rb") as src, fs.open(remote_path, "wb", **extra_kwargs) as dest:
                shutil.copyfileobj(src, dest, length=_CHUNK_SIZE)

        try:
            _copy_to_remote(open_kwargs)
        except TypeError:
            if not open_kwargs:
                raise
            _copy_to_remote({})
        etag = ""
        try:
            info = fs.info(remote_path)
            etag = str(info.get("ETag") or info.get("etag") or "")
        except Exception:
            etag = ""
        return StoredObject(
            backend=self.backend,
            uri=self._uri(full_key),
            bucket=self.bucket,
            key=full_key,
            etag=etag,
            size=size,
            content_type=content_type,
            sha1=sha1,
        )

    def exists(self, ref: BlobRef | str) -> bool:
        ref = self.resolve_ref(ref)
        if ref.backend == "local":
            return Path(ref.key or ref.uri.removeprefix("file://")).exists()
        return bool(self._filesystem().exists(f"{ref.bucket}/{ref.key}"))

    def delete(self, ref: BlobRef | str) -> None:
        ref = self.resolve_ref(ref)
        if ref.backend == "local":
            Path(ref.key or ref.uri.removeprefix("file://")).unlink(missing_ok=True)
            return
        try:
            self._filesystem().rm(f"{ref.bucket}/{ref.key}")
        except FileNotFoundError:
            return

    def open_read(self, ref: BlobRef | str) -> BinaryIO:
        ref = self.resolve_ref(ref)
        if ref.backend == "local":
            return Path(ref.key or ref.uri.removeprefix("file://")).open("rb")
        return self._filesystem().open(f"{ref.bucket}/{ref.key}", "rb")

    def iter_bytes(self, ref: BlobRef | str, *, chunk_size: int = _CHUNK_SIZE) -> Iterator[bytes]:
        with self.open_read(ref) as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    def materialize_to_path(self, ref: BlobRef | str, destination: Path | None = None) -> Path:
        ref = self.resolve_ref(ref)
        if ref.backend == "local":
            path = Path(ref.key or ref.uri.removeprefix("file://"))
            if destination is None:
                return path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            return destination

        cache_key = _safe_key_part(ref.key) or quote(ref.uri, safe="")
        destination = destination or self.cache_dir / ref.backend / quote(ref.bucket, safe="") / cache_key
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self.open_read(ref) as src, destination.open("wb") as dest:
            shutil.copyfileobj(src, dest, length=_CHUNK_SIZE)
        return destination

    def resolve_ref(self, ref: BlobRef | str) -> BlobRef:
        if isinstance(ref, BlobRef):
            return ref
        uri = str(ref or "").strip()
        if uri.startswith("file://"):
            path = Path(uri.removeprefix("file://"))
            return BlobRef(backend="local", uri=uri, key=str(path))
        parsed = urlparse(uri)
        if parsed.scheme == "s3":
            return BlobRef(backend="s3", uri=uri, bucket=parsed.netloc, key=parsed.path.lstrip("/"))
        if parsed.scheme in {"az", "abfs"}:
            return BlobRef(backend="azure_blob", uri=uri, bucket=parsed.netloc, key=parsed.path.lstrip("/"))
        path = Path(uri)
        return BlobRef(backend="local", uri=path.resolve().as_uri(), key=str(path))


def build_blob_store(settings: Any) -> BlobStore:
    data_dir = Path(getattr(settings, "data_dir", Path("data")))
    return BlobStore(
        backend=str(getattr(settings, "object_store_backend", "local") or "local"),
        local_root=Path(getattr(settings, "uploads_dir", data_dir / "uploads")),
        cache_dir=Path(getattr(settings, "object_store_cache_dir", data_dir / "cache" / "blob-store")),
        bucket=str(
            getattr(settings, "object_store_bucket", "")
            or getattr(settings, "object_store_container", "")
            or ""
        ),
        prefix=str(getattr(settings, "object_store_prefix", "") or ""),
        endpoint_url=str(getattr(settings, "object_store_endpoint_url", "") or ""),
        region=str(getattr(settings, "object_store_region", "") or ""),
        access_key=str(getattr(settings, "object_store_access_key", "") or ""),
        secret_key=str(getattr(settings, "object_store_secret_key", "") or ""),
        session_token=str(getattr(settings, "object_store_session_token", "") or ""),
        azure_connection_string=str(getattr(settings, "azure_blob_connection_string", "") or ""),
        azure_account_name=str(getattr(settings, "azure_blob_account_name", "") or ""),
        azure_account_key=str(getattr(settings, "azure_blob_account_key", "") or ""),
    )
