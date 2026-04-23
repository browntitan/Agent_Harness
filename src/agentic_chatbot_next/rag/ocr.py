"""rag/ocr.py — PaddleOCR wrapper for image files and scanned PDF pages.

Design constraints
------------------
- PaddleOCR is an optional dependency. If not installed, a warning is logged
  and callers receive an empty list of Documents.
- The OCR engine is initialised once (singleton) so the model-load cost is
  paid exactly once per process lifetime.
- Thread-safety: a threading.Lock guards singleton construction only. Once
  built, PaddleOCR.ocr() is called without holding the lock because
  PaddleOCR's CPU inference is re-entrant.
- PyMuPDF (fitz) is used to render individual PDF pages to numpy arrays for
  OCR. It is also an optional dep — scanned PDF pages are skipped if absent.
- No LLM calls; no external I/O beyond reading the supplied file.
"""
from __future__ import annotations

import logging
import inspect
import threading
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from langchain_core.documents import Document

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported image suffixes
# ---------------------------------------------------------------------------

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif"}

# ---------------------------------------------------------------------------
# Singleton management
# ---------------------------------------------------------------------------

_ocr_lock: threading.Lock = threading.Lock()
_ocr_engine: Optional[object] = None   # PaddleOCR instance once initialised
_ocr_available: Optional[bool] = None  # None = not yet probed


def _is_paddle_available() -> bool:
    """Return True if both paddleocr and paddlepaddle are importable."""
    global _ocr_available
    if _ocr_available is not None:
        return _ocr_available
    try:
        import paddleocr  # noqa: F401
        import paddle      # noqa: F401
        _ocr_available = True
    except ImportError:
        _ocr_available = False
        logger.warning(
            "PaddleOCR is not installed. "
            "Image files and scanned PDFs will be skipped. "
            "Install with: pip install paddlepaddle paddleocr"
        )
    return _ocr_available


def get_ocr_engine(language: str = "en", use_gpu: bool = False) -> Optional[object]:
    """Return the PaddleOCR singleton, creating it on first call.

    Returns None if PaddleOCR is not available.
    """
    global _ocr_engine
    if not _is_paddle_available():
        return None

    if _ocr_engine is not None:
        return _ocr_engine

    with _ocr_lock:
        # Double-checked locking: another thread may have initialised while
        # we were waiting for the lock.
        if _ocr_engine is None:
            from paddleocr import PaddleOCR  # type: ignore
            logger.info(
                "Initialising PaddleOCR (lang=%s, gpu=%s). "
                "First-run model download may take a moment.",
                language,
                use_gpu,
            )
            init_kwargs = {
                "use_angle_cls": True,  # auto-correct rotated/skewed text
                "lang": language,
                "use_gpu": use_gpu,
            }
            try:
                if "show_log" in inspect.signature(PaddleOCR.__init__).parameters:
                    # Newer PaddleOCR builds accept this and keep stdout quieter.
                    init_kwargs["show_log"] = False
            except (TypeError, ValueError):
                # Some builds expose a wrapped or non-inspectable initializer.
                pass
            _ocr_engine = PaddleOCR(**init_kwargs)
            logger.info("PaddleOCR ready.")
    return _ocr_engine


# ---------------------------------------------------------------------------
# Core OCR helper
# ---------------------------------------------------------------------------

def _ocr_numpy_image(img: "np.ndarray", engine: object) -> str:
    """Run OCR on a numpy HWC uint8 RGB array and return concatenated text.

    PaddleOCR returns:
        result[0] = list of [bbox, (text, confidence)] sorted top-to-bottom.

    Lines are joined with newlines preserving reading order.
    """
    result = engine.ocr(img, cls=True)  # type: ignore[union-attr]
    if not result or result[0] is None:
        return ""

    lines: List[str] = []
    for line in result[0]:
        # line = [[[x1,y1],...], ("text", confidence)]
        text_info = line[1]
        if text_info:
            text = text_info[0].strip()
            if text:
                lines.append(text)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Image-file entry point
# ---------------------------------------------------------------------------

def load_image_documents(
    path: Path,
    language: str = "en",
    use_gpu: bool = False,
) -> List[Document]:
    """OCR a standalone image file and return a single-element List[Document].

    Returns an empty list if PaddleOCR is unavailable or the file cannot
    be read.

    Metadata on the returned Document:
        source     : str(path)
        page       : 0  (consistent with PyPDFLoader page numbering)
        ocr_source : "paddleocr"
    """
    engine = get_ocr_engine(language=language, use_gpu=use_gpu)
    if engine is None:
        return []

    try:
        import numpy as np
        from PIL import Image

        img_pil = Image.open(str(path)).convert("RGB")
        img_np = np.array(img_pil)
    except Exception as exc:
        logger.warning("Failed to open image %s: %s", path, exc)
        return []

    text = _ocr_numpy_image(img_np, engine)
    if not text.strip():
        logger.debug("PaddleOCR returned empty text for image: %s", path)
        return []

    return [
        Document(
            page_content=text,
            metadata={
                "source": str(path),
                "page": 0,
                "ocr_source": "paddleocr",
            },
        )
    ]


# ---------------------------------------------------------------------------
# PDF entry point — hybrid native text + OCR
# ---------------------------------------------------------------------------

def load_pdf_documents_with_ocr(
    path: Path,
    min_page_chars: int = 50,
    language: str = "en",
    use_gpu: bool = False,
) -> List[Document]:
    """Load a PDF using PyPDF for text-rich pages and PaddleOCR for image-only pages.

    Algorithm per page
    ------------------
    1. Extract text with pypdf. If len(text) >= min_page_chars: use it (tag
       ocr_source="pypdf").
    2. Otherwise: render the page to numpy via PyMuPDF and run PaddleOCR
       (tag ocr_source="paddleocr").
    3. Pages that yield nothing from either path are silently skipped.

    Falls back to pure PyPDF output if PyMuPDF or PaddleOCR are not installed.

    Returns one Document per page with metadata:
        source     : str(path)
        page       : zero-based page index (int) — matches PyPDFLoader convention
        ocr_source : "pypdf" | "paddleocr"
    """
    # Check optional rendering dep
    try:
        import fitz  # type: ignore  # noqa: F401
        fitz_available = True
    except ImportError:
        fitz_available = False
        logger.warning(
            "PyMuPDF (fitz) not installed; cannot OCR scanned PDF pages. "
            "Install with: pip install pymupdf"
        )

    engine = (
        get_ocr_engine(language=language, use_gpu=use_gpu)
        if fitz_available
        else None
    )

    # Always load with pypdf first
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
    except Exception as exc:
        logger.warning("pypdf failed to open %s: %s — skipping.", path, exc)
        return []

    documents: List[Document] = []

    for page_idx, pdf_page in enumerate(reader.pages):
        native_text = (pdf_page.extract_text() or "").strip()

        if len(native_text) >= min_page_chars:
            # Sufficient native text
            documents.append(
                Document(
                    page_content=native_text,
                    metadata={
                        "source": str(path),
                        "page": page_idx,
                        "ocr_source": "pypdf",
                    },
                )
            )
        elif engine is not None and fitz_available:
            # Image-only page — render and OCR
            ocr_text = _render_and_ocr_page(path, page_idx, engine)
            if ocr_text.strip():
                documents.append(
                    Document(
                        page_content=ocr_text,
                        metadata={
                            "source": str(path),
                            "page": page_idx,
                            "ocr_source": "paddleocr",
                        },
                    )
                )
            else:
                logger.debug(
                    "Page %d of %s: no text from PyPDF or PaddleOCR — skipping page.",
                    page_idx,
                    path,
                )
        else:
            # OCR unavailable; keep the sparse native text if there is any
            if native_text:
                documents.append(
                    Document(
                        page_content=native_text,
                        metadata={
                            "source": str(path),
                            "page": page_idx,
                            "ocr_source": "pypdf",
                        },
                    )
                )

    return documents


def _render_and_ocr_page(
    path: Path,
    page_idx: int,
    engine: object,
    dpi: int = 200,
) -> str:
    """Render a single PDF page to a numpy image and run OCR.

    Args:
        path:     Path to the PDF file.
        page_idx: Zero-based page index.
        engine:   PaddleOCR engine (already initialised).
        dpi:      Rendering resolution (200 = good balance of speed vs quality).
                  Use 300 for denser/smaller text.

    Returns:
        Extracted text string (empty string if nothing detected or on error).
    """
    import fitz  # type: ignore
    import numpy as np

    try:
        doc = fitz.open(str(path))
        page = doc[page_idx]
        # Matrix controls resolution: scale = dpi / 72 (PDF unit = 1/72 inch)
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, 3
        )
        doc.close()
    except Exception as exc:
        logger.warning("Failed to render page %d of %s: %s", page_idx, path, exc)
        return ""

    return _ocr_numpy_image(img_np, engine)
