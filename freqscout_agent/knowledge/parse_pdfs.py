"""
PDF Parser
----------
Parses clinical reference PDFs into hierarchy-preserving chunks for the knowledge base.
Uses pdfplumber (lightweight, pure Python) for layout-aware text extraction.
Falls back to pypdf if pdfplumber is unavailable.

Chunking strategy:
  - Split at section headings (bold/large text or lines matching heading patterns).
  - Max chunk size ~400 tokens, with 50-token overlap.
  - Each chunk carries: source, section, page, chunk_id, text.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

MAX_CHUNK_CHARS = 1600   # ~400 tokens at 4 chars/token
OVERLAP_CHARS   = 200    # ~50 tokens overlap


def parse_pdf(pdf_path: str) -> List[Dict]:
    """
    Parse a PDF and return a list of chunk dicts:
    {chunk_id, source, section, page, text}

    Corrupt / HTML-masquerading downloads are skipped with a warning
    (common when wget saves an error page as .pdf).
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        logger.error("PDF not found: %s", pdf_path)
        return []

    if not _looks_like_pdf(pdf_path):
        logger.warning(
            "Skipping %s — not a valid PDF (likely an HTML/error page saved as .pdf). "
            "Re-download it or remove it from the pdfs/ folder.",
            pdf_path.name,
        )
        return []

    try:
        text_by_page = _extract_text(pdf_path)
    except Exception as exc:
        logger.warning("Skipping %s — parse failed: %s", pdf_path.name, exc)
        return []

    if not text_by_page or all(not (t or "").strip() for t in text_by_page):
        logger.warning("Skipping %s — no extractable text.", pdf_path.name)
        return []

    full_text = "\n".join(f"[PAGE {i+1}]\n{t}" for i, t in enumerate(text_by_page))
    chunks = _chunk_text(full_text, str(pdf_path))
    logger.info("Parsed %s: %d chunks from %d pages.", pdf_path.name, len(chunks), len(text_by_page))
    return chunks


def parse_markdown(md_path: str) -> List[Dict]:
    """Parse the clinical_spec.md as structured chunks."""
    md_path = Path(md_path)
    if not md_path.exists():
        logger.error("Markdown not found: %s", md_path)
        return []

    text = md_path.read_text(encoding="utf-8")
    sections = re.split(r"(?m)^#{1,3} ", text)
    chunks = []
    chunk_id = 0
    source = md_path.stem
    for sec in sections:
        if not sec.strip():
            continue
        lines = sec.strip().splitlines()
        section_title = lines[0].strip() if lines else "General"
        body = "\n".join(lines[1:]).strip()
        for sub_chunk in _split_into_chunks(body):
            chunks.append({
                "chunk_id": f"{source}_{chunk_id}",
                "source": source,
                "section": section_title,
                "page": None,
                "text": sub_chunk.strip(),
            })
            chunk_id += 1
    logger.info("Parsed %s: %d chunks.", md_path.name, len(chunks))
    return chunks


def parse_all(pdf_dir: str, spec_path: Optional[str] = None) -> List[Dict]:
    """Parse all PDFs in pdf_dir plus the clinical spec markdown."""
    chunks = []
    pdf_dir = Path(pdf_dir)
    for pdf_file in sorted(pdf_dir.glob("*.pdf")):
        chunks.extend(parse_pdf(str(pdf_file)))

    if spec_path:
        chunks.extend(parse_markdown(spec_path))

    if not chunks:
        logger.warning("No PDFs found in %s and no spec provided. Knowledge base will be minimal.", pdf_dir)
        chunks = _build_fallback_chunks()

    logger.info("Total chunks for knowledge base: %d", len(chunks))
    return chunks


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _looks_like_pdf(pdf_path: Path) -> bool:
    """True if the file starts with the PDF magic bytes (%PDF)."""
    try:
        with open(pdf_path, "rb") as f:
            header = f.read(8)
        return header.startswith(b"%PDF")
    except OSError:
        return False


def _extract_text(pdf_path: Path) -> List[str]:
    """Extract per-page text. Tries pdfplumber, then pypdf. Raises on total failure."""
    last_err: Optional[Exception] = None

    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                pages.append(page.extract_text() or "")
        return pages
    except ImportError:
        pass
    except Exception as exc:
        last_err = exc
        logger.debug("pdfplumber failed on %s: %s — trying pypdf.", pdf_path.name, exc)

    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        return [page.extract_text() or "" for page in reader.pages]
    except ImportError:
        pass
    except Exception as exc:
        last_err = exc

    if last_err is not None:
        raise last_err

    raise RuntimeError("Neither pdfplumber nor pypdf available. Cannot parse PDF.")


def _chunk_text(text: str, source: str) -> List[Dict]:
    """Chunk a long text string into overlapping segments."""
    # Split on page markers first
    page_sections = re.split(r"\[PAGE (\d+)\]", text)
    chunks = []
    chunk_id = 0
    current_page = 1

    for part in page_sections:
        m = re.fullmatch(r"\d+", part.strip())
        if m:
            current_page = int(m.group())
            continue
        if not part.strip():
            continue
        for sub in _split_into_chunks(part):
            if sub.strip():
                chunks.append({
                    "chunk_id": f"{Path(source).stem}_{chunk_id}",
                    "source": Path(source).stem,
                    "section": _extract_section(sub),
                    "page": current_page,
                    "text": sub.strip(),
                })
                chunk_id += 1
    return chunks


def _split_into_chunks(text: str) -> List[str]:
    """Split text into MAX_CHUNK_CHARS segments with OVERLAP_CHARS overlap."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + MAX_CHUNK_CHARS
        # Try to break at sentence boundary
        if end < len(text):
            break_point = text.rfind(". ", start, end)
            if break_point == -1:
                break_point = text.rfind("\n", start, end)
            if break_point != -1:
                end = break_point + 1
        chunks.append(text[start:end])
        start = end - OVERLAP_CHARS
    return chunks


def _extract_section(text: str) -> str:
    """Guess section heading from first line."""
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    if len(first_line) < 80 and re.match(r"^[A-Z\d]", first_line):
        return first_line
    return "General"


def _build_fallback_chunks() -> List[Dict]:
    """Return hard-coded essential knowledge chunks when no PDFs are available."""
    texts = [
        {
            "chunk_id": "fallback_0",
            "source": "fallback",
            "section": "bSSFP Banding Artifacts",
            "page": None,
            "text": (
                "In balanced steady-state free precession (bSSFP) MRI, dark banding artifacts "
                "appear at off-resonance frequencies equal to ±1/(2·TR) from the transmitter. "
                "At 3.0T, the wider B0 inhomogeneity means these bands are more likely to "
                "overlap the myocardium compared to 1.5T. A frequency scout acquisition presents "
                "the same anatomical slice at multiple RF offsets so the radiologist can select "
                "the offset that pushes bands away from the heart."
            ),
        },
        {
            "chunk_id": "fallback_1",
            "source": "fallback",
            "section": "Optimal Offset Selection",
            "page": None,
            "text": (
                "The optimal frequency offset is that at which no dark banding artifact overlaps "
                "the left ventricular myocardium or blood pool. Myocardial signal uniformity "
                "(low coefficient of variation, CoV) and high blood-myocardium contrast-to-noise "
                "ratio (CNR) are the primary quantitative markers of a good offset. "
                "There is often a plateau of 1–4 consecutive acceptable offsets; the central "
                "offset of this plateau is the most robust choice."
            ),
        },
        {
            "chunk_id": "fallback_2",
            "source": "fallback",
            "section": "Banding Location and Severity",
            "page": None,
            "text": (
                "Banding severity is graded as: none (no detectable dark stripe), mild "
                "(faint band outside the cardiac silhouette), moderate (band touching or "
                "grazing the epicardial border), or severe (band transecting the myocardium "
                "or blood pool). An offset is clinically rejected if banding severity is "
                "moderate or severe within the myocardial region."
            ),
        },
        {
            "chunk_id": "fallback_3",
            "source": "fallback",
            "section": "SCMR Quality Standards",
            "page": None,
            "text": (
                "The Society for Cardiovascular Magnetic Resonance (SCMR) recommends that "
                "bSSFP CINE acquisitions show clear delineation of the endocardial and "
                "epicardial borders, with no banding artifacts crossing the myocardium. "
                "At 3.0T, frequency scouting is a mandatory step in the protocol to ensure "
                "this quality standard is met before acquiring the diagnostic CINE series."
            ),
        },
        {
            "chunk_id": "fallback_4",
            "source": "fallback",
            "section": "Frequency Scouting Protocol",
            "page": None,
            "text": (
                "Standard frequency scout: 13 images acquired at offsets –150, –125, –100, "
                "–75, –50, –25, 0, +25, +50, +75, +100, +125, +150 Hz relative to the "
                "nominal water Larmor frequency. Some systems use 7 images at 50 Hz steps. "
                "The acquisition takes only a few seconds. The radiologist or MRI technologist "
                "reviews the series and selects the best offset visually, which is then set "
                "as the transmitter frequency for the diagnostic CINE acquisition."
            ),
        },
        {
            "chunk_id": "fallback_5",
            "source": "fallback",
            "section": "Myocardial Quality Metrics",
            "page": None,
            "text": (
                "Quantitative assessment of frequency-scout image quality uses: "
                "(1) Coefficient of Variation (CoV) of myocardial pixel intensities — lower is "
                "better, values < 0.15 indicate excellent uniformity; "
                "(2) Blood-myocardium CNR = (mean_blood − mean_myo) / sigma_background — higher "
                "is better, CNR > 10 is excellent for bSSFP at 3T; "
                "(3) SNR of myocardium = mean_myo / sigma_background. "
                "Banding artifacts cause CoV to increase and CNR to decrease in affected slices."
            ),
        },
    ]
    return texts
