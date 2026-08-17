"""
Build the knowledge base from PDFs + clinical spec.
Run once before using the pipeline.

Usage:
    python -m freqscout_agent.knowledge.build_kb
    # OR
    python freqscout_agent/knowledge/build_kb.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

SPEC_PATH = Path(__file__).parent / "clinical_spec.md"
PDF_DIR   = Path(__file__).parent / "pdfs"
INDEX_DIR = Path(__file__).parent / "index_store"


def main():
    from freqscout_agent.knowledge.parse_pdfs import parse_all
    from freqscout_agent.knowledge.index import KnowledgeBaseBuilder

    logger.info("Parsing clinical spec and PDFs…")
    chunks = parse_all(str(PDF_DIR), str(SPEC_PATH))

    logger.info("Building index…")
    builder = KnowledgeBaseBuilder(index_dir=str(INDEX_DIR))
    builder.build(chunks, force="--force" in sys.argv)

    logger.info("Done. Knowledge base ready at %s.", INDEX_DIR)


if __name__ == "__main__":
    main()
