"""Document parsing via Docling.

Converts PDF/DOCX files to markdown text. Useful for parsing
SEC EDGAR filings and other financial documents.

Requires: pip install quant-tools[docs]
"""

from __future__ import annotations

import logging
from pathlib import Path

from quant_tools.errors import DataError

logger = logging.getLogger(__name__)


class DocumentParser:
    """Parse PDF and DOCX files to markdown text via Docling.

    Raises ImportError if docling is not installed.
    """

    def __init__(self) -> None:
        try:
            from docling.document_converter import DocumentConverter
            self._converter = DocumentConverter()
        except ImportError:
            raise ImportError(
                "docling is required for document parsing. "
                "Install with: pip install quant-tools[docs]"
            )

    def parse_pdf(self, path: str | Path) -> str:
        """Parse a PDF file and return markdown text.

        Args:
            path: Path to the PDF file.

        Returns:
            Markdown-formatted text content.

        Raises:
            DataError: If the file doesn't exist or parsing fails.
        """
        return self._parse(path, expected_suffix=".pdf")

    def parse_docx(self, path: str | Path) -> str:
        """Parse a DOCX file and return markdown text.

        Args:
            path: Path to the DOCX file.

        Returns:
            Markdown-formatted text content.

        Raises:
            DataError: If the file doesn't exist or parsing fails.
        """
        return self._parse(path, expected_suffix=".docx")

    def parse(self, path: str | Path) -> str:
        """Parse any supported file and return markdown text.

        Args:
            path: Path to the document file.

        Returns:
            Markdown-formatted text content.

        Raises:
            DataError: If the file doesn't exist or parsing fails.
        """
        return self._parse(path)

    def _parse(self, path: str | Path, expected_suffix: str | None = None) -> str:
        """Internal parse method.

        Args:
            path: Path to the document.
            expected_suffix: Expected file extension (e.g. ".pdf"). None to accept any.

        Returns:
            Markdown text.

        Raises:
            DataError: On file-not-found or parse failure.
        """
        filepath = Path(path)
        if not filepath.exists():
            raise DataError(f"File not found: {filepath}")

        if expected_suffix and filepath.suffix.lower() != expected_suffix:
            raise DataError(
                f"Expected {expected_suffix} file, got {filepath.suffix}: {filepath}"
            )

        try:
            result = self._converter.convert(str(filepath))
            markdown = result.document.export_to_markdown()
            logger.info("Parsed %s (%d chars)", filepath.name, len(markdown))
            return markdown
        except Exception as exc:
            raise DataError(f"Failed to parse {filepath.name}: {exc}") from exc
