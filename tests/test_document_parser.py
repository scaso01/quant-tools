"""Tests for document parser (Docling).

All external APIs are mocked. No real file parsing.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from quant_tools.errors import DataError


class TestDocumentParser:
    """Tests for DocumentParser."""

    def _make_parser(self):
        """Create a DocumentParser with a mocked converter injected."""
        mock_dc_module = MagicMock()
        mock_converter = MagicMock()
        mock_dc_module.DocumentConverter.return_value = mock_converter

        with patch.dict("sys.modules", {
            "docling": MagicMock(),
            "docling.document_converter": mock_dc_module,
        }):
            # Import fresh inside the patched context
            from quant_tools.data.document_parser import DocumentParser
            parser = DocumentParser()
        return parser, mock_converter

    def test_parse_pdf_success(self, tmp_path):
        parser, mock_converter = self._make_parser()

        pdf_file = tmp_path / "report.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake content")

        mock_result = MagicMock()
        mock_result.document.export_to_markdown.return_value = "# Annual Report\n\nContent here."
        mock_converter.convert.return_value = mock_result

        result = parser.parse_pdf(str(pdf_file))

        assert "Annual Report" in result
        mock_converter.convert.assert_called_once_with(str(pdf_file))

    def test_parse_docx_success(self, tmp_path):
        parser, mock_converter = self._make_parser()

        docx_file = tmp_path / "document.docx"
        docx_file.write_bytes(b"PK fake docx")

        mock_result = MagicMock()
        mock_result.document.export_to_markdown.return_value = "# Meeting Notes\n\nDetails."
        mock_converter.convert.return_value = mock_result

        result = parser.parse_docx(str(docx_file))

        assert "Meeting Notes" in result

    def test_parse_pdf_file_not_found(self):
        parser, _ = self._make_parser()

        with pytest.raises(DataError, match="File not found"):
            parser.parse_pdf("/nonexistent/file.pdf")

    def test_parse_pdf_wrong_extension(self, tmp_path):
        parser, _ = self._make_parser()

        txt_file = tmp_path / "report.txt"
        txt_file.write_text("plain text")

        with pytest.raises(DataError, match="Expected .pdf"):
            parser.parse_pdf(str(txt_file))

    def test_parse_docx_wrong_extension(self, tmp_path):
        parser, _ = self._make_parser()

        pdf_file = tmp_path / "report.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        with pytest.raises(DataError, match="Expected .docx"):
            parser.parse_docx(str(pdf_file))

    def test_parse_generic_any_extension(self, tmp_path):
        parser, mock_converter = self._make_parser()

        html_file = tmp_path / "filing.html"
        html_file.write_text("<html>content</html>")

        mock_result = MagicMock()
        mock_result.document.export_to_markdown.return_value = "# Filing\n\nParsed content."
        mock_converter.convert.return_value = mock_result

        result = parser.parse(str(html_file))

        assert "Filing" in result

    def test_parse_converter_exception(self, tmp_path):
        parser, mock_converter = self._make_parser()

        pdf_file = tmp_path / "corrupt.pdf"
        pdf_file.write_bytes(b"corrupt data")

        mock_converter.convert.side_effect = RuntimeError("parse error")

        with pytest.raises(DataError, match="Failed to parse"):
            parser.parse_pdf(str(pdf_file))

    def test_import_error_without_docling(self):
        """When docling is not installed, ImportError should be raised."""
        with patch.dict("sys.modules", {
            "docling": None,
            "docling.document_converter": None,
        }):
            from quant_tools.data.document_parser import DocumentParser
            with pytest.raises(ImportError, match="docling is required"):
                DocumentParser()
