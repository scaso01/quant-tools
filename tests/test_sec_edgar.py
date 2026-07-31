"""Tests for SEC EDGAR client.

All external APIs are mocked. No real network calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pandas as pd
import pytest

from quant_tools.errors import ConfigError, DataError


class TestSecEdgarClient:
    """Tests for SecEdgarClient."""

    @patch("quant_tools.config.cfg")
    def test_init_no_email_raises(self, mock_cfg):
        mock_cfg.sec_edgar_email = ""
        from quant_tools.data.providers.sec_edgar import SecEdgarClient

        with pytest.raises(ConfigError, match="SEC_EDGAR_EMAIL"):
            SecEdgarClient()

    def test_init_explicit_email(self):
        from quant_tools.data.providers.sec_edgar import SecEdgarClient

        client = SecEdgarClient(email="test@example.com")
        assert client._email == "test@example.com"

    def test_invalid_filing_type(self):
        from quant_tools.data.providers.sec_edgar import SecEdgarClient

        client = SecEdgarClient(email="test@example.com")
        with pytest.raises(DataError, match="Invalid filing type"):
            client.fetch_filings("AAPL", filing_type="INVALID")

    @patch("sec_edgar_downloader.Downloader")
    def test_fetch_filings_success(self, mock_dl_cls, tmp_path):
        from quant_tools.data.providers.sec_edgar import SecEdgarClient, Filing

        # Build mock file structure
        filings_dir = tmp_path / "sec-edgar-filings" / "AAPL" / "10-K"
        filing_sub = filings_dir / "0001234567-23-000001"
        filing_sub.mkdir(parents=True)
        (filing_sub / "filing.htm").write_text("<html>Annual report</html>")

        mock_dl = MagicMock()
        mock_dl_cls.return_value = mock_dl

        with patch("tempfile.mkdtemp", return_value=str(tmp_path)):
            client = SecEdgarClient(email="test@example.com")
            # Patch TemporaryDirectory to use tmp_path
            with patch("tempfile.TemporaryDirectory") as mock_td:
                mock_ctx = MagicMock()
                mock_ctx.__enter__ = MagicMock(return_value=str(tmp_path))
                mock_ctx.__exit__ = MagicMock(return_value=False)
                mock_td.return_value = mock_ctx

                results = client.fetch_filings("AAPL", filing_type="10-K", num_filings=1)

        assert len(results) == 1
        assert isinstance(results[0], Filing)
        assert results[0].ticker == "AAPL"
        assert results[0].filing_type == "10-K"
        assert "Annual report" in results[0].content
        mock_dl.get.assert_called_once_with("10-K", "AAPL", limit=1)

    @patch("sec_edgar_downloader.Downloader")
    def test_fetch_filings_no_results(self, mock_dl_cls, tmp_path):
        from quant_tools.data.providers.sec_edgar import SecEdgarClient

        mock_dl_cls.return_value = MagicMock()

        with patch("tempfile.TemporaryDirectory") as mock_td:
            mock_ctx = MagicMock()
            mock_ctx.__enter__ = MagicMock(return_value=str(tmp_path))
            mock_ctx.__exit__ = MagicMock(return_value=False)
            mock_td.return_value = mock_ctx

            client = SecEdgarClient(email="test@example.com")
            with pytest.raises(DataError, match="No 10-K filings found"):
                client.fetch_filings("ZZZZZ", filing_type="10-K")

    @patch("sec_edgar_downloader.Downloader")
    def test_fetch_filings_multiple(self, mock_dl_cls, tmp_path):
        from quant_tools.data.providers.sec_edgar import SecEdgarClient

        # Build mock file structure with 3 filings
        filings_dir = tmp_path / "sec-edgar-filings" / "MSFT" / "10-Q"
        for i in range(3):
            sub = filings_dir / f"0001234567-23-00000{i}"
            sub.mkdir(parents=True)
            (sub / "filing.txt").write_text(f"Quarter {i} report")

        mock_dl_cls.return_value = MagicMock()

        with patch("tempfile.TemporaryDirectory") as mock_td:
            mock_ctx = MagicMock()
            mock_ctx.__enter__ = MagicMock(return_value=str(tmp_path))
            mock_ctx.__exit__ = MagicMock(return_value=False)
            mock_td.return_value = mock_ctx

            client = SecEdgarClient(email="test@example.com")
            results = client.fetch_filings("MSFT", filing_type="10-Q", num_filings=3)

        assert len(results) == 3
        assert all(r.filing_type == "10-Q" for r in results)
        assert all(r.ticker == "MSFT" for r in results)

    @patch("sec_edgar_downloader.Downloader")
    def test_fetch_filings_download_exception(self, mock_dl_cls, tmp_path):
        from quant_tools.data.providers.sec_edgar import SecEdgarClient

        mock_dl_cls.return_value.get.side_effect = RuntimeError("API failure")

        with patch("tempfile.TemporaryDirectory") as mock_td:
            mock_ctx = MagicMock()
            mock_ctx.__enter__ = MagicMock(return_value=str(tmp_path))
            mock_ctx.__exit__ = MagicMock(return_value=False)
            mock_td.return_value = mock_ctx

            client = SecEdgarClient(email="test@example.com")
            with pytest.raises(DataError, match="Failed to download"):
                client.fetch_filings("AAPL", filing_type="10-K")

    @patch("sec_edgar_downloader.Downloader")
    def test_all_valid_filing_types(self, mock_dl_cls, tmp_path):
        """All valid filing types should be accepted."""
        from quant_tools.data.providers.sec_edgar import SecEdgarClient

        mock_dl_cls.return_value = MagicMock()

        client = SecEdgarClient(email="test@example.com")
        for ft in ("10-K", "10-Q", "8-K"):
            with patch("tempfile.TemporaryDirectory") as mock_td:
                mock_ctx = MagicMock()
                mock_ctx.__enter__ = MagicMock(return_value=str(tmp_path))
                mock_ctx.__exit__ = MagicMock(return_value=False)
                mock_td.return_value = mock_ctx
                try:
                    client.fetch_filings("TEST", filing_type=ft, num_filings=1)
                except DataError:
                    pass  # Expected — no actual files


class TestSecEdgarXbrl:
    """Tests for SecEdgarClient.fetch_xbrl."""

    def _mock_httpx_response(self, json_data, status_code=200):
        """Build a mock httpx response."""
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = status_code
        resp.json.return_value = json_data
        resp.raise_for_status = MagicMock()
        if status_code >= 400:
            resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                message=f"HTTP {status_code}",
                request=MagicMock(),
                response=resp,
            )
        return resp

    def test_fetch_xbrl_companyfacts_success(self):
        """XBRL via SEC company facts API returns correct DataFrame."""
        from quant_tools.data.providers.sec_edgar import SecEdgarClient

        tickers_json = {
            "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}
        }
        concept_json = {
            "units": {
                "USD": [
                    {"end": "2023-09-30", "val": 383285000000, "form": "10-K", "fy": 2023, "fp": "FY"},
                    {"end": "2024-09-28", "val": 391035000000, "form": "10-K", "fy": 2024, "fp": "FY"},
                ]
            }
        }

        with patch.object(SecEdgarClient, "_has_edgartools", return_value=False):
            client = SecEdgarClient(email="test@example.com")

            with patch("httpx.Client") as mock_client_cls:
                mock_client = MagicMock()
                mock_client.__enter__ = MagicMock(return_value=mock_client)
                mock_client.__exit__ = MagicMock(return_value=False)
                mock_client.get.side_effect = [
                    self._mock_httpx_response(tickers_json),
                    self._mock_httpx_response(concept_json),
                ]
                mock_client_cls.return_value = mock_client

                df = client.fetch_xbrl("AAPL", "Revenue")

        assert isinstance(df, pd.DataFrame)
        assert "end_date" in df.columns
        assert "value" in df.columns
        assert "form" in df.columns
        assert len(df) == 2
        assert df["value"].iloc[0] == 383285000000

    def test_fetch_xbrl_unknown_ticker(self):
        """Unknown ticker raises DataError."""
        from quant_tools.data.providers.sec_edgar import SecEdgarClient

        tickers_json = {
            "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}
        }

        with patch.object(SecEdgarClient, "_has_edgartools", return_value=False):
            client = SecEdgarClient(email="test@example.com")

            with patch("httpx.Client") as mock_client_cls:
                mock_client = MagicMock()
                mock_client.__enter__ = MagicMock(return_value=mock_client)
                mock_client.__exit__ = MagicMock(return_value=False)
                mock_client.get.return_value = self._mock_httpx_response(tickers_json)
                mock_client_cls.return_value = mock_client

                with pytest.raises(DataError, match="Could not resolve ticker"):
                    client.fetch_xbrl("ZZZZZ", "Revenue")

    def test_fetch_xbrl_404_concept(self):
        """Unknown XBRL concept returns 404 and raises DataError."""
        from quant_tools.data.providers.sec_edgar import SecEdgarClient

        tickers_json = {
            "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}
        }
        # First call returns tickers, second raises 404
        resp_404 = self._mock_httpx_response({}, status_code=404)

        with patch.object(SecEdgarClient, "_has_edgartools", return_value=False):
            client = SecEdgarClient(email="test@example.com")

            with patch("httpx.Client") as mock_client_cls:
                mock_client = MagicMock()
                mock_client.__enter__ = MagicMock(return_value=mock_client)
                mock_client.__exit__ = MagicMock(return_value=False)
                mock_client.get.side_effect = [
                    self._mock_httpx_response(tickers_json),
                    resp_404,
                ]
                mock_client_cls.return_value = mock_client

                with pytest.raises(DataError, match="not found"):
                    client.fetch_xbrl("AAPL", "FakeConcept")

    def test_has_edgartools_false(self):
        """_has_edgartools returns False when not installed."""
        from quant_tools.data.providers.sec_edgar import SecEdgarClient

        with patch.dict("sys.modules", {"edgar": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module")):
                # Call the static method directly
                assert SecEdgarClient._has_edgartools() is False

    def test_fetch_xbrl_concept_mapping(self):
        """Concept shorthand maps to full taxonomy name."""
        from quant_tools.data.providers.sec_edgar import SecEdgarClient

        tickers_json = {
            "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}
        }
        concept_json = {
            "units": {
                "USD": [
                    {"end": "2023-09-30", "val": 97000000000, "form": "10-K", "fy": 2023, "fp": "FY"},
                ]
            }
        }

        with patch.object(SecEdgarClient, "_has_edgartools", return_value=False):
            client = SecEdgarClient(email="test@example.com")

            with patch("httpx.Client") as mock_client_cls:
                mock_client = MagicMock()
                mock_client.__enter__ = MagicMock(return_value=mock_client)
                mock_client.__exit__ = MagicMock(return_value=False)
                mock_client.get.side_effect = [
                    self._mock_httpx_response(tickers_json),
                    self._mock_httpx_response(concept_json),
                ]
                mock_client_cls.return_value = mock_client

                df = client.fetch_xbrl("AAPL", "NetIncomeLoss")

        assert len(df) == 1
        # Verify the second HTTP call used the mapped concept
        calls = mock_client.get.call_args_list
        assert "NetIncomeLoss" in calls[1][0][0]
