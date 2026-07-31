"""SEC EDGAR filing downloader and XBRL data client.

Standalone helper (not a full DataProvider) for fetching SEC filings
(10-K, 10-Q, 8-K) by ticker symbol. Supports XBRL financial concept
extraction via edgartools (optional) with sec-edgar-downloader fallback.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quant_tools.errors import ConfigError, DataError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Filing:
    """A single SEC filing."""

    ticker: str
    filing_type: str
    filename: str
    content: str


class SecEdgarClient:
    """Download SEC filings via sec-edgar-downloader.

    Args:
        email: Email for SEC fair-use policy. Loaded from config if not provided.
    """

    def __init__(self, email: str | None = None) -> None:
        if email is None:
            from quant_tools.config import cfg
            email = cfg.sec_edgar_email

        if not email:
            raise ConfigError(
                "SEC_EDGAR_EMAIL is required. Set it in .env or pass email= argument."
            )
        self._email = email

    def fetch_filings(
        self,
        ticker: str,
        filing_type: str = "10-K",
        num_filings: int = 5,
    ) -> list[Filing]:
        """Download recent SEC filings for a ticker.

        Args:
            ticker: Stock ticker symbol (e.g. "AAPL").
            filing_type: Filing type — "10-K", "10-Q", or "8-K".
            num_filings: Number of recent filings to fetch.

        Returns:
            List of Filing dataclass instances with content.

        Raises:
            DataError: If no filings are found or download fails.
            ConfigError: If email is not configured.
        """
        valid_types = {"10-K", "10-Q", "8-K"}
        if filing_type not in valid_types:
            raise DataError(f"Invalid filing type '{filing_type}'. Must be one of {valid_types}")

        try:
            from sec_edgar_downloader import Downloader

            with tempfile.TemporaryDirectory() as tmpdir:
                # The signature is (company_name, email_address, download_folder).
                # Passing two arguments sent the temp directory as the SEC
                # user-agent email and left download_folder unset, so the filings
                # landed in whatever directory the user happened to run from
                # while the code kept looking inside tmpdir and reported that
                # nothing was found.
                dl = Downloader("quant-tools", self._email, tmpdir)
                dl.get(filing_type, ticker, limit=num_filings)

                filings_dir = Path(tmpdir) / "sec-edgar-filings" / ticker / filing_type
                if not filings_dir.exists():
                    raise DataError(f"No {filing_type} filings found for {ticker}")

                results: list[Filing] = []
                for filing_dir in sorted(filings_dir.iterdir()):
                    if not filing_dir.is_dir():
                        continue
                    for f in filing_dir.iterdir():
                        if f.suffix in (".txt", ".htm", ".html"):
                            content = f.read_text(encoding="utf-8", errors="replace")
                            results.append(Filing(
                                ticker=ticker,
                                filing_type=filing_type,
                                filename=f.name,
                                content=content,
                            ))
                            break  # One file per filing directory

                if not results:
                    raise DataError(f"No {filing_type} filings found for {ticker}")

                logger.info("Fetched %d %s filings for %s", len(results), filing_type, ticker)
                return results

        except (ConfigError, DataError):
            raise
        except Exception as exc:
            logger.error("SEC EDGAR download failed for %s: %s", ticker, exc)
            raise DataError(f"Failed to download {filing_type} for {ticker}: {exc}") from exc

    def fetch_xbrl(
        self,
        ticker: str,
        concept: str = "Revenue",
    ) -> pd.DataFrame:
        """Fetch XBRL financial concept data from SEC EDGAR.

        Uses edgartools for structured XBRL parsing. Falls back to a simple
        SEC EDGAR company facts API call if edgartools is not installed.

        Common concepts:
            Revenue, NetIncomeLoss, Assets, Liabilities,
            StockholdersEquity, EarningsPerShareBasic,
            OperatingIncomeLoss, CashAndCashEquivalentsAtCarryingValue

        Args:
            ticker: Stock ticker symbol (e.g. "AAPL").
            concept: XBRL concept name (e.g. "Revenue", "NetIncomeLoss").

        Returns:
            DataFrame with columns: end_date, value, form, fiscal_year, fiscal_period.

        Raises:
            DataError: If no XBRL data found or concept is invalid.
        """
        # Map common shorthand to full US-GAAP taxonomy names
        concept_map = {
            "Revenue": "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues": "us-gaap:Revenues",
            "NetIncomeLoss": "us-gaap:NetIncomeLoss",
            "Assets": "us-gaap:Assets",
            "Liabilities": "us-gaap:Liabilities",
            "StockholdersEquity": "us-gaap:StockholdersEquity",
            "EarningsPerShareBasic": "us-gaap:EarningsPerShareBasic",
            "OperatingIncomeLoss": "us-gaap:OperatingIncomeLoss",
            "CashAndCashEquivalentsAtCarryingValue": (
                "us-gaap:CashAndCashEquivalentsAtCarryingValue"
            ),
        }

        if self._has_edgartools():
            return self._fetch_xbrl_edgartools(ticker, concept, concept_map)
        return self._fetch_xbrl_companyfacts(ticker, concept, concept_map)

    def _fetch_xbrl_edgartools(
        self,
        ticker: str,
        concept: str,
        concept_map: dict[str, str],
    ) -> pd.DataFrame:
        """Fetch XBRL data using edgartools library.

        Args:
            ticker: Stock ticker symbol.
            concept: XBRL concept name.
            concept_map: Mapping of shorthand to full taxonomy names.

        Returns:
            DataFrame with XBRL data.

        Raises:
            DataError: If extraction fails.
        """
        try:
            from edgar import Company, set_identity

            set_identity(self._email)
            company = Company(ticker)

            # Try to get financials from most recent 10-K
            filings = company.get_filings(form="10-K").latest(1)
            if not filings:
                raise DataError(f"No 10-K filings found for {ticker} via edgartools")

            filing = filings[0] if hasattr(filings, '__getitem__') else filings
            xbrl = filing.xbrl()
            if xbrl is None:
                raise DataError(f"No XBRL data in latest 10-K for {ticker}")

            # Resolve concept name
            full_concept = concept_map.get(concept, concept)
            # Strip prefix for edgartools lookup
            concept_name = full_concept.split(":")[-1] if ":" in full_concept else full_concept

            # Try to find the concept in XBRL facts
            facts = xbrl.facts
            matching = [f for f in facts if concept_name.lower() in str(f).lower()]

            if not matching:
                raise DataError(
                    f"XBRL concept '{concept}' not found for {ticker} via edgartools"
                )

            rows = []
            for fact in matching:
                rows.append({
                    "end_date": str(getattr(fact, "end_date", "")),
                    "value": float(getattr(fact, "value", 0)),
                    "form": getattr(fact, "form", "10-K"),
                    "fiscal_year": getattr(fact, "fiscal_year", ""),
                    "fiscal_period": getattr(fact, "fiscal_period", ""),
                })

            if not rows:
                raise DataError(f"No XBRL values for '{concept}' in {ticker}")

            df = pd.DataFrame(rows).sort_values("end_date").reset_index(drop=True)
            logger.info("XBRL (edgartools): %s/%s — %d entries", ticker, concept, len(df))
            return df

        except DataError:
            raise
        except Exception as exc:
            logger.warning("edgartools XBRL failed for %s/%s: %s", ticker, concept, exc)
            # Fall back to company facts API
            return self._fetch_xbrl_companyfacts(ticker, concept, concept_map)

    def _fetch_xbrl_companyfacts(
        self,
        ticker: str,
        concept: str,
        concept_map: dict[str, str],
    ) -> pd.DataFrame:
        """Fetch XBRL data using SEC EDGAR company facts REST API.

        This is the fallback when edgartools is not installed.
        Uses the free SEC EDGAR API (no auth, just User-Agent header).

        Args:
            ticker: Stock ticker symbol.
            concept: XBRL concept name.
            concept_map: Mapping of shorthand to full taxonomy names.

        Returns:
            DataFrame with XBRL data.

        Raises:
            DataError: If no data found.
        """
        import httpx

        full_concept = concept_map.get(concept, concept)

        # Parse taxonomy and concept name
        if ":" in full_concept:
            taxonomy, concept_name = full_concept.split(":", 1)
            taxonomy = taxonomy.replace("-", "")  # "us-gaap" -> "usgaap"
        else:
            taxonomy = "usgaap"
            concept_name = full_concept

        # First, resolve ticker to CIK
        try:
            with httpx.Client(
                timeout=self._timeout if hasattr(self, "_timeout") else 30.0,
                headers={"User-Agent": f"quant-tools {self._email}"},
            ) as client:
                # Ticker -> CIK lookup
                tickers_url = "https://www.sec.gov/files/company_tickers.json"
                resp = client.get(tickers_url)
                resp.raise_for_status()
                tickers_data = resp.json()

                cik = None
                for entry in tickers_data.values():
                    if entry.get("ticker", "").upper() == ticker.upper():
                        cik = str(entry["cik_str"]).zfill(10)
                        break

                if not cik:
                    raise DataError(f"Could not resolve ticker '{ticker}' to CIK")

                # Fetch company concept
                concept_url = (
                    f"https://data.sec.gov/api/xbrl/companyconcept/"
                    f"CIK{cik}/{taxonomy}/{concept_name}.json"
                )
                resp = client.get(concept_url)
                resp.raise_for_status()
                data = resp.json()

            # Extract units (typically USD or shares)
            units = data.get("units", {})
            if not units:
                raise DataError(f"No XBRL units data for {ticker}/{concept}")

            rows = []
            for unit_type, entries in units.items():
                for entry in entries:
                    val = entry.get("val")
                    if val is None:
                        continue
                    rows.append({
                        "end_date": entry.get("end", ""),
                        "value": float(val),
                        "form": entry.get("form", ""),
                        "fiscal_year": entry.get("fy", ""),
                        "fiscal_period": entry.get("fp", ""),
                    })

            if not rows:
                raise DataError(f"No XBRL values for '{concept}' in {ticker}")

            df = pd.DataFrame(rows)
            # Deduplicate: keep only annual/quarterly (10-K, 10-Q)
            if "form" in df.columns:
                annual_quarterly = df[df["form"].isin(["10-K", "10-Q"])]
                if not annual_quarterly.empty:
                    df = annual_quarterly

            df = df.sort_values("end_date").drop_duplicates(
                subset=["end_date", "form"], keep="last"
            ).reset_index(drop=True)

            logger.info(
                "XBRL (SEC API): %s/%s — %d entries", ticker, concept, len(df),
            )
            return df

        except DataError:
            raise
        except httpx.TimeoutException as exc:
            raise DataError(f"SEC EDGAR API timeout for {ticker}/{concept}: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise DataError(
                    f"XBRL concept '{concept}' not found for {ticker}. "
                    f"Try a different concept name."
                ) from exc
            raise DataError(
                f"SEC EDGAR API error {exc.response.status_code} for {ticker}/{concept}"
            ) from exc
        except Exception as exc:
            raise DataError(
                f"XBRL fetch failed for {ticker}/{concept}: {exc}"
            ) from exc

    @staticmethod
    def _has_edgartools() -> bool:
        """Check if edgartools library is available.

        Actually importing it is the check. ``find_spec`` would call a
        half-installed package available, and edgartools has enough transitive
        dependencies for that to be a real outcome.

        Returns:
            True if edgartools is importable, False otherwise.
        """
        try:
            import edgar
        except ImportError:
            return False
        return edgar is not None
