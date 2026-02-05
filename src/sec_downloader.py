"""
SEC EDGAR Data Acquisition Module

Downloads 10-K and 10-Q filings from SEC EDGAR for a given company
within a 2-year lookback window from a user-provided reference date.
"""

import os
import re
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional, Tuple
from sec_edgar_downloader import Downloader

from .config import sec_config, FILINGS_DIR

logger = logging.getLogger(__name__)


@dataclass
class FilingMetadata:
    """Metadata for a downloaded SEC filing."""
    ticker: str
    cik: str
    filing_type: str
    filing_date: str
    fiscal_year: str
    fiscal_quarter: Optional[str]
    file_path: Path
    accession_number: str


class SECDownloader:
    """
    Downloads SEC EDGAR filings for a given company.
    
    Supports both ticker symbols (e.g., "AAPL") and CIK numbers (e.g., "0000320193").
    Complies with SEC fair access policies by setting proper user-agent headers.
    """
    
    def __init__(self, download_dir: Optional[Path] = None):
        """
        Initialize the SEC downloader.
        
        Args:
            download_dir: Directory to store downloaded filings. 
                         Defaults to data/filings.
        """
        self.download_dir = download_dir or FILINGS_DIR
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # Parse user agent for sec-edgar-downloader
        # Expected format: "CompanyName email@example.com"
        user_agent_parts = sec_config.user_agent.split()
        if len(user_agent_parts) >= 2:
            company_name = " ".join(user_agent_parts[:-1])
            email = user_agent_parts[-1]
        else:
            company_name = "FinanceAnalyzer"
            email = "contact@example.com"
        
        self.downloader = Downloader(company_name, email, str(self.download_dir))
        
    def calculate_date_range(
        self, 
        reference_date: datetime, 
        years_back: int = 2
    ) -> Tuple[datetime, datetime]:
        """
        Calculate the date range for filing lookback.
        
        Args:
            reference_date: The end date for the lookback window.
            years_back: Number of years to look back.
            
        Returns:
            Tuple of (start_date, end_date).
        """
        end_date = reference_date
        start_date = reference_date - timedelta(days=years_back * 365)
        return start_date, end_date
    
    def normalize_ticker(self, identifier: str) -> str:
        """
        Normalize ticker symbol or CIK number.
        
        Args:
            identifier: Ticker symbol (e.g., "AAPL") or CIK (e.g., "0000320193").
            
        Returns:
            Normalized identifier (uppercase ticker or padded CIK).
        """
        # Check if it's a CIK number
        if identifier.isdigit():
            # Pad CIK to 10 digits
            return identifier.zfill(10)
        # Otherwise treat as ticker
        return identifier.upper().strip()
    
    def download_filings(
        self,
        ticker: str,
        reference_date: datetime,
        filing_types: Optional[List[str]] = None,
        years_back: int = 2
    ) -> List[FilingMetadata]:
        """
        Download SEC filings for a company within the specified date range.
        
        Args:
            ticker: Company ticker symbol or CIK number.
            reference_date: End date for the lookback window.
            filing_types: List of filing types to download (default: ["10-K", "10-Q"]).
            years_back: Number of years to look back from reference_date.
            
        Returns:
            List of FilingMetadata objects for downloaded filings.
        """
        ticker = self.normalize_ticker(ticker)
        filing_types = filing_types or sec_config.filing_types
        start_date, end_date = self.calculate_date_range(reference_date, years_back)
        
        logger.info(
            f"Downloading filings for {ticker} from {start_date.date()} to {end_date.date()}"
        )
        
        downloaded_filings: List[FilingMetadata] = []
        
        for filing_type in filing_types:
            try:
                logger.info(f"Fetching {filing_type} filings for {ticker}")
                
                # Download filings using sec-edgar-downloader
                # The library handles rate limiting internally
                self.downloader.get(
                    filing_type,
                    ticker,
                    after=start_date.strftime("%Y-%m-%d"),
                    before=end_date.strftime("%Y-%m-%d"),
                    download_details=True
                )
                
                # Respect SEC rate limits
                time.sleep(sec_config.rate_limit_delay)
                
                # Find the downloaded files
                filings = self._find_downloaded_filings(ticker, filing_type)
                downloaded_filings.extend(filings)
                
            except Exception as e:
                logger.error(f"Error downloading {filing_type} for {ticker}: {e}")
                continue
        
        logger.info(f"Downloaded {len(downloaded_filings)} filings for {ticker}")
        return downloaded_filings
    
    def _find_downloaded_filings(
        self, 
        ticker: str, 
        filing_type: str
    ) -> List[FilingMetadata]:
        """
        Find downloaded filing files and extract metadata.
        
        Args:
            ticker: Company ticker symbol.
            filing_type: Type of filing (10-K or 10-Q).
            
        Returns:
            List of FilingMetadata for found filings.
        """
        filings = []
        
        # sec-edgar-downloader creates structure: download_dir/sec-edgar-filings/TICKER/FILING_TYPE/
        ticker_dir = self.download_dir / "sec-edgar-filings" / ticker / filing_type
        
        if not ticker_dir.exists():
            logger.warning(f"Filing directory not found: {ticker_dir}")
            return filings
        
        # Each filing is in a subdirectory named by accession number
        for accession_dir in ticker_dir.iterdir():
            if not accession_dir.is_dir():
                continue
                
            accession_number = accession_dir.name
            
            # Find the main filing document (usually full-submission.txt or primary-document.html)
            filing_file = self._find_main_document(accession_dir)
            if not filing_file:
                logger.warning(f"No main document found in {accession_dir}")
                continue
            
            # Extract metadata from the filing
            metadata = self._extract_filing_metadata(
                ticker=ticker,
                filing_type=filing_type,
                accession_number=accession_number,
                file_path=filing_file
            )
            
            if metadata:
                filings.append(metadata)
        
        return filings
    
    def _find_main_document(self, accession_dir: Path) -> Optional[Path]:
        """
        Find the main filing document in an accession directory.
        
        Args:
            accession_dir: Path to the accession number directory.
            
        Returns:
            Path to the main document, or None if not found.
        """
        # Priority order for finding the main document
        priority_patterns = [
            "primary-document.html",
            "primary-document.htm",
            "*10-k*.htm",
            "*10-q*.htm",
            "full-submission.txt",
        ]
        
        for pattern in priority_patterns:
            matches = list(accession_dir.glob(pattern))
            if matches:
                return matches[0]
        
        # Fallback: find any HTML file
        html_files = list(accession_dir.glob("*.htm*"))
        if html_files:
            return html_files[0]
        
        return None
    
    def _extract_filing_metadata(
        self,
        ticker: str,
        filing_type: str,
        accession_number: str,
        file_path: Path
    ) -> Optional[FilingMetadata]:
        """
        Extract metadata from a filing.
        
        Args:
            ticker: Company ticker symbol.
            filing_type: Type of filing.
            accession_number: SEC accession number.
            file_path: Path to the filing document.
            
        Returns:
            FilingMetadata object, or None if extraction fails.
        """
        try:
            # Parse accession number for date info
            # Format: 0000320193-24-000081 (CIK-YY-SEQUENCE)
            parts = accession_number.split("-")
            
            cik = parts[0] if parts else "unknown"
            
            # Try to extract year from accession number
            year_str = parts[1] if len(parts) > 1 else None
            if year_str and len(year_str) == 2:
                year = 2000 + int(year_str) if int(year_str) < 50 else 1900 + int(year_str)
            else:
                year = datetime.now().year
            
            # Determine fiscal quarter for 10-Q
            fiscal_quarter = None
            if filing_type == "10-Q":
                # Try to determine quarter from file content or name
                fiscal_quarter = self._determine_fiscal_quarter(file_path)
            
            # Get file modification time as approximate filing date
            filing_date = datetime.fromtimestamp(file_path.stat().st_mtime)
            
            return FilingMetadata(
                ticker=ticker,
                cik=cik,
                filing_type=filing_type,
                filing_date=filing_date.strftime("%Y-%m-%d"),
                fiscal_year=str(year),
                fiscal_quarter=fiscal_quarter,
                file_path=file_path,
                accession_number=accession_number
            )
            
        except Exception as e:
            logger.error(f"Error extracting metadata from {file_path}: {e}")
            return None
    
    def _determine_fiscal_quarter(self, file_path: Path) -> Optional[str]:
        """
        Attempt to determine the fiscal quarter from the filing.
        
        Args:
            file_path: Path to the filing document.
            
        Returns:
            Quarter string (Q1, Q2, Q3) or None.
        """
        try:
            # Read first portion of file to find quarter reference
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(50000)  # Read first 50KB
            
            # Look for quarter patterns
            quarter_patterns = [
                r'quarterly report.*?(?:first|1st|q1)',
                r'quarterly report.*?(?:second|2nd|q2)',
                r'quarterly report.*?(?:third|3rd|q3)',
                r'for the quarter.*?(?:march|april)',   # Q1 typically ends March
                r'for the quarter.*?(?:june|july)',     # Q2 typically ends June
                r'for the quarter.*?(?:september|october)',  # Q3 typically ends September
            ]
            
            content_lower = content.lower()
            
            if re.search(quarter_patterns[0], content_lower) or re.search(quarter_patterns[3], content_lower):
                return "Q1"
            elif re.search(quarter_patterns[1], content_lower) or re.search(quarter_patterns[4], content_lower):
                return "Q2"
            elif re.search(quarter_patterns[2], content_lower) or re.search(quarter_patterns[5], content_lower):
                return "Q3"
            
        except Exception as e:
            logger.warning(f"Could not determine fiscal quarter: {e}")
        
        return None
    
    def get_cached_filings(self, ticker: str) -> List[FilingMetadata]:
        """
        Get already downloaded filings for a ticker.
        
        Args:
            ticker: Company ticker symbol.
            
        Returns:
            List of FilingMetadata for cached filings.
        """
        ticker = self.normalize_ticker(ticker)
        filings = []
        
        for filing_type in sec_config.filing_types:
            filings.extend(self._find_downloaded_filings(ticker, filing_type))
        
        return filings


def download_filings(
    ticker: str,
    reference_date: datetime,
    filing_types: Optional[List[str]] = None
) -> List[FilingMetadata]:
    """
    Convenience function to download SEC filings.
    
    Args:
        ticker: Company ticker symbol or CIK number.
        reference_date: End date for the 2-year lookback window.
        filing_types: Optional list of filing types (default: ["10-K", "10-Q"]).
        
    Returns:
        List of FilingMetadata objects for downloaded filings.
    """
    downloader = SECDownloader()
    return downloader.download_filings(ticker, reference_date, filing_types)


if __name__ == "__main__":
    # Test mode
    import argparse
    
    parser = argparse.ArgumentParser(description="Download SEC filings")
    parser.add_argument("--ticker", required=True, help="Ticker symbol or CIK")
    parser.add_argument("--date", required=True, help="Reference date (YYYY-MM-DD)")
    parser.add_argument("--test", action="store_true", help="Test mode")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    ref_date = datetime.strptime(args.date, "%Y-%m-%d")
    filings = download_filings(args.ticker, ref_date)
    
    print(f"\nDownloaded {len(filings)} filings:")
    for f in filings:
        print(f"  - {f.filing_type} ({f.fiscal_year}): {f.file_path.name}")
