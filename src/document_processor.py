"""
Document Processing & Chunking Pipeline

Extracts text from SEC filing HTML documents and applies semantic chunking
with section awareness to preserve financial context and clause relationships.
"""

import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from bs4 import BeautifulSoup
import html2text

from .config import chunking_config
from .sec_downloader import FilingMetadata

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """A semantic chunk from a SEC filing document."""
    content: str
    ticker: str
    cik: str
    filing_type: str
    filing_date: str
    fiscal_year: str
    fiscal_quarter: Optional[str]
    section: str
    chunk_index: int
    accession_number: str
    
    def to_metadata(self) -> Dict:
        """Convert to metadata dictionary for ChromaDB."""
        return {
            "ticker": self.ticker,
            "cik": self.cik,
            "filing_type": self.filing_type,
            "filing_date": self.filing_date,
            "fiscal_year": self.fiscal_year,
            "fiscal_quarter": self.fiscal_quarter or "",
            "section": self.section,
            "chunk_index": self.chunk_index,
            "accession_number": self.accession_number,
        }
    
    def get_id(self) -> str:
        """Generate a unique ID for this chunk."""
        return f"{self.ticker}_{self.filing_type}_{self.fiscal_year}_{self.section}_{self.chunk_index}".replace(" ", "_")


class DocumentProcessor:
    """
    Processes SEC filing documents with semantic chunking.
    
    Extracts text from HTML filings and chunks by section (Item 1, Item 7, etc.)
    while preserving financial context and table structures.
    """
    
    # Pattern to identify SEC filing sections
    SECTION_PATTERNS = [
        # 10-K sections
        (r'item\s*1[.\s]*[-–—]?\s*business', 'Item 1 - Business'),
        (r'item\s*1a[.\s]*[-–—]?\s*risk\s*factors', 'Item 1A - Risk Factors'),
        (r'item\s*1b[.\s]*[-–—]?\s*unresolved', 'Item 1B - Unresolved Staff Comments'),
        (r'item\s*2[.\s]*[-–—]?\s*properties', 'Item 2 - Properties'),
        (r'item\s*3[.\s]*[-–—]?\s*legal', 'Item 3 - Legal Proceedings'),
        (r'item\s*4[.\s]*[-–—]?\s*mine', 'Item 4 - Mine Safety'),
        (r'item\s*5[.\s]*[-–—]?\s*market', 'Item 5 - Market Information'),
        (r'item\s*6[.\s]*[-–—]?\s*(?:selected|reserved)', 'Item 6 - Selected Financial Data'),
        (r'item\s*7[.\s]*[-–—]?\s*management', 'Item 7 - MD&A'),
        (r'item\s*7a[.\s]*[-–—]?\s*quantitative', 'Item 7A - Market Risk'),
        (r'item\s*8[.\s]*[-–—]?\s*financial\s*statements', 'Item 8 - Financial Statements'),
        (r'item\s*9[.\s]*[-–—]?\s*changes', 'Item 9 - Changes in Accountants'),
        (r'item\s*9a[.\s]*[-–—]?\s*controls', 'Item 9A - Controls and Procedures'),
        (r'item\s*9b[.\s]*[-–—]?\s*other', 'Item 9B - Other Information'),
        (r'item\s*10[.\s]*[-–—]?\s*directors', 'Item 10 - Directors'),
        (r'item\s*11[.\s]*[-–—]?\s*executive\s*compensation', 'Item 11 - Executive Compensation'),
        (r'item\s*12[.\s]*[-–—]?\s*security', 'Item 12 - Security Ownership'),
        (r'item\s*13[.\s]*[-–—]?\s*certain\s*relationships', 'Item 13 - Related Transactions'),
        (r'item\s*14[.\s]*[-–—]?\s*principal\s*account', 'Item 14 - Accountant Fees'),
        (r'item\s*15[.\s]*[-–—]?\s*exhibits', 'Item 15 - Exhibits'),
        # 10-Q specific
        (r'part\s*i[.\s]*[-–—]?\s*financial\s*information', 'Part I - Financial Information'),
        (r'part\s*ii[.\s]*[-–—]?\s*other\s*information', 'Part II - Other Information'),
    ]
    
    def __init__(self):
        """Initialize the document processor."""
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = True
        self.html_converter.ignore_images = True
        self.html_converter.ignore_emphasis = False
        self.html_converter.body_width = 0  # No line wrapping
    
    def process_filing(self, filing: FilingMetadata) -> List[DocumentChunk]:
        """
        Process a single SEC filing into semantic chunks.
        
        Args:
            filing: Metadata for the filing to process.
            
        Returns:
            List of DocumentChunk objects.
        """
        logger.info(f"Processing {filing.filing_type} for {filing.ticker} ({filing.fiscal_year})")
        
        # Read and parse the HTML file
        try:
            with open(filing.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
        except Exception as e:
            logger.error(f"Error reading filing {filing.file_path}: {e}")
            return []
        
        # Extract clean text
        clean_text = self._extract_text(html_content)
        
        # Split into sections
        sections = self._split_into_sections(clean_text)
        
        # Create chunks from sections
        chunks = []
        for section_name, section_text in sections.items():
            section_chunks = self._create_section_chunks(
                section_text, 
                section_name,
                filing
            )
            chunks.extend(section_chunks)
        
        logger.info(f"Created {len(chunks)} chunks from {filing.file_path.name}")
        return chunks
    
    def _extract_text(self, html_content: str) -> str:
        """
        Extract clean text from HTML content.
        
        Args:
            html_content: Raw HTML string.
            
        Returns:
            Clean text with preserved structure.
        """
        # Parse HTML
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Remove script and style elements
        for element in soup(['script', 'style', 'meta', 'link']):
            element.decompose()
        
        # Preserve table structure by adding markers
        for table in soup.find_all('table'):
            # Add table markers
            table.insert_before('[TABLE_START]')
            table.insert_after('[TABLE_END]')
        
        # Convert to markdown for better structure preservation
        text = self.html_converter.handle(str(soup))
        
        # Clean up the text
        text = self._clean_text(text)
        
        return text
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing artifacts.
        
        Args:
            text: Raw extracted text.
            
        Returns:
            Cleaned text.
        """
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)  # Standalone page numbers
        text = re.sub(r'Table of Contents', '', text, flags=re.IGNORECASE)
        
        # Remove common SEC filing artifacts
        text = re.sub(r'\*{3,}', '', text)  # Asterisk lines
        text = re.sub(r'-{5,}', '', text)   # Dash lines
        text = re.sub(r'_{5,}', '', text)   # Underscore lines
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        return text.strip()
    
    def _split_into_sections(self, text: str) -> Dict[str, str]:
        """
        Split document text into sections based on SEC filing structure.
        
        Args:
            text: Full document text.
            
        Returns:
            Dictionary mapping section names to content.
        """
        sections = {}
        text_lower = text.lower()
        
        # Find all section positions
        section_positions: List[Tuple[int, str]] = []
        
        for pattern, section_name in self.SECTION_PATTERNS:
            for match in re.finditer(pattern, text_lower):
                section_positions.append((match.start(), section_name))
        
        # Sort by position
        section_positions.sort(key=lambda x: x[0])
        
        if not section_positions:
            # No sections found - treat entire document as one section
            sections["Full Document"] = text
            return sections
        
        # Extract each section's content
        for i, (pos, name) in enumerate(section_positions):
            # Find the end position (start of next section or end of document)
            if i < len(section_positions) - 1:
                end_pos = section_positions[i + 1][0]
            else:
                end_pos = len(text)
            
            section_content = text[pos:end_pos].strip()
            
            # Skip very short sections (likely just headers)
            if len(section_content) > 100:
                sections[name] = section_content
        
        return sections
    
    def _create_section_chunks(
        self,
        section_text: str,
        section_name: str,
        filing: FilingMetadata
    ) -> List[DocumentChunk]:
        """
        Create semantic chunks from a section.
        
        Args:
            section_text: Text content of the section.
            section_name: Name of the section.
            filing: Filing metadata.
            
        Returns:
            List of DocumentChunk objects.
        """
        chunks = []
        
        # Determine minimum chunk length based on section type
        min_length = self._get_min_chunk_length(section_name)
        
        # Split section into paragraphs
        paragraphs = self._split_into_paragraphs(section_text)
        
        # Group paragraphs into chunks
        current_chunk = ""
        chunk_index = 0
        
        for para in paragraphs:
            # Check if adding this paragraph would create a good chunk
            if len(current_chunk) + len(para) < min_length:
                current_chunk += "\n\n" + para if current_chunk else para
            else:
                # Save current chunk if it meets minimum length
                if len(current_chunk) >= min_length:
                    chunk = DocumentChunk(
                        content=current_chunk.strip(),
                        ticker=filing.ticker,
                        cik=filing.cik,
                        filing_type=filing.filing_type,
                        filing_date=filing.filing_date,
                        fiscal_year=filing.fiscal_year,
                        fiscal_quarter=filing.fiscal_quarter,
                        section=section_name,
                        chunk_index=chunk_index,
                        accession_number=filing.accession_number,
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    current_chunk = para
                else:
                    current_chunk += "\n\n" + para
        
        # Don't forget the last chunk
        if current_chunk and len(current_chunk) >= min_length // 2:  # Allow last chunk to be shorter
            chunk = DocumentChunk(
                content=current_chunk.strip(),
                ticker=filing.ticker,
                cik=filing.cik,
                filing_type=filing.filing_type,
                filing_date=filing.filing_date,
                fiscal_year=filing.fiscal_year,
                fiscal_quarter=filing.fiscal_quarter,
                section=section_name,
                chunk_index=chunk_index,
                accession_number=filing.accession_number,
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs, preserving tables.
        
        Args:
            text: Section text.
            
        Returns:
            List of paragraph strings.
        """
        paragraphs = []
        
        # Split by double newlines, but keep tables together
        in_table = False
        current = ""
        
        for line in text.split('\n'):
            if '[TABLE_START]' in line:
                if current.strip():
                    paragraphs.append(current.strip())
                    current = ""
                in_table = True
                current = line
            elif '[TABLE_END]' in line:
                current += '\n' + line
                paragraphs.append(current.strip())
                current = ""
                in_table = False
            elif in_table:
                current += '\n' + line
            elif line.strip() == "":
                if current.strip():
                    paragraphs.append(current.strip())
                    current = ""
            else:
                current += '\n' + line if current else line
        
        if current.strip():
            paragraphs.append(current.strip())
        
        return paragraphs
    
    def _get_min_chunk_length(self, section_name: str) -> int:
        """
        Get minimum chunk length based on section type.
        
        Args:
            section_name: Name of the section.
            
        Returns:
            Minimum chunk length in characters.
        """
        if 'MD&A' in section_name or 'Management' in section_name.lower():
            return chunking_config.min_chunk_length_mda
        elif 'Item 1' in section_name and 'Risk' not in section_name:
            return chunking_config.min_chunk_length_item1
        else:
            return chunking_config.min_chunk_length_default


def process_filings(filings: List[FilingMetadata]) -> List[DocumentChunk]:
    """
    Process multiple SEC filings into chunks.
    
    Args:
        filings: List of filing metadata objects.
        
    Returns:
        List of all document chunks.
    """
    processor = DocumentProcessor()
    all_chunks = []
    
    for filing in filings:
        chunks = processor.process_filing(filing)
        all_chunks.extend(chunks)
    
    return all_chunks


if __name__ == "__main__":
    # Test mode
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        test_file = Path(sys.argv[1])
        if test_file.exists():
            # Create a mock filing metadata
            mock_filing = FilingMetadata(
                ticker="TEST",
                cik="0000000000",
                filing_type="10-K",
                filing_date="2024-01-01",
                fiscal_year="2024",
                fiscal_quarter=None,
                file_path=test_file,
                accession_number="test-001"
            )
            
            processor = DocumentProcessor()
            chunks = processor.process_filing(mock_filing)
            
            print(f"\nProcessed {len(chunks)} chunks:")
            for chunk in chunks[:3]:  # Show first 3
                print(f"\n--- {chunk.section} (chunk {chunk.chunk_index}) ---")
                print(chunk.content[:500] + "..." if len(chunk.content) > 500 else chunk.content)
