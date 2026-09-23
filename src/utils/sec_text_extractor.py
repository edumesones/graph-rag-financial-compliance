"""
SEC Text Extractor - Intelligent extraction from SEC filings

Handles multiple file formats and finds the best content in filing directories.
"""

import os
import glob
from typing import Optional
from pathlib import Path


class SECTextExtractor:
    """
    Intelligent SEC filing text extractor.

    Handles HTML, TXT, and HTM files with smart content selection.
    """

    def __init__(self):
        """Initialize SEC text extractor."""
        pass

    def extract_from_filing_directory(self, filing_dir: str) -> Optional[str]:
        """
        Extract text from a filing directory.

        Tries to find the best file (usually full-submission.txt or .html files).

        Args:
            filing_dir: Path to filing directory

        Returns:
            Extracted text or None if extraction fails
        """
        if not os.path.exists(filing_dir):
            print(f"   Directory not found: {filing_dir}")
            return None

        # Priority order for file types
        file_priorities = [
            ("full-submission.txt", self._extract_txt),
            ("*.txt", self._extract_txt),
            ("*.html", self._extract_html),
            ("*.htm", self._extract_html),
        ]

        for pattern, extractor in file_priorities:
            files = glob.glob(os.path.join(filing_dir, pattern))
            if files:
                # Use first matching file
                file_path = files[0]
                print(f"   Extracting from: {os.path.basename(file_path)}")
                text = extractor(file_path)
                if text and len(text.strip()) > 100:
                    return text

        print(f"   No readable files found in {filing_dir}")
        return None

    def _extract_txt(self, file_path: str) -> Optional[str]:
        """Extract text from TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            print(f"   Error reading TXT: {e}")
            return None

    def _extract_html(self, file_path: str) -> Optional[str]:
        """Extract text from HTML file."""
        try:
            # Try with BeautifulSoup for better HTML parsing
            try:
                from bs4 import BeautifulSoup
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    soup = BeautifulSoup(f.read(), 'html.parser')

                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()

                    # Get text
                    text = soup.get_text()

                    # Clean up whitespace
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = '\n'.join(chunk for chunk in chunks if chunk)

                    return text
            except ImportError:
                # Fallback to simple text extraction
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
        except Exception as e:
            print(f"   Error reading HTML: {e}")
            return None
