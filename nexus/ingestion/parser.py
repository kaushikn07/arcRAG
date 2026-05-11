"""
Layout-Aware PDF Parser for NEXUS

Extracts text, structure, and tables from financial documents using
pymupdf4llm for markdown conversion and pdfplumber for table extraction.
"""

import re
import sys
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ParsedSection:
    """Represents a parsed document section."""
    section_id: str
    doc_id: str
    heading: str
    heading_level: int  # 1, 2, or 3
    sequence_order: int
    content_type: str  # paragraph, table, list, mixed
    raw_text: str
    table_data: Optional[dict] = None  # {headers, rows, markdown} or None
    page_number: Optional[int] = None


@dataclass
class ParsedDocument:
    """Represents a fully parsed document."""
    doc_id: str
    company: str
    year: int
    doc_type: str
    filename: str
    page_count: int
    sections: list[ParsedSection] = field(default_factory=list)


class LayoutAwareParser:
    """
    Layout-aware PDF parser using pymupdf4llm and pdfplumber.
    
    Pass 1: pymupdf4llm.to_markdown() to get heading hierarchy via # markers
    Pass 2: pdfplumber to extract tables as {headers, rows, markdown}
    Merge: replace raw markdown table blocks with structured pdfplumber table data
    """

    def __init__(self):
        self._import_dependencies()

    def _import_dependencies(self):
        """Import required dependencies."""
        global pymupdf4llm, pdfplumber, fitz
        
        try:
            import pymupdf4llm
        except ImportError:
            raise ImportError("pymupdf4llm is required. Install with: pip install pymupdf4llm")
        
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("pdfplumber is required. Install with: pip install pdfplumber")
        
        try:
            import fitz
        except ImportError:
            raise ImportError("PyMuPDF is required. Install with: pip install pymupdf")

    def _generate_doc_id(self, company: str, doc_type: str, year: int) -> str:
        """Generate deterministic document ID."""
        company_slug = company.lower().replace(' ', '_')
        doc_type_slug = doc_type.lower()
        return f"{company_slug}_{doc_type_slug}_{year}"

    def _generate_section_id(self, doc_id: str, sequence_order: int, heading: str) -> str:
        """Generate deterministic section ID slug."""
        # Create slug from heading: lowercase, replace spaces with underscores, limit to 40 chars
        heading_slug = re.sub(r'[^a-zA-Z0-9\s]', '', heading).lower().strip()
        heading_slug = re.sub(r'\s+', '_', heading_slug)[:40]
        if not heading_slug:
            heading_slug = "section"
        return f"{doc_id}_{sequence_order:04d}_{heading_slug}"

    def _detect_heading_level(self, line: str) -> Optional[int]:
        """Detect heading level from markdown line by counting leading # characters."""
        stripped = line.lstrip()
        if not stripped.startswith('#'):
            return None
        
        # Count consecutive # characters
        match = re.match(r'^(#+)\s', stripped)
        if match:
            level = len(match.group(1))
            # Only accept levels 1-3
            if 1 <= level <= 3:
                return level
        return None

    def _extract_heading_text(self, line: str) -> str:
        """Extract heading text from markdown heading line."""
        stripped = line.lstrip()
        # Remove leading # characters and whitespace
        text = re.sub(r'^#+\s*', '', stripped).strip()
        return text

    def _normalize_table_cells(self, headers: list, rows: list) -> tuple[list, list]:
        """Replace None cells with empty strings in headers and rows."""
        normalized_headers = ['' if cell is None else str(cell) for cell in headers]
        normalized_rows = []
        for row in rows:
            normalized_row = ['' if cell is None else str(cell) for cell in row]
            normalized_rows.append(normalized_row)
        return normalized_headers, normalized_rows

    def _table_to_markdown(self, headers: list, rows: list) -> str:
        """Convert table data to markdown format."""
        if not headers:
            return ""
        
        lines = []
        # Header row
        header_line = '| ' + ' | '.join(headers) + ' |'
        lines.append(header_line)
        
        # Separator row
        separator = '| ' + ' | '.join(['---'] * len(headers)) + ' |'
        lines.append(separator)
        
        # Data rows
        for row in rows:
            # Pad row if needed
            padded_row = row + [''] * (len(headers) - len(row))
            row_line = '| ' + ' | '.join(padded_row[:len(headers)]) + ' |'
            lines.append(row_line)
        
        return '\n'.join(lines)

    def _extract_tables_from_pdf(self, pdf_path: Path) -> dict[int, list[dict]]:
        """
        Extract tables from PDF using pdfplumber.
        Returns dict mapping page numbers to list of table data.
        """
        tables_by_page = {}
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_tables = page.extract_tables()
                
                if page_tables:
                    tables_by_page[page_num] = []
                    
                    for table in page_tables:
                        if table and len(table) > 0:
                            # First row is typically headers
                            headers = table[0] if table else []
                            rows = table[1:] if len(table) > 1 else []
                            
                            # Normalize None cells
                            headers, rows = self._normalize_table_cells(headers, rows)
                            
                            # Generate markdown representation
                            markdown = self._table_to_markdown(headers, rows)
                            
                            table_data = {
                                'headers': headers,
                                'rows': rows,
                                'markdown': markdown
                            }
                            tables_by_page[page_num].append(table_data)
        
        return tables_by_page

    def _parse_markdown_sections(self, md_text: str, doc_id: str) -> list[dict]:
        """
        Parse markdown text into sections based on heading hierarchy.
        Returns list of section dicts with heading info and raw text.
        """
        lines = md_text.split('\n')
        sections = []
        
        current_heading = None
        current_level = 1
        current_text_lines = []
        seq_order = 0
        current_start_page = 0
        
        # Track approximate page boundaries (every ~50 lines as rough estimate)
        lines_per_page_estimate = 50
        
        for i, line in enumerate(lines):
            estimated_page = i // lines_per_page_estimate
            
            heading_level = self._detect_heading_level(line)
            
            if heading_level is not None:
                # Save previous section if exists
                if current_heading is not None or current_text_lines:
                    raw_text = '\n'.join(current_text_lines).strip()
                    if raw_text or current_heading is not None:
                        sections.append({
                            'heading': current_heading if current_heading else 'Document',
                            'heading_level': current_level,
                            'sequence_order': seq_order,
                            'raw_text': raw_text,
                            'page_number': current_start_page
                        })
                        seq_order += 1
                
                # Start new section
                current_heading = self._extract_heading_text(line)
                current_level = heading_level
                current_text_lines = []
                current_start_page = estimated_page
            else:
                current_text_lines.append(line)
        
        # Save last section
        if current_heading is not None or current_text_lines:
            raw_text = '\n'.join(current_text_lines).strip()
            sections.append({
                'heading': current_heading if current_heading else 'Document',
                'heading_level': current_level,
                'sequence_order': seq_order,
                'raw_text': raw_text,
                'page_number': current_start_page
            })
        
        return sections

    def _detect_content_type(self, raw_text: str, has_table: bool) -> str:
        """Detect content type based on text characteristics."""
        if has_table:
            return 'table'
        
        lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
        
        # Check for list patterns
        list_pattern = re.compile(r'^[\s]*[-•*]\s+')
        list_items = sum(1 for line in lines if list_pattern.match(line))
        
        if list_items > len(lines) * 0.5 and len(lines) > 2:
            return 'list'
        
        return 'paragraph'

    def _merge_tables_into_sections(
        self, 
        sections: list[dict], 
        tables_by_page: dict[int, list[dict]],
        doc_id: str
    ) -> list[ParsedSection]:
        """
        Merge extracted tables into sections, replacing markdown table blocks
        with structured table data.
        """
        parsed_sections = []
        
        for section in sections:
            seq_order = section['sequence_order']
            page_num = section.get('page_number', 0)
            
            # Check if this page has tables
            page_tables = tables_by_page.get(page_num, [])
            
            # Determine if section contains table content
            has_table = False
            table_data = None
            
            # Simple heuristic: if section text looks like a table (has | characters)
            # and there are tables on this page, associate them
            raw_text = section['raw_text']
            if '|' in raw_text and page_tables:
                has_table = True
                # Use first table on the page for this section
                if page_tables:
                    table_data = page_tables[0]
            
            content_type = self._detect_content_type(raw_text, has_table)
            
            # If it's a table section but we don't have table data yet,
            # try to find one from nearby pages
            if content_type == 'table' and table_data is None:
                for offset in range(-1, 2):
                    check_page = page_num + offset
                    if check_page in tables_by_page and tables_by_page[check_page]:
                        table_data = tables_by_page[check_page][0]
                        has_table = True
                        break
            
            section_id = self._generate_section_id(doc_id, seq_order, section['heading'])
            
            parsed_section = ParsedSection(
                section_id=section_id,
                doc_id=doc_id,
                heading=section['heading'],
                heading_level=section['heading_level'],
                sequence_order=seq_order,
                content_type=content_type,
                raw_text=raw_text,
                table_data=table_data,
                page_number=page_num
            )
            parsed_sections.append(parsed_section)
        
        return parsed_sections

    def parse(
        self, 
        pdf_path: str, 
        company: str, 
        year: int, 
        doc_type: str = '10-K'
    ) -> ParsedDocument:
        """
        Parse a PDF document with layout awareness.
        
        Args:
            pdf_path: Path to the PDF file
            company: Company name
            year: Document year
            doc_type: Document type (default: '10-K')
        
        Returns:
            ParsedDocument with extracted sections
        """
        path = Path(pdf_path)
        
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Generate document ID
        doc_id = self._generate_doc_id(company, doc_type, year)
        
        # Get page count using PyMuPDF
        doc = fitz.open(path)
        page_count = len(doc)
        doc.close()
        
        # Check for scanned PDF (no text extracted)
        test_doc = fitz.open(path)
        has_text = False
        for page in test_doc:
            text = page.get_text().strip()
            if text:
                has_text = True
                break
        test_doc.close()
        
        if not has_text:
            # Scanned PDF - return single section indicating OCR needed
            section_id = self._generate_section_id(doc_id, 0, 'Document')
            section = ParsedSection(
                section_id=section_id,
                doc_id=doc_id,
                heading='Document',
                heading_level=1,
                sequence_order=0,
                content_type='paragraph',
                raw_text='[SCANNED PDF - OCR required]',
                table_data=None,
                page_number=0
            )
            return ParsedDocument(
                doc_id=doc_id,
                company=company,
                year=year,
                doc_type=doc_type,
                filename=path.name,
                page_count=page_count,
                sections=[section]
            )
        
        # Pass 1: Extract markdown structure using pymupdf4llm
        md_text = pymupdf4llm.to_markdown(str(path))
        
        # Parse markdown into sections
        markdown_sections = self._parse_markdown_sections(md_text, doc_id)
        
        # Pass 2: Extract tables using pdfplumber
        tables_by_page = self._extract_tables_from_pdf(path)
        
        # Merge tables into sections
        parsed_sections = self._merge_tables_into_sections(
            markdown_sections, tables_by_page, doc_id
        )
        
        # Handle edge case: no headings detected
        if len(parsed_sections) == 0 or (
            len(parsed_sections) == 1 and 
            parsed_sections[0].heading == 'Document' and
            parsed_sections[0].heading_level == 1
        ):
            # Check if we actually have content or just empty
            if parsed_sections and parsed_sections[0].raw_text.strip():
                pass  # Keep as single section
            elif len(parsed_sections) == 0:
                # Create single section with all content
                section_id = self._generate_section_id(doc_id, 0, 'Document')
                section = ParsedSection(
                    section_id=section_id,
                    doc_id=doc_id,
                    heading='Document',
                    heading_level=1,
                    sequence_order=0,
                    content_type='paragraph',
                    raw_text=md_text.strip(),
                    table_data=None,
                    page_number=0
                )
                parsed_sections = [section]
        
        return ParsedDocument(
            doc_id=doc_id,
            company=company,
            year=year,
            doc_type=doc_type,
            filename=path.name,
            page_count=page_count,
            sections=parsed_sections
        )


def main():
    """CLI entry point for parsing a PDF file."""
    if len(sys.argv) < 2:
        print("Usage: python -m ingestion.parser <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # Default values for demo purposes
    company = "Sample Corp"
    year = 2024
    doc_type = "10-K"
    
    parser = LayoutAwareParser()
    doc = parser.parse(pdf_path, company, year, doc_type)
    
    # Calculate statistics
    total_sections = len(doc.sections)
    table_sections = sum(1 for s in doc.sections if s.content_type == 'table')
    paragraph_sections = sum(1 for s in doc.sections if s.content_type == 'paragraph')
    list_sections = sum(1 for s in doc.sections if s.content_type == 'list')
    mixed_sections = sum(1 for s in doc.sections if s.content_type == 'mixed')
    
    print(f"doc_id: {doc.doc_id}")
    print(f"Total sections: {total_sections}")
    print(f"Table sections: {table_sections}")
    print(f"Paragraph sections: {paragraph_sections}")
    if list_sections:
        print(f"List sections: {list_sections}")
    if mixed_sections:
        print(f"Mixed sections: {mixed_sections}")
    
    print("\nFirst 3 section headings:")
    for i, section in enumerate(doc.sections[:3]):
        print(f"  {i+1}. [Level {section.heading_level}] {section.heading}")


if __name__ == "__main__":
    main()
