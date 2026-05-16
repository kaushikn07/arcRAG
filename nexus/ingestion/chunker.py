"""
Hierarchical Chunker for NEXUS

Splits parsed document sections into L1 (section-level) and L2 (paragraph-level) chunks.
"""

import re
from dataclasses import dataclass
from typing import Optional

import tiktoken

from ingestion.parser import ParsedDocument, ParsedSection


@dataclass
class Chunk:
    """Represents a text chunk for retrieval."""
    chunk_id: str
    doc_id: str
    section_id: str
    chunk_level: int  # 1 (section-level) or 2 (paragraph-level)
    parent_chunk_id: Optional[str]
    text: str
    token_count: int
    page_number: Optional[int]


class HierarchicalChunker:
    """
    Creates hierarchical chunks from parsed documents.
    
    Level 1: One chunk per section (full section text)
    Level 2: Paragraph-level chunks with max 512 tokens, 1-sentence overlap
    """

    MAX_L2_TOKENS = 512
    MIN_PARAGRAPH_TOKENS = 30
    SENTENCE_OVERLAP = 1

    def __init__(self, encoding_name: str = "cl100k_base"):
        """Initialize the chunker with tiktoken encoder."""
        self.encoder = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using cl100k_base encoding."""
        return len(self.encoder.encode(text))

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences using regex."""
        # Match sentence endings followed by space or end of string
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _merge_tiny_paragraphs(self, paragraphs: list[str]) -> list[str]:
        """Merge paragraphs with < 30 tokens into next paragraph."""
        if not paragraphs:
            return []

        merged = []
        i = 0
        while i < len(paragraphs):
            current = paragraphs[i]
            current_tokens = self.count_tokens(current)

            # If tiny paragraph, merge with next until >= 30 tokens
            if current_tokens < self.MIN_PARAGRAPH_TOKENS:
                combined = current
                i += 1
                while i < len(paragraphs):
                    next_para = paragraphs[i]
                    combined_tokens = self.count_tokens(combined + ' ' + next_para)
                    if combined_tokens >= self.MIN_PARAGRAPH_TOKENS:
                        break
                    combined += ' ' + next_para
                    i += 1
                merged.append(combined)
            else:
                merged.append(current)
                i += 1

        return merged

    def _create_l2_chunks_from_paragraph(
        self,
        paragraph: str,
        section: ParsedSection,
        l1_chunk_id: str,
        heading_prefix: str
    ) -> list[Chunk]:
        """Create L2 chunks from a single paragraph, splitting if > 512 tokens."""
        chunks = []
        prefixed_text = heading_prefix + paragraph

        # Check if entire paragraph fits in one chunk
        total_tokens = self.count_tokens(prefixed_text)
        if total_tokens <= self.MAX_L2_TOKENS:
            chunk_id = f"{section.section_id}_L2_{len(chunks):04d}"
            chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_id=section.doc_id,
                section_id=section.section_id,
                chunk_level=2,
                parent_chunk_id=l1_chunk_id,
                text=prefixed_text,
                token_count=total_tokens,
                page_number=section.page_number
            ))
            return chunks

        # Need to split - work with sentences
        sentences = self._split_into_sentences(paragraph)
        if not sentences:
            return chunks

        current_sentences = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            test_tokens = self.count_tokens(heading_prefix + ' '.join(current_sentences + [sentence]))

            if test_tokens <= self.MAX_L2_TOKENS:
                current_sentences.append(sentence)
                current_tokens = test_tokens
            else:
                # Current batch is full, create chunk
                if current_sentences:
                    chunk_text = heading_prefix + ' '.join(current_sentences)
                    chunk_id = f"{section.section_id}_L2_{len(chunks):04d}"
                    chunks.append(Chunk(
                        chunk_id=chunk_id,
                        doc_id=section.doc_id,
                        section_id=section.section_id,
                        chunk_level=2,
                        parent_chunk_id=l1_chunk_id,
                        text=chunk_text,
                        token_count=self.count_tokens(chunk_text),
                        page_number=section.page_number
                    ))

                # Start new batch with overlap (last SENTENCE_OVERLAP sentences)
                overlap_sentences = current_sentences[-self.SENTENCE_OVERLAP:] if len(current_sentences) >= self.SENTENCE_OVERLAP else []
                current_sentences = overlap_sentences + [sentence]

        # Create final chunk if there are remaining sentences
        if current_sentences:
            chunk_text = heading_prefix + ' '.join(current_sentences)
            chunk_id = f"{section.section_id}_L2_{len(chunks):04d}"
            chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_id=section.doc_id,
                section_id=section.section_id,
                chunk_level=2,
                parent_chunk_id=l1_chunk_id,
                text=chunk_text,
                token_count=self.count_tokens(chunk_text),
                page_number=section.page_number
            ))

        return chunks

    def chunk_document(self, doc: ParsedDocument) -> list[Chunk]:
        """
        Chunk a parsed document into L1 and L2 chunks.

        Level 1: One chunk per section (full section text)
        Level 2: Paragraph-level chunks

        Args:
            doc: ParsedDocument to chunk

        Returns:
            List of Chunk objects (L1 followed by L2)
        """
        all_chunks = []

        for section in doc.sections:
            # Skip empty sections
            if not section.raw_text.strip() and section.content_type != 'table':
                continue

            # Create L1 chunk (section-level)
            l1_chunk_id = f"{section.section_id}_L1"
            l1_text = section.raw_text.strip()
            l1_tokens = self.count_tokens(l1_text)

            l1_chunk = Chunk(
                chunk_id=l1_chunk_id,
                doc_id=section.doc_id,
                section_id=section.section_id,
                chunk_level=1,
                parent_chunk_id=None,
                text=l1_text,
                token_count=l1_tokens,
                page_number=section.page_number
            )
            all_chunks.append(l1_chunk)

            # Create L2 chunks (paragraph-level)
            heading_prefix = f"[Section: {section.heading}] "

            if section.content_type == 'table' and section.table_data:
                # Tables: single L2 chunk using markdown representation
                table_markdown = section.table_data.get('markdown', '')
                if table_markdown:
                    prefixed_text = heading_prefix + table_markdown
                    chunk_id = f"{section.section_id}_L2_0000"
                    l2_chunk = Chunk(
                        chunk_id=chunk_id,
                        doc_id=section.doc_id,
                        section_id=section.section_id,
                        chunk_level=2,
                        parent_chunk_id=l1_chunk_id,
                        text=prefixed_text,
                        token_count=self.count_tokens(prefixed_text),
                        page_number=section.page_number
                    )
                    all_chunks.append(l2_chunk)
            else:
                # Paragraphs: split by sentence boundaries
                # First, split raw text into paragraphs
                paragraphs = [p.strip() for p in section.raw_text.split('\n\n') if p.strip()]

                # Merge tiny paragraphs
                merged_paragraphs = self._merge_tiny_paragraphs(paragraphs)

                # Create L2 chunks from each paragraph
                for para in merged_paragraphs:
                    l2_chunks = self._create_l2_chunks_from_paragraph(
                        para, section, l1_chunk_id, heading_prefix
                    )
                    all_chunks.extend(l2_chunks)

        return all_chunks


def chunk_document(doc: ParsedDocument) -> list[Chunk]:
    """Convenience function to chunk a document."""
    chunker = HierarchicalChunker()
    return chunker.chunk_document(doc)
