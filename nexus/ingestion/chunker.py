"""
Text Chunker for NEXUS

Splits document sections into retrievable chunks with configurable size and overlap.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    logger.warning("tiktoken not installed. Install with: pip install tiktoken")

from config import settings


@dataclass
class Chunk:
    """Represents a text chunk."""
    text: str
    token_count: int
    section_id: Optional[int] = None
    doc_id: Optional[int] = None
    chunk_level: int = 0
    parent_chunk_id: Optional[int] = None
    page_number: Optional[int] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TextChunker:
    """
    Split text into chunks for retrieval.
    
    Supports multiple chunking strategies:
    - Fixed-size chunks with overlap
    - Semantic chunking (by sentences/paragraphs)
    - Hierarchical chunking (parent-child relationships)
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        encoding_name: str = "cl100k_base"
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Token overlap between consecutive chunks
            encoding_name: Tiktoken encoding to use
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        if HAS_TIKTOKEN:
            self.encoder = tiktoken.get_encoding(encoding_name)
        else:
            self.encoder = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.encoder:
            return len(self.encoder.encode(text))
        else:
            # Fallback: approximate by character count
            return len(text) // 4

    def chunk_text(
        self,
        text: str,
        section_id: Optional[int] = None,
        doc_id: Optional[int] = None,
        page_number: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Split text into chunks.
        
        Args:
            text: Text to chunk
            section_id: Optional section identifier
            doc_id: Optional document identifier
            page_number: Optional page number
            metadata: Optional metadata dict
            
        Returns:
            List of Chunk objects
        """
        if not text.strip():
            return []

        tokens = self._tokenize(text)
        chunks = []
        
        start = 0
        chunk_idx = 0
        
        while start < len(tokens):
            end = start + self.chunk_size
            
            # If we're at the end, just take what's left
            if end >= len(tokens):
                chunk_tokens = tokens[start:]
            else:
                # Try to break at sentence boundary
                chunk_text = self._detokenize(tokens[start:end])
                
                # Look for sentence boundaries near the end
                last_period = chunk_text.rfind('. ')
                last_newline = chunk_text.rfind('\n')
                
                if last_period > self.chunk_size // 2:
                    end = start + last_period + 1
                elif last_newline > self.chunk_size // 2:
                    end = start + last_newline + 1
                    
                chunk_tokens = tokens[start:end]
            
            chunk_text = self._detokenize(chunk_tokens)
            token_count = len(chunk_tokens)
            
            chunks.append(Chunk(
                text=chunk_text,
                token_count=token_count,
                section_id=section_id,
                doc_id=doc_id,
                chunk_level=0,
                page_number=page_number,
                metadata=metadata or {}
            ))
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start <= 0:
                break
            chunk_idx += 1
        
        logger.debug(f"Created {len(chunks)} chunks from {len(tokens)} tokens")
        return chunks

    def hierarchical_chunk(
        self,
        text: str,
        parent_chunk_id: Optional[int] = None,
        section_id: Optional[int] = None,
        doc_id: Optional[int] = None,
        levels: int = 2
    ) -> List[Chunk]:
        """
        Create hierarchical chunks (parent-child relationships).
        
        Args:
            text: Text to chunk hierarchically
            parent_chunk_id: Optional parent chunk ID
            section_id: Optional section identifier
            doc_id: Optional document identifier
            levels: Number of hierarchy levels
            
        Returns:
            List of Chunk objects at all levels
        """
        all_chunks = []
        
        # First level: larger chunks
        large_chunk_size = self.chunk_size * 2
        original_chunk_size = self.chunk_size
        
        self.chunk_size = large_chunk_size
        parent_chunks = self.chunk_text(text, section_id, doc_id)
        self.chunk_size = original_chunk_size
        
        for p_chunk in parent_chunks:
            all_chunks.append(p_chunk)
            
            # Create child chunks if more levels needed
            if levels > 1:
                child_chunks = self.chunk_text(
                    p_chunk.text,
                    section_id=section_id,
                    doc_id=doc_id,
                    metadata={"parent_chunk_id": p_chunk}
                )
                
                for c_chunk in child_chunks:
                    c_chunk.parent_chunk_id = parent_chunk_id
                    c_chunk.chunk_level = 1
                    all_chunks.append(c_chunk)
        
        return all_chunks

    def _tokenize(self, text: str) -> List[int]:
        """Tokenize text."""
        if self.encoder:
            return self.encoder.encode(text)
        else:
            # Simple whitespace tokenization fallback
            return text.split()

    def _detokenize(self, tokens: List) -> str:
        """Detokenize to text."""
        if self.encoder:
            return self.encoder.decode(tokens)
        else:
            return ' '.join(tokens)


def chunk_document(
    text: str,
    section_id: Optional[int] = None,
    doc_id: Optional[int] = None,
    **kwargs
) -> List[Chunk]:
    """
    Convenience function to chunk text.
    
    Args:
        text: Text to chunk
        section_id: Optional section identifier
        doc_id: Optional document identifier
        **kwargs: Additional arguments for TextChunker
        
    Returns:
        List of Chunk objects
    """
    chunker = TextChunker(**kwargs)
    return chunker.chunk_text(text, section_id, doc_id)
