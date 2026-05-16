"""
Storage module for NEXUS ingestion pipeline.

Provides async functions to store parsed documents, sections, and chunks with embeddings.
"""

from typing import List

from asyncpg import Record

from db.neon import execute, executemany
from ingestion.parser import ParsedDocument, ParsedSection
from ingestion.chunker import Chunk


async def store_document(doc: ParsedDocument) -> None:
    """
    Store a parsed document in the database.
    
    Uses ON CONFLICT DO NOTHING to avoid duplicates.
    
    Args:
        doc: ParsedDocument to store
    """
    query = """
        INSERT INTO documents (doc_id, company, year, doc_type, filename, page_count, ingested_at, stage_completed)
        VALUES ($1, $2, $3, $4, $5, $6, NOW(), 'parsed')
        ON CONFLICT (doc_id) DO NOTHING
    """
    await execute(
        query,
        doc.doc_id,
        doc.company,
        doc.year,
        doc.doc_type,
        doc.filename,
        doc.page_count
    )


async def store_sections(sections: List[ParsedSection]) -> None:
    """
    Store parsed sections in batch.
    
    Uses ON CONFLICT DO NOTHING to avoid duplicates.
    
    Args:
        sections: List of ParsedSection objects to store
    """
    if not sections:
        return
    
    query = """
        INSERT INTO sections (section_id, doc_id, heading, heading_level, sequence_order, content_type, raw_text, token_count, page_number)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        ON CONFLICT (section_id) DO NOTHING
    """
    
    # Prepare batch parameters
    params = [
        (
            section.section_id,
            section.doc_id,
            section.heading,
            section.heading_level,
            section.sequence_order,
            section.content_type,
            section.raw_text,
            len(section.raw_text.split()),  # Approximate token count
            section.page_number
        )
        for section in sections
    ]
    
    await executemany(query, params)


async def store_chunks_with_embeddings(chunks: List[Chunk], embeddings: List[List[float]]) -> None:
    """
    Store chunks with their embeddings in batch.
    
    Uses ON CONFLICT DO NOTHING to avoid duplicates.
    
    Args:
        chunks: List of Chunk objects
        embeddings: List of embedding vectors (must match chunks length)
    """
    if not chunks or not embeddings:
        return
    
    if len(chunks) != len(embeddings):
        raise ValueError(f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must have same length")
    
    query = """
        INSERT INTO chunks (chunk_id, doc_id, section_id, chunk_level, parent_chunk_id, text, token_count, embedding, page_number)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        ON CONFLICT (chunk_id) DO NOTHING
    """
    
    # Prepare batch parameters
    params = [
        (
            chunk.chunk_id,
            chunk.doc_id,
            chunk.section_id,
            chunk.chunk_level,
            chunk.parent_chunk_id,
            chunk.text,
            chunk.token_count,
            embeddings[i],  # pgvector will handle list[float] -> vector conversion
            chunk.page_number
        )
        for i, chunk in enumerate(chunks)
    ]
    
    await executemany(query, params)


async def store_hypothetical_questions(questions: List[tuple]) -> None:
    """
    Store hypothetical questions with embeddings in batch.
    
    Args:
        questions: List of tuples (q_id, chunk_id, doc_id, question_text, embedding)
    """
    if not questions:
        return
    
    query = """
        INSERT INTO hypothetical_questions (q_id, chunk_id, doc_id, question_text, embedding)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (q_id) DO NOTHING
    """
    
    await executemany(query, questions)


async def store_financial_metrics(metrics: List[tuple]) -> None:
    """
    Store financial metrics in batch.
    
    Args:
        metrics: List of tuples (metric_id, doc_id, company, year, period, metric_name, value, unit, source_section_id)
    """
    if not metrics:
        return
    
    query = """
        INSERT INTO financial_metrics (metric_id, doc_id, company, year, period, metric_name, value, unit, source_section_id)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        ON CONFLICT (metric_id) DO NOTHING
    """
    
    await executemany(query, metrics)
