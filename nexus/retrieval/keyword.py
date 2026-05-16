"""Keyword retrieval using PostgreSQL full-text search."""
import asyncio
from dataclasses import dataclass
from typing import List, Optional
from db.neon import fetch


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    parent_text: Optional[str]
    score: float
    doc_id: str
    section_id: str
    source: str = "keyword"


async def bm25_search(query_text: str, top_k: int = 10) -> List[RetrievedChunk]:
    """
    Use Postgres FTS: WHERE text_search @@ plainto_tsquery('english', $1)
    ORDER BY ts_rank_cd(text_search, plainto_tsquery('english', $1)) DESC
    """
    query = """
        SELECT 
            c.chunk_id,
            c.text,
            c.doc_id,
            c.section_id,
            c.parent_chunk_id,
            ts_rank_cd(c.text_search, plainto_tsquery('english', $1)) AS score
        FROM chunks c
        WHERE c.text_search @@ plainto_tsquery('english', $1)
        ORDER BY score DESC
        LIMIT $2
    """
    
    rows = await fetch(query, query_text, top_k)
    
    results = []
    for row in rows:
        parent_text = None
        if row["parent_chunk_id"]:
            parent_row = await fetch(
                "SELECT text FROM chunks WHERE chunk_id = $1",
                row["parent_chunk_id"]
            )
            if parent_row and len(parent_row) > 0:
                parent_text = parent_row[0]["text"]
        
        results.append(RetrievedChunk(
            chunk_id=row["chunk_id"],
            text=row["text"],
            parent_text=parent_text,
            score=float(row["score"]),
            doc_id=row["doc_id"],
            section_id=row["section_id"],
            source="keyword"
        ))
    
    return results
