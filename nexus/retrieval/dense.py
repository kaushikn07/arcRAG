"""Dense vector retrieval using pgvector."""
import asyncio
from dataclasses import dataclass
from typing import List, Optional
from db.neon import fetch, fetchrow


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    parent_text: Optional[str]
    score: float
    doc_id: str
    section_id: str
    source: str = "dense"


async def pgvector_search(
    query_embedding: List[float],
    top_k: int = 10,
    doc_filter: Optional[str] = None
) -> List[RetrievedChunk]:
    """
    Search chunks.embedding with cosine similarity (<=> operator).
    Also search hypothetical_questions.embedding, join back to chunks.
    Deduplicate by chunk_id, take best score per chunk.
    Resolve parent: for each L2 result, fetch its L1 parent_chunk_id and attach parent_text.
    """
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
    
    # Build query with optional doc filter
    if doc_filter:
        chunk_filter = f"AND c.doc_id = '{doc_filter}'"
        hyde_filter = f"AND c.doc_id = '{doc_filter}'"
    else:
        chunk_filter = ""
        hyde_filter = ""
    
    query = f"""
        WITH chunk_scores AS (
            SELECT 
                c.chunk_id,
                c.text,
                c.doc_id,
                c.section_id,
                c.parent_chunk_id,
                1 - (c.embedding <=> $1::vector) AS score
            FROM chunks c
            WHERE 1 - (c.embedding <=> $1::vector) > 0
            {chunk_filter}
            ORDER BY score DESC
            LIMIT $2
        ),
        hyde_scores AS (
            SELECT 
                c.chunk_id,
                c.text,
                c.doc_id,
                c.section_id,
                c.parent_chunk_id,
                MAX(1 - (hq.embedding <=> $1::vector)) AS score
            FROM hypothetical_questions hq
            JOIN chunks c ON hq.chunk_id = c.chunk_id
            WHERE 1 - (hq.embedding <=> $1::vector) > 0
            {hyde_filter}
            GROUP BY c.chunk_id, c.text, c.doc_id, c.section_id, c.parent_chunk_id
        ),
        combined AS (
            SELECT * FROM chunk_scores
            UNION ALL
            SELECT * FROM hyde_scores
            WHERE chunk_id NOT IN (SELECT chunk_id FROM chunk_scores)
        ),
        deduplicated AS (
            SELECT DISTINCT ON (chunk_id)
                chunk_id, text, doc_id, section_id, parent_chunk_id, score
            FROM combined
            ORDER BY chunk_id, score DESC
        )
        SELECT * FROM deduplicated
        ORDER BY score DESC
        LIMIT $2
    """
    
    rows = await fetch(query, embedding_str, top_k)
    
    results = []
    for row in rows:
        parent_text = None
        if row["parent_chunk_id"]:
            parent_row = await fetchrow(
                "SELECT text FROM chunks WHERE chunk_id = $1",
                row["parent_chunk_id"]
            )
            if parent_row:
                parent_text = parent_row["text"]
        
        results.append(RetrievedChunk(
            chunk_id=row["chunk_id"],
            text=row["text"],
            parent_text=parent_text,
            score=float(row["score"]),
            doc_id=row["doc_id"],
            section_id=row["section_id"],
            source="dense"
        ))
    
    return results
