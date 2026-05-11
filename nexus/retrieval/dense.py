"""
Dense Retrieval for NEXUS

Retrieves chunks using vector similarity search.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from loguru import logger

from db.neon import fetch, get_db
from ingestion.embedder import Embedder, embed_query
from config import settings


class DenseRetriever:
    """
    Retrieve chunks using dense vector embeddings.
    
    Uses pgvector for efficient cosine similarity search
    with IVFFlat index for scalability.
    """

    def __init__(self, top_k: int = None):
        """
        Initialize the dense retriever.
        
        Args:
            top_k: Number of results to return
        """
        self.top_k = top_k or settings.top_k_dense
        self.embedder = Embedder()

    async def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks similar to the query.
        
        Args:
            query: Query text
            filters: Optional filters (company, year, doc_type, etc.)
            top_k: Override default top_k
            
        Returns:
            List of chunk dicts with metadata and scores
        """
        k = top_k or self.top_k
        
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)
        embedding_str = "[" + ",".join(map(str, query_embedding.tolist())) + "]"
        
        # Build SQL query with optional filters
        where_clauses = []
        params = [embedding_str, k]
        param_idx = 3
        
        if filters:
            if filters.get('company'):
                where_clauses.append(f"d.company = ${param_idx}")
                params.append(filters['company'])
                param_idx += 1
            
            if filters.get('year'):
                where_clauses.append(f"d.year = ${param_idx}")
                params.append(filters['year'])
                param_idx += 1
            
            if filters.get('doc_type'):
                where_clauses.append(f"d.doc_type = ${param_idx}")
                params.append(filters['doc_type'])
                param_idx += 1
        
        where_clause = " AND ".join(where_clauses)
        if where_clause:
            where_clause = f"WHERE {where_clause}"
        
        sql = f"""
        SELECT 
            c.chunk_id,
            c.text,
            c.token_count,
            c.page_number,
            d.doc_id,
            d.company,
            d.year,
            d.doc_type,
            s.section_id,
            s.heading,
            1 - (c.embedding <=> $1::vector) AS similarity_score
        FROM chunks c
        JOIN documents d ON c.doc_id = d.doc_id
        LEFT JOIN sections s ON c.section_id = s.section_id
        {where_clause}
        ORDER BY c.embedding <=> $1::vector
        LIMIT $2
        """
        
        try:
            results = await fetch(sql, *params)
            
            return [
                {
                    "chunk_id": r["chunk_id"],
                    "text": r["text"],
                    "token_count": r["token_count"],
                    "page_number": r["page_number"],
                    "doc_id": r["doc_id"],
                    "company": r["company"],
                    "year": r["year"],
                    "doc_type": r["doc_type"],
                    "section_id": r["section_id"],
                    "heading": r["heading"],
                    "similarity_score": float(r["similarity_score"]),
                    "retrieval_method": "dense"
                }
                for r in results
            ]
            
        except Exception as e:
            logger.error(f"Dense retrieval error: {e}")
            raise

    async def retrieve_with_rerank(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = None,
        rerank_top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve and optionally re-rank results.
        
        Args:
            query: Query text
            filters: Optional filters
            top_k: Initial retrieval count
            rerank_top_k: Final count after re-ranking
            
        Returns:
            List of chunk dicts
        """
        # Retrieve more candidates for re-ranking
        initial_k = (rerank_top_k or top_k or self.top_k) * 3
        results = await self.retrieve(query, filters, initial_k)
        
        # Simple re-ranking by boosting exact matches
        query_lower = query.lower()
        for result in results:
            text_lower = result["text"].lower()
            
            # Boost score for exact phrase matches
            if query_lower in text_lower:
                result["similarity_score"] *= 1.2
            
            # Boost for heading matches
            if result["heading"] and query_lower in result["heading"].lower():
                result["similarity_score"] *= 1.3
        
        # Sort by boosted score
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return results[:rerank_top_k or top_k or self.top_k]


# Global retriever instance
_dense_retriever: Optional[DenseRetriever] = None


def get_dense_retriever() -> DenseRetriever:
    """Get the global dense retriever instance."""
    global _dense_retriever
    if _dense_retriever is None:
        _dense_retriever = DenseRetriever()
    return _dense_retriever


async def retrieve_dense(
    query: str,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Convenience function for dense retrieval.
    
    Args:
        query: Query text
        **kwargs: Additional arguments
        
    Returns:
        List of chunk dicts
    """
    retriever = get_dense_retriever()
    return await retriever.retrieve(query, **kwargs)
