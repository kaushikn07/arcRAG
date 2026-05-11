"""
Hybrid Retrieval for NEXUS

Combines dense and keyword retrieval with reciprocal rank fusion.
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from retrieval.dense import DenseRetriever
from retrieval.keyword import KeywordRetriever
from config import settings


class HybridRetriever:
    """
    Retrieve chunks using hybrid approach.
    
    Combines dense vector search and keyword search
    using Reciprocal Rank Fusion (RRF) for ranking.
    """

    def __init__(self, top_k: int = None, k_param: int = 60):
        """
        Initialize the hybrid retriever.
        
        Args:
            top_k: Number of results to return
            k_param: RRF constant (smaller = more weight to rank)
        """
        self.top_k = top_k or settings.top_k_hybrid
        self.k_param = k_param
        self.dense_retriever = DenseRetriever()
        self.keyword_retriever = KeywordRetriever()

    async def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks using hybrid search.
        
        Args:
            query: Query text
            filters: Optional filters
            top_k: Override default top_k
            
        Returns:
            List of chunk dicts with fused scores
        """
        k = top_k or self.top_k
        
        # Get results from both retrievers
        dense_results = await self.dense_retriever.retrieve(
            query, filters, top_k=k * 2
        )
        keyword_results = await self.keyword_retriever.retrieve(
            query, filters, top_k=k * 2
        )
        
        logger.debug(f"Dense results: {len(dense_results)}, Keyword results: {len(keyword_results)}")
        
        # Apply Reciprocal Rank Fusion
        fused_scores = {}
        
        # Score dense results
        for rank, result in enumerate(dense_results):
            chunk_id = result["chunk_id"]
            rrf_score = 1.0 / (self.k_param + rank)
            
            if chunk_id not in fused_scores:
                fused_scores[chunk_id] = {
                    **result,
                    "rrf_score": 0,
                    "dense_rank": rank + 1,
                    "keyword_rank": None,
                    "dense_score": result.get("similarity_score", 0),
                    "keyword_score": 0
                }
            else:
                fused_scores[chunk_id]["rrf_score"] += rrf_score
                fused_scores[chunk_id]["dense_rank"] = rank + 1
                fused_scores[chunk_id]["dense_score"] = result.get("similarity_score", 0)
        
        # Score keyword results
        for rank, result in enumerate(keyword_results):
            chunk_id = result["chunk_id"]
            rrf_score = 1.0 / (self.k_param + rank)
            
            if chunk_id not in fused_scores:
                fused_scores[chunk_id] = {
                    **result,
                    "rrf_score": 0,
                    "dense_rank": None,
                    "keyword_rank": rank + 1,
                    "dense_score": 0,
                    "keyword_score": result.get("relevance_score", 0)
                }
            else:
                fused_scores[chunk_id]["rrf_score"] += rrf_score
                fused_scores[chunk_id]["keyword_rank"] = rank + 1
                fused_scores[chunk_id]["keyword_score"] = result.get("relevance_score", 0)
        
        # Sort by RRF score
        ranked = sorted(
            fused_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )[:k]
        
        # Add retrieval method label
        for result in ranked:
            if result["dense_rank"] and result["keyword_rank"]:
                result["retrieval_method"] = "hybrid"
            elif result["dense_rank"]:
                result["retrieval_method"] = "dense"
            else:
                result["retrieval_method"] = "keyword"
        
        return ranked


# Global retriever instance
_hybrid_retriever: Optional[HybridRetriever] = None


def get_hybrid_retriever() -> HybridRetriever:
    """Get the global hybrid retriever instance."""
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever()
    return _hybrid_retriever


async def retrieve_hybrid(
    query: str,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Convenience function for hybrid retrieval.
    
    Args:
        query: Query text
        **kwargs: Additional arguments
        
    Returns:
        List of chunk dicts
    """
    retriever = get_hybrid_retriever()
    return await retriever.retrieve(query, **kwargs)
