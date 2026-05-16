"""Hybrid retrieval with Reciprocal Rank Fusion (RRF)."""
import asyncio
from typing import List, Dict, Any

from retrieval.dense import pgvector_search, RetrievedChunk as DenseChunk
from retrieval.keyword import bm25_search, RetrievedChunk as KeywordChunk


def reciprocal_rank_fusion(
    result_lists: List[List[Any]],
    weights: List[float],
    k: int = 60
) -> List[str]:
    """
    Implement Reciprocal Rank Fusion (RRF).
    
    RRF score = sum over lists of (weight / (k + rank))
    
    Args:
        result_lists: List of result lists from different retrievers
        weights: Weight for each result list (should sum to 1.0)
        k: RRF constant (default 60)
    
    Returns:
        List of chunk_ids sorted by RRF score
    """
    # Map chunk_id to RRF score
    rrf_scores: Dict[str, float] = {}
    
    for results, weight in zip(result_lists, weights):
        for rank, item in enumerate(results):
            # Extract chunk_id from the result object
            if hasattr(item, 'chunk_id'):
                chunk_id = item.chunk_id
            elif isinstance(item, dict):
                chunk_id = item.get('chunk_id')
            else:
                chunk_id = str(item)
            
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = 0.0
            
            rrf_scores[chunk_id] += weight / (k + rank)
    
    # Sort by RRF score descending
    sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    return [chunk_id for chunk_id, _ in sorted_chunks]


async def hybrid_retrieve(
    query: str,
    query_embedding: List[float],
    top_k: int = 10,
    weights: List[float] = None
) -> List[DenseChunk]:
    """
    Run dense + keyword retrieval in parallel and merge with RRF.
    
    Args:
        query: Query text for keyword search
        query_embedding: Query embedding for dense search
        top_k: Number of final results to return
        weights: Weights for [dense, keyword], default [0.6, 0.4]
    
    Returns:
        List of RetrievedChunk objects (from dense retriever structure)
    """
    if weights is None:
        weights = [0.6, 0.4]
    
    # Run both retrievals in parallel
    dense_results, keyword_results = await asyncio.gather(
        pgvector_search(query_embedding, top_k=top_k * 2),
        bm25_search(query, top_k=top_k * 2)
    )
    
    # Get RRF ranking of chunk_ids
    rrf_chunk_ids = reciprocal_rank_fusion(
        [dense_results, keyword_results],
        weights,
        k=60
    )
    
    # Build a map of all results by chunk_id
    all_results: Dict[str, Any] = {}
    for chunk in dense_results:
        all_results[chunk.chunk_id] = chunk
    for chunk in keyword_results:
        if chunk.chunk_id not in all_results:
            # Create a DenseChunk-like object for keyword-only results
            all_results[chunk.chunk_id] = DenseChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                parent_text=chunk.parent_text,
                score=chunk.score,
                doc_id=chunk.doc_id,
                section_id=chunk.section_id,
                source="hybrid"
            )
        else:
            # Update source to indicate hybrid match
            all_results[chunk.chunk_id].source = "hybrid"
    
    # Return top_k results in RRF order
    final_results = []
    for chunk_id in rrf_chunk_ids[:top_k]:
        if chunk_id in all_results:
            result = all_results[chunk_id]
            # Ensure source is marked correctly
            if result.source == "dense":
                result.source = "hybrid"
            final_results.append(result)
    
    return final_results
