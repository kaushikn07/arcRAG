"""NEXUS API Routes."""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel

from retrieval.dense import retrieve_dense
from retrieval.keyword import retrieve_keyword
from retrieval.hybrid import retrieve_hybrid


router = APIRouter(tags=["retrieval"])


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str
    company: Optional[str] = None
    year: Optional[int] = None
    doc_type: Optional[str] = None
    top_k: int = 10
    method: str = "hybrid"  # dense, keyword, hybrid


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    results: List[Dict[str, Any]]
    query: str
    method: str
    count: int


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query financial documents using RAG.
    
    Supports multiple retrieval methods:
    - dense: Vector similarity search
    - keyword: Full-text search
    - hybrid: Combined approach (default)
    """
    filters = {}
    if request.company:
        filters["company"] = request.company
    if request.year:
        filters["year"] = request.year
    if request.doc_type:
        filters["doc_type"] = request.doc_type
    
    try:
        if request.method == "dense":
            results = await retrieve_dense(
                request.query,
                filters=filters,
                top_k=request.top_k
            )
        elif request.method == "keyword":
            from retrieval.keyword import get_keyword_retriever
            retriever = get_keyword_retriever()
            results = await retriever.retrieve(
                request.query,
                filters=filters,
                top_k=request.top_k
            )
        else:  # hybrid
            results = await retrieve_hybrid(
                request.query,
                filters=filters,
                top_k=request.top_k
            )
        
        return QueryResponse(
            results=results,
            query=request.query,
            method=request.method,
            count=len(results)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents")
async def list_documents(
    company: Optional[str] = Query(None),
    year: Optional[int] = Query(None),
    doc_type: Optional[str] = Query(None),
    limit: int = Query(50, le=200)
):
    """List ingested documents with optional filters."""
    from db.neon import fetch
    
    conditions = []
    params = []
    param_idx = 1
    
    if company:
        conditions.append(f"company = ${param_idx}")
        params.append(company)
        param_idx += 1
    
    if year:
        conditions.append(f"year = ${param_idx}")
        params.append(year)
        param_idx += 1
    
    if doc_type:
        conditions.append(f"doc_type = ${param_idx}")
        params.append(doc_type)
        param_idx += 1
    
    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
    
    query = f"""
    SELECT doc_id, company, year, doc_type, filename, 
           page_count, ingested_at, stage_completed
    FROM documents
    {where_clause}
    ORDER BY ingested_at DESC
    LIMIT ${param_idx}
    """
    params.append(limit)
    
    try:
        results = await fetch(query, *params)
        return {"documents": [dict(r) for r in results], "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{doc_id}")
async def get_document(doc_id: int):
    """Get details of a specific document."""
    from db.neon import fetchrow
    
    query = """
    SELECT doc_id, company, year, doc_type, filename,
           page_count, ingested_at, stage_completed
    FROM documents
    WHERE doc_id = $1
    """
    
    result = await fetchrow(query, doc_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {"document": dict(result)}


@router.get("/metrics")
async def get_metrics(
    company: str = Query(...),
    year: Optional[int] = Query(None),
    metric_name: Optional[str] = Query(None)
):
    """Get financial metrics for a company."""
    from db.neon import fetch
    
    conditions = ["company = $1"]
    params = [company]
    param_idx = 2
    
    if year:
        conditions.append(f"year = ${param_idx}")
        params.append(year)
        param_idx += 1
    
    if metric_name:
        conditions.append(f"metric_name = ${param_idx}")
        params.append(metric_name)
        param_idx += 1
    
    where_clause = "WHERE " + " AND ".join(conditions)
    
    query = f"""
    SELECT metric_id, company, year, period, metric_name, 
           value, unit
    FROM financial_metrics
    {where_clause}
    ORDER BY year, period, metric_name
    """
    
    try:
        results = await fetch(query, *params)
        return {"metrics": [dict(r) for r in results], "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
