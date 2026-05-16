"""Graph retrieval from Neo4j knowledge graph."""
from typing import List, Optional
from neo4j import AsyncGraphDatabase
from config import settings


async def graph_retrieve(
    entities: List[str],
    query_type: str,
    max_results: int = 20
) -> List[str]:
    """
    Retrieve relationships from Neo4j graph.
    
    Args:
        entities: List of entity names to search for
        query_type: 'relational' for 1-hop, 'multi_hop' for up to 3 hops
        max_results: Maximum number of results (LIMIT in Cypher)
    
    Returns:
        List of formatted strings: "EntityA RELATION_TYPE EntityB (source: doc_id, year: year)"
    """
    if not entities:
        return []
    
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password)
    )
    
    results = []
    
    try:
        async with driver.session() as session:
            if query_type == 'relational':
                # 1-hop query
                cypher = """
                MATCH (a)-[r]-(b)
                WHERE a.name IN $entities
                RETURN a.name AS from_entity, 
                       type(r) AS relation_type, 
                       b.name AS to_entity,
                       a.doc_id AS doc_id,
                       a.year AS year
                LIMIT $limit
                """
            else:
                # Multi-hop query (up to 3 hops)
                cypher = """
                MATCH path=(a)-[*1..3]-(b)
                WHERE a.name IN $entities
                WITH a, b, relationships(path) AS rels
                UNWIND rels AS r
                RETURN a.name AS from_entity,
                       type(r) AS relation_type,
                       b.name AS to_entity,
                       a.doc_id AS doc_id,
                       a.year AS year
                LIMIT $limit
                """
            
            result = await session.run(
                cypher,
                entities=entities,
                limit=max_results
            )
            
            async for record in result:
                from_entity = record["from_entity"]
                relation_type = record["relation_type"]
                to_entity = record["to_entity"]
                doc_id = record.get("doc_id", "unknown")
                year = record.get("year", "unknown")
                
                formatted = f"{from_entity} {relation_type} {to_entity} (source: {doc_id}, year: {year})"
                results.append(formatted)
    
    except Exception as e:
        # Log error but return empty list - graph is non-blocking
        print(f"Graph retrieval error: {e}")
        results = []
    
    finally:
        await driver.close()
    
    return results
