import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any

import httpx
from loguru import logger
from redis.asyncio import Redis

from config import settings


@dataclass
class QueryPlan:
    query_text: str
    query_type: str  # factual/comparative/relational/multi_hop/metric/out_of_scope
    requires_graph: bool = False
    requires_vector: bool = False
    requires_sql: bool = False
    requires_keyword: bool = False
    entities: List[str] = field(default_factory=list)
    time_range: Optional[Tuple[int, int]] = None
    keyword_heavy: bool = False
    estimated_hops: int = 1


class QueryPlanClassifier:
    def __init__(self):
        self.redis: Optional[Redis] = None
        self._redis_initialized = False
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
            'dare', 'ought', 'used', 'it', 'its', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'we', 'they', 'what', 'which', 'who', 'whom',
            'whose', 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both',
            'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'also'
        }

    async def _get_redis(self) -> Optional[Redis]:
        if self._redis_initialized:
            return self.redis
        try:
            self.redis = await Redis.from_url(settings.redis_url)
            self._redis_initialized = True
            return self.redis
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self._redis_initialized = True
            return None

    def _extract_entities_simple(self, query: str) -> List[str]:
        """Extract capitalized words that aren't stop words as entity hints."""
        words = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b', query)
        entities = []
        for word in words:
            # Skip single letters unless they're likely part of an acronym
            if len(word) == 1 and word not in {'US', 'UK', 'EU'}:
                continue
            # Check if it's a stop word (case-insensitive)
            if word.lower() not in self.stop_words:
                # Avoid duplicates while preserving order
                if word not in entities:
                    entities.append(word)
        return entities

    def _generate_cache_key(self, query: str) -> str:
        return f"qplan:{hashlib.sha256(query.encode()).hexdigest()}"

    async def classify(self, query: str) -> QueryPlan:
        # Try cache first
        redis_client = await self._get_redis()
        if redis_client:
            try:
                cache_key = self._generate_cache_key(query)
                cached = await redis_client.get(cache_key)
                if cached:
                    logger.debug(f"Query plan cache hit for query: {query[:50]}...")
                    data = json.loads(cached)
                    # Reconstruct QueryPlan from cached dict
                    return QueryPlan(
                        query_text=data['query_text'],
                        query_type=data['query_type'],
                        requires_graph=data['requires_graph'],
                        requires_vector=data['requires_vector'],
                        requires_sql=data['requires_sql'],
                        requires_keyword=data['requires_keyword'],
                        entities=data['entities'],
                        time_range=tuple(data['time_range']) if data['time_range'] else None,
                        keyword_heavy=data['keyword_heavy'],
                        estimated_hops=data['estimated_hops']
                    )
            except Exception as e:
                logger.warning(f"Cache retrieval error: {e}")

        # Extract entity hints before LLM call
        entity_hints = self._extract_entities_simple(query)

        # Call LLM for classification
        system_prompt = """You are a query routing classifier for a financial document intelligence system.
Analyze the user query and determine the optimal retrieval strategy.

Return ONLY a valid JSON object with these exact fields:
{
  "query_type": "factual" | "comparative" | "relational" | "multi_hop" | "metric" | "out_of_scope",
  "requires_graph": boolean,
  "requires_vector": boolean,
  "requires_sql": boolean,
  "requires_keyword": boolean,
  "entities": ["list", "of", "extracted", "entities"],
  "time_range": [start_year, end_year] or null,
  "keyword_heavy": boolean,
  "estimated_hops": 1 | 2 | 3
}

Routing rules:
- Contains alphanumeric codes (e.g., "ASC 842", "IFRS 9", "GAAP") → keyword_heavy=true, requires_keyword=true
- "how is X connected/related to Y", "relationship between", "link between" → relational, requires_graph=true
- "compare X and Y", "difference between", "versus" → comparative, requires_sql=true, requires_vector=true
- "revenue", "earnings", "EPS", "growth rate", "net income", "profit margin" → metric, requires_sql=true
- "suppliers of X also supply Y", "companies that both...", "shared investors" → multi_hop, all paths true, estimated_hops=3
- Questions about politics, sports, entertainment unrelated to business/finance → out_of_scope
- Simple factual questions about a single company/document → factual, requires_vector=true

Entity extraction:
- Include company names, people, products, regulations mentioned
- Use the provided hints but refine based on context

Time range:
- Extract years mentioned (e.g., "2020 to 2023" → [2020, 2023])
- "last 3 years" → calculate from current year (2024)
- If no time mentioned, use null

estimated_hops:
- 1: Single direct lookup
- 2: Requires joining two pieces of information
- 3: Multi-hop reasoning across multiple entities

Return ONLY the JSON. No preamble, no markdown code blocks."""

        user_prompt = f"""Query: "{query}"

Entity hints detected: {entity_hints}

Classify this query according to the routing rules."""

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{settings.openrouter_base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {settings.openrouter_api_key}",
                        "HTTP-Referer": "https://github.com/nexus-rag",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": settings.llm_model_classify,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": 0.1,
                        "max_tokens": 500
                    }
                )
                response.raise_for_status()
                result = response.json()
                content = result["choices"][0]["message"]["content"]

                # Parse JSON from response (handle potential markdown wrapping)
                content = content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

                data = json.loads(content)

                # Validate and construct QueryPlan
                query_plan = QueryPlan(
                    query_text=query,
                    query_type=data.get("query_type", "factual"),
                    requires_graph=data.get("requires_graph", False),
                    requires_vector=data.get("requires_vector", False),
                    requires_sql=data.get("requires_sql", False),
                    requires_keyword=data.get("requires_keyword", False),
                    entities=data.get("entities", entity_hints),
                    time_range=tuple(data["time_range"]) if data.get("time_range") else None,
                    keyword_heavy=data.get("keyword_heavy", False),
                    estimated_hops=min(max(data.get("estimated_hops", 1), 1), 3)
                )

                # Cache the result
                if redis_client:
                    try:
                        cache_data = {
                            'query_text': query_plan.query_text,
                            'query_type': query_plan.query_type,
                            'requires_graph': query_plan.requires_graph,
                            'requires_vector': query_plan.requires_vector,
                            'requires_sql': query_plan.requires_sql,
                            'requires_keyword': query_plan.requires_keyword,
                            'entities': query_plan.entities,
                            'time_range': list(query_plan.time_range) if query_plan.time_range else None,
                            'keyword_heavy': query_plan.keyword_heavy,
                            'estimated_hops': query_plan.estimated_hops
                        }
                        await redis_client.setex(
                            self._generate_cache_key(query),
                            1800,  # 30 minutes TTL
                            json.dumps(cache_data)
                        )
                    except Exception as e:
                        logger.warning(f"Cache storage error: {e}")

                return query_plan

        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            # Return default QueryPlan on failure
            return QueryPlan(
                query_text=query,
                query_type="factual",
                requires_vector=True,
                requires_graph=False,
                requires_sql=False,
                requires_keyword=False,
                entities=entity_hints,
                time_range=None,
                keyword_heavy=False,
                estimated_hops=1
            )


async def main():
    classifier = QueryPlanClassifier()
    
    test_queries = [
        "What was Apple's revenue in 2023?",
        "How is Microsoft connected to OpenAI?",
        "Compare the profit margins of Tesla and Ford",
        "What does ASC 842 say about lease accounting?",
        "Which suppliers of Apple also supply Samsung?",
        "Who won the Super Bowl in 2024?",
        "Show me the growth rate of net income for Google from 2020 to 2023"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        plan = await classifier.classify(query)
        print(f"  Type: {plan.query_type}")
        print(f"  Graph: {plan.requires_graph}, Vector: {plan.requires_vector}, SQL: {plan.requires_sql}, Keyword: {plan.requires_keyword}")
        print(f"  Entities: {plan.entities}")
        print(f"  Time Range: {plan.time_range}")
        print(f"  Keyword Heavy: {plan.keyword_heavy}, Hops: {plan.estimated_hops}")


if __name__ == "__main__":
    asyncio.run(main())
