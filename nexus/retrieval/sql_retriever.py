"""SQL-based retrieval for financial metrics using NL-to-SQL."""
import httpx
from typing import List, Optional
from db.neon import fetch
from config import settings


async def sql_retrieve(
    nl_query: str,
    entities: List[str],
    company_filter: Optional[str] = None
) -> str:
    """
    Generate SQL from natural language query and execute it.
    
    Args:
        nl_query: Natural language query
        entities: List of extracted entities (companies, etc.)
        company_filter: Optional explicit company filter
    
    Returns:
        Formatted result as markdown table or "No data found."
    
    Raises:
        ValueError: If generated SQL contains DROP/DELETE/UPDATE/INSERT
    """
    # Determine company filter
    if company_filter:
        company = company_filter
    elif entities:
        company = entities[0]  # Use first entity as company
    else:
        company = None
    
    # System prompt for SQL generation
    system_prompt = """You are a SQL expert for financial data queries.
    
Schema:
- financial_metrics(metric_id, doc_id, company, year, period, metric_name, value NUMERIC, unit, source_section_id)
- documents(doc_id, company, year, doc_type, filename, page_count, ingested_at, stage_completed)

Rules:
1. SELECT only - NEVER generate DROP, DELETE, UPDATE, INSERT
2. Only query the financial_metrics table
3. Must filter by company if one is provided
4. Return ONLY the SQL query, no explanations
5. Use parameterized-style syntax with $1, $2, etc. for values

Example:
Q: "What was Apple's revenue in 2023?"
A: SELECT metric_name, value, unit, period FROM financial_metrics WHERE company = $1 AND year = $2 AND metric_name = 'revenue'
"""
    
    # Build user prompt
    if company:
        user_prompt = f"""Generate SQL for: {nl_query}
Company filter: {company}
Year: extract from query if mentioned, otherwise omit."""
    else:
        user_prompt = f"""Generate SQL for: {nl_query}
No specific company filter."""
    
    # Call OpenRouter API
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.openrouter_base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "HTTP-Referer": "https://github.com/nexus-rag",
                "Content-Type": "application/json"
            },
            json={
                "model": settings.llm_model_fast,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": 200
            }
        )
        response.raise_for_status()
        data = response.json()
    
    # Extract generated SQL
    generated_sql = data["choices"][0]["message"]["content"].strip()
    
    # Remove markdown code blocks if present
    if generated_sql.startswith("```sql"):
        generated_sql = generated_sql[6:]
    if generated_sql.startswith("```"):
        generated_sql = generated_sql[3:]
    if generated_sql.endswith("```"):
        generated_sql = generated_sql[:-3]
    generated_sql = generated_sql.strip()
    
    # Security validation
    forbidden_keywords = ["DROP", "DELETE", "UPDATE", "INSERT"]
    sql_upper = generated_sql.upper()
    for keyword in forbidden_keywords:
        if keyword in sql_upper:
            raise ValueError(f"Generated SQL contains forbidden keyword: {keyword}")
    
    # Execute SQL with parameters
    params = []
    if company:
        params.append(company)
        # Try to extract year from query
        import re
        year_match = re.search(r'\b(20\d{2}|19\d{2})\b', nl_query)
        if year_match:
            params.append(int(year_match.group(1)))
    
    try:
        results = await fetch(generated_sql, *params)
        
        if not results:
            return "No data found."
        
        # Format as markdown table
        if len(results) == 0:
            return "No data found."
        
        # Get column names from first result
        columns = list(results[0].keys())
        
        # Build markdown table
        header = " | ".join(columns)
        separator = " | ".join(["---"] * len(columns))
        rows = []
        for row in results:
            row_values = [str(row.get(col, "")) for col in columns]
            rows.append(" | ".join(row_values))
        
        markdown_table = f"| {header} |\n| {separator} |\n"
        for row in rows:
            markdown_table += f"| {row} |\n"
        
        return markdown_table
    
    except Exception as e:
        return f"Error executing query: {str(e)}"
