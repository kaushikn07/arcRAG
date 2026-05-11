# NEXUS - Agentic RAG System for Financial Document Intelligence

## Project Structure

```
nexus/
  config.py                  # pydantic-settings, loads from .env
  db/
    neon.py                  # asyncpg connection pool, fetch/execute helpers
    schema.sql               # all CREATE TABLE / CREATE EXTENSION statements
  ingestion/
    parser.py
    chunker.py
    embedder.py
    hyde.py
    metrics_extractor.py
    graph_builder.py
    orchestrator.py
  retrieval/
    dense.py
    keyword.py
    hybrid.py
    graph.py
    sql_retriever.py
  agent/
    tools.py
    react_loop.py
    query_plan.py
  guardrails/
    input_guard.py
    output_guard.py
  evaluation/
    eval_runner.py
    trajectory_scorer.py
  api/
    main.py
    routes.py
  docker-compose.yml
  .env.example
  requirements.txt
```

## Setup Instructions

1. Copy `.env.example` to `.env` and fill in your credentials
2. Start Docker services: `docker-compose up -d`
3. Install dependencies: `pip install -r requirements.txt`
4. Initialize database with schema.sql
