-- NEXUS Database Schema for Neon (Postgres with pgvector)
-- This schema supports financial document intelligence with RAG capabilities

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ============================================
-- DOCUMENTS TABLE
-- Stores metadata about ingested financial documents
-- ============================================
CREATE TABLE IF NOT EXISTS documents (
    doc_id SERIAL PRIMARY KEY,
    company VARCHAR(255) NOT NULL,
    year INTEGER NOT NULL,
    doc_type VARCHAR(100) NOT NULL,  -- e.g., '10-K', '10-Q', 'Annual Report'
    filename VARCHAR(500) NOT NULL,
    page_count INTEGER DEFAULT 0,
    ingested_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    stage_completed VARCHAR(50) DEFAULT 'uploaded'  -- uploaded, parsed, chunked, embedded, indexed
);

CREATE INDEX idx_documents_company_year ON documents(company, year);
CREATE INDEX idx_documents_doc_type ON documents(doc_type);
CREATE INDEX idx_documents_ingested_at ON documents(ingested_at);

-- ============================================
-- SECTIONS TABLE
-- Stores document sections extracted during parsing
-- ============================================
CREATE TABLE IF NOT EXISTS sections (
    section_id SERIAL PRIMARY KEY,
    doc_id INTEGER NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    heading VARCHAR(500),
    heading_level INTEGER DEFAULT 1,
    sequence_order INTEGER NOT NULL,
    content_type VARCHAR(50) DEFAULT 'text',  -- text, table, figure, etc.
    raw_text TEXT,
    token_count INTEGER DEFAULT 0
);

CREATE INDEX idx_sections_doc_id ON sections(doc_id);
CREATE INDEX idx_sections_sequence ON sections(doc_id, sequence_order);

-- ============================================
-- CHUNKS TABLE
-- Stores text chunks for retrieval with embeddings
-- ============================================
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id SERIAL PRIMARY KEY,
    doc_id INTEGER NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    section_id INTEGER REFERENCES sections(section_id) ON DELETE SET NULL,
    chunk_level INTEGER DEFAULT 0,  -- 0 = top-level, higher = sub-chunks
    parent_chunk_id INTEGER REFERENCES chunks(chunk_id) ON DELETE SET NULL,
    text TEXT NOT NULL,
    token_count INTEGER DEFAULT 0,
    embedding vector(384),
    text_search tsvector GENERATED ALWAYS AS (to_tsvector('english', COALESCE(text, ''))) STORED,
    page_number INTEGER DEFAULT 0
);

CREATE INDEX idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX idx_chunks_section_id ON chunks(section_id);
CREATE INDEX idx_chunks_parent ON chunks(parent_chunk_id);
CREATE INDEX idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_chunks_text_search ON chunks USING GIN (text_search);
CREATE INDEX idx_chunks_page ON chunks(doc_id, page_number);

-- ============================================
-- HYPOTHETICAL QUESTIONS TABLE
-- Stores HyDE-generated questions for improved retrieval
-- ============================================
CREATE TABLE IF NOT EXISTS hypothetical_questions (
    q_id SERIAL PRIMARY KEY,
    chunk_id INTEGER NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    doc_id INTEGER NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    question_text TEXT NOT NULL,
    embedding vector(384)
);

CREATE INDEX idx_hypothetical_questions_chunk_id ON hypothetical_questions(chunk_id);
CREATE INDEX idx_hypothetical_questions_doc_id ON hypothetical_questions(doc_id);
CREATE INDEX idx_hypothetical_questions_embedding ON hypothetical_questions USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ============================================
-- FINANCIAL METRICS TABLE
-- Stores extracted financial metrics for structured queries
-- ============================================
CREATE TABLE IF NOT EXISTS financial_metrics (
    metric_id SERIAL PRIMARY KEY,
    doc_id INTEGER NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    company VARCHAR(255) NOT NULL,
    year INTEGER NOT NULL,
    period VARCHAR(50),  -- e.g., 'Q1', 'Q2', 'FY', 'TTM'
    metric_name VARCHAR(200) NOT NULL,  -- e.g., 'Revenue', 'Net Income', 'EBITDA'
    value NUMERIC NOT NULL,
    unit VARCHAR(50) DEFAULT 'USD',  -- e.g., 'USD', 'EUR', 'millions', 'billions'
    source_section_id INTEGER REFERENCES sections(section_id) ON DELETE SET NULL
);

CREATE INDEX idx_financial_metrics_doc_id ON financial_metrics(doc_id);
CREATE INDEX idx_financial_metrics_company_year ON financial_metrics(company, year);
CREATE INDEX idx_financial_metrics_metric_name ON financial_metrics(metric_name);
CREATE INDEX idx_financial_metrics_period ON financial_metrics(period);

-- ============================================
-- UTILITY FUNCTIONS
-- ============================================

-- Function to update stage_completed timestamp
CREATE OR REPLACE FUNCTION update_stage_completed()
RETURNS TRIGGER AS $$
BEGIN
    NEW.stage_completed := CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- INITIAL DATA (Optional seed data)
-- ============================================

-- Comments for documentation
COMMENT ON TABLE documents IS 'Stores metadata about ingested financial documents';
COMMENT ON TABLE sections IS 'Stores document sections extracted during parsing';
COMMENT ON TABLE chunks IS 'Stores text chunks for retrieval with vector embeddings';
COMMENT ON TABLE hypothetical_questions IS 'Stores HyDE-generated questions for improved retrieval';
COMMENT ON TABLE financial_metrics IS 'Stores extracted financial metrics for structured queries';
