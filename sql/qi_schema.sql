CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE SCHEMA IF NOT EXISTS qi;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'entity_type' AND typnamespace = 'qi'::regnamespace) THEN
        CREATE TYPE qi.entity_type AS ENUM (
            'stock',
            'etf',
            'fund',
            'index',
            'industry',
            'macro_indicator',
            'financial_metric'
        );
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'source_type' AND typnamespace = 'qi'::regnamespace) THEN
        CREATE TYPE qi.source_type AS ENUM (
            'news',
            'announcement',
            'faq',
            'product_doc',
            'research_note',
            'market_api',
            'industry_sql',
            'fundamental_sql',
            'macro_sql'
        );
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'product_type' AND typnamespace = 'qi'::regnamespace) THEN
        CREATE TYPE qi.product_type AS ENUM (
            'stock',
            'etf',
            'fund',
            'index',
            'macro',
            'generic_market',
            'unknown'
        );
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'time_scope' AND typnamespace = 'qi'::regnamespace) THEN
        CREATE TYPE qi.time_scope AS ENUM (
            'today',
            'recent_3d',
            'recent_1w',
            'recent_1m',
            'recent_1q',
            'long_term',
            'unspecified'
        );
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'question_style' AND typnamespace = 'qi'::regnamespace) THEN
        CREATE TYPE qi.question_style AS ENUM ('fact', 'why', 'compare', 'advice', 'forecast');
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'action_type' AND typnamespace = 'qi'::regnamespace) THEN
        CREATE TYPE qi.action_type AS ENUM ('buy', 'sell', 'hold', 'reduce', 'observe', 'unknown');
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS qi.entity_master (
    entity_id BIGSERIAL PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    symbol TEXT NULL,
    market TEXT NOT NULL DEFAULT 'CN',
    exchange TEXT NULL,
    entity_type qi.entity_type NOT NULL,
    industry_name TEXT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (symbol, entity_type, market)
);

CREATE INDEX IF NOT EXISTS idx_qi_entity_symbol ON qi.entity_master(symbol);
CREATE INDEX IF NOT EXISTS idx_qi_entity_type ON qi.entity_master(entity_type);
CREATE INDEX IF NOT EXISTS idx_qi_entity_name_trgm ON qi.entity_master USING GIN (normalized_name gin_trgm_ops);

CREATE TABLE IF NOT EXISTS qi.entity_alias (
    alias_id BIGSERIAL PRIMARY KEY,
    entity_id BIGINT NOT NULL REFERENCES qi.entity_master(entity_id),
    alias_text TEXT NOT NULL,
    normalized_alias TEXT NOT NULL,
    alias_type TEXT NOT NULL,
    priority SMALLINT NOT NULL DEFAULT 100,
    is_official BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE (entity_id, normalized_alias)
);

CREATE INDEX IF NOT EXISTS idx_qi_alias_trgm ON qi.entity_alias USING GIN (normalized_alias gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_qi_alias_priority ON qi.entity_alias(priority);

CREATE TABLE IF NOT EXISTS qi.macro_term_dict (
    term_id BIGSERIAL PRIMARY KEY,
    term_text TEXT NOT NULL,
    normalized_term TEXT NOT NULL,
    indicator_code TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    unit TEXT NULL,
    frequency TEXT NULL
);

CREATE TABLE IF NOT EXISTS qi.label_dictionary (
    label_type TEXT NOT NULL,
    label_key TEXT NOT NULL,
    label_name TEXT NOT NULL,
    description TEXT NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    PRIMARY KEY (label_type, label_key)
);

CREATE TABLE IF NOT EXISTS qi.query_sample (
    query_id UUID PRIMARY KEY,
    raw_query TEXT NOT NULL,
    normalized_query TEXT NULL,
    source TEXT NOT NULL,
    split TEXT NOT NULL CHECK (split IN ('train', 'valid', 'test')),
    dialog_context JSONB NOT NULL DEFAULT '[]'::jsonb,
    user_profile JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS qi.query_label_product (
    query_id UUID PRIMARY KEY REFERENCES qi.query_sample(query_id),
    product_type qi.product_type NOT NULL,
    confidence NUMERIC(4, 3) NULL
);

CREATE TABLE IF NOT EXISTS qi.query_label_intent (
    query_id UUID NOT NULL REFERENCES qi.query_sample(query_id),
    label_key TEXT NOT NULL,
    is_positive BOOLEAN NOT NULL DEFAULT TRUE,
    PRIMARY KEY (query_id, label_key)
);

CREATE TABLE IF NOT EXISTS qi.query_label_topic (
    query_id UUID NOT NULL REFERENCES qi.query_sample(query_id),
    label_key TEXT NOT NULL,
    is_positive BOOLEAN NOT NULL DEFAULT TRUE,
    PRIMARY KEY (query_id, label_key)
);

CREATE TABLE IF NOT EXISTS qi.query_slot_gold (
    query_id UUID PRIMARY KEY REFERENCES qi.query_sample(query_id),
    question_style qi.question_style NOT NULL,
    time_scope qi.time_scope NOT NULL,
    forecast_horizon TEXT NOT NULL,
    action_type qi.action_type NOT NULL,
    comparison_target TEXT NULL,
    metrics_requested JSONB NOT NULL DEFAULT '[]'::jsonb,
    missing_slots JSONB NOT NULL DEFAULT '[]'::jsonb
);

CREATE TABLE IF NOT EXISTS qi.query_entity_gold (
    query_id UUID NOT NULL REFERENCES qi.query_sample(query_id),
    mention TEXT NOT NULL,
    start_offset INTEGER NOT NULL,
    end_offset INTEGER NOT NULL,
    entity_type qi.entity_type NOT NULL,
    canonical_entity_id BIGINT NULL REFERENCES qi.entity_master(entity_id),
    PRIMARY KEY (query_id, start_offset, end_offset)
);

CREATE TABLE IF NOT EXISTS qi.query_source_plan_gold (
    query_id UUID NOT NULL REFERENCES qi.query_sample(query_id),
    source_key qi.source_type NOT NULL,
    priority_rank SMALLINT NOT NULL,
    PRIMARY KEY (query_id, source_key)
);

CREATE TABLE IF NOT EXISTS qi.annotation_meta (
    query_id UUID NOT NULL REFERENCES qi.query_sample(query_id),
    annotator TEXT NOT NULL,
    version TEXT NOT NULL,
    agreement_score NUMERIC(4, 3) NULL CHECK (agreement_score BETWEEN 0 AND 1),
    notes TEXT NULL,
    PRIMARY KEY (query_id, annotator, version)
);

CREATE TABLE IF NOT EXISTS qi.document_store (
    doc_id BIGSERIAL PRIMARY KEY,
    source_type qi.source_type NOT NULL,
    source_name TEXT NOT NULL,
    source_url TEXT NULL,
    title TEXT NOT NULL,
    summary TEXT NULL,
    body TEXT NOT NULL,
    publish_time TIMESTAMPTZ NULL,
    product_type qi.product_type NOT NULL DEFAULT 'unknown',
    credibility_score NUMERIC(4, 3) NOT NULL DEFAULT 0.500 CHECK (credibility_score BETWEEN 0 AND 1),
    language TEXT NOT NULL DEFAULT 'zh',
    fts_document TSVECTOR NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_qi_document_fts ON qi.document_store USING GIN (fts_document);
CREATE INDEX IF NOT EXISTS idx_qi_document_source_time ON qi.document_store(source_type, publish_time DESC);
CREATE INDEX IF NOT EXISTS idx_qi_document_title_trgm ON qi.document_store USING GIN (title gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_qi_document_summary_trgm ON qi.document_store USING GIN (COALESCE(summary, '') gin_trgm_ops);

CREATE TABLE IF NOT EXISTS qi.document_entity_map (
    doc_id BIGINT NOT NULL REFERENCES qi.document_store(doc_id),
    entity_id BIGINT NOT NULL REFERENCES qi.entity_master(entity_id),
    match_strength NUMERIC(4, 3) NOT NULL DEFAULT 1.000 CHECK (match_strength BETWEEN 0 AND 1),
    PRIMARY KEY (doc_id, entity_id)
);

CREATE TABLE IF NOT EXISTS qi.market_daily (
    symbol TEXT NOT NULL,
    trade_date DATE NOT NULL,
    open NUMERIC(18, 4),
    high NUMERIC(18, 4),
    low NUMERIC(18, 4),
    close NUMERIC(18, 4),
    pct_change NUMERIC(8, 4),
    volume NUMERIC(20, 2),
    amount NUMERIC(20, 2),
    PRIMARY KEY (symbol, trade_date)
);

CREATE TABLE IF NOT EXISTS qi.fundamental_quarterly (
    symbol TEXT NOT NULL,
    report_date DATE NOT NULL,
    revenue NUMERIC(20, 2),
    net_profit NUMERIC(20, 2),
    roe NUMERIC(8, 4),
    gross_margin NUMERIC(8, 4),
    pe_ttm NUMERIC(12, 4),
    pb NUMERIC(12, 4),
    PRIMARY KEY (symbol, report_date)
);

CREATE TABLE IF NOT EXISTS qi.industry_daily (
    industry_name TEXT NOT NULL,
    trade_date DATE NOT NULL,
    pct_change NUMERIC(8, 4),
    pe NUMERIC(12, 4),
    pb NUMERIC(12, 4),
    turnover NUMERIC(12, 4),
    PRIMARY KEY (industry_name, trade_date)
);

CREATE TABLE IF NOT EXISTS qi.macro_indicator (
    indicator_code TEXT NOT NULL,
    metric_date DATE NOT NULL,
    metric_value NUMERIC(20, 6) NOT NULL,
    unit TEXT NULL,
    PRIMARY KEY (indicator_code, metric_date)
);

CREATE TABLE IF NOT EXISTS qi.retrieval_query (
    query_id UUID PRIMARY KEY REFERENCES qi.query_sample(query_id),
    nlu_snapshot JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS qi.retrieval_candidate (
    query_id UUID NOT NULL REFERENCES qi.retrieval_query(query_id),
    source_type qi.source_type NOT NULL,
    source_record_id TEXT NOT NULL,
    retrieval_score NUMERIC(8, 4) NULL,
    feature_json JSONB NOT NULL,
    PRIMARY KEY (query_id, source_type, source_record_id)
);

CREATE TABLE IF NOT EXISTS qi.retrieval_qrel (
    query_id UUID NOT NULL REFERENCES qi.retrieval_query(query_id),
    source_type qi.source_type NOT NULL,
    source_record_id TEXT NOT NULL,
    relevance SMALLINT NOT NULL CHECK (relevance IN (0, 1, 2)),
    rationale TEXT NULL,
    PRIMARY KEY (query_id, source_type, source_record_id)
);
