-- ================================
-- Fintech Agentic RAG - PostgreSQL Schema
-- ================================
-- This script initializes the database schema for:
-- - Prompt logging (LLM interactions)
-- - Error tracking
-- - Query metrics
-- - System configuration
-- ================================

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ================================
-- Prompt Logs Table
-- ================================
-- Stores all LLM prompts and responses for observability
CREATE TABLE IF NOT EXISTS prompt_logs (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    layer VARCHAR(50) NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    latency_ms FLOAT NOT NULL,
    metadata JSONB DEFAULT '{}',
    session_id VARCHAR(100) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Indexes for common queries
    CONSTRAINT prompt_logs_layer_check CHECK (layer IN (
        'layer1_indexing',
        'layer2_routing',
        'layer3_query_builder',
        'layer4_retrieval',
        'layer5_generation',
        'orchestrator',
        'other'
    ))
);

-- Indexes for prompt_logs
CREATE INDEX idx_prompt_logs_timestamp ON prompt_logs(timestamp DESC);
CREATE INDEX idx_prompt_logs_layer ON prompt_logs(layer);
CREATE INDEX idx_prompt_logs_session_id ON prompt_logs(session_id);
CREATE INDEX idx_prompt_logs_created_at ON prompt_logs(created_at DESC);
CREATE INDEX idx_prompt_logs_metadata_gin ON prompt_logs USING gin(metadata);

-- ================================
-- Error Logs Table
-- ================================
-- Stores all application errors with context
CREATE TABLE IF NOT EXISTS error_logs (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    layer VARCHAR(50) NOT NULL,
    error_type VARCHAR(100) NOT NULL,
    error_message TEXT NOT NULL,
    traceback TEXT,
    context JSONB DEFAULT '{}',
    severity VARCHAR(20) NOT NULL,
    session_id VARCHAR(100),
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT error_logs_severity_check CHECK (severity IN ('critical', 'error', 'warning', 'info'))
);

-- Indexes for error_logs
CREATE INDEX idx_error_logs_timestamp ON error_logs(timestamp DESC);
CREATE INDEX idx_error_logs_layer ON error_logs(layer);
CREATE INDEX idx_error_logs_severity ON error_logs(severity);
CREATE INDEX idx_error_logs_error_type ON error_logs(error_type);
CREATE INDEX idx_error_logs_resolved ON error_logs(resolved);
CREATE INDEX idx_error_logs_created_at ON error_logs(created_at DESC);
CREATE INDEX idx_error_logs_context_gin ON error_logs USING gin(context);

-- ================================
-- Query Metrics Table
-- ================================
-- Stores metrics for each RAG query
CREATE TABLE IF NOT EXISTS query_metrics (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    query TEXT NOT NULL,
    company VARCHAR(255),
    total_time_ms FLOAT NOT NULL,
    confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1),
    citations_count INTEGER DEFAULT 0,
    documents_retrieved INTEGER DEFAULT 0,
    routing_decision VARCHAR(50),
    metadata JSONB DEFAULT '{}',
    user_feedback INTEGER CHECK (user_feedback >= 1 AND user_feedback <= 5),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT query_metrics_routing_check CHECK (routing_decision IN (
        'semantic',
        'graph',
        'hybrid',
        'keyword',
        'other'
    ))
);

-- Indexes for query_metrics
CREATE INDEX idx_query_metrics_timestamp ON query_metrics(timestamp DESC);
CREATE INDEX idx_query_metrics_company ON query_metrics(company);
CREATE INDEX idx_query_metrics_confidence ON query_metrics(confidence);
CREATE INDEX idx_query_metrics_routing ON query_metrics(routing_decision);
CREATE INDEX idx_query_metrics_created_at ON query_metrics(created_at DESC);
CREATE INDEX idx_query_metrics_metadata_gin ON query_metrics USING gin(metadata);

-- ================================
-- System Configuration Table
-- ================================
-- Stores system-wide configuration key-value pairs
CREATE TABLE IF NOT EXISTS system_config (
    key VARCHAR(100) PRIMARY KEY,
    value TEXT NOT NULL,
    description TEXT,
    value_type VARCHAR(20) DEFAULT 'string',
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT system_config_type_check CHECK (value_type IN ('string', 'number', 'boolean', 'json'))
);

-- Index for system_config
CREATE INDEX idx_system_config_updated_at ON system_config(updated_at DESC);

-- ================================
-- Cache Statistics Table
-- ================================
-- Stores Redis cache performance metrics
CREATE TABLE IF NOT EXISTS cache_stats (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    cache_key VARCHAR(255) NOT NULL,
    hit BOOLEAN NOT NULL,
    ttl INTEGER,
    size_bytes INTEGER,
    operation VARCHAR(20) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT cache_stats_operation_check CHECK (operation IN ('get', 'set', 'delete', 'exists'))
);

-- Indexes for cache_stats
CREATE INDEX idx_cache_stats_timestamp ON cache_stats(timestamp DESC);
CREATE INDEX idx_cache_stats_cache_key ON cache_stats(cache_key);
CREATE INDEX idx_cache_stats_hit ON cache_stats(hit);
CREATE INDEX idx_cache_stats_operation ON cache_stats(operation);
CREATE INDEX idx_cache_stats_created_at ON cache_stats(created_at DESC);

-- ================================
-- Initial System Configuration
-- ================================
INSERT INTO system_config (key, value, description, value_type) VALUES
    ('app_version', '2.0.0', 'Application version', 'string'),
    ('architecture', '5-layer-agentic-rag', 'System architecture type', 'string'),
    ('max_workers', '4', 'Maximum Gunicorn workers', 'number'),
    ('request_timeout', '300', 'Request timeout in seconds', 'number'),
    ('cache_enabled', 'true', 'Enable Redis caching', 'boolean'),
    ('cache_ttl', '3600', 'Default cache TTL in seconds', 'number'),
    ('monitoring_enabled', 'true', 'Enable Prometheus monitoring', 'boolean'),
    ('langsmith_enabled', 'false', 'Enable LangSmith tracing', 'boolean')
ON CONFLICT (key) DO NOTHING;

-- ================================
-- Helpful Views
-- ================================

-- View: Recent errors summary
CREATE OR REPLACE VIEW recent_errors AS
SELECT
    layer,
    error_type,
    severity,
    COUNT(*) as count,
    MAX(timestamp) as last_occurrence,
    MIN(timestamp) as first_occurrence
FROM error_logs
WHERE timestamp > NOW() - INTERVAL '24 hours'
GROUP BY layer, error_type, severity
ORDER BY count DESC;

-- View: Query performance summary
CREATE OR REPLACE VIEW query_performance AS
SELECT
    routing_decision,
    COUNT(*) as total_queries,
    AVG(total_time_ms) as avg_time_ms,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_time_ms) as p50_time_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_time_ms) as p95_time_ms,
    AVG(confidence) as avg_confidence,
    AVG(documents_retrieved) as avg_docs_retrieved
FROM query_metrics
WHERE timestamp > NOW() - INTERVAL '24 hours'
GROUP BY routing_decision;

-- View: Cache hit rate
CREATE OR REPLACE VIEW cache_hit_rate AS
SELECT
    DATE_TRUNC('hour', timestamp) as hour,
    COUNT(*) FILTER (WHERE hit = true) as hits,
    COUNT(*) FILTER (WHERE hit = false) as misses,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE hit = true) / NULLIF(COUNT(*), 0),
        2
    ) as hit_rate_percent
FROM cache_stats
WHERE timestamp > NOW() - INTERVAL '24 hours'
GROUP BY hour
ORDER BY hour DESC;

-- ================================
-- Maintenance Functions
-- ================================

-- Function: Clean old logs (keep last 30 days)
CREATE OR REPLACE FUNCTION cleanup_old_logs()
RETURNS void AS $$
BEGIN
    DELETE FROM prompt_logs WHERE created_at < NOW() - INTERVAL '30 days';
    DELETE FROM error_logs WHERE created_at < NOW() - INTERVAL '30 days';
    DELETE FROM query_metrics WHERE created_at < NOW() - INTERVAL '30 days';
    DELETE FROM cache_stats WHERE created_at < NOW() - INTERVAL '30 days';
END;
$$ LANGUAGE plpgsql;

-- Function: Get system health
CREATE OR REPLACE FUNCTION get_system_health()
RETURNS TABLE (
    metric VARCHAR,
    value TEXT,
    status VARCHAR
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        'total_queries_24h'::VARCHAR,
        COUNT(*)::TEXT,
        CASE WHEN COUNT(*) > 0 THEN 'healthy' ELSE 'warning' END
    FROM query_metrics WHERE timestamp > NOW() - INTERVAL '24 hours'
    UNION ALL
    SELECT
        'total_errors_24h'::VARCHAR,
        COUNT(*)::TEXT,
        CASE WHEN COUNT(*) < 100 THEN 'healthy' ELSE 'critical' END
    FROM error_logs WHERE timestamp > NOW() - INTERVAL '24 hours' AND severity IN ('critical', 'error')
    UNION ALL
    SELECT
        'avg_confidence_24h'::VARCHAR,
        ROUND(AVG(confidence)::NUMERIC, 2)::TEXT,
        CASE WHEN AVG(confidence) > 0.7 THEN 'healthy' ELSE 'warning' END
    FROM query_metrics WHERE timestamp > NOW() - INTERVAL '24 hours'
    UNION ALL
    SELECT
        'cache_hit_rate_24h'::VARCHAR,
        ROUND(
            100.0 * COUNT(*) FILTER (WHERE hit = true) / NULLIF(COUNT(*), 0)::NUMERIC,
            2
        )::TEXT || '%',
        CASE
            WHEN 100.0 * COUNT(*) FILTER (WHERE hit = true) / NULLIF(COUNT(*), 0) > 50 THEN 'healthy'
            WHEN 100.0 * COUNT(*) FILTER (WHERE hit = true) / NULLIF(COUNT(*), 0) > 30 THEN 'warning'
            ELSE 'critical'
        END
    FROM cache_stats WHERE timestamp > NOW() - INTERVAL '24 hours';
END;
$$ LANGUAGE plpgsql;

-- ================================
-- Grants (for application user)
-- ================================
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO fintech_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO fintech_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO fintech_user;

-- ================================
-- Completion Message
-- ================================
DO $$
BEGIN
    RAISE NOTICE '✅ Database schema initialized successfully';
    RAISE NOTICE '✅ Tables created: prompt_logs, error_logs, query_metrics, system_config, cache_stats';
    RAISE NOTICE '✅ Views created: recent_errors, query_performance, cache_hit_rate';
    RAISE NOTICE '✅ Functions created: cleanup_old_logs(), get_system_health()';
END $$;
