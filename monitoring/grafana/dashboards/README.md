# Grafana Dashboards

Este directorio contiene los dashboards de Grafana para el sistema Fintech Agentic RAG.

## Dashboards Disponibles

Los dashboards se provisionan automáticamente cuando se inicia Grafana. Puedes agregar dashboards adicionales colocando archivos JSON en este directorio.

## Crear un Dashboard

### Opción 1: Desde la UI de Grafana

1. Accede a Grafana: http://localhost:3000
2. Login: admin / [GRAFANA_PASSWORD from .env]
3. Crea tu dashboard manualmente
4. Exporta el dashboard como JSON (Share → Export → Save to file)
5. Coloca el archivo JSON en este directorio
6. Grafana lo cargará automáticamente

### Opción 2: Dashboard Básico Recomendado

Crea un archivo `rag-system-overview.json` con los siguientes paneles:

**Métricas Clave**:
- Request Rate (requests/sec)
- Request Latency (p50, p95, p99)
- Error Rate (errors/min)
- Active Requests
- Cache Hit Rate
- Query Confidence Score
- Documents Retrieved per Query

**Queries Prometheus Útiles**:

```promql
# Request rate
rate(rag_requests_total[5m])

# Request latency p95
histogram_quantile(0.95, rate(rag_request_duration_seconds_bucket[5m]))

# Error rate
rate(rag_errors_total[5m])

# Active requests
rag_active_requests

# Cache hit rate
rate(rag_cache_hits_total[5m]) / (rate(rag_cache_hits_total[5m]) + rate(rag_cache_misses_total[5m]))
```

**Queries PostgreSQL Útiles**:

```sql
-- Recent errors (últimas 24h)
SELECT * FROM recent_errors ORDER BY count DESC LIMIT 10;

-- Query performance por routing decision
SELECT * FROM query_performance ORDER BY total_queries DESC;

-- Cache hit rate por hora
SELECT * FROM cache_hit_rate ORDER BY hour DESC LIMIT 24;

-- System health
SELECT * FROM get_system_health();
```

## Estructura de Archivos

```
monitoring/grafana/dashboards/
├── dashboard.yml              # Provisioning config
├── README.md                  # Este archivo
└── [tu-dashboard].json        # Tus dashboards personalizados
```

## Dashboards Recomendados

### 1. System Overview
- Request metrics (rate, latency, errors)
- Resource usage (CPU, memory, connections)
- Cache performance
- Database metrics

### 2. RAG Pipeline Performance
- Layer execution times
- Document retrieval metrics
- Confidence scores distribution
- Routing decision breakdown

### 3. Error Tracking
- Error rate por layer
- Error types distribution
- Recent errors table
- Error trends

### 4. Business Metrics
- Queries por company
- User feedback distribution
- Query complexity analysis
- Popular queries

## Tips

- Usa variables para filtrar por company, layer, etc.
- Configura alertas para métricas críticas
- Usa template queries para dashboards dinámicos
- Exporta y versionea tus dashboards en git
