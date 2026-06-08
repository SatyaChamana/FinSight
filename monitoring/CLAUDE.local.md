# Observability and Dashboards -- monitoring/

## Directory Structure

- `dashboards/` -- Grafana dashboard JSON definitions
- `rules/` -- Prometheus alerting and recording rules

## Conventions

- Dashboards are version-controlled as JSON, provisioned via Grafana config.
- Prometheus rules use standard PromQL.
- Alert severity levels: `critical`, `warning`, `info`.
- Every service-level metric exposed by Go service should have a corresponding dashboard panel.
- Key metrics to track: inference latency (p50/p95/p99), error rate, cache hit ratio, per-tenant request volume, model version.
