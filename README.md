# FinSight

Multi-tenant portfolio risk prediction service. Go gRPC backend serves ONNX model inference; Python ML pipeline handles training and export.

## Architecture

```
Client -> gRPC -> Go Service -> ONNX Runtime -> Risk Prediction
                      |
                PostgreSQL + Redis
```

- **Go service** loads ONNX models, serves predictions over gRPC, handles multi-tenancy
- **Python pipeline** trains LSTM/Transformer models on financial data, exports to ONNX
- **Infrastructure** runs on AWS (EKS, RDS, ElastiCache) via Terraform

## Project Structure

```
cmd/server/              Go entrypoint
internal/                Go packages (server, service, model, tenant, store, cache, auth, ratelimit, telemetry)
proto/finsight/v1/       Protobuf definitions
ml/                      Python ML pipeline (data, models, training, export, notebooks)
deploy/                  Kubernetes manifests (Kustomize base + overlays)
terraform/               AWS infrastructure (modules + environments)
monitoring/              Grafana dashboards, Prometheus rules
scripts/                 Dev tooling
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Service | Go 1.22+, gRPC, protobuf |
| ML Pipeline | Python 3.11+, PyTorch, ONNX |
| Model Serving | ONNX Runtime (onnxruntime-go) |
| Database | PostgreSQL 15+ (pgx) |
| Cache | Redis 7+ (go-redis) |
| Infrastructure | Terraform on AWS (EKS, RDS, ElastiCache, S3) |
| Orchestration | Kubernetes 1.28+ with Kustomize |
| Observability | OpenTelemetry, Prometheus, Grafana, Jaeger |
| CI/CD | GitHub Actions |

## Quick Start

```bash
# Build Go service
go build -o bin/finsight ./cmd/server

# Run Go service
go run ./cmd/server

# Run Go tests
go test ./...

# Run Python ML tests
cd ml && python -m pytest

# Generate protobuf code
./scripts/generate-proto.sh

# Full stack via Docker
docker-compose up -d
```

## Development

```bash
# Lint Go
golangci-lint run

# Format Go
gofmt -w .

# Lint Python
cd ml && ruff check .

# Terraform
cd terraform/environments/dev && terraform plan
```

## License

Private repository.
