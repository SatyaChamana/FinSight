# FinSight -- Multi-Tenant Portfolio Risk Predictor

## Developer Context

I am learning Go from scratch through this project. I know Python well but have zero Go experience. Every Go concept should be explained when first introduced. Move at a teaching pace: one new concept per task, not five. Prefer working code with clear explanations over clever/idiomatic code in early phases. We will refactor toward idiomatic Go as my understanding grows.

**Do not rush. Do not skip explanations. Do not assume I know Go idioms.**

## Project Overview

A Go-based gRPC microservice that serves portfolio risk predictions to multiple tenants. Python ML pipeline trains LSTM/Transformer models on historical financial data, exports to ONNX. Go service loads ONNX models and serves inference over gRPC with full observability.

See @FINSIGHT_PROJECT_PLAN.md for the full phased build plan with architecture details.

## Tech Stack

- **Service Layer:** Go 1.22+, gRPC, protobuf
- **ML Pipeline:** Python 3.11+, PyTorch, ONNX export
- **Model Serving:** ONNX Runtime via onnxruntime-go
- **Database:** PostgreSQL 15+ (pgx driver, no ORM)
- **Cache:** Redis 7+ (go-redis)
- **Infra:** Terraform 1.7+ on AWS (EKS, RDS, ElastiCache, S3, ECR)
- **Orchestration:** Kubernetes 1.28+ with Kustomize overlays
- **Observability:** OpenTelemetry + Prometheus + Grafana + Jaeger
- **CI/CD:** GitHub Actions

## Commands

- Build: `go build -o bin/finsight ./cmd/server`
- Run: `go run ./cmd/server`
- Test: `go test ./...`
- Test verbose: `go test -v ./...`
- Test single package: `go test -v ./internal/<package>`
- Lint: `golangci-lint run`
- Format: `gofmt -w .`
- Proto generate: `./scripts/generate-proto.sh`
- Python ML tests: `cd ml && python -m pytest`
- Python lint: `cd ml && ruff check .`
- Docker build: `docker build -t finsight:dev .`
- Docker compose (full stack): `docker-compose up -d`
- Terraform plan: `cd terraform/environments/dev && terraform plan`
- Terraform apply: `cd terraform/environments/dev && terraform apply`

## Repository Layout

```
finsight/
  cmd/server/          -- Go entrypoint (main.go)
  internal/            -- Private Go packages (server, service, model, tenant, store, cache, auth, ratelimit, telemetry)
  proto/finsight/v1/   -- Protobuf definitions
  ml/                  -- Python ML pipeline (data, models, training, export, notebooks)
  deploy/              -- Kubernetes manifests (base + overlays for dev/prod)
  terraform/           -- AWS IaC (modules + environments)
  monitoring/          -- Grafana dashboards, Prometheus rules
  scripts/             -- Dev tooling (proto generation, seed data)
```

## Module-Specific Instructions

Claude Code should read the relevant CLAUDE.local.md when working in a specific directory. These contain module-specific conventions and context:

- `internal/CLAUDE.local.md` -- Go service layer conventions
- `ml/CLAUDE.local.md` -- Python ML pipeline conventions
- `proto/CLAUDE.local.md` -- Protobuf and gRPC conventions
- `deploy/CLAUDE.local.md` -- Kubernetes deployment conventions
- `terraform/CLAUDE.local.md` -- Terraform IaC conventions
- `monitoring/CLAUDE.local.md` -- Observability and dashboards conventions

**Read the relevant module file before making changes in that directory.**

## Conventions

### Go Code

- Always handle errors explicitly. Never use `_` to discard errors unless there is a documented reason.
- Use `context.Context` as the first parameter in any function that does I/O or crosses a service boundary.
- Naming: packages are lowercase single words, interfaces end with `-er` when sensible (e.g., `RiskScorer`), exported types are PascalCase.
- Tests live next to the code they test (e.g., `store.go` and `store_test.go` in the same package).
- Use table-driven tests wherever multiple input/output combinations exist.
- No global mutable state. Pass dependencies via constructor injection.
- Database queries use raw SQL with pgx. No ORM.

### Python Code

- ML pipeline uses PyTorch for training, ONNX for export.
- All data processing uses pandas and numpy.
- Type hints on all function signatures.
- Tests use pytest.
- Linting with ruff, formatting with black.

### Protobuf

- All proto files live under `proto/finsight/v1/`.
- Package name: `finsight.v1`.
- Use `google.protobuf.Timestamp` for time fields, never strings.
- Run `./scripts/generate-proto.sh` after any proto change.

### Git

- Branch names: `phase-<N>/<short-description>` (e.g., `phase-1/grpc-skeleton`)
- Commit messages: conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`)
- Always run `go test ./... && golangci-lint run` before committing Go changes.
- Never commit generated protobuf Go code without regenerating from source protos first.

### Multi-Tenancy

- Every database table has a `tenant_id` column.
- Every query must filter on `tenant_id`. No exceptions.
- Tenant ID is extracted from gRPC metadata in an interceptor and stored in `context.Context`.
- Use Postgres Row-Level Security (RLS) as a safety net on top of application-level filtering.

## Phase Tracking

Current phase: **Phase 0 -- Foundation and Go Basics**

When starting a new phase, I will say: "Starting Phase X". Read the relevant section of @FINSIGHT_PROJECT_PLAN.md and walk me through it step by step.

## Writing Style Constraint

IMPORTANT: Never use em dashes in any generated code comments, documentation, commit messages, or explanations. Use commas, periods, semicolons, or parentheses instead.

## Skills (Matt Pocock)

This project uses skills from `mattpocock/skills`. Install them with:

```bash
npx skills@latest add mattpocock/skills
```

Key skills available:
- `/grill-with-docs` -- Use before starting any new phase or major feature. Updates CONTEXT.md and ADRs.
- `/tdd` -- Use for all Go code. Red-green-refactor loop.
- `/diagnose` -- Use when debugging hard issues.
- `/to-prd` -- Use to formalize a feature into a PRD issue.
- `/to-issues` -- Use to break a phase into GitHub issues.
- `/improve-codebase-architecture` -- Run every few days to check for design drift.
- `/zoom-out` -- Use when Go code structure is confusing.
- `/setup-matt-pocock-skills` -- Run once after installing to configure issue tracker and labels.

After installing, run `/setup-matt-pocock-skills` in Claude Code to configure the issue tracker (GitHub), triage labels, and docs location for this project.
