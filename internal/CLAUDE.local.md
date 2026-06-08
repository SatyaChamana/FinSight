# Go Service Layer -- internal/

## Developer Reminder

The developer is learning Go. Explain every new pattern, interface, or concurrency primitive when first introduced. Link to relevant Go documentation or "Effective Go" sections when appropriate.

## Package Responsibilities

- `server/` -- gRPC server setup, listener configuration, graceful shutdown
- `service/` -- Business logic. Orchestrates calls between model, store, and cache. No direct DB or gRPC imports here.
- `model/` -- ONNX model loading, inference execution, feature preprocessing, model hot-reload
- `tenant/` -- Tenant context extraction, propagation, and isolation enforcement
- `store/` -- Postgres access via pgx. Raw SQL only. Repository pattern with interfaces.
- `cache/` -- Redis operations via go-redis. Cache get/set/invalidate with TTL.
- `auth/` -- Tenant API key validation. gRPC interceptor that rejects unauthenticated requests.
- `ratelimit/` -- Per-tenant rate limiting using Redis token bucket algorithm.
- `telemetry/` -- OpenTelemetry provider setup, custom metric definitions, span helpers.

## Dependency Flow

```
server -> service -> model, store, cache
server -> auth (interceptor)
server -> ratelimit (interceptor)
server -> telemetry (provider init)
tenant -> used by all packages via context
```

Dependencies flow inward. `service` never imports `server`. `store` never imports `service`. Use interfaces at package boundaries so packages depend on abstractions, not concrete types.

## Patterns

### Constructor Injection

Every package exposes a `New()` constructor that accepts its dependencies as interfaces:

```go
// store/risk_store.go
type RiskStore struct {
    db *pgxpool.Pool
}

func NewRiskStore(db *pgxpool.Pool) *RiskStore {
    return &RiskStore{db: db}
}
```

### Interface Definitions

Define interfaces where they are consumed, not where they are implemented. If `service` needs to call `store`, the interface lives in `service/`:

```go
// service/interfaces.go
type PortfolioReader interface {
    GetPortfolio(ctx context.Context, tenantID, portfolioID string) (*Portfolio, error)
}
```

### Error Handling

Wrap errors with context using `fmt.Errorf("getting portfolio: %w", err)`. Never return bare errors from internal packages. gRPC status codes are mapped in the `server/` package only, not in `service/` or `store/`.

### Testing

- Unit tests: mock interfaces using hand-written mocks or a minimal mock library.
- Integration tests: use testcontainers-go to spin up real Postgres and Redis in Docker.
- Every exported function has at least one test.
- Use table-driven tests for functions with multiple input variations.

## ONNX Model Serving (internal/model/)

- Load model once at startup, hold in memory behind a sync.RWMutex.
- Hot-reload: watch S3 for new model versions, swap atomically using write lock.
- Feature preprocessing must exactly match the Python training pipeline. Document any preprocessing step with a comment referencing the Python equivalent.
- Log model version, inference latency, and confidence score on every prediction.
