# FinSight Project Plan

Multi-tenant portfolio risk prediction service. LSTM + attention model predicts VaR, CVaR, and volatility. Go gRPC service serves ONNX inference. Full AWS infrastructure.

## Decision Log

All decisions made during Phase 0 design sessions.

| Area | Decision | Rationale |
|------|----------|-----------|
| Risk metrics | Multi-target: VaR (95%), CVaR (95%), Volatility | No single metric captures full risk picture. VaR alone is flawed (not subadditive). CVaR covers tail severity. Volatility is foundational. |
| Model architecture | LSTM (2-layer) + self-attention (4-head) | LSTM captures local temporal dynamics in daily returns. Attention finds long-range dependencies (earnings cycles, regime echoes). Full Transformer overfits on <10yr financial data. |
| Hidden dimension | 128 | Sufficient capacity for ~50 input features without overfitting |
| Lookback window | 63 trading days (~3 months) | Captures one earnings cycle. Shorter misses regime changes, longer adds noise. |
| Dropout | 0.3 (LSTM), 0.1 (attention) | Financial data is noisy. Regularization prevents memorizing specific market regimes. |
| Max portfolio size | 50 assets | Reasonable for real portfolios. Beyond 50, feature dimension explodes. |
| Loss function | Combined weighted: Huber(volatility) + QuantileLoss(VaR, alpha=0.95) + TruncatedMSE(CVaR) | Each target has different statistical nature. MSE is wrong for quantile targets. Pinball loss is correct for VaR. Huber handles outlier days. |
| Data source | yfinance (real OHLCV) + synthetic stress scenarios | Real data = real interview stories. Synthetic = validate tail behavior and regime shifts. |
| Multi-tenancy model | Shared model + tenant config | Risk dynamics are market-universal. Tenants differ in portfolio composition and confidence level, not in how risk works. One model in memory, simple ops. |
| Caching | Redis, model_version in cache key, 15m TTL | Model version in key = automatic miss on reload, no flush needed. 15m balances freshness vs compute. |
| Tenant isolation | 3-layer: auth interceptor + app WHERE clause + Postgres RLS | Defense in depth. Even a buggy query can't leak data because RLS is the floor. |
| Infra | EKS + RDS Postgres + ElastiCache Redis + S3 + ECR | EKS = auto-scaling, rolling deploys. RDS = RLS for multi-tenancy. Redis = shared cache + rate limiting. S3 = versioned model storage. |

---

## Phase 0: Foundation (Education + Design)

**Goal:** Understand every technical decision deeply enough to defend it in an interview. No code written yet. Pure learning.

### 0A: Domain Understanding

- What is portfolio risk? Why do firms measure it?
- Value at Risk (VaR): definition, confidence levels, limitations (not subadditive, ignores tail shape)
- Conditional VaR (Expected Shortfall): fixes VaR's weakness, measures average loss beyond VaR
- Volatility: standard deviation of returns, why log returns not raw returns, annualization factor (sqrt(252))
- How these three metrics complement each other
- Real-world usage: regulatory requirements (Basel III), internal risk limits, client reporting

### 0B: ML Deep Dive

**Data:**
- OHLCV data (Open, High, Low, Close, Volume) from yfinance
- Feature engineering: daily returns, log returns, rolling volatility (5d/21d/63d), EWMA volatility, Garman-Klass volatility, RSI, Bollinger Band width, ATR
- Portfolio-level features: weighted returns, correlation matrix, Herfindahl concentration index
- Macro features (optional): VIX, yield curve slope, credit spreads
- Why stationarity matters: raw prices are non-stationary, returns are stationary. Models train on stationary data.
- Train/validation/test split: walk-forward (never peek into the future). NOT random split.

**Model architecture:**
```
Input: [batch, 63 days, P * F + portfolio_features]
  |
  v
LSTM Layer 1 (128 hidden, dropout 0.3)
  |
  v
LSTM Layer 2 (128 hidden, dropout 0.3)
  |
  v
Self-Attention (4 heads, dropout 0.1)
  Learns which past days matter most for today's risk prediction
  |
  v
Context vector [batch, 128]
  |
  v
Three parallel linear heads:
  Head 1 -> Volatility (1 value)
  Head 2 -> VaR at 95% (1 value)
  Head 3 -> CVaR at 95% (1 value)
  |
  v
Output: [batch, 3]
```

**Loss function:**
```
L_total = w1 * Huber(volatility_pred, volatility_true)
        + w2 * QuantileLoss(var_pred, var_true, alpha=0.95)
        + w3 * TruncatedMSE(cvar_pred, cvar_true)

QuantileLoss (Pinball Loss):
  L(y, q) = alpha * max(y - q, 0) + (1 - alpha) * max(q - y, 0)
  Penalizes underestimation of VaR more than overestimation at the 95% level.

Huber Loss:
  Quadratic for small errors, linear for large errors.
  Robust to outlier days (flash crashes, circuit breakers).

TruncatedMSE:
  MSE computed only on observations where actual loss exceeds predicted VaR.
  Focuses CVaR learning on the tail region where it matters.
```

**Python libraries:**
- `torch` -- model definition, training loop, autograd
- `pandas` / `numpy` -- data loading, feature engineering
- `yfinance` -- historical market data download
- `scikit-learn` -- preprocessing (StandardScaler), walk-forward split utilities
- `onnx` / `torch.onnx` -- model export to ONNX format
- `onnxruntime` -- validate exported model matches PyTorch output
- `ruff` -- linting
- `black` -- formatting
- `pytest` -- testing

**Training concepts to understand:**
- Epochs, batches, learning rate, Adam optimizer
- Walk-forward validation (why random split is wrong for time series)
- Early stopping (monitor validation loss, patience parameter)
- Gradient clipping (prevents exploding gradients in LSTM)
- Overfitting signs: training loss drops, validation loss rises

### 0C: Go Service Architecture

**Why Go for this service:**
- Static binary, fast startup (matters for K8s pod scaling)
- Built-in concurrency (goroutines for parallel predictions)
- Strong typing catches bugs at compile time
- gRPC + protobuf are first-class in Go ecosystem
- ONNX Runtime has Go bindings

**Request flow:**
```
Client -> gRPC -> [Interceptor Chain] -> server -> service -> model -> response
                   |
                   Telemetry (start span)
                   Auth (API key -> tenant_id in context)
                   RateLimit (Redis token bucket per tenant)
```

**Package dependency flow:**
```
server -> service -> model, store, cache
server -> auth (interceptor)
server -> ratelimit (interceptor)
server -> telemetry (provider init)
tenant -> used by all packages via context.Context
```

Dependencies flow inward. Packages depend on interfaces, not concrete types. Interfaces defined where consumed, not where implemented.

**Go concepts learned in this project (by phase):**
- Phase 1: modules, packages, func main, struct, method, error handling, if/else
- Phase 2: protobuf codegen, interfaces, context.Context, gRPC server/handler
- Phase 3: pgx database driver, SQL queries, connection pooling, transactions
- Phase 4: (Python phase, Go only loads ONNX)
- Phase 5: sync.RWMutex, goroutines, channels, ONNX Runtime FFI
- Phase 6: middleware pattern, interceptors, Redis client
- Phase 7: OpenTelemetry SDK, metric types, span creation

### 0D: gRPC API Design

```protobuf
service RiskPredictionService {
  rpc PredictRisk(PredictRiskRequest) returns (PredictRiskResponse);
  rpc GetPortfolio(GetPortfolioRequest) returns (GetPortfolioResponse);
  rpc GetModelInfo(GetModelInfoRequest) returns (GetModelInfoResponse);
}

message PredictRiskRequest {
  string portfolio_id = 1;
  int32 lookback_days = 2;          // optional, default 63
  float confidence_level = 3;       // optional, default 0.95
}

message PredictRiskResponse {
  float var_95 = 1;
  float cvar_95 = 2;
  float volatility = 3;
  string model_version = 4;
  google.protobuf.Timestamp predicted_at = 5;
  map<string, float> asset_contributions = 6;
}
```

Note: tenant_id is NOT in the request. Extracted from API key by auth interceptor, stored in context.

### 0E: Infrastructure Understanding

**AWS architecture:**
```
Internet -> ALB -> EKS (Go pods)
                    |
              RDS PostgreSQL (tenant data, portfolios, predictions)
              ElastiCache Redis (cache, rate limiting)
              S3 (ONNX model storage, versioned)
              ECR (container images)
```

**Why each component:**
- EKS over EC2: auto-scaling, rolling deploys, health-check routing, zero-downtime model updates
- RDS Postgres over DynamoDB: relational queries, Row-Level Security for multi-tenancy, rich SQL
- ElastiCache Redis over in-process cache: shared across pods, survives pod restart, token bucket rate limiting
- S3 for models: versioned, cheap, event notifications trigger hot-reload
- ECR: private registry, native EKS IAM integration, no credential management

**Scaling behavior:**
- Low load: 2 pods, each holding ONNX model in memory
- Spike: HPA scales pods based on CPU or custom metric (prediction queue depth)
- Model deploy: rolling update, new pods load new model, old pods drain
- Pod failure: K8s restarts, pod loads latest model from S3
- DB failure: predictions fail gracefully, cache still serves recent results

### 0F: Security and Multi-Tenancy

**Three-layer tenant isolation:**

Layer 1 (Auth Interceptor):
- Client sends API key in gRPC metadata ("authorization" header)
- Interceptor hashes key, looks up tenant_id in api_keys table
- Rejects if revoked or not found
- Stores tenant_id in context.Context

Layer 2 (Application Queries):
- Every store method requires tenant_id parameter
- Every SQL query includes WHERE tenant_id = $1
- No method exists without tenant_id filtering

Layer 3 (Postgres RLS):
- Row-Level Security policy on every table
- SET LOCAL app.tenant_id = '...' per transaction
- Even if application code has a bug, Postgres filters rows

### 0G: Observability

**Metrics (Prometheus):**
- `prediction_latency_seconds` (histogram) -- core SLO: p95 < 200ms
- `prediction_total` (counter by tenant, status) -- volume and error rate
- `cache_hit_ratio` (gauge) -- below 0.6 = investigate
- `model_inference_seconds` (histogram) -- isolate ONNX time
- `active_model_version` (gauge label) -- which model is serving
- `ratelimit_rejected_total` (counter by tenant) -- noisy tenant detection

**Traces (Jaeger via OpenTelemetry):**
- One trace per request
- Spans: auth interceptor, cache lookup, DB query, ONNX inference, cache write
- Trace ID propagated through context.Context

**Alerts:**
- Critical: error_rate > 5% for 5 min
- Warning: p99 latency > 500ms for 10 min
- Warning: cache_hit_ratio < 0.3 for 15 min
- Info: model_age > 24h

### 0H: Caching Strategy

**Redis cache design:**

| Data | Key pattern | TTL |
|------|------------|-----|
| Risk predictions | `predict:{tenant_id}:{portfolio_id}:{model_version}` | 15 min |
| Portfolio composition | `portfolio:{tenant_id}:{portfolio_id}` | 5 min |
| Model metadata | `model:info:{version}` | Until invalidated |

**Invalidation events:**
- Portfolio updated -> delete portfolio key + prediction keys for that portfolio
- Model hot-reload -> no action needed (model_version in key = automatic miss)
- Tenant deleted -> delete all keys matching tenant_id

---

## Phase 1: Go Basics + Project Setup

**Goal:** Get a Go binary compiling, serving HTTP health check, shutting down gracefully.

**Go concepts introduced:** modules (go.mod), packages, func main, structs, methods, error handling, goroutines (basic), os/signal, context.Context (basic).

### Tasks:
1. Initialize Go module: `go mod init github.com/SatyaChamana/FinSight`
2. Create `cmd/server/main.go` with basic HTTP server
3. Health check endpoint: `GET /healthz` returns 200 OK
4. Graceful shutdown: listen for SIGINT/SIGTERM, drain connections, exit cleanly
5. Structured logging setup (slog package)
6. Configuration via environment variables (port, log level)
7. Write tests for health check handler
8. Set up golangci-lint configuration
9. Create Dockerfile for Go service
10. Set up GitHub Actions CI (build + test + lint)

---

## Phase 2: gRPC + Protobuf

**Goal:** Replace HTTP with gRPC. Define protobuf schema. Serve a dummy PredictRisk that returns hardcoded values.

**Go concepts introduced:** interfaces, protobuf code generation, gRPC server/handler, context.Context (request-scoped values), interceptors (basic).

### Tasks:
1. Write proto files: `proto/finsight/v1/risk_service.proto`
2. Create `scripts/generate-proto.sh`
3. Generate Go code from protos
4. Implement gRPC server in `internal/server/`
5. Dummy `PredictRisk` handler returning hardcoded VaR/CVaR/volatility
6. gRPC health check service
7. gRPC reflection (for debugging with grpcurl)
8. Write tests using gRPC test helpers
9. Update Dockerfile and CI

---

## Phase 3: Domain + PostgreSQL Storage

**Goal:** Real database. Tenant and portfolio CRUD. Multi-tenant query enforcement.

**Go concepts introduced:** pgx driver, connection pooling, SQL queries in Go, transactions, Row-Level Security, repository pattern with interfaces.

### Tasks:
1. Design database schema (tenants, api_keys, portfolios, portfolio_assets, predictions)
2. SQL migration files
3. Implement `internal/store/` with pgx
4. Tenant CRUD operations
5. Portfolio CRUD operations (always filtered by tenant_id)
6. Set up Postgres RLS policies
7. Implement `internal/tenant/` context extraction and propagation
8. Integration tests with testcontainers-go (real Postgres in Docker)
9. Docker Compose with Postgres for local dev

---

## Phase 4: Python ML Pipeline

**Goal:** Train LSTM + attention model on real financial data. Export to ONNX. Validate export.

### Tasks:
1. Data download script using yfinance (S&P 500 stocks, ETFs)
2. Feature engineering pipeline (returns, rolling volatility, RSI, ATR, portfolio features)
3. Walk-forward train/validation/test split
4. Dataset class (PyTorch DataLoader)
5. Model definition: LSTM + self-attention + 3-head output
6. Combined loss function (Huber + QuantileLoss + TruncatedMSE)
7. Training loop with early stopping, gradient clipping, Adam optimizer
8. Evaluation metrics and validation
9. ONNX export script with metadata embedding
10. ONNX validation: compare PyTorch vs ONNX output within tolerance
11. Synthetic data generator for stress scenarios
12. pytest test suite for data pipeline and model

---

## Phase 5: ONNX Inference in Go

**Goal:** Load ONNX model in Go service. Serve real predictions behind gRPC.

**Go concepts introduced:** sync.RWMutex, goroutines, channels, CGo/FFI (onnxruntime-go), feature preprocessing in Go.

### Tasks:
1. Implement `internal/model/` with onnxruntime-go
2. Model loading from local file (later S3)
3. Feature preprocessing (must match Python pipeline exactly)
4. Inference execution with RWMutex for concurrent reads
5. Wire model into service layer
6. PredictRisk now returns real predictions
7. Asset contribution calculation (per-asset risk decomposition)
8. Model hot-reload: watch for new version, swap atomically
9. Benchmarks: measure inference latency
10. Tests with a small test ONNX model

---

## Phase 6: Auth + Rate Limiting + Redis Cache

**Goal:** Production-ready request pipeline. Auth, rate limits, caching.

**Go concepts introduced:** gRPC interceptors (unary + stream), Redis client (go-redis), token bucket algorithm, bcrypt hashing.

### Tasks:
1. Implement `internal/auth/` interceptor
2. API key hashing and validation
3. Implement `internal/ratelimit/` using Redis token bucket
4. Per-tenant rate limit configuration
5. Implement `internal/cache/` with go-redis
6. Cache predictions with model_version in key
7. Cache invalidation on portfolio update
8. Wire all interceptors into gRPC server chain
9. Integration tests with testcontainers (Redis + Postgres)

---

## Phase 7: Observability

**Goal:** Full OpenTelemetry instrumentation. Prometheus metrics. Grafana dashboards. Jaeger traces.

**Go concepts introduced:** OpenTelemetry SDK, metric instruments (counter, histogram, gauge), span creation, context propagation.

### Tasks:
1. Implement `internal/telemetry/` provider setup
2. Instrument gRPC interceptor with traces
3. Add custom Prometheus metrics (latency, error rate, cache hit ratio, model version)
4. Create Grafana dashboard JSON (prediction latency, per-tenant volume, cache performance)
5. Create Prometheus alerting rules
6. Add span attributes (tenant_id, portfolio_id, model_version)
7. Structured log correlation with trace IDs
8. Docker Compose with Prometheus + Grafana + Jaeger for local dev

---

## Phase 8: Infrastructure (Terraform + Kubernetes)

**Goal:** Deploy to AWS. IaC everything. Kubernetes manifests with Kustomize.

### Tasks:
1. Terraform modules: VPC, EKS, RDS, ElastiCache, S3, ECR, IAM
2. Dev environment root module
3. Prod environment root module
4. Remote state backend (S3 + DynamoDB)
5. Kubernetes base manifests (Deployment, Service, ConfigMap, Secrets)
6. Kustomize overlays for dev and prod
7. HPA configuration (auto-scaling policy)
8. Liveness and readiness probes
9. S3 model loading (replace local file)
10. GitHub Actions CD pipeline (build, push to ECR, deploy to EKS)

---

## Phase 9: Production Hardening

**Goal:** Battle-test everything. Load test, chaos test, optimize.

### Tasks:
1. Load testing with k6 or ghz (gRPC load tester)
2. Establish SLOs: p95 < 200ms, error rate < 0.1%, availability > 99.9%
3. Chaos testing: kill pods, simulate DB failure, Redis failure
4. Model A/B testing capability (serve two model versions, compare)
5. Graceful degradation: serve cached results when model/DB is down
6. Database connection pool tuning
7. ONNX Runtime optimization (thread count, execution providers)
8. Security audit: API key rotation, TLS, network policies
9. Documentation: runbook, architecture diagram, API docs
10. Cost optimization: right-size instances, spot instances for non-prod
