# Phase 5C: The Service Layer, Interfaces Where Consumed, and the Adapter Pattern

How the model, the database, and the cache get wired together without any of them depending on each other. This is the Go architecture lesson of the phase.

## The layers

```
gRPC handler (server)  ->  service.RiskService  ->  Scorer, FeatureBuilder, PortfolioReader, Cache, Writer
                                                     (interfaces, not concrete types)
```

`service.RiskService.PredictRisk` orchestrates one prediction:

1. Validate inputs (apply defaults: lookback 63, confidence 0.95).
2. Load the portfolio (PortfolioReader).
3. Check the cache (Cache); on a hit, return immediately.
4. On a miss: build features (FeatureBuilder), run the model (Scorer).
5. Write the result to the cache and persist it (Writer), both best effort.
6. Return the prediction.

The service contains the business logic and nothing else: no SQL, no gRPC types, no ONNX. That separation is the point.

## Interfaces are defined where they are consumed

This is the Go idiom that trips up people coming from Java or Python. You do not define an interface next to the thing that implements it. You define it next to the code that uses it, listing only the methods that caller actually needs.

The service needs to read a portfolio, so the service package declares:

```go
// internal/service/interfaces.go
type PortfolioReader interface {
    GetPortfolioWithAssets(ctx context.Context, portfolioID string) (*store.Portfolio, error)
}
```

The `store.Store` type already has a `GetPortfolioWithAssets` method. It therefore satisfies `service.PortfolioReader` automatically, with no `implements` keyword and without the store package even knowing the interface exists. Go interfaces are satisfied structurally (duck typing checked at compile time).

Why this is powerful:

- The dependency arrow points inward. `service` depends on its own small interface, not on the concrete `store`. The store could be swapped for a different backend and the service would not change.
- Tests become trivial: a hand written fake with the right method signature is a valid dependency. Our `service_test.go` uses fakes for every interface, so the service is unit tested with zero database, zero Redis, zero ONNX.
- Interfaces stay small. `PortfolioReader` has one method, not the store's entire surface. Small interfaces are easy to fake and hard to misuse. ("The bigger the interface, the weaker the abstraction.")

## The adapter pattern: bridging two packages that must not know each other

Problem: the service defines `Scorer` returning `*service.ScoreResult`. The model defines `Engine.Predict` returning `*model.Prediction`. These are different named types. The model does not import the service, the service does not import the model (neither should depend on the other). So `*model.Engine` does not satisfy `service.Scorer` directly.

Solution: a thin adapter in a third package, `internal/inference`, whose only job is translation:

```go
// internal/inference/scorer.go
type Scorer struct{ engine *model.Engine }

var _ service.Scorer = (*Scorer)(nil) // compile time proof it satisfies the interface

func (s *Scorer) Predict(ctx context.Context, features [][]float32, weights map[string]float64) (*service.ScoreResult, error) {
    p, err := s.engine.Predict(ctx, features, weights)
    if err != nil {
        return nil, fmt.Errorf("model inference: %w", err)
    }
    return &service.ScoreResult{          // model.Prediction -> service.ScoreResult
        Volatility: p.Volatility, VaR95: p.VaR95, CVaR95: p.CVaR95,
        ModelVersion: p.ModelVersion, AssetContributions: p.AssetContributions,
    }, nil
}
```

The `var _ service.Scorer = (*Scorer)(nil)` line is a Go trick: it compiles only if `*Scorer` satisfies `service.Scorer`. It is a free, compile time assertion that the adapter is correct, with no runtime cost (it assigns nil to the blank identifier).

`main.go` is the composition root. It is the only place that imports model, service, and inference together and wires them:

```go
engine, _ := model.NewEngine(ctx, model.Config{ModelPath: cfg.ModelPath})
predictor := service.NewRiskService(service.Options{
    Portfolios: portfolios,                  // *store.Store
    Scorer:     inference.NewScorer(engine), // adapter
    Features:   inference.NewFeatureBuilder(),
})
```

This keeps the dependency graph acyclic. model, store, service are independent leaves; inference and main are the glue.

## Errors: sentinels in the service, gRPC codes in the server

The service returns plain Go errors, including named sentinel errors:

```go
var ErrPortfolioNotFound = errors.New("service: portfolio not found")
var ErrInvalidConfidence = errors.New("service: invalid confidence level")
```

The service must not know about gRPC status codes; that is a transport concern. Mapping happens in exactly one place, the server handler:

```go
func (s *RiskService) mapPredictError(ctx context.Context, err error) error {
    switch {
    case errors.Is(err, service.ErrPortfolioNotFound):
        return status.Error(codes.NotFound, "portfolio not found")
    case errors.Is(err, service.ErrInvalidConfidence):
        return status.Error(codes.InvalidArgument, err.Error())
    default:
        return status.Error(codes.Internal, "prediction failed")
    }
}
```

`errors.Is` walks the wrapped error chain (errors wrapped with `%w`), so a deeply wrapped `ErrPortfolioNotFound` still matches. This is why we wrap with `fmt.Errorf("...: %w", err)` everywhere: it preserves the chain for `errors.Is` while adding context to the message.

## Optional dependencies and graceful degradation

Cache and Writer may be nil. The service checks for nil and skips them, logging at debug, rather than requiring Redis and a predictions table to exist. Cache and persistence errors are swallowed (logged, request still succeeds): a cache outage should slow the service, not break it.

At a higher level, `main.go` does the same with the model itself. If the ONNX model or its native library cannot load, the server still starts and `PredictRisk` returns documented dummy values, with a warning logged. Local dev and machines without the dylib are never blocked. The real predictor is wired only when both a model and a database are present.

## Honest gaps (documented in code)

- `inference.FeatureBuilder` is a placeholder. There is no OHLCV price history store yet, so it synthesizes a deterministic `[63][49]` matrix seeded from the portfolio (so the same portfolio yields stable, cacheable features). The real, Python parity pipeline (`model.BuildFeatureMatrix`) exists and is tested, but it needs market bars we cannot supply until a price data source lands.
- Asset contributions use a simple weight share of VaR, not a true marginal VaR decomposition, because the model does not expose gradients.

## Flashcards

- Q: Where do you define a Go interface? A: In the package that consumes it, listing only the methods that caller needs, not next to the implementation.
- Q: How does a type satisfy a Go interface? A: Structurally and at compile time; if it has the methods it satisfies the interface, with no implements keyword and no import of the interface.
- Q: What does var _ Iface = (*T)(nil) do? A: A zero cost compile time assertion that *T satisfies Iface; the file fails to compile if it does not.
- Q: Why an adapter in internal/inference? A: model and service must not depend on each other, so a third package translates model.Prediction into service.ScoreResult and satisfies service.Scorer.
- Q: What is the composition root? A: main.go, the single place that imports the concrete packages and wires them together, keeping the rest of the graph acyclic.
- Q: Why sentinel errors plus errors.Is? A: The service returns transport agnostic named errors; the server maps them to gRPC codes in one place. errors.Is matches through %w wrapping so context can be added without breaking the match.
- Q: Why wrap errors with %w? A: It preserves the error chain so errors.Is and errors.As still work, while adding human context to the message.
- Q: How does the service degrade gracefully? A: Cache and Writer are optional (nil safe) and their errors are swallowed; if the model or DB is absent the server serves dummy predictions instead of failing to start.
- Q: Benefit of small consumer defined interfaces for testing? A: Dependencies can be replaced with tiny hand written fakes, so the service is unit tested with no DB, Redis, or ONNX.
