package service

import (
	"context"
	"time"

	"github.com/SatyaChamana/FinSight/internal/store"
)

// This file defines the consumer interfaces that the service package depends
// on. In Go the idiom is to define interfaces where they are consumed, not
// where they are implemented. That keeps this package decoupled from the
// concrete model, store, and cache packages. The integrating agent writes
// thin adapters around the real implementations so they satisfy these shapes.

// PortfolioReader loads a tenant-scoped portfolio. The tenant id is carried
// in ctx (an upstream interceptor sets it), so it is not a parameter here.
// The implementation enforces tenant scoping using the value in ctx.
type PortfolioReader interface {
	GetPortfolioWithAssets(ctx context.Context, portfolioID string) (*store.Portfolio, error)
}

// ScoreResult is the raw model output the service consumes. It is produced by
// a Scorer and translated by the service into a Prediction.
type ScoreResult struct {
	Volatility         float64
	VaR95              float64
	CVaR95             float64
	ModelVersion       string
	AssetContributions map[string]float64
}

// Scorer runs model inference on a prepared feature matrix. features is the
// [lookback][feature] input matrix and weights maps each asset symbol to its
// portfolio weight (used for per-asset risk decomposition).
type Scorer interface {
	Predict(ctx context.Context, features [][]float32, weights map[string]float64) (*ScoreResult, error)
	// Version reports the version of the currently loaded model. It is used
	// in the cache key so that a model reload automatically misses the cache.
	Version() string
}

// FeatureBuilder turns a portfolio into the model input matrix (for example a
// [63][49] matrix) plus the per-asset weight map. lookbackDays controls how
// many trailing trading days of history make up the first dimension.
type FeatureBuilder interface {
	BuildFeatures(ctx context.Context, p *store.Portfolio, lookbackDays int) (features [][]float32, weights map[string]float64, err error)
}

// Cache is the optional prediction cache. The service is nil-safe with
// respect to the cache: if no cache is injected, the service simply skips
// caching. Tenant scoping is handled inside the cache implementation, which
// prefixes the tenant id (from ctx) onto the key, so callers pass a key that
// omits the tenant.
type Cache interface {
	// GetPrediction returns the cached prediction for key. The boolean is
	// true on a cache hit and false on a miss. A non-nil error signals a
	// cache failure (for example a Redis outage), which the service treats
	// as a best-effort miss rather than a fatal error.
	GetPrediction(ctx context.Context, key string) (*Prediction, bool, error)
	// SetPrediction stores p under key with the given ttl.
	SetPrediction(ctx context.Context, key string, p *Prediction, ttl time.Duration) error
}

// PredictionWriter persists a computed prediction. The tenant id is carried
// in ctx. This dependency is optional and nil-safe: if no writer is injected,
// the service skips persistence.
type PredictionWriter interface {
	SavePrediction(ctx context.Context, portfolioID string, p *Prediction) error
}
