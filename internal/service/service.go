package service

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"time"

	"github.com/SatyaChamana/FinSight/internal/store"
)

// Sentinel errors. Callers (specifically the server package) use errors.Is to
// detect these and map them to gRPC status codes. The service itself never
// imports gRPC or maps status codes.
var (
	// ErrInvalidPortfolioID is returned when the portfolio id is empty.
	ErrInvalidPortfolioID = errors.New("service: invalid portfolio id")
	// ErrInvalidConfidence is returned when the confidence level is out of
	// the valid range (must be less than 1).
	ErrInvalidConfidence = errors.New("service: invalid confidence level")
	// ErrPortfolioNotFound is returned when the requested portfolio does not
	// exist for the current tenant.
	ErrPortfolioNotFound = errors.New("service: portfolio not found")
)

// Default request parameters, sourced from the project plan decision log.
const (
	defaultLookbackDays    = 63   // ~3 months / one earnings cycle
	defaultConfidenceLevel = 0.95 // VaR / CVaR at the 95% level
	defaultCacheTTL        = 15 * time.Minute
)

// Prediction is the service-level domain result returned to callers. It is
// distinct from ScoreResult (the raw model output) so the service owns the
// shape it exposes and can add fields like PredictedAt.
type Prediction struct {
	VaR95              float64
	CVaR95             float64
	Volatility         float64
	ModelVersion       string
	PredictedAt        time.Time
	AssetContributions map[string]float64
}

// Options bundles the dependencies for NewRiskService. Cache and Writer may be
// nil (the service degrades gracefully and just skips that step). Now and
// Logger are optional and default to time.Now and slog.Default when nil.
type Options struct {
	Portfolios PortfolioReader
	Scorer     Scorer
	Features   FeatureBuilder
	Cache      Cache            // optional, may be nil
	Writer     PredictionWriter // optional, may be nil
	CacheTTL   time.Duration    // optional, defaults to 15m when zero
	Now        func() time.Time // optional, defaults to time.Now (injected for tests)
	Logger     *slog.Logger     // optional, defaults to slog.Default
}

// RiskService orchestrates the portfolio risk prediction flow across the
// store (load portfolio), feature builder, model (score), and cache.
type RiskService struct {
	portfolios PortfolioReader
	scorer     Scorer
	features   FeatureBuilder
	cache      Cache
	writer     PredictionWriter
	cacheTTL   time.Duration
	now        func() time.Time
	logger     *slog.Logger
}

// NewRiskService constructs a RiskService from the given Options, filling in
// defaults for the optional fields.
func NewRiskService(opts Options) *RiskService {
	now := opts.Now
	if now == nil {
		now = time.Now
	}
	logger := opts.Logger
	if logger == nil {
		logger = slog.Default()
	}
	ttl := opts.CacheTTL
	if ttl <= 0 {
		ttl = defaultCacheTTL
	}
	return &RiskService{
		portfolios: opts.Portfolios,
		scorer:     opts.Scorer,
		features:   opts.Features,
		cache:      opts.Cache,
		writer:     opts.Writer,
		cacheTTL:   ttl,
		now:        now,
		logger:     logger,
	}
}

// PredictRisk loads a portfolio, runs (or reuses a cached) risk prediction,
// and returns the result. The tenant id is carried in ctx and enforced by the
// store and cache layers; this method never handles tenant ids directly.
func (s *RiskService) PredictRisk(ctx context.Context, portfolioID string, lookbackDays int, confidenceLevel float64) (*Prediction, error) {
	// 1. Validate and default inputs.
	if portfolioID == "" {
		return nil, ErrInvalidPortfolioID
	}
	if lookbackDays <= 0 {
		lookbackDays = defaultLookbackDays
	}
	if confidenceLevel <= 0 {
		confidenceLevel = defaultConfidenceLevel
	}
	// A confidence level of 1 (or more) is meaningless for VaR/CVaR.
	if confidenceLevel >= 1 {
		return nil, fmt.Errorf("confidence %v out of range: %w", confidenceLevel, ErrInvalidConfidence)
	}

	// 2. Load the portfolio. Map the store's not-found sentinel to ours.
	portfolio, err := s.portfolios.GetPortfolioWithAssets(ctx, portfolioID)
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			return nil, fmt.Errorf("loading portfolio %q: %w", portfolioID, ErrPortfolioNotFound)
		}
		return nil, fmt.Errorf("loading portfolio %q: %w", portfolioID, err)
	}

	// 3. Build the cache key. The model version is part of the key so that a
	// model reload automatically misses the cache (no explicit flush needed).
	// Tenant scoping is added by the cache implementation from ctx.
	modelVersion := s.scorer.Version()
	cacheKey := fmt.Sprintf("predict:%s:%s", portfolioID, modelVersion)

	// Cache lookup (best effort). A cache error is logged and treated as a
	// miss rather than failing the request.
	if s.cache != nil {
		cached, hit, cacheErr := s.cache.GetPrediction(ctx, cacheKey)
		if cacheErr != nil {
			s.logger.WarnContext(ctx, "cache get failed, treating as miss",
				slog.String("key", cacheKey), slog.Any("error", cacheErr))
		} else if hit && cached != nil {
			s.logger.DebugContext(ctx, "cache hit", slog.String("key", cacheKey))
			return cached, nil
		}
	} else {
		s.logger.DebugContext(ctx, "no cache configured, skipping lookup")
	}

	// 4. Cache miss: build features, then score.
	features, weights, err := s.features.BuildFeatures(ctx, portfolio, lookbackDays)
	if err != nil {
		return nil, fmt.Errorf("building features for portfolio %q: %w", portfolioID, err)
	}

	result, err := s.scorer.Predict(ctx, features, weights)
	if err != nil {
		return nil, fmt.Errorf("scoring portfolio %q: %w", portfolioID, err)
	}

	prediction := &Prediction{
		VaR95:              result.VaR95,
		CVaR95:             result.CVaR95,
		Volatility:         result.Volatility,
		ModelVersion:       result.ModelVersion,
		PredictedAt:        s.now(),
		AssetContributions: result.AssetContributions,
	}

	// 5a. Write to cache (best effort). A failure here does not fail the
	// request: the prediction is already computed and correct.
	if s.cache != nil {
		if cacheErr := s.cache.SetPrediction(ctx, cacheKey, prediction, s.cacheTTL); cacheErr != nil {
			s.logger.WarnContext(ctx, "cache set failed, continuing",
				slog.String("key", cacheKey), slog.Any("error", cacheErr))
		}
	} else {
		s.logger.DebugContext(ctx, "no cache configured, skipping store")
	}

	// 5b. Persist the prediction (best effort, same rationale as caching).
	if s.writer != nil {
		if writeErr := s.writer.SavePrediction(ctx, portfolioID, prediction); writeErr != nil {
			s.logger.WarnContext(ctx, "persisting prediction failed, continuing",
				slog.String("portfolio_id", portfolioID), slog.Any("error", writeErr))
		}
	} else {
		s.logger.DebugContext(ctx, "no prediction writer configured, skipping persist")
	}

	// 6. Return the freshly computed prediction.
	return prediction, nil
}
