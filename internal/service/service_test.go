package service

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/SatyaChamana/FinSight/internal/store"
)

// --- Hand-written fakes implementing the consumer interfaces. ---

// fakePortfolioReader returns a fixed portfolio or a configured error.
type fakePortfolioReader struct {
	portfolio *store.Portfolio
	err       error
	calls     int
}

func (f *fakePortfolioReader) GetPortfolioWithAssets(_ context.Context, _ string) (*store.Portfolio, error) {
	f.calls++
	if f.err != nil {
		return nil, f.err
	}
	return f.portfolio, nil
}

// fakeScorer records calls and returns a fixed result or error.
type fakeScorer struct {
	result  *ScoreResult
	err     error
	version string
	calls   int
}

func (f *fakeScorer) Predict(_ context.Context, _ [][]float32, _ map[string]float64) (*ScoreResult, error) {
	f.calls++
	if f.err != nil {
		return nil, f.err
	}
	return f.result, nil
}

func (f *fakeScorer) Version() string { return f.version }

// fakeFeatureBuilder records the lookback it was called with.
type fakeFeatureBuilder struct {
	features     [][]float32
	weights      map[string]float64
	err          error
	calls        int
	lastLookback int
}

func (f *fakeFeatureBuilder) BuildFeatures(_ context.Context, _ *store.Portfolio, lookbackDays int) ([][]float32, map[string]float64, error) {
	f.calls++
	f.lastLookback = lookbackDays
	if f.err != nil {
		return nil, nil, f.err
	}
	return f.features, f.weights, nil
}

// fakeCache simulates get hits/misses and set behavior.
type fakeCache struct {
	getResult *Prediction
	getHit    bool
	getErr    error
	setErr    error

	getCalls int
	setCalls int
	lastKey  string
	lastTTL  time.Duration
}

func (f *fakeCache) GetPrediction(_ context.Context, key string) (*Prediction, bool, error) {
	f.getCalls++
	f.lastKey = key
	if f.getErr != nil {
		return nil, false, f.getErr
	}
	return f.getResult, f.getHit, nil
}

func (f *fakeCache) SetPrediction(_ context.Context, key string, p *Prediction, ttl time.Duration) error {
	f.setCalls++
	f.lastKey = key
	f.lastTTL = ttl
	return f.setErr
}

// fakePredictionWriter records save calls.
type fakePredictionWriter struct {
	err   error
	calls int
}

func (f *fakePredictionWriter) SavePrediction(_ context.Context, _ string, _ *Prediction) error {
	f.calls++
	return f.err
}

// --- Test helpers. ---

func samplePortfolio() *store.Portfolio {
	return &store.Portfolio{
		ID:       "pf-1",
		TenantID: "tenant-1",
		Name:     "Growth",
		Assets:   []store.Asset{{Symbol: "AAPL", Weight: 0.6}, {Symbol: "MSFT", Weight: 0.4}},
	}
}

func sampleScoreResult() *ScoreResult {
	return &ScoreResult{
		Volatility:         0.21,
		VaR95:              -0.034,
		CVaR95:             -0.051,
		ModelVersion:       "v1.2.3",
		AssetContributions: map[string]float64{"AAPL": 0.6, "MSFT": 0.4},
	}
}

// fixedNow returns a deterministic clock for asserting PredictedAt.
func fixedNow() func() time.Time {
	t := time.Date(2026, 6, 8, 12, 0, 0, 0, time.UTC)
	return func() time.Time { return t }
}

// --- Tests. ---

func TestPredictRisk_HappyPath(t *testing.T) {
	reader := &fakePortfolioReader{portfolio: samplePortfolio()}
	scorer := &fakeScorer{result: sampleScoreResult(), version: "v1.2.3"}
	builder := &fakeFeatureBuilder{
		features: [][]float32{{1, 2}, {3, 4}},
		weights:  map[string]float64{"AAPL": 0.6, "MSFT": 0.4},
	}
	cache := &fakeCache{getHit: false}
	writer := &fakePredictionWriter{}
	now := fixedNow()

	svc := NewRiskService(Options{
		Portfolios: reader,
		Scorer:     scorer,
		Features:   builder,
		Cache:      cache,
		Writer:     writer,
		Now:        now,
	})

	got, err := svc.PredictRisk(context.Background(), "pf-1", 63, 0.95)
	if err != nil {
		t.Fatalf("PredictRisk returned error: %v", err)
	}

	if got.VaR95 != -0.034 || got.CVaR95 != -0.051 || got.Volatility != 0.21 {
		t.Errorf("unexpected risk values: %+v", got)
	}
	if got.ModelVersion != "v1.2.3" {
		t.Errorf("ModelVersion = %q, want v1.2.3", got.ModelVersion)
	}
	if !got.PredictedAt.Equal(now()) {
		t.Errorf("PredictedAt = %v, want %v", got.PredictedAt, now())
	}
	if got.AssetContributions["AAPL"] != 0.6 {
		t.Errorf("AssetContributions[AAPL] = %v, want 0.6", got.AssetContributions["AAPL"])
	}

	// Verify the full flow ran exactly once each.
	if builder.calls != 1 {
		t.Errorf("builder calls = %d, want 1", builder.calls)
	}
	if scorer.calls != 1 {
		t.Errorf("scorer calls = %d, want 1", scorer.calls)
	}
	if cache.setCalls != 1 {
		t.Errorf("cache set calls = %d, want 1", cache.setCalls)
	}
	if writer.calls != 1 {
		t.Errorf("writer calls = %d, want 1", writer.calls)
	}

	// Cache key must include portfolio id and model version, and the TTL
	// should default to 15 minutes.
	wantKey := "predict:pf-1:v1.2.3"
	if cache.lastKey != wantKey {
		t.Errorf("cache key = %q, want %q", cache.lastKey, wantKey)
	}
	if cache.lastTTL != defaultCacheTTL {
		t.Errorf("cache TTL = %v, want %v", cache.lastTTL, defaultCacheTTL)
	}
}

func TestPredictRisk_CacheHitShortCircuits(t *testing.T) {
	cachedPrediction := &Prediction{
		VaR95:        -0.01,
		CVaR95:       -0.02,
		Volatility:   0.1,
		ModelVersion: "v1.2.3",
		PredictedAt:  fixedNow()(),
	}
	reader := &fakePortfolioReader{portfolio: samplePortfolio()}
	scorer := &fakeScorer{result: sampleScoreResult(), version: "v1.2.3"}
	builder := &fakeFeatureBuilder{}
	cache := &fakeCache{getResult: cachedPrediction, getHit: true}
	writer := &fakePredictionWriter{}

	svc := NewRiskService(Options{
		Portfolios: reader,
		Scorer:     scorer,
		Features:   builder,
		Cache:      cache,
		Writer:     writer,
		Now:        fixedNow(),
	})

	got, err := svc.PredictRisk(context.Background(), "pf-1", 63, 0.95)
	if err != nil {
		t.Fatalf("PredictRisk returned error: %v", err)
	}
	if got != cachedPrediction {
		t.Errorf("expected cached prediction to be returned verbatim")
	}

	// On a cache hit, the scorer, builder, cache set, and writer must NOT run.
	if scorer.calls != 0 {
		t.Errorf("scorer calls = %d, want 0 on cache hit", scorer.calls)
	}
	if builder.calls != 0 {
		t.Errorf("builder calls = %d, want 0 on cache hit", builder.calls)
	}
	if cache.setCalls != 0 {
		t.Errorf("cache set calls = %d, want 0 on cache hit", cache.setCalls)
	}
	if writer.calls != 0 {
		t.Errorf("writer calls = %d, want 0 on cache hit", writer.calls)
	}
}

func TestPredictRisk_PortfolioNotFound(t *testing.T) {
	reader := &fakePortfolioReader{err: store.ErrNotFound}
	scorer := &fakeScorer{version: "v1.0.0"}
	builder := &fakeFeatureBuilder{}

	svc := NewRiskService(Options{
		Portfolios: reader,
		Scorer:     scorer,
		Features:   builder,
		Now:        fixedNow(),
	})

	_, err := svc.PredictRisk(context.Background(), "missing", 63, 0.95)
	if !errors.Is(err, ErrPortfolioNotFound) {
		t.Fatalf("error = %v, want ErrPortfolioNotFound", err)
	}
	if scorer.calls != 0 || builder.calls != 0 {
		t.Errorf("downstream deps should not run when portfolio missing")
	}
}

func TestPredictRisk_InvalidInputs(t *testing.T) {
	tests := []struct {
		name            string
		portfolioID     string
		confidenceLevel float64
		wantErr         error
	}{
		{name: "empty portfolio id", portfolioID: "", confidenceLevel: 0.95, wantErr: ErrInvalidPortfolioID},
		{name: "confidence equals one", portfolioID: "pf-1", confidenceLevel: 1.0, wantErr: ErrInvalidConfidence},
		{name: "confidence above one", portfolioID: "pf-1", confidenceLevel: 1.5, wantErr: ErrInvalidConfidence},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			reader := &fakePortfolioReader{portfolio: samplePortfolio()}
			scorer := &fakeScorer{result: sampleScoreResult(), version: "v1.0.0"}
			builder := &fakeFeatureBuilder{}

			svc := NewRiskService(Options{
				Portfolios: reader,
				Scorer:     scorer,
				Features:   builder,
				Now:        fixedNow(),
			})

			_, err := svc.PredictRisk(context.Background(), tc.portfolioID, 63, tc.confidenceLevel)
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("error = %v, want %v", err, tc.wantErr)
			}
		})
	}
}

func TestPredictRisk_NilCacheAndWriter(t *testing.T) {
	reader := &fakePortfolioReader{portfolio: samplePortfolio()}
	scorer := &fakeScorer{result: sampleScoreResult(), version: "v1.0.0"}
	builder := &fakeFeatureBuilder{
		features: [][]float32{{1}},
		weights:  map[string]float64{"AAPL": 1.0},
	}

	svc := NewRiskService(Options{
		Portfolios: reader,
		Scorer:     scorer,
		Features:   builder,
		Cache:      nil,
		Writer:     nil,
		Now:        fixedNow(),
	})

	got, err := svc.PredictRisk(context.Background(), "pf-1", 63, 0.95)
	if err != nil {
		t.Fatalf("PredictRisk with nil cache/writer returned error: %v", err)
	}
	if got == nil || got.ModelVersion != "v1.2.3" {
		t.Errorf("expected a valid prediction, got %+v", got)
	}
	if scorer.calls != 1 || builder.calls != 1 {
		t.Errorf("expected full flow to run with nil cache/writer")
	}
}

func TestPredictRisk_CacheAndPersistErrorsSwallowed(t *testing.T) {
	reader := &fakePortfolioReader{portfolio: samplePortfolio()}
	scorer := &fakeScorer{result: sampleScoreResult(), version: "v1.0.0"}
	builder := &fakeFeatureBuilder{
		features: [][]float32{{1}},
		weights:  map[string]float64{"AAPL": 1.0},
	}
	// Cache get errors (treated as miss), cache set errors, and writer errors
	// must all be swallowed.
	cache := &fakeCache{getErr: errors.New("redis down"), setErr: errors.New("redis down")}
	writer := &fakePredictionWriter{err: errors.New("db down")}

	svc := NewRiskService(Options{
		Portfolios: reader,
		Scorer:     scorer,
		Features:   builder,
		Cache:      cache,
		Writer:     writer,
		Now:        fixedNow(),
	})

	got, err := svc.PredictRisk(context.Background(), "pf-1", 63, 0.95)
	if err != nil {
		t.Fatalf("PredictRisk should swallow cache/persist errors, got: %v", err)
	}
	if got == nil {
		t.Fatal("expected a prediction despite cache/persist failures")
	}
	// Even with a get error (treated as miss), the flow must still score.
	if scorer.calls != 1 {
		t.Errorf("scorer calls = %d, want 1", scorer.calls)
	}
	if cache.setCalls != 1 {
		t.Errorf("cache set calls = %d, want 1 (best-effort)", cache.setCalls)
	}
	if writer.calls != 1 {
		t.Errorf("writer calls = %d, want 1 (best-effort)", writer.calls)
	}
}

func TestPredictRisk_DefaultsApplied(t *testing.T) {
	reader := &fakePortfolioReader{portfolio: samplePortfolio()}
	scorer := &fakeScorer{result: sampleScoreResult(), version: "v1.0.0"}
	builder := &fakeFeatureBuilder{
		features: [][]float32{{1}},
		weights:  map[string]float64{"AAPL": 1.0},
	}

	svc := NewRiskService(Options{
		Portfolios: reader,
		Scorer:     scorer,
		Features:   builder,
		Now:        fixedNow(),
	})

	// lookback 0 -> 63, confidence 0 -> 0.95 (no error).
	_, err := svc.PredictRisk(context.Background(), "pf-1", 0, 0)
	if err != nil {
		t.Fatalf("PredictRisk with zero defaults returned error: %v", err)
	}
	if builder.lastLookback != defaultLookbackDays {
		t.Errorf("lookback passed to builder = %d, want %d", builder.lastLookback, defaultLookbackDays)
	}
}
