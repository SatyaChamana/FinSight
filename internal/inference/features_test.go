package inference

import (
	"context"
	"math"
	"os"
	"testing"

	"github.com/SatyaChamana/FinSight/internal/marketdata"
	"github.com/SatyaChamana/FinSight/internal/model"
	"github.com/SatyaChamana/FinSight/internal/store"
)

// fakeBars is a BarReader stub. Its Bars method is never reached by the
// universe-validation tests (validation happens before any bar load).
type fakeBars struct{}

func (fakeBars) Bars(context.Context, string) ([]model.OHLCV, error) { return nil, nil }

var modelSymbols = []string{"AAPL", "MSFT", "GOOG", "AMZN", "META"}

func TestBuildFeatures_RejectsOutsideUniverse(t *testing.T) {
	b := NewFeatureBuilder(fakeBars{}, modelSymbols)
	p := &store.Portfolio{
		ID: "energy",
		Assets: []store.Asset{
			{Symbol: "XOM", Weight: 0.2}, {Symbol: "CVX", Weight: 0.2},
			{Symbol: "COP", Weight: 0.2}, {Symbol: "SLB", Weight: 0.2},
			{Symbol: "EOG", Weight: 0.2},
		},
	}
	if _, _, err := b.BuildFeatures(context.Background(), p, 63); err == nil {
		t.Fatal("want error for out-of-universe portfolio, got nil")
	}
}

func TestBuildFeatures_RejectsWrongCount(t *testing.T) {
	b := NewFeatureBuilder(fakeBars{}, modelSymbols)
	p := &store.Portfolio{
		ID:     "partial",
		Assets: []store.Asset{{Symbol: "AAPL", Weight: 0.5}, {Symbol: "MSFT", Weight: 0.5}},
	}
	if _, _, err := b.BuildFeatures(context.Background(), p, 63); err == nil {
		t.Fatal("want error for wrong asset count, got nil")
	}
}

func TestBuildFeatures_RejectsNoSymbolMetadata(t *testing.T) {
	b := NewFeatureBuilder(fakeBars{}, nil)
	p := &store.Portfolio{ID: "x", Assets: []store.Asset{{Symbol: "AAPL", Weight: 1.0}}}
	if _, _, err := b.BuildFeatures(context.Background(), p, 63); err == nil {
		t.Fatal("want error when model has no symbol metadata, got nil")
	}
}

// TestBuildFeatures_TrueNumbers is the end-to-end regression lock: the real
// ONNX engine, the committed OHLCV in data/bars, and the feature builder must
// reproduce the Python ground-truth prediction for the trained portfolio.
// Values verified against ml onnxruntime on 2025-12-30 data (see the Python
// reference: volatility 0.268855, var_95 0.065819, cvar_95 0.037074).
func TestBuildFeatures_TrueNumbers(t *testing.T) {
	const (
		modelPath = "../../ml/models/artifacts/model.onnx"
		barsDir   = "../../data/bars"
	)
	if _, err := os.Stat(modelPath); err != nil {
		t.Skip("model file not present, skipping true-numbers regression")
	}
	if _, err := os.Stat(barsDir); err != nil {
		t.Skip("data/bars not present, skipping true-numbers regression")
	}

	engine, err := model.NewEngine(context.Background(), model.Config{ModelPath: modelPath})
	if err != nil {
		t.Skipf("engine unavailable (likely no ONNX Runtime lib): %v", err)
	}
	defer func() { _ = engine.Close() }()

	b := NewFeatureBuilder(marketdata.NewCSVBarStore(barsDir), engine.Symbols())
	// Assets intentionally alphabetical; the builder must reorder to training order.
	p := &store.Portfolio{
		ID: "flagship",
		Assets: []store.Asset{
			{Symbol: "AAPL", Weight: 0.2}, {Symbol: "AMZN", Weight: 0.2},
			{Symbol: "GOOG", Weight: 0.2}, {Symbol: "META", Weight: 0.2},
			{Symbol: "MSFT", Weight: 0.2},
		},
	}

	features, weights, err := b.BuildFeatures(context.Background(), p, 63)
	if err != nil {
		t.Fatalf("BuildFeatures: %v", err)
	}
	if len(features) != 63 || len(features[0]) != 49 {
		t.Fatalf("feature dims: got %dx%d, want 63x49", len(features), len(features[0]))
	}
	if len(weights) != 5 || weights["AAPL"] != 0.2 {
		t.Errorf("weights mismatch: %v", weights)
	}

	pred, err := engine.Predict(context.Background(), features, weights)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}

	const tol = 1e-4
	checks := []struct {
		name string
		got  float64
		want float64
	}{
		{"volatility", pred.Volatility, 0.268855},
		{"var_95", pred.VaR95, 0.065819},
		{"cvar_95", pred.CVaR95, 0.037074},
	}
	for _, c := range checks {
		if math.Abs(c.got-c.want) > tol {
			t.Errorf("%s: got %.6f, want %.6f (tol %g)", c.name, c.got, c.want, tol)
		}
	}
}
