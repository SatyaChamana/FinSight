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

// genBars is a BarReader that synthesizes a deterministic, varying OHLCV series
// per symbol so every feature is finite. Used to exercise the generalized path
// without the ONNX runtime.
type genBars struct{ n int }

func (g genBars) Bars(_ context.Context, sym string) ([]model.OHLCV, error) {
	var seed float64
	for _, c := range sym {
		seed += float64(c)
	}
	bars := make([]model.OHLCV, g.n)
	price := 100.0 + seed/10.0
	for i := 0; i < g.n; i++ {
		drift := math.Sin(float64(i)/7.0+seed) * 0.5
		price *= 1.0 + drift/100.0
		open := price * (1.0 + math.Cos(float64(i)+seed)/200.0)
		bars[i] = model.OHLCV{
			Open: open, High: price * 1.01, Low: price * 0.99,
			Close: price, AdjClose: price, Volume: 1e6,
		}
	}
	return bars, nil
}

func TestBuildFeatures_GeneralizedPath(t *testing.T) {
	// nil model symbols => generalized model: any 5-asset portfolio is accepted
	// and laid out in sorted symbol order.
	b := NewFeatureBuilder(genBars{n: 200}, nil)
	p := &store.Portfolio{
		ID: "mix",
		Assets: []store.Asset{
			{Symbol: "MSFT", Weight: 0.3}, {Symbol: "AAPL", Weight: 0.1},
			{Symbol: "GOOG", Weight: 0.2}, {Symbol: "AMZN", Weight: 0.2},
			{Symbol: "META", Weight: 0.2},
		},
	}
	features, weights, err := b.BuildFeatures(context.Background(), p, 63)
	if err != nil {
		t.Fatalf("BuildFeatures (generalized): %v", err)
	}
	if len(features) != 63 || len(features[0]) != 49 {
		t.Fatalf("feature dims: got %dx%d, want 63x49", len(features), len(features[0]))
	}
	if len(weights) != 5 || weights["MSFT"] != 0.3 {
		t.Errorf("weights mismatch: %v", weights)
	}
}

// TestBuildFeatures_TrueNumbers is the end-to-end regression lock for the
// generalized v1.0.0 model: the real ONNX engine, the committed OHLCV in
// data/bars, and the feature builder must reproduce the Python onnxruntime
// ground truth for two distinct portfolios (verified on 2025-12-30 data). It
// also asserts the heads are internally consistent (CVaR >= VaR) and that the
// two portfolios produce different risk, which the generalized model should.
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

	bars := marketdata.NewCSVBarStore(barsDir)
	// engine.Symbols() is nil for the generalized model; the builder then orders
	// the portfolio's own holdings (sorted). Assets passed unsorted on purpose.
	b := NewFeatureBuilder(bars, engine.Symbols())

	cases := []struct {
		name                       string
		assets                     []store.Asset
		wantVol, wantVar, wantCVaR float64
	}{
		{
			name: "tech",
			assets: []store.Asset{
				{Symbol: "MSFT", Weight: 0.2}, {Symbol: "AAPL", Weight: 0.2},
				{Symbol: "GOOG", Weight: 0.2}, {Symbol: "AMZN", Weight: 0.2},
				{Symbol: "META", Weight: 0.2},
			},
			wantVol: 0.221184, wantVar: 0.035225, wantCVaR: 0.036550,
		},
		{
			name: "financials",
			assets: []store.Asset{
				{Symbol: "JPM", Weight: 0.2}, {Symbol: "BAC", Weight: 0.2},
				{Symbol: "GS", Weight: 0.2}, {Symbol: "MS", Weight: 0.2},
				{Symbol: "C", Weight: 0.2},
			},
			wantVol: 0.162252, wantVar: 0.024381, wantCVaR: 0.025154,
		},
	}

	const tol = 1e-4
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			features, weights, err := b.BuildFeatures(context.Background(), &store.Portfolio{ID: tc.name, Assets: tc.assets}, 63)
			if err != nil {
				t.Fatalf("BuildFeatures: %v", err)
			}
			if len(features) != 63 || len(features[0]) != 49 {
				t.Fatalf("feature dims: got %dx%d, want 63x49", len(features), len(features[0]))
			}
			if len(weights) != 5 {
				t.Errorf("weights len: got %d, want 5", len(weights))
			}
			pred, err := engine.Predict(context.Background(), features, weights)
			if err != nil {
				t.Fatalf("Predict: %v", err)
			}
			for _, c := range []struct {
				metric    string
				got, want float64
			}{
				{"volatility", pred.Volatility, tc.wantVol},
				{"var_95", pred.VaR95, tc.wantVar},
				{"cvar_95", pred.CVaR95, tc.wantCVaR},
			} {
				if math.Abs(c.got-c.want) > tol {
					t.Errorf("%s: got %.6f, want %.6f (tol %g)", c.metric, c.got, c.want, tol)
				}
			}
			if pred.CVaR95 < pred.VaR95 {
				t.Errorf("CVaR (%.6f) < VaR (%.6f): heads inconsistent", pred.CVaR95, pred.VaR95)
			}
		})
	}
}
