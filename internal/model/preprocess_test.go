package model

import (
	"math"
	"testing"
)

// approxEqual reports whether a and b are within tol. NaN is treated as a match
// only when both are NaN.
func approxEqual(a, b, tol float64) bool {
	if math.IsNaN(a) || math.IsNaN(b) {
		return math.IsNaN(a) && math.IsNaN(b)
	}
	return math.Abs(a-b) <= tol
}

func TestLogReturn(t *testing.T) {
	// Golden values produced by ml/data/features.py _log_return on this series.
	adj := []float64{100.0, 101.0, 99.0, 102.0, 103.0, 98.0}
	got := logReturn(adj)

	want := []float64{
		math.NaN(),
		0.0099503309,
		-0.0200006667,
		0.0298529631,
		0.0097561749,
		-0.0497615096,
	}

	if len(got) != len(want) {
		t.Fatalf("logReturn length = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if !approxEqual(got[i], want[i], 1e-9) {
			t.Errorf("logReturn[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestRSI(t *testing.T) {
	// Golden values produced by ml/data/features.py _rsi (Wilder, period=3) on
	// this close series. The first `period` rows are NaN (min_periods).
	close := []float64{100.0, 101.0, 99.0, 102.0, 103.0, 98.0, 99.0, 101.0, 104.0, 103.0}
	got := rsi(close, 3)

	want := []float64{
		math.NaN(), math.NaN(), math.NaN(),
		76.47058824,
		81.39534884,
		31.67420814,
		42.25621415,
		60.57441253,
		76.99485812,
		63.72448578,
	}

	if len(got) != len(want) {
		t.Fatalf("rsi length = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if !approxEqual(got[i], want[i], 1e-6) {
			t.Errorf("rsi[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestRSISaturatesAt100(t *testing.T) {
	// Strictly increasing closes -> avg_loss is zero -> RSI saturates at 100.
	close := []float64{10, 11, 12, 13, 14, 15, 16}
	got := rsi(close, 3)
	// First finite value is at index 3 (period=3) and onward.
	for i := 3; i < len(got); i++ {
		if !approxEqual(got[i], 100.0, 1e-9) {
			t.Errorf("rsi[%d] = %v, want 100 (monotonic gains)", i, got[i])
		}
	}
}

func TestSampleStd(t *testing.T) {
	// Sample std (ddof=1) of [2,4,4,4,5,5,7,9] = 2.13808993...
	vals := []float64{2, 4, 4, 4, 5, 5, 7, 9}
	got, ok := sampleStd(vals)
	if !ok {
		t.Fatalf("sampleStd returned ok=false")
	}
	want := 2.138089935299395
	if !approxEqual(got, want, 1e-9) {
		t.Errorf("sampleStd = %v, want %v", got, want)
	}

	if _, ok := sampleStd([]float64{1.0}); ok {
		t.Errorf("sampleStd of single element should return ok=false")
	}
	if _, ok := sampleStd([]float64{1.0, math.NaN()}); ok {
		t.Errorf("sampleStd with NaN should return ok=false")
	}
}

func TestGarmanKlassNonNegative(t *testing.T) {
	open := []float64{100, 101}
	high := []float64{102, 103}
	low := []float64{99, 100}
	close := []float64{101, 102}
	got := garmanKlass(open, high, low, close)
	for i, v := range got {
		if v < 0 || math.IsNaN(v) {
			t.Errorf("garmanKlass[%d] = %v, want finite non-negative", i, v)
		}
	}
}

func TestUpperTriMeanCorr(t *testing.T) {
	// Two perfectly correlated columns -> mean upper-tri correlation = 1.0.
	a := []float64{1, 2, 3, 4, 5}
	b := []float64{2, 4, 6, 8, 10}
	got := upperTriMeanCorr([][]float64{a, b})
	if !approxEqual(got, 1.0, 1e-9) {
		t.Errorf("upperTriMeanCorr (perfectly correlated) = %v, want 1.0", got)
	}

	// Perfectly anti-correlated -> -1.0.
	c := []float64{5, 4, 3, 2, 1}
	got = upperTriMeanCorr([][]float64{a, c})
	if !approxEqual(got, -1.0, 1e-9) {
		t.Errorf("upperTriMeanCorr (anti-correlated) = %v, want -1.0", got)
	}
}

// makeBars builds a deterministic OHLCV series of length n. Prices follow a
// gentle wave so features are all well-defined (no constant columns).
func makeBars(n int, seed float64) []OHLCV {
	bars := make([]OHLCV, n)
	for i := 0; i < n; i++ {
		base := 100.0 + seed + 5.0*math.Sin(float64(i)/7.0) + 0.05*float64(i)
		bars[i] = OHLCV{
			Open:     base,
			High:     base * 1.01,
			Low:      base * 0.99,
			Close:    base * 1.002,
			AdjClose: base * 1.002,
			Volume:   1_000_000 + float64(i),
		}
	}
	return bars
}

func TestBuildFeatureMatrixDims(t *testing.T) {
	// Need enough history so the 63-day rolling correlation has 63 valid rows
	// plus 63 lookback rows. log_return drops 1 row, so >= 1 + 63 + 63 bars.
	const n = 200
	symbols := []string{"AAA", "BBB", "CCC", "DDD", "EEE"}
	bars := make(map[string][]OHLCV, len(symbols))
	for i, s := range symbols {
		bars[s] = makeBars(n, float64(i)*3.0)
	}
	weights := map[string]float64{
		"AAA": 0.2, "BBB": 0.2, "CCC": 0.2, "DDD": 0.2, "EEE": 0.2,
	}

	matrix, err := BuildFeatureMatrix(bars, weights)
	if err != nil {
		t.Fatalf("BuildFeatureMatrix error: %v", err)
	}
	if len(matrix) != lookbackDays {
		t.Fatalf("matrix rows = %d, want %d", len(matrix), lookbackDays)
	}
	for r, row := range matrix {
		if len(row) != featureCols {
			t.Fatalf("matrix row %d cols = %d, want %d", r, len(row), featureCols)
		}
		for c, v := range row {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Errorf("matrix[%d][%d] = %v, want finite", r, c, v)
			}
		}
	}

	// Concentration column (index 47, the 3rd portfolio feature) is Herfindahl of
	// equal weights: 5 * 0.2^2 = 0.2. Constant across rows.
	concCol := expectedSymbolCount*perSymbolFeatures + 2
	for r := range matrix {
		if !approxEqual(float64(matrix[r][concCol]), 0.2, 1e-6) {
			t.Errorf("concentration[%d] = %v, want 0.2", r, matrix[r][concCol])
		}
	}
}

func TestBuildFeatureMatrixValidation(t *testing.T) {
	good := map[string]float64{"AAA": 0.2, "BBB": 0.2, "CCC": 0.2, "DDD": 0.2, "EEE": 0.2}

	tests := []struct {
		name    string
		symbols []string
		weights map[string]float64
		nBars   int
		wantErr bool
	}{
		{
			name:    "wrong symbol count",
			symbols: []string{"AAA", "BBB", "CCC"},
			weights: map[string]float64{"AAA": 0.34, "BBB": 0.33, "CCC": 0.33},
			nBars:   200,
			wantErr: true,
		},
		{
			name:    "weights do not sum to one",
			symbols: []string{"AAA", "BBB", "CCC", "DDD", "EEE"},
			weights: map[string]float64{"AAA": 0.5, "BBB": 0.2, "CCC": 0.2, "DDD": 0.2, "EEE": 0.2},
			nBars:   200,
			wantErr: true,
		},
		{
			name:    "not enough history",
			symbols: []string{"AAA", "BBB", "CCC", "DDD", "EEE"},
			weights: good,
			nBars:   60,
			wantErr: true,
		},
		{
			name:    "valid",
			symbols: []string{"AAA", "BBB", "CCC", "DDD", "EEE"},
			weights: good,
			nBars:   200,
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bars := make(map[string][]OHLCV, len(tt.symbols))
			for i, s := range tt.symbols {
				bars[s] = makeBars(tt.nBars, float64(i)*3.0)
			}
			_, err := BuildFeatureMatrix(bars, tt.weights)
			if tt.wantErr && err == nil {
				t.Errorf("expected error, got nil")
			}
			if !tt.wantErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}
