package inference

import (
	"context"
	"hash/fnv"
	"math"

	"github.com/SatyaChamana/FinSight/internal/service"
	"github.com/SatyaChamana/FinSight/internal/store"
)

const (
	// lookbackRows and featureCols match the exported ONNX model input
	// shape [63, 49] (see ml/models/artifacts/model.manifest.json).
	lookbackRows = 63
	featureCols  = 49
)

// FeatureBuilder produces the [63][49] model input from a portfolio.
//
// NOTE (Phase 5 gap): FinSight has no OHLCV price-history data source
// yet. The store holds portfolio composition only (symbols and weights),
// not market bars. model.BuildFeatureMatrix is the real, Python-parity
// feature pipeline, but it requires per-symbol OHLCV history we cannot
// supply here.
//
// Until a price-history store lands, this builder synthesizes a
// deterministic feature matrix seeded from the portfolio so that:
//  1. inference runs end to end against the real ONNX model, and
//  2. the same portfolio always yields the same features, which keeps
//     the prediction cache stable (model_version + portfolio_id key).
//
// Replace this with an OHLCV-backed builder that calls
// model.BuildFeatureMatrix once market data is wired (a later phase).
type FeatureBuilder struct{}

// compile-time check that *FeatureBuilder satisfies service.FeatureBuilder.
var _ service.FeatureBuilder = (*FeatureBuilder)(nil)

// NewFeatureBuilder constructs the placeholder feature builder.
func NewFeatureBuilder() *FeatureBuilder {
	return &FeatureBuilder{}
}

// BuildFeatures returns a deterministic [63][49] matrix and the asset
// weight map. lookbackDays is accepted for interface compatibility but
// the model input is fixed at 63 rows, so it is not used to resize.
func (b *FeatureBuilder) BuildFeatures(_ context.Context, p *store.Portfolio, _ int) ([][]float32, map[string]float64, error) {
	weights := make(map[string]float64, len(p.Assets))
	for _, a := range p.Assets {
		weights[a.Symbol] = a.Weight
	}

	seed := seedFor(p)
	matrix := make([][]float32, lookbackRows)
	for i := range matrix {
		row := make([]float32, featureCols)
		for j := range row {
			row[j] = pseudoFeature(seed, i, j)
		}
		matrix[i] = row
	}
	return matrix, weights, nil
}

// seedFor derives a stable uint64 seed from the portfolio identity and
// composition, so the same portfolio always produces the same features.
func seedFor(p *store.Portfolio) uint64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(p.ID))
	for _, a := range p.Assets {
		_, _ = h.Write([]byte(a.Symbol))
		// fold the weight in so re-weighting changes the features.
		bits := math.Float64bits(a.Weight)
		var buf [8]byte
		for k := 0; k < 8; k++ {
			buf[k] = byte(bits >> (8 * k))
		}
		_, _ = h.Write(buf[:])
	}
	return h.Sum64()
}

// pseudoFeature produces a small bounded deterministic value in roughly
// [-0.05, 0.05], resembling the scale of returns and volatilities the
// model was trained on. It is a cheap SplitMix64-style mix of the seed
// and cell coordinates, not real market data.
func pseudoFeature(seed uint64, row, col int) float32 {
	x := seed
	x ^= uint64(row+1) * 0x9E3779B97F4A7C15
	x ^= uint64(col+1) * 0xC2B2AE3D27D4EB4F
	x ^= x >> 30
	x *= 0xBF58476D1CE4E5B9
	x ^= x >> 27
	x *= 0x94D049BB133111EB
	x ^= x >> 31
	// map the top 53 bits to [0, 1), then to [-0.05, 0.05].
	unit := float64(x>>11) / float64(uint64(1)<<53)
	return float32((unit - 0.5) * 0.1)
}
