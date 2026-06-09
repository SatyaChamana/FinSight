package inference

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"github.com/SatyaChamana/FinSight/internal/marketdata"
	"github.com/SatyaChamana/FinSight/internal/model"
	"github.com/SatyaChamana/FinSight/internal/service"
	"github.com/SatyaChamana/FinSight/internal/store"
)

// FeatureBuilder turns a portfolio into the model's [63][49] input by loading
// real OHLCV history for each holding and running the Python-parity feature
// pipeline (model.BuildFeatureMatrixOrdered).
//
// It supports two kinds of model:
//
//   - Specific-universe model (symbols non-empty, from engine.Symbols()): the
//     model was trained on one fixed 5-symbol portfolio in a fixed column
//     order. The builder imposes that exact order and rejects portfolios whose
//     holdings fall outside the trained universe.
//
//   - Generalized model (symbols empty: the manifest omits the symbols field):
//     the model was trained on many sampled portfolios with shuffled column
//     order, so it scores any 5-asset portfolio. The builder accepts the
//     portfolio's own holdings and lays them out in a deterministic order
//     (sorted, matching the store's ORDER BY symbol), as long as bar data is
//     available for each.
type FeatureBuilder struct {
	bars    marketdata.BarReader
	symbols []string // trained symbol order, or empty for a generalized model
}

// compile-time check that *FeatureBuilder satisfies service.FeatureBuilder.
var _ service.FeatureBuilder = (*FeatureBuilder)(nil)

// NewFeatureBuilder constructs a builder over a bar source and the model's
// trained symbol order (from engine.Symbols()).
func NewFeatureBuilder(bars marketdata.BarReader, symbols []string) *FeatureBuilder {
	return &FeatureBuilder{bars: bars, symbols: symbols}
}

// BuildFeatures loads bars for each model symbol, builds the feature matrix in
// the model's column order, and returns it with the portfolio's weight map.
// lookbackDays is accepted for interface compatibility but the model input is
// fixed at 63 rows.
func (b *FeatureBuilder) BuildFeatures(ctx context.Context, p *store.Portfolio, _ int) ([][]float32, map[string]float64, error) {
	weights := make(map[string]float64, len(p.Assets))
	for _, a := range p.Assets {
		weights[a.Symbol] = a.Weight
	}

	// Decide the column order. A specific-universe model dictates the order and
	// restricts the holdings; a generalized model accepts the portfolio's own
	// holdings in a deterministic (sorted) order.
	var order []string
	if len(b.symbols) > 0 {
		if err := checkUniverse(b.symbols, p.Assets); err != nil {
			return nil, nil, err
		}
		order = b.symbols
	} else {
		order = sortedSymbols(p.Assets)
	}

	symbolBars := make(map[string][]model.OHLCV, len(order))
	for _, sym := range order {
		bars, err := b.bars.Bars(ctx, sym)
		if err != nil {
			return nil, nil, fmt.Errorf("loading bars for %q: %w", sym, err)
		}
		symbolBars[sym] = bars
	}

	features, err := model.BuildFeatureMatrixOrdered(order, symbolBars, weights)
	if err != nil {
		return nil, nil, fmt.Errorf("building feature matrix: %w", err)
	}
	return features, weights, nil
}

// sortedSymbols returns the portfolio's symbols in ascending order (matching the
// store's ORDER BY symbol), used as the column order for a generalized model.
func sortedSymbols(assets []store.Asset) []string {
	out := make([]string, len(assets))
	for i, a := range assets {
		out[i] = a.Symbol
	}
	sort.Strings(out)
	return out
}

// checkUniverse verifies the portfolio holds exactly the model's symbol set.
func checkUniverse(modelSymbols []string, assets []store.Asset) error {
	if len(assets) != len(modelSymbols) {
		return fmt.Errorf(
			"portfolio has %d holdings but the model serves %d symbols [%s]",
			len(assets), len(modelSymbols), strings.Join(modelSymbols, ", "),
		)
	}
	want := make(map[string]bool, len(modelSymbols))
	for _, s := range modelSymbols {
		want[s] = true
	}
	missing := make([]string, 0)
	for _, a := range assets {
		if !want[a.Symbol] {
			missing = append(missing, a.Symbol)
		}
	}
	if len(missing) > 0 {
		sort.Strings(missing)
		return fmt.Errorf(
			"portfolio holdings [%s] are outside the model's trained universe [%s]",
			strings.Join(missing, ", "), strings.Join(modelSymbols, ", "),
		)
	}
	return nil
}
