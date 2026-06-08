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
// The model is shared and was trained on one fixed symbol universe in a fixed
// column order (engine.Symbols()). The per-symbol feature blocks must be laid
// out in that exact order, which is insertion order from training, NOT
// alphabetical. This builder therefore imposes the model's symbol order
// regardless of how the store returns the portfolio's assets, and rejects
// portfolios whose holdings fall outside the model's trained universe (a shared
// single-portfolio model cannot meaningfully score arbitrary holdings).
type FeatureBuilder struct {
	bars    marketdata.BarReader
	symbols []string // model's trained symbol order (column layout)
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
	if len(b.symbols) == 0 {
		return nil, nil, fmt.Errorf("feature builder: model exposes no symbol metadata; cannot order feature columns")
	}
	if err := checkUniverse(b.symbols, p.Assets); err != nil {
		return nil, nil, err
	}

	weights := make(map[string]float64, len(p.Assets))
	for _, a := range p.Assets {
		weights[a.Symbol] = a.Weight
	}

	symbolBars := make(map[string][]model.OHLCV, len(b.symbols))
	for _, sym := range b.symbols {
		bars, err := b.bars.Bars(ctx, sym)
		if err != nil {
			return nil, nil, fmt.Errorf("loading bars for %q: %w", sym, err)
		}
		symbolBars[sym] = bars
	}

	features, err := model.BuildFeatureMatrixOrdered(b.symbols, symbolBars, weights)
	if err != nil {
		return nil, nil, fmt.Errorf("building feature matrix: %w", err)
	}
	return features, weights, nil
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
