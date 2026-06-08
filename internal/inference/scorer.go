// Package inference wires the model package (raw ONNX execution) to the
// service package (orchestration). The service package defines the
// Scorer and FeatureBuilder interfaces where it consumes them; this
// package provides concrete adapters that satisfy those interfaces.
//
// Keeping the adapters here (not in service, not in model) preserves the
// dependency direction: service depends only on its own interfaces,
// model knows nothing about service, and the composition root (main)
// glues them together through this package.
package inference

import (
	"context"
	"fmt"

	"github.com/SatyaChamana/FinSight/internal/model"
	"github.com/SatyaChamana/FinSight/internal/service"
)

// Scorer adapts a *model.Engine to service.Scorer. It translates the
// model package's Prediction into the service package's ScoreResult so
// neither package needs to import the other.
type Scorer struct {
	engine *model.Engine
}

// compile-time check that *Scorer satisfies the interface the service
// layer consumes.
var _ service.Scorer = (*Scorer)(nil)

// NewScorer wraps a loaded model engine.
func NewScorer(engine *model.Engine) *Scorer {
	return &Scorer{engine: engine}
}

// Predict runs ONNX inference and maps the result into the service-level
// ScoreResult shape.
func (s *Scorer) Predict(ctx context.Context, features [][]float32, weights map[string]float64) (*service.ScoreResult, error) {
	p, err := s.engine.Predict(ctx, features, weights)
	if err != nil {
		return nil, fmt.Errorf("model inference: %w", err)
	}
	return &service.ScoreResult{
		Volatility:         p.Volatility,
		VaR95:              p.VaR95,
		CVaR95:             p.CVaR95,
		ModelVersion:       p.ModelVersion,
		AssetContributions: p.AssetContributions,
	}, nil
}

// Version reports the loaded model version (used in the cache key).
func (s *Scorer) Version() string {
	return s.engine.Version()
}
