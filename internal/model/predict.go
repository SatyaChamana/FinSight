package model

import (
	"context"
	"errors"
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
)

// Prediction is the result of one inference call. VaR95 and CVaR95 are positive
// loss magnitudes (the model is trained that way, see ml/data/dataset.py).
type Prediction struct {
	Volatility float64
	VaR95      float64
	CVaR95     float64

	ModelVersion string

	// AssetContributions decomposes VaR95 across assets. See the comment on
	// Predict for the (deliberately simple) decomposition used. The map values
	// sum to approximately VaR95. It is an empty (non-nil) map when no weights
	// are supplied.
	AssetContributions map[string]float64
}

// Predict runs one forward pass of the model on a single [63, 49] feature
// window and returns the three risk metrics plus a per-asset VaR decomposition.
//
// features must be exactly lookbackDays (63) rows, each with featureCols (49)
// columns, in the column order produced by BuildFeatureMatrix. weights maps
// asset symbol to portfolio weight; it is used only for AssetContributions and
// may be empty.
//
// Concurrency: Predict holds a read lock for the duration of the inference call,
// so many predictions run in parallel and a Reload cannot swap the session out
// from under an in-flight Run.
//
// ctx is accepted for cancellation/consistency; the ONNX Run itself is a
// synchronous C call and is not interrupted mid-flight, but we check ctx before
// starting work.
func (e *Engine) Predict(
	ctx context.Context,
	features [][]float32,
	weights map[string]float64,
) (*Prediction, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	flat, err := flattenFeatures(features)
	if err != nil {
		return nil, err
	}

	e.mu.RLock()
	defer e.mu.RUnlock()

	if e.session == nil {
		return nil, errors.New("model: engine is closed")
	}

	// Input tensor of shape [1, 63, 49] (batch size 1).
	inputShape := ort.NewShape(1, lookbackDays, featureCols)
	inputTensor, err := ort.NewTensor(inputShape, flat)
	if err != nil {
		return nil, fmt.Errorf("creating input tensor: %w", err)
	}
	defer func() { _ = inputTensor.Destroy() }()

	// Pre-allocate the output tensor of shape [1, 3]. Pre-allocating (rather than
	// passing a nil output) keeps ownership of the result buffer on our side so
	// we can read it directly after Run.
	outputShape := ort.NewShape(1, outputCols)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, fmt.Errorf("creating output tensor: %w", err)
	}
	defer func() { _ = outputTensor.Destroy() }()

	if err := e.session.Run(
		[]ort.Value{inputTensor},
		[]ort.Value{outputTensor},
	); err != nil {
		return nil, fmt.Errorf("running ONNX inference: %w", err)
	}

	out := outputTensor.GetData()
	if len(out) != outputCols {
		return nil, fmt.Errorf("unexpected output length: got %d, want %d", len(out), outputCols)
	}

	// Column order is fixed by the model: [volatility, var_95, cvar_95].
	pred := &Prediction{
		Volatility:         float64(out[0]),
		VaR95:              float64(out[1]),
		CVaR95:             float64(out[2]),
		ModelVersion:       e.version,
		AssetContributions: assetContributions(weights, float64(out[1])),
	}
	return pred, nil
}

// flattenFeatures validates the [63][49] shape and flattens it row-major into a
// single []float32 of length 63*49, which is the layout the input tensor expects.
func flattenFeatures(features [][]float32) ([]float32, error) {
	if len(features) != lookbackDays {
		return nil, fmt.Errorf("features must have %d rows, got %d", lookbackDays, len(features))
	}
	flat := make([]float32, 0, lookbackDays*featureCols)
	for i, row := range features {
		if len(row) != featureCols {
			return nil, fmt.Errorf("features row %d must have %d columns, got %d", i, featureCols, len(row))
		}
		flat = append(flat, row...)
	}
	return flat, nil
}

// assetContributions performs a simple, documented VaR decomposition.
//
// SIMPLIFICATION: a rigorous risk decomposition would use marginal VaR
// (the partial derivative of portfolio VaR with respect to each asset weight,
// which needs the asset covariance matrix). The model does not expose those
// gradients, so we approximate each asset's contribution as its weight share of
// total VaR: contribution_i = (weight_i / sum_weights) * VaR95. The map values
// therefore sum to approximately VaR95. This is a proportional attribution, not
// a true marginal-risk decomposition, and is documented as such for callers.
//
// If weights is empty, an empty (non-nil) map is returned.
func assetContributions(weights map[string]float64, var95 float64) map[string]float64 {
	contributions := make(map[string]float64, len(weights))
	if len(weights) == 0 {
		return contributions
	}

	var total float64
	for _, w := range weights {
		total += w
	}
	if total == 0 {
		// Degenerate weights: split VaR evenly so the map still sums to VaR95.
		even := var95 / float64(len(weights))
		for sym := range weights {
			contributions[sym] = even
		}
		return contributions
	}

	for sym, w := range weights {
		contributions[sym] = (w / total) * var95
	}
	return contributions
}
