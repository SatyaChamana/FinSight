package model

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
)

// testModelPath is the trained model produced by the Python pipeline.
const testModelPath = "/Users/satyachamana/My Projects/FinSight/ml/models/artifacts/model.onnx"

// testVenvPython is the venv interpreter used for the Python cross-check.
const testVenvPython = "/Users/satyachamana/My Projects/FinSight/ml/.venv/bin/python"

// resolveTestLibPath returns the dylib path the same way NewEngine does. The
// runtime tests skip when the lib or model is absent so CI without the native
// library still passes.
func resolveTestLibPath(t *testing.T) string {
	t.Helper()
	libPath, err := resolveLibPath("")
	if err != nil {
		t.Skipf("skipping: ONNX runtime shared library not available: %v", err)
	}
	if _, err := os.Stat(testModelPath); err != nil {
		t.Skipf("skipping: model file not available: %v", err)
	}
	return libPath
}

// constantFeatures returns a deterministic [63][49] matrix filled with v.
func constantFeatures(v float32) [][]float32 {
	m := make([][]float32, lookbackDays)
	for i := range m {
		row := make([]float32, featureCols)
		for j := range row {
			row[j] = v
		}
		m[i] = row
	}
	return m
}

func newTestEngine(t *testing.T) *Engine {
	t.Helper()
	resolveTestLibPath(t)
	eng, err := NewEngine(context.Background(), Config{ModelPath: testModelPath})
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}
	return eng
}

func TestPredictGolden(t *testing.T) {
	eng := newTestEngine(t)
	defer func() {
		if err := eng.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
	}()

	if got := eng.Version(); got != "0.1.0" {
		t.Errorf("Version = %q, want %q", got, "0.1.0")
	}

	features := constantFeatures(0.01)
	weights := map[string]float64{
		"AAA": 0.2, "BBB": 0.2, "CCC": 0.2, "DDD": 0.2, "EEE": 0.2,
	}

	pred, err := eng.Predict(context.Background(), features, weights)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}

	// Shape / sanity: all three outputs finite.
	for name, v := range map[string]float64{
		"Volatility": pred.Volatility,
		"VaR95":      pred.VaR95,
		"CVaR95":     pred.CVaR95,
	} {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("%s = %v, want finite", name, v)
		}
	}

	if pred.ModelVersion != "0.1.0" {
		t.Errorf("ModelVersion = %q, want %q", pred.ModelVersion, "0.1.0")
	}

	// AssetContributions should sum to approximately VaR95.
	if len(pred.AssetContributions) != len(weights) {
		t.Errorf("AssetContributions has %d entries, want %d", len(pred.AssetContributions), len(weights))
	}
	var sum float64
	for _, c := range pred.AssetContributions {
		sum += c
	}
	if math.Abs(sum-pred.VaR95) > 1e-6 {
		t.Errorf("AssetContributions sum = %v, want ~VaR95 %v", sum, pred.VaR95)
	}

	// Cross-check against Python onnxruntime if the venv exists.
	pyOut, ok := pythonReference(t, features)
	if !ok {
		t.Log("python cross-check skipped (venv/onnxruntime unavailable)")
		return
	}
	const tol = 1e-4
	if math.Abs(pyOut[0]-pred.Volatility) > tol {
		t.Errorf("Volatility Go=%v Python=%v (tol %g)", pred.Volatility, pyOut[0], tol)
	}
	if math.Abs(pyOut[1]-pred.VaR95) > tol {
		t.Errorf("VaR95 Go=%v Python=%v (tol %g)", pred.VaR95, pyOut[1], tol)
	}
	if math.Abs(pyOut[2]-pred.CVaR95) > tol {
		t.Errorf("CVaR95 Go=%v Python=%v (tol %g)", pred.CVaR95, pyOut[2], tol)
	}
	t.Logf("golden cross-check passed: Go=%v Python=%v", []float64{pred.Volatility, pred.VaR95, pred.CVaR95}, pyOut)
}

// pythonReference runs the same constant input through Python onnxruntime and
// returns the [volatility, var_95, cvar_95] output. ok is false (test continues)
// if the venv python or onnxruntime is unavailable.
func pythonReference(t *testing.T, features [][]float32) ([]float64, bool) {
	t.Helper()
	if _, err := os.Stat(testVenvPython); err != nil {
		return nil, false
	}

	// The input is constant, so we only need the fill value and dimensions.
	fill := features[0][0]
	script := `
import sys, json
try:
    import numpy as np
    import onnxruntime as ort
except Exception:
    print("NO_ORT")
    sys.exit(0)
fill = ` + formatFloat(fill) + `
x = np.full((1, ` + itoa(lookbackDays) + `, ` + itoa(featureCols) + `), fill, dtype=np.float32)
sess = ort.InferenceSession(` + jsonString(testModelPath) + `)
out = sess.run(["output"], {"input": x})[0]
print(json.dumps([float(v) for v in out[0].tolist()]))
`
	cmd := exec.Command(testVenvPython, "-c", script)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Logf("python reference failed (skipping cross-check): %v\n%s", err, out)
		return nil, false
	}
	text := strings.TrimSpace(string(out))
	if text == "NO_ORT" || text == "" {
		return nil, false
	}
	// Take the last line in case of warnings.
	lines := strings.Split(text, "\n")
	last := strings.TrimSpace(lines[len(lines)-1])
	var vals []float64
	if err := json.Unmarshal([]byte(last), &vals); err != nil {
		t.Logf("could not parse python output %q: %v", last, err)
		return nil, false
	}
	if len(vals) != outputCols {
		return nil, false
	}
	return vals, true
}

func TestPredictWrongDims(t *testing.T) {
	eng := newTestEngine(t)
	defer func() { _ = eng.Close() }()

	tests := []struct {
		name     string
		features [][]float32
	}{
		{name: "too few rows", features: make([][]float32, 10)},
		{name: "empty", features: [][]float32{}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := eng.Predict(context.Background(), tt.features, nil); err == nil {
				t.Errorf("expected error for %s", tt.name)
			}
		})
	}

	// Right rows, wrong columns.
	bad := make([][]float32, lookbackDays)
	for i := range bad {
		bad[i] = make([]float32, featureCols-1)
	}
	if _, err := eng.Predict(context.Background(), bad, nil); err == nil {
		t.Errorf("expected error for wrong column count")
	}
}

func TestPredictEmptyWeights(t *testing.T) {
	eng := newTestEngine(t)
	defer func() { _ = eng.Close() }()

	pred, err := eng.Predict(context.Background(), constantFeatures(0.01), nil)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}
	if pred.AssetContributions == nil {
		t.Errorf("AssetContributions should be non-nil empty map, got nil")
	}
	if len(pred.AssetContributions) != 0 {
		t.Errorf("AssetContributions should be empty, got %v", pred.AssetContributions)
	}
}

// TestConcurrentPredict proves that many goroutines can call Predict at once
// without data races or panics. Run with -race to catch races. It skips if the
// native library is unavailable.
func TestConcurrentPredict(t *testing.T) {
	eng := newTestEngine(t)
	defer func() { _ = eng.Close() }()

	const goroutines = 50
	features := constantFeatures(0.01)
	weights := map[string]float64{"AAA": 0.2, "BBB": 0.2, "CCC": 0.2, "DDD": 0.2, "EEE": 0.2}

	var wg sync.WaitGroup
	errCh := make(chan error, goroutines)
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			// Mix in Version() reads to exercise the RLock from two call sites.
			_ = eng.Version()
			if _, err := eng.Predict(context.Background(), features, weights); err != nil {
				errCh <- err
			}
		}()
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Errorf("concurrent Predict error: %v", err)
	}
}

func TestNewEngineErrors(t *testing.T) {
	// Empty model path is rejected before any native call.
	if _, err := NewEngine(context.Background(), Config{ModelPath: ""}); err == nil {
		t.Errorf("expected error for empty ModelPath")
	}

	// Cancelled context is rejected up front.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := NewEngine(ctx, Config{ModelPath: testModelPath}); err == nil {
		t.Errorf("expected error for cancelled context")
	}
}

func TestReadVersionFromManifest(t *testing.T) {
	dir := t.TempDir()
	modelPath := filepath.Join(dir, "model.onnx")
	manifestPath := filepath.Join(dir, "model.manifest.json")

	if err := os.WriteFile(manifestPath, []byte(`{"model_version":"9.9.9"}`), 0o600); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	v, err := readVersionFromManifest(modelPath)
	if err != nil {
		t.Fatalf("readVersionFromManifest: %v", err)
	}
	if v != "9.9.9" {
		t.Errorf("version = %q, want %q", v, "9.9.9")
	}

	// Missing manifest -> "unknown", no error.
	v, err = readVersionFromManifest(filepath.Join(t.TempDir(), "model.onnx"))
	if err != nil {
		t.Fatalf("readVersionFromManifest (missing): %v", err)
	}
	if v != "unknown" {
		t.Errorf("missing manifest version = %q, want %q", v, "unknown")
	}
}

// Small formatting helpers to keep the Python script generation dependency-free.

func formatFloat(f float32) string {
	return strconv.FormatFloat(float64(f), 'g', -1, 32)
}

func itoa(i int) string { return strconv.Itoa(i) }

func jsonString(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}
