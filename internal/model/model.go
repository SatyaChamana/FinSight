// Package model loads a trained ONNX risk-prediction model and serves
// inference from the Go service. The model is the LSTM + self-attention network
// trained by the Python pipeline (see ml/models/architecture.py). It takes a
// [63, 49] feature window and returns three risk numbers
// (volatility, VaR95, CVaR95).
//
// Go concepts in this package (teaching notes):
//   - sync.RWMutex: many goroutines call Predict concurrently (read path), but
//     Reload happens rarely (write path). An RWMutex lets all readers proceed in
//     parallel while a single writer gets exclusive access during a swap. A plain
//     sync.Mutex would serialize every prediction, which we do not want.
//   - sync.Once: the ONNX Runtime C library must be initialized exactly once per
//     process. sync.Once guarantees that even if many goroutines build an Engine
//     at the same time, the shared environment is set up a single time.
//   - cgo/FFI: onnxruntime_go is a thin wrapper over the ONNX Runtime C library.
//     We must point it at the shared library (.dylib) before first use.
package model

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// inputTensorName, outputTensorName, and the input dimensions are fixed by the
// exported model (see ml/models/artifacts/model.manifest.json). Changing the
// Python export means changing these constants too.
const (
	inputTensorName  = "input"
	outputTensorName = "output"

	lookbackDays = 63 // rows in the feature window (T)
	featureCols  = 49 // columns per row (P*F + portfolio features)
	outputCols   = 3  // [volatility, var_95, cvar_95]

	// fallbackLibPath is used when ONNXRUNTIME_LIB_PATH is not set. It points at
	// the dylib that ships inside the Python venv on this machine.
	fallbackLibPath = "/Users/satyachamana/My Projects/FinSight/ml/.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.1.26.0.dylib"

	// libPathEnv is the environment variable checked first for the dylib path.
	libPathEnv = "ONNXRUNTIME_LIB_PATH"
)

// ortInitOnce guards one-time initialization of the shared ONNX Runtime
// environment. The environment is process-wide, so it is initialized once and
// never destroyed during normal operation (the OS reclaims it at exit). We keep
// the init error so later callers see the same failure.
var (
	ortInitOnce sync.Once
	ortInitErr  error
)

// Config holds the inputs needed to construct an Engine. Dependencies are passed
// in explicitly (constructor injection) rather than read from globals.
type Config struct {
	// ModelPath is the absolute path to the .onnx file.
	ModelPath string

	// SharedLibPath optionally overrides the ONNX Runtime shared library path.
	// If empty, NewEngine resolves it from ONNXRUNTIME_LIB_PATH, then falls back
	// to the bundled venv dylib.
	SharedLibPath string

	// Version optionally overrides the model version. If empty, NewEngine reads
	// it from model.manifest.json next to the model file, defaulting to
	// "unknown" if no manifest is present.
	Version string
}

// manifest mirrors the fields we read from model.manifest.json. Only
// ModelVersion is required here; the rest of the file is ignored.
type manifest struct {
	ModelVersion string `json:"model_version"`
}

// Engine owns the ONNX inference session and serves predictions. It is safe for
// concurrent use: Predict and Version take a read lock, Reload takes the write
// lock.
type Engine struct {
	// mu protects session, version, and modelPath. RWMutex is chosen because
	// reads (Predict) are frequent and concurrent while writes (Reload) are rare.
	mu sync.RWMutex

	session   *ort.DynamicAdvancedSession
	version   string
	modelPath string
}

// resolveLibPath returns the shared library path to use, preferring the explicit
// override, then the environment variable, then the bundled fallback. It returns
// an error if the chosen path does not exist on disk.
func resolveLibPath(override string) (string, error) {
	candidate := override
	if candidate == "" {
		candidate = os.Getenv(libPathEnv)
	}
	if candidate == "" {
		candidate = fallbackLibPath
	}

	info, err := os.Stat(candidate)
	if err != nil {
		return "", fmt.Errorf("onnx runtime shared library not found at %q: %w", candidate, err)
	}
	if info.IsDir() {
		return "", fmt.Errorf("onnx runtime shared library path %q is a directory, not a file", candidate)
	}
	return candidate, nil
}

// initRuntime initializes the process-wide ONNX Runtime environment exactly once.
// It is safe to call from multiple goroutines.
func initRuntime(libPath string) error {
	ortInitOnce.Do(func() {
		ort.SetSharedLibraryPath(libPath)
		ortInitErr = ort.InitializeEnvironment()
	})
	return ortInitErr
}

// readVersionFromManifest looks for model.manifest.json next to the model file
// and returns its model_version. If the manifest is missing it returns
// "unknown" with no error (a missing manifest is not fatal). Malformed JSON is
// reported as an error so the caller knows something is wrong.
func readVersionFromManifest(modelPath string) (string, error) {
	manifestPath := filepath.Join(filepath.Dir(modelPath), "model.manifest.json")
	raw, err := os.ReadFile(manifestPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "unknown", nil
		}
		return "", fmt.Errorf("reading manifest %q: %w", manifestPath, err)
	}

	var m manifest
	if err := json.Unmarshal(raw, &m); err != nil {
		return "", fmt.Errorf("parsing manifest %q: %w", manifestPath, err)
	}
	if m.ModelVersion == "" {
		return "unknown", nil
	}
	return m.ModelVersion, nil
}

// newSession creates a DynamicAdvancedSession for the given model path. It
// assumes the ONNX Runtime environment is already initialized.
func newSession(modelPath string) (*ort.DynamicAdvancedSession, error) {
	if _, err := os.Stat(modelPath); err != nil {
		return nil, fmt.Errorf("model file not found at %q: %w", modelPath, err)
	}

	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{inputTensorName},
		[]string{outputTensorName},
		nil, // default session options
	)
	if err != nil {
		return nil, fmt.Errorf("creating ONNX session for %q: %w", modelPath, err)
	}
	return session, nil
}

// NewEngine constructs an Engine: it resolves and initializes the ONNX Runtime
// shared library, opens an inference session on cfg.ModelPath, and resolves the
// model version. The ctx is accepted for interface consistency and future
// cancellation support; the underlying ONNX session creation is synchronous.
func NewEngine(ctx context.Context, cfg Config) (*Engine, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if cfg.ModelPath == "" {
		return nil, errors.New("model: Config.ModelPath must not be empty")
	}

	libPath, err := resolveLibPath(cfg.SharedLibPath)
	if err != nil {
		return nil, err
	}
	if err := initRuntime(libPath); err != nil {
		return nil, fmt.Errorf("initializing ONNX runtime environment: %w", err)
	}

	version := cfg.Version
	if version == "" {
		version, err = readVersionFromManifest(cfg.ModelPath)
		if err != nil {
			return nil, err
		}
	}

	session, err := newSession(cfg.ModelPath)
	if err != nil {
		return nil, err
	}

	return &Engine{
		session:   session,
		version:   version,
		modelPath: cfg.ModelPath,
	}, nil
}

// Version returns the loaded model version. It takes a read lock so it is safe
// to call concurrently with Predict and during a Reload.
func (e *Engine) Version() string {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.version
}

// Close releases the ONNX session. The process-wide ONNX Runtime environment is
// intentionally left initialized (it is shared and reclaimed at process exit).
// Close is safe to call once; calling it again is a no-op.
func (e *Engine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.session == nil {
		return nil
	}
	err := e.session.Destroy()
	e.session = nil
	if err != nil {
		return fmt.Errorf("destroying ONNX session: %w", err)
	}
	return nil
}
