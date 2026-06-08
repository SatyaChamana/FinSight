package model

import (
	"context"
	"math"
	"testing"
)

// TestReloadSameModel loads the real model, reloads the same file, and verifies
// the version is unchanged and predictions still work. It skips when the native
// library or model file is unavailable.
func TestReloadSameModel(t *testing.T) {
	eng := newTestEngine(t)
	defer func() { _ = eng.Close() }()

	before := eng.Version()
	if before != "0.1.0" {
		t.Fatalf("Version before reload = %q, want %q", before, "0.1.0")
	}

	if err := eng.Reload(context.Background(), testModelPath); err != nil {
		t.Fatalf("Reload: %v", err)
	}

	after := eng.Version()
	if after != before {
		t.Errorf("Version after reload = %q, want unchanged %q", after, before)
	}

	pred, err := eng.Predict(context.Background(), constantFeatures(0.01), nil)
	if err != nil {
		t.Fatalf("Predict after reload: %v", err)
	}
	if math.IsNaN(pred.VaR95) || math.IsInf(pred.VaR95, 0) {
		t.Errorf("VaR95 after reload = %v, want finite", pred.VaR95)
	}
	if pred.ModelVersion != before {
		t.Errorf("ModelVersion after reload = %q, want %q", pred.ModelVersion, before)
	}
}

func TestReloadValidation(t *testing.T) {
	eng := newTestEngine(t)
	defer func() { _ = eng.Close() }()

	// Empty path is rejected.
	if err := eng.Reload(context.Background(), ""); err == nil {
		t.Errorf("expected error for empty modelPath")
	}

	// Cancelled context is rejected.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := eng.Reload(ctx, testModelPath); err == nil {
		t.Errorf("expected error for cancelled context")
	}

	// Nonexistent path is rejected and leaves the live model intact.
	if err := eng.Reload(context.Background(), "/nonexistent/model.onnx"); err == nil {
		t.Errorf("expected error for nonexistent model path")
	}
	if _, err := eng.Predict(context.Background(), constantFeatures(0.01), nil); err != nil {
		t.Errorf("engine should still serve after failed reload: %v", err)
	}
}
