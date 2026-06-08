package model

import (
	"context"
	"errors"
	"fmt"
)

// Reload swaps the engine's ONNX session to the model at modelPath without
// disrupting in-flight predictions. It loads the new model into a fresh session
// first (the slow, fallible part), and only then takes the write lock to swap
// the session and version atomically and close the old session.
//
// Concurrency: Reload takes the exclusive write lock (Lock), so it cannot run
// while any Predict holds the read lock and no Predict can start mid-swap. This
// prevents a torn read where a prediction would see the new session but the old
// version, or run against a session being destroyed.
//
// If anything before the swap fails, the engine keeps serving the old model and
// Reload returns an error.
func (e *Engine) Reload(ctx context.Context, modelPath string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if modelPath == "" {
		return errors.New("model: Reload modelPath must not be empty")
	}

	// Resolve the new manifest (version + symbols) before building the session so
	// a bad manifest does not leave us with a session but no metadata.
	m, err := readManifest(modelPath)
	if err != nil {
		return err
	}
	newVersion := m.ModelVersion
	if newVersion == "" {
		newVersion = "unknown"
	}

	// Build the new session outside the lock (this is the expensive step). The
	// ONNX Runtime environment is already initialized by NewEngine.
	newSess, err := newSession(modelPath)
	if err != nil {
		return fmt.Errorf("reload: %w", err)
	}

	// Atomic swap under the write lock.
	e.mu.Lock()
	oldSess := e.session
	e.session = newSess
	e.version = newVersion
	e.symbols = m.Symbols
	e.modelPath = modelPath
	e.mu.Unlock()

	// Close the old session after releasing the lock; no reader can still hold a
	// reference to it because they all went through the read lock we just excluded.
	if oldSess != nil {
		if err := oldSess.Destroy(); err != nil {
			return fmt.Errorf("reload: new model is live but closing old session failed: %w", err)
		}
	}
	return nil
}
