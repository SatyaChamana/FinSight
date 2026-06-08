# Phase 5B: Concurrency, sync.RWMutex, and Atomic Hot Reload

The Go concurrency concepts this phase introduced, grounded in the model engine.

## The setup: one model, many concurrent requests

A gRPC server handles many requests at once. Each request runs in its own goroutine (the gRPC library spawns them; you do not). All of those goroutines share one `*model.Engine` holding one loaded ONNX session in memory. Shared mutable state read by many goroutines at once is the classic data race. We need to make concurrent access safe without making it slow.

### Goroutine, in one line

A goroutine is a function running concurrently, started with `go f()`. It is much cheaper than an OS thread; a server can have thousands. You rarely start them yourself here, but you must assume your handler code runs in many goroutines simultaneously.

## Mutex vs RWMutex

A `sync.Mutex` gives exclusive access: one goroutine holds it, everyone else waits. If every prediction took the mutex, predictions would run one at a time, serializing the whole server. That is wrong for our access pattern.

Our access pattern is: predictions happen constantly (reads of the model), reloads happen rarely (writes that replace the model). This is a classic many readers, rare writer case. `sync.RWMutex` is built for it:

- `RLock` / `RUnlock`: any number of readers can hold it at the same time.
- `Lock` / `Unlock`: a writer gets exclusive access; it waits for in flight readers to finish, and new readers wait for it.

So all predictions run in parallel, and only a reload pauses them, briefly.

```go
type Engine struct {
    mu      sync.RWMutex
    session *ort.DynamicAdvancedSession
    version string
}

func (e *Engine) Predict(...) (*Prediction, error) {
    e.mu.RLock()
    defer e.mu.RUnlock()
    // ... run inference using e.session ...
}

func (e *Engine) Version() string {
    e.mu.RLock()
    defer e.mu.RUnlock()
    return e.version
}
```

`defer` schedules the unlock to run when the function returns, even on an early return or panic. It is the idiomatic way to guarantee you never leak a held lock.

## Atomic hot reload: swap the model without dropping requests

Requirement from the plan: when a new model version is published, swap it in with zero downtime, no torn reads (no request should ever see a half loaded model).

The naive approach is wrong: lock, close the old session, load the new one, unlock. That holds the write lock for the entire slow load, blocking every prediction for hundreds of milliseconds, and if the load fails you have already destroyed the working model.

The correct approach, in `reload.go`, is build then swap:

```go
func (e *Engine) Reload(ctx context.Context, modelPath string) error {
    newSession, newVersion, err := loadSession(modelPath) // slow, no lock held
    if err != nil {
        return fmt.Errorf("reload: %w", err)              // old model still serving
    }

    e.mu.Lock()                  // exclusive, but only for the swap
    old := e.session
    e.session = newSession
    e.version = newVersion
    e.mu.Unlock()

    old.Destroy()                // free the old one after readers are gone
    return nil
}
```

Why this is correct:

1. The expensive load happens with no lock held, so predictions keep flowing.
2. The write lock is held only for three pointer assignments, microseconds.
3. If the new model fails to load, we return early and the old model keeps serving. A bad deploy cannot take the service down.
4. The swap is atomic from a reader's perspective: a prediction either uses the whole old session or the whole new one, never a mix.

This build then swap pattern is the same idea behind copy on write and atomic pointer publication. It is worth recognizing by name.

## One time init with sync.Once

ONNX Runtime's `InitializeEnvironment` is process global and must run exactly once even if several engines are created (for example in tests). `sync.Once` guarantees a function body runs a single time no matter how many goroutines call it:

```go
var initOnce sync.Once
initOnce.Do(func() {
    ort.SetSharedLibraryPath(libPath)
    ort.InitializeEnvironment()
})
```

Every caller after the first is a no op and blocks until the first finishes, so initialization is both single and safe.

## Proving it is race free

Concurrency bugs are intermittent, so we do not rely on reading the code. The test runs 50 goroutines calling `Predict` at once and the suite runs under the race detector:

```
go test -race ./internal/model/...
```

The Go race detector instruments memory access and fails the test if two goroutines touch the same memory without synchronization and at least one writes. Our suite passes clean under `-race`, which is real evidence the RWMutex usage is correct.

## Flashcards

- Q: Why RWMutex instead of Mutex for the model engine? A: Access is many readers (predictions), rare writer (reload). RWMutex lets all reads run in parallel and only blocks them during a reload; a plain Mutex would serialize every prediction.
- Q: What does defer mu.Unlock() guarantee? A: The unlock runs when the function returns, including early returns and panics, so a held lock is never leaked.
- Q: Describe zero downtime model hot reload. A: Load the new session with no lock held, then take the write lock only to swap pointers and version, then destroy the old session. Load failures leave the old model serving.
- Q: Why not lock around the whole reload? A: It would block all predictions for the duration of a slow load, and a failed load would leave you with a destroyed model.
- Q: What is sync.Once for here? A: Running ONNX Runtime's process global InitializeEnvironment exactly once, safely, across goroutines and multiple engine constructions.
- Q: How did you verify thread safety? A: 50 concurrent Predict goroutines plus the whole suite run under go test -race, which flags unsynchronized shared access; it passed clean.
- Q: What is a data race? A: Two goroutines access the same memory concurrently with at least one writing and no synchronization between them; results are undefined.
