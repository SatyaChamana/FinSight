# Phase 5A: Running an ONNX Model from Go (CGo / FFI)

How the Go service loads the model Python trained and runs inference on it. This is the single hardest integration in the project because it crosses a language boundary.

## The problem

Python trained an LSTM + attention model and exported it to `model.onnx`. Go has to load that file and run predictions. Go cannot run PyTorch. So we use ONNX Runtime, a C++ library that executes any ONNX graph, and call into it from Go.

```
PyTorch (train)  ->  model.onnx  ->  ONNX Runtime (C++)  <-  Go (cgo bindings)
```

## What is FFI and cgo

FFI (Foreign Function Interface) means calling functions written in another language. ONNX Runtime is C++. Go calls it through cgo, the mechanism that lets Go call C functions. The library `github.com/yalue/onnxruntime_go` wraps the C API so we write Go, not C.

Key consequence: cgo code links against a native shared library at runtime (a `.dylib` on macOS, `.so` on Linux). That file is not bundled in the Go binary. The program must find it on the machine, or inference fails at startup. This is the number one operational gotcha of the whole phase.

### How we resolve the shared library

In `internal/model/model.go` we set the path before using the runtime, with a fallback chain:

1. `ONNXRUNTIME_LIB_PATH` env var, if set.
2. Otherwise the dylib bundled inside the Python venv (`ml/.venv/.../libonnxruntime.*.dylib`).
3. If neither exists, `NewEngine` returns a clear error instead of crashing.

```go
ort.SetSharedLibraryPath(libPath)
ort.InitializeEnvironment()
// ... later, at process shutdown:
ort.DestroyEnvironment()
```

`InitializeEnvironment` and `DestroyEnvironment` are process global. They must run exactly once. We guard initialization with `sync.Once` (see 05B) so multiple engines or test runs do not double initialize.

## The model contract (read it from the file, do not hardcode)

The exported model has a fixed input and output shape, recorded in `model.manifest.json`:

| Tensor | Name | Shape | Meaning |
|--------|------|-------|---------|
| input  | `input`  | `[batch, 63, 49]` | 63 trading days x 49 features |
| output | `output` | `[batch, 3]` | `[volatility, var_95, cvar_95]` |

The output column order is not guessable. It is defined in `ml/models/architecture.py` (`torch.stack([vol, var, cvar])`). Getting it wrong silently returns VaR where you expect volatility. The Go code maps it explicitly and a comment cites the Python source:

```go
// output columns are [volatility, var_95, cvar_95] (ml/models/architecture.py)
pred.Volatility = out[0]
pred.VaR95      = out[1]
pred.CVaR95     = out[2]
```

There is no scaler (`scaler_path: null`), so features go in raw, no standardization. This is the one thing that made Go side preprocessing tractable (see 05C).

## Running one inference

We use `DynamicAdvancedSession`, which lets us pass input tensors per call (good for batch size 1, one portfolio at a time). The flow inside `Predict`:

1. Validate the feature matrix is exactly 63 x 49. Wrong dimensions return an error, never a panic.
2. Flatten the `[][]float32` into the contiguous `[1*63*49]float32` backing array ONNX Runtime expects.
3. Build an input tensor of shape `[1, 63, 49]`.
4. Run the session.
5. Read the `[1, 3]` output tensor, map the three columns by position.

A Python idiom to unlearn: in numpy you pass an `ndarray` and shape is metadata. In Go with this binding you manage a flat slice plus an explicit shape. The shape and the length of the slice must agree or the call errors.

## Proving the Go output equals Python (golden test)

A reimplementation across a language boundary is only trustworthy if you check it against the original. `model_test.go` runs a golden test:

1. Feed a fixed deterministic input (every value 0.01) to the Go engine.
2. Shell out to `ml/.venv/bin/python` and run the same input through Python `onnxruntime`.
3. Assert the two outputs match within tolerance.

Result: they matched to full float32 precision, byte identical (`[0.10403532, 0.02424913, 0.00348946]`). That is the proof the FFI path, tensor layout, and output ordering are all correct.

The test is defensive: it calls `t.Skip` if the dylib, the model file, or the Python venv is missing, so CI without the native library still passes. The non FFI tests (preprocessing math) always run.

## Why this matters in an interview

You can say: "I served a PyTorch model from a Go service without a Python sidecar. I used ONNX as the portable format and ONNX Runtime via cgo for execution. The risk in any cross language port is silent numerical drift, so I wrote a golden test that runs identical input through both the Go path and Python onnxruntime and asserts equality. They matched bit for bit." That is a concrete, defensible engineering story.

## Flashcards

- Q: Why ONNX instead of calling PyTorch from Go? A: Go cannot run PyTorch. ONNX is a portable model format; ONNX Runtime (C++) executes it and has Go cgo bindings, so no Python sidecar is needed in production.
- Q: What is cgo and what operational cost does it add? A: cgo lets Go call C/C++. The cost is a runtime dependency on a native shared library (.dylib/.so) that is not embedded in the Go binary and must be found on the host.
- Q: How do you locate the ONNX Runtime library? A: `ort.SetSharedLibraryPath` with a fallback chain (env var, then bundled venv dylib), and a clear error if none exists.
- Q: Why must InitializeEnvironment run once? A: It is process global state in the C++ runtime; double init or use after destroy is undefined. We guard it with sync.Once.
- Q: How is tensor data passed across the boundary? A: As a flat contiguous []float32 plus an explicit shape [1,63,49], unlike numpy where shape is attached to the array.
- Q: How did you trust the Go reimplementation? A: A golden test comparing Go output against Python onnxruntime on identical input; they matched to full float32 precision.
- Q: Why is the output column order dangerous? A: [volatility, var_95, cvar_95] is fixed by the Python model and not inferable from the tensor; mismapping silently returns the wrong metric.
