# Protobuf and gRPC -- proto/

## Directory Structure

- `finsight/v1/` -- All proto definitions for v1 API

## Conventions

- Package name: `finsight.v1`
- Use `google.protobuf.Timestamp` for time fields, never strings.
- Service names are PascalCase (e.g., `RiskPredictionService`).
- RPC names are PascalCase verbs (e.g., `PredictRisk`, `GetPortfolio`).
- Field names are snake_case.
- Every message field has a comment explaining its purpose.
- Run `./scripts/generate-proto.sh` after any proto change.
- Never edit generated `.pb.go` files directly.

## Code Generation

Generated Go code goes to `gen/go/finsight/v1/` (gitignored until generated).
Proto files are the source of truth. Always regenerate before committing.
