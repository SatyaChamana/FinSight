#!/usr/bin/env bash
# Regenerate Go gRPC stubs from proto/ definitions.
# Source of truth: proto/finsight/v1/*.proto.
# Output: gen/go/finsight/v1/*.pb.go (gitignored, regenerated on demand).
#
# Requires:
#   protoc                  (brew install protobuf)
#   protoc-gen-go           (go install google.golang.org/protobuf/cmd/protoc-gen-go@latest)
#   protoc-gen-go-grpc      (go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest)
#
# Make sure $(go env GOPATH)/bin is on PATH so protoc can find the plugins.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTO_DIR="${REPO_ROOT}/proto"
OUT_DIR="${REPO_ROOT}/gen/go"

export PATH="$(go env GOPATH)/bin:${PATH}"

for bin in protoc protoc-gen-go protoc-gen-go-grpc; do
  if ! command -v "${bin}" >/dev/null 2>&1; then
    echo "error: ${bin} not on PATH" >&2
    exit 1
  fi
done

rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"

PROTO_FILES=()
while IFS= read -r f; do
  PROTO_FILES+=("${f}")
done < <(find "${PROTO_DIR}" -name "*.proto" -print)

if [[ ${#PROTO_FILES[@]} -eq 0 ]]; then
  echo "error: no .proto files found under ${PROTO_DIR}" >&2
  exit 1
fi

protoc \
  -I "${PROTO_DIR}" \
  --go_out="${OUT_DIR}" \
  --go_opt=paths=source_relative \
  --go-grpc_out="${OUT_DIR}" \
  --go-grpc_opt=paths=source_relative \
  "${PROTO_FILES[@]}"

echo "generated:"
find "${OUT_DIR}" -name "*.go" -print
