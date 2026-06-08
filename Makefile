.PHONY: help db-up db-down db-reset db-migrate db-rollback test test-integration build lint proto ml-setup ml-test ml-lint ml-fmt ml-train ml-export ml-clean

help: ## Show available targets.
	@echo "FinSight Makefile targets:"
	@echo "  help              Show this help."
	@echo "  db-up             Start postgres and redis via docker compose."
	@echo "  db-down           Stop the local stack (preserves volumes)."
	@echo "  db-reset          Drop volumes, restart stack, re-apply migrations."
	@echo "  db-migrate        Apply all up migrations."
	@echo "  db-rollback       Apply all down migrations."
	@echo "  test              Run unit tests with the race detector."
	@echo "  test-integration  Run integration tests (build tag: integration)."
	@echo "  build             Build the finsight server binary into bin/."
	@echo "  lint              Run golangci-lint across the module."
	@echo "  proto             Regenerate protobuf code."

db-up:
	scripts/db-up.sh

db-down:
	scripts/db-down.sh

db-reset:
	scripts/db-down.sh --volumes
	scripts/db-up.sh
	scripts/migrate.sh up

db-migrate:
	scripts/migrate.sh up

db-rollback:
	scripts/migrate.sh down

test:
	go test -race ./...

test-integration:
	go test -race -tags=integration ./...

build:
	go build -o bin/finsight ./cmd/server

lint:
	golangci-lint run ./...

proto:
	./scripts/generate-proto.sh

.PHONY: ml-setup ml-test ml-lint ml-fmt ml-train ml-export ml-clean

ml-setup:        ## Create venv (one-time) and install ml deps
	cd ml && python -m venv .venv && .venv/bin/python -m pip install -e ".[dev]"

ml-test:         ## Run python tests under ml/
	cd ml && .venv/bin/python -m pytest

ml-lint:         ## Run ruff over ml/
	cd ml && .venv/bin/python -m ruff check .

ml-fmt:          ## Run black over ml/
	cd ml && .venv/bin/python -m black .

ml-train:        ## Train (placeholder; integration step writes the real loop)
	cd ml && .venv/bin/finsight-ml train --config configs/dev.yaml

ml-export:       ## Export the latest checkpoint to ONNX
	cd ml && .venv/bin/finsight-ml export --checkpoint models/artifacts/model.pt --out models/artifacts/model.onnx

ml-clean:        ## Wipe build artifacts
	cd ml && rm -rf .venv .pytest_cache __pycache__ models/artifacts/*.onnx
