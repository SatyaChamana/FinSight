# syntax=docker/dockerfile:1.7

FROM golang:1.26-alpine AS builder
WORKDIR /src

# protoc and bash for the generate script. Plugins installed via go install below.
RUN apk add --no-cache protoc bash

COPY go.mod go.sum ./
RUN go mod download

# Install proto plugins. Pinned to the same versions we develop against.
RUN go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.36.11 && \
    go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.6.2

COPY . .

# Generated code is .gitignore'd; regenerate inside the build context.
RUN ./scripts/generate-proto.sh

RUN CGO_ENABLED=0 GOOS=linux go build \
    -trimpath \
    -ldflags="-s -w" \
    -o /out/finsight \
    ./cmd/server

FROM gcr.io/distroless/static-debian12:nonroot
WORKDIR /app
COPY --from=builder /out/finsight /app/finsight

USER nonroot:nonroot
EXPOSE 9090

ENV PORT=9090 \
    ENV=prod \
    LOG_LEVEL=info

ENTRYPOINT ["/app/finsight"]
