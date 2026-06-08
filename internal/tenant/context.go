// Package tenant carries the per-request tenant identity through
// context.Context and exposes gRPC interceptors that populate it from
// incoming metadata. The same helpers are used by the dev interceptor in
// the current phase and by the real auth interceptor in Phase 6.
package tenant

import (
	"context"
	"errors"
)

// ErrMissing is returned by FromContext when no tenant id is bound to ctx.
var ErrMissing = errors.New("tenant: no tenant id in context")

// MetadataKey is the gRPC metadata header that carries the tenant id.
// gRPC normalises metadata keys to lowercase on the wire, so this value
// must stay lowercase.
const MetadataKey = "x-tenant-id"

// ctxKey is an unexported type used as the context.Value key for the
// tenant id. Using a private struct type avoids collisions with any
// other package that might store a value in the context.
type ctxKey struct{}

// WithTenant returns a copy of ctx that carries the given tenant id.
// WithTenant does not validate the id; callers (typically an interceptor)
// are responsible for rejecting empty or malformed values before binding.
func WithTenant(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, ctxKey{}, id)
}

// FromContext extracts the tenant id from ctx. It returns ErrMissing if
// none was set.
func FromContext(ctx context.Context) (string, error) {
	v := ctx.Value(ctxKey{})
	id, ok := v.(string)
	if !ok {
		return "", ErrMissing
	}
	return id, nil
}
