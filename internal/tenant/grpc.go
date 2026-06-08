package tenant

import (
	"context"
	"strings"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

// tenantFromIncoming reads the MetadataKey header from the incoming
// gRPC metadata on ctx, trims surrounding whitespace, and returns the
// resulting tenant id. A missing header, an empty value, or a value
// that is only whitespace is rejected with codes.Unauthenticated.
func tenantFromIncoming(ctx context.Context) (string, error) {
	md, ok := metadata.FromIncomingContext(ctx)
	if !ok {
		return "", status.Errorf(codes.Unauthenticated, "tenant: missing metadata")
	}
	values := md.Get(MetadataKey)
	if len(values) == 0 {
		return "", status.Errorf(codes.Unauthenticated, "tenant: missing %s header", MetadataKey)
	}
	id := strings.TrimSpace(values[0])
	if id == "" {
		return "", status.Errorf(codes.Unauthenticated, "tenant: empty %s header", MetadataKey)
	}
	return id, nil
}

// UnaryServerInterceptor returns a gRPC unary interceptor that reads
// the tenant id from incoming metadata, attaches it to the context via
// WithTenant, and forwards to the handler. If the metadata is missing
// or empty it returns codes.Unauthenticated.
func UnaryServerInterceptor() grpc.UnaryServerInterceptor {
	return func(
		ctx context.Context,
		req interface{},
		_ *grpc.UnaryServerInfo,
		handler grpc.UnaryHandler,
	) (interface{}, error) {
		id, err := tenantFromIncoming(ctx)
		if err != nil {
			return nil, err
		}
		return handler(WithTenant(ctx, id), req)
	}
}

// tenantServerStream wraps grpc.ServerStream so that downstream
// handlers see a context that carries the tenant id.
type tenantServerStream struct {
	grpc.ServerStream
	ctx context.Context
}

// Context returns the wrapped context so the handler observes the
// tenant id bound by the interceptor.
func (s *tenantServerStream) Context() context.Context {
	return s.ctx
}

// StreamServerInterceptor mirrors UnaryServerInterceptor for streaming RPCs.
func StreamServerInterceptor() grpc.StreamServerInterceptor {
	return func(
		srv interface{},
		ss grpc.ServerStream,
		_ *grpc.StreamServerInfo,
		handler grpc.StreamHandler,
	) error {
		id, err := tenantFromIncoming(ss.Context())
		if err != nil {
			return err
		}
		wrapped := &tenantServerStream{
			ServerStream: ss,
			ctx:          WithTenant(ss.Context(), id),
		}
		return handler(srv, wrapped)
	}
}
