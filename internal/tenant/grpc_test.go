package tenant

import (
	"context"
	"errors"
	"testing"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

// fakeServerStream is a minimal grpc.ServerStream implementation that
// only needs to expose Context() for the stream interceptor tests.
// Every other method is unused; they panic if the test ever reaches
// them so we notice immediately.
type fakeServerStream struct {
	ctx context.Context
}

func (f *fakeServerStream) Context() context.Context     { return f.ctx }
func (f *fakeServerStream) SetHeader(metadata.MD) error  { panic("not implemented") }
func (f *fakeServerStream) SendHeader(metadata.MD) error { panic("not implemented") }
func (f *fakeServerStream) SetTrailer(metadata.MD)       { panic("not implemented") }
func (f *fakeServerStream) SendMsg(interface{}) error    { panic("not implemented") }
func (f *fakeServerStream) RecvMsg(interface{}) error    { panic("not implemented") }

func TestUnaryServerInterceptor(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		md          metadata.MD // nil means do not attach metadata at all
		wantID      string
		wantCode    codes.Code
		wantHandler bool
	}{
		{
			name:        "happy path passes tenant to handler",
			md:          metadata.Pairs(MetadataKey, "tenant-abc"),
			wantID:      "tenant-abc",
			wantCode:    codes.OK,
			wantHandler: true,
		},
		{
			name:        "no metadata is unauthenticated",
			md:          nil,
			wantCode:    codes.Unauthenticated,
			wantHandler: false,
		},
		{
			name:        "empty value is unauthenticated",
			md:          metadata.Pairs(MetadataKey, ""),
			wantCode:    codes.Unauthenticated,
			wantHandler: false,
		},
		{
			name:        "whitespace only value is unauthenticated",
			md:          metadata.Pairs(MetadataKey, "   \t  "),
			wantCode:    codes.Unauthenticated,
			wantHandler: false,
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			ctx := context.Background()
			if tc.md != nil {
				ctx = metadata.NewIncomingContext(ctx, tc.md)
			}

			var (
				handlerCalled bool
				observedID    string
				observedErr   error
			)
			handler := func(ctx context.Context, _ interface{}) (interface{}, error) {
				handlerCalled = true
				observedID, observedErr = FromContext(ctx)
				return "ok", nil
			}

			interceptor := UnaryServerInterceptor()
			resp, err := interceptor(ctx, nil, &grpc.UnaryServerInfo{}, handler)

			if tc.wantCode == codes.OK {
				if err != nil {
					t.Fatalf("interceptor returned unexpected error: %v", err)
				}
				if resp != "ok" {
					t.Fatalf("interceptor returned resp %v, want %q", resp, "ok")
				}
			} else {
				if err == nil {
					t.Fatalf("interceptor returned nil error, want code %s", tc.wantCode)
				}
				if got := status.Code(err); got != tc.wantCode {
					t.Fatalf("interceptor returned code %s, want %s", got, tc.wantCode)
				}
			}

			if handlerCalled != tc.wantHandler {
				t.Fatalf("handler called = %v, want %v", handlerCalled, tc.wantHandler)
			}
			if tc.wantHandler {
				if observedErr != nil {
					t.Fatalf("handler observed FromContext error %v, want nil", observedErr)
				}
				if observedID != tc.wantID {
					t.Fatalf("handler observed tenant id %q, want %q", observedID, tc.wantID)
				}
			}
		})
	}
}

func TestStreamServerInterceptor(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		md          metadata.MD
		wantID      string
		wantCode    codes.Code
		wantHandler bool
	}{
		{
			name:        "happy path passes tenant to streaming handler",
			md:          metadata.Pairs(MetadataKey, "tenant-stream"),
			wantID:      "tenant-stream",
			wantCode:    codes.OK,
			wantHandler: true,
		},
		{
			name:        "no metadata is unauthenticated",
			md:          nil,
			wantCode:    codes.Unauthenticated,
			wantHandler: false,
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			ctx := context.Background()
			if tc.md != nil {
				ctx = metadata.NewIncomingContext(ctx, tc.md)
			}
			ss := &fakeServerStream{ctx: ctx}

			var (
				handlerCalled bool
				observedID    string
				observedErr   error
			)
			handler := func(_ interface{}, stream grpc.ServerStream) error {
				handlerCalled = true
				observedID, observedErr = FromContext(stream.Context())
				return nil
			}

			interceptor := StreamServerInterceptor()
			err := interceptor(nil, ss, &grpc.StreamServerInfo{}, handler)

			if tc.wantCode == codes.OK {
				if err != nil {
					t.Fatalf("interceptor returned unexpected error: %v", err)
				}
			} else {
				if err == nil {
					t.Fatalf("interceptor returned nil error, want code %s", tc.wantCode)
				}
				if got := status.Code(err); got != tc.wantCode {
					t.Fatalf("interceptor returned code %s, want %s", got, tc.wantCode)
				}
			}

			if handlerCalled != tc.wantHandler {
				t.Fatalf("handler called = %v, want %v", handlerCalled, tc.wantHandler)
			}
			if tc.wantHandler {
				if observedErr != nil {
					t.Fatalf("handler observed FromContext error %v, want nil", observedErr)
				}
				if observedID != tc.wantID {
					t.Fatalf("handler observed tenant id %q, want %q", observedID, tc.wantID)
				}
			}
		})
	}
}

// Sanity check that ErrMissing remains a sentinel that callers can
// match with errors.Is; this guards against accidental rewrapping in
// FromContext.
func TestErrMissingIsSentinel(t *testing.T) {
	t.Parallel()

	_, err := FromContext(context.Background())
	if !errors.Is(err, ErrMissing) {
		t.Fatalf("FromContext error %v is not ErrMissing", err)
	}
}
