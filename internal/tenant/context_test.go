package tenant

import (
	"context"
	"errors"
	"testing"
)

func TestWithTenantFromContextRoundTrip(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		id   string
	}{
		{name: "normal id", id: "tenant-abc"},
		{name: "uuid-like id", id: "11111111-2222-3333-4444-555555555555"},
		{name: "empty id is still stored", id: ""},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			ctx := WithTenant(context.Background(), tc.id)
			got, err := FromContext(ctx)
			if err != nil {
				t.Fatalf("FromContext returned unexpected error: %v", err)
			}
			if got != tc.id {
				t.Fatalf("FromContext returned %q, want %q", got, tc.id)
			}
		})
	}
}

func TestFromContextBareContextReturnsErrMissing(t *testing.T) {
	t.Parallel()

	got, err := FromContext(context.Background())
	if got != "" {
		t.Fatalf("FromContext returned id %q, want empty string", got)
	}
	if !errors.Is(err, ErrMissing) {
		t.Fatalf("FromContext returned err %v, want ErrMissing", err)
	}
}
