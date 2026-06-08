package store

import (
	"errors"
	"time"
)

// ErrNotFound is returned when a row lookup matches zero rows. This
// includes the case where RLS filtered the row out (which from the
// application side is indistinguishable from a missing row, by design).
var ErrNotFound = errors.New("store: not found")

// Portfolio is the domain view of a portfolios row plus its assets.
type Portfolio struct {
	ID        string
	TenantID  string
	Name      string
	Assets    []Asset
	CreatedAt time.Time
	UpdatedAt time.Time
}

// Asset is one position within a portfolio. Weights are in [0, 1] and
// across a portfolio sum to 1.
type Asset struct {
	Symbol string
	Weight float64
}
