package store

import (
	"context"
	"fmt"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
)

// CreateTenant inserts a new tenant and returns its UUID. This is a
// bootstrap operation; it does not require a tenant id in ctx.
//
// The generated UUID is bound to app.tenant_id inside the tx so the
// WITH CHECK clause of tenants_tenant_isolation passes.
func (s *Store) CreateTenant(ctx context.Context, name string) (string, error) {
	if name == "" {
		return "", fmt.Errorf("create tenant: name is required")
	}
	tenantID := uuid.NewString()

	err := s.withAdminTx(ctx, tenantID, func(tx pgx.Tx) error {
		_, err := tx.Exec(ctx,
			`INSERT INTO tenants (id, name) VALUES ($1, $2)`,
			tenantID, name,
		)
		return err
	})
	if err != nil {
		return "", fmt.Errorf("insert tenant: %w", err)
	}
	return tenantID, nil
}
