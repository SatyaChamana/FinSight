// Package store is the Postgres data-access layer for FinSight. It
// uses pgx (no ORM) and assumes the schema produced by the migrations
// under internal/store/migrations.
//
// Every method that is called from a per-request handler reads the
// tenant id from context.Context (via the tenant package) and runs
// its query inside a transaction that:
//
//  1. SET LOCAL ROLE finsight_app    -- limited DML role
//  2. SELECT set_config('app.tenant_id', $1, true)  -- engages RLS
//
// Layers 1 and 2 are application-layer enforcement; the Postgres RLS
// policies installed by 002_rls.up.sql are the third, defense-in-depth
// floor.
package store

import (
	"context"
	"fmt"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Store wraps a pgxpool.Pool and exposes domain repositories.
type Store struct {
	pool *pgxpool.Pool
}

// New constructs a Store around an already-opened pool. The caller
// owns the pool lifecycle.
func New(pool *pgxpool.Pool) *Store {
	return &Store{pool: pool}
}

// NewPool opens a pgx connection pool against dsn and pings it.
// The caller must Close the pool when done.
func NewPool(ctx context.Context, dsn string) (*pgxpool.Pool, error) {
	cfg, err := pgxpool.ParseConfig(dsn)
	if err != nil {
		return nil, fmt.Errorf("parse dsn: %w", err)
	}
	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("open pool: %w", err)
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("ping: %w", err)
	}
	return pool, nil
}

// withTenantTx runs fn inside a transaction that switches to the
// limited finsight_app role and sets the tenant id GUC, so RLS
// policies engage. The transaction is committed if fn returns nil.
//
// SET LOCAL does not accept parameters, so the tenant id is bound
// via set_config() with is_local=true (per-transaction setting).
func (s *Store) withTenantTx(ctx context.Context, tenantID string, fn func(pgx.Tx) error) (err error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin tx: %w", err)
	}
	defer func() {
		if err != nil {
			_ = tx.Rollback(ctx)
		}
	}()

	if _, err = tx.Exec(ctx, "SET LOCAL ROLE finsight_app"); err != nil {
		return fmt.Errorf("set role: %w", err)
	}
	if _, err = tx.Exec(ctx, "SELECT set_config('app.tenant_id', $1, true)", tenantID); err != nil {
		return fmt.Errorf("set tenant: %w", err)
	}
	if err = fn(tx); err != nil {
		return err
	}
	if err = tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit tx: %w", err)
	}
	return nil
}

// withAdminTx is the bootstrap path used by CreateTenant: the tenant
// row does not exist yet, so there is no SET ROLE to a limited role.
// We still set app.tenant_id so the FORCE RLS policy on tenants
// accepts the INSERT (id::text = current_setting).
func (s *Store) withAdminTx(ctx context.Context, tenantID string, fn func(pgx.Tx) error) (err error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin tx: %w", err)
	}
	defer func() {
		if err != nil {
			_ = tx.Rollback(ctx)
		}
	}()

	if _, err = tx.Exec(ctx, "SELECT set_config('app.tenant_id', $1, true)", tenantID); err != nil {
		return fmt.Errorf("set tenant: %w", err)
	}
	if err = fn(tx); err != nil {
		return err
	}
	if err = tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit tx: %w", err)
	}
	return nil
}
