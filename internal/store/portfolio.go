package store

import (
	"context"
	"errors"
	"fmt"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"

	"github.com/SatyaChamana/FinSight/internal/tenant"
)

// CreatePortfolio inserts a portfolio together with its assets in a
// single transaction. The tenant id is read from ctx (set by the
// tenant interceptor).
//
// Returns the new portfolio's UUID.
func (s *Store) CreatePortfolio(ctx context.Context, name string, assets []Asset) (string, error) {
	tenantID, err := tenant.FromContext(ctx)
	if err != nil {
		return "", err
	}
	if name == "" {
		return "", fmt.Errorf("create portfolio: name is required")
	}

	portfolioID := uuid.NewString()

	err = s.withTenantTx(ctx, tenantID, func(tx pgx.Tx) error {
		_, err := tx.Exec(ctx,
			`INSERT INTO portfolios (id, tenant_id, name) VALUES ($1, $2, $3)`,
			portfolioID, tenantID, name,
		)
		if err != nil {
			return fmt.Errorf("insert portfolio: %w", err)
		}
		for _, a := range assets {
			_, err := tx.Exec(ctx,
				`INSERT INTO portfolio_assets (portfolio_id, symbol, weight) VALUES ($1, $2, $3)`,
				portfolioID, a.Symbol, a.Weight,
			)
			if err != nil {
				return fmt.Errorf("insert asset %q: %w", a.Symbol, err)
			}
		}
		return nil
	})
	if err != nil {
		return "", err
	}
	return portfolioID, nil
}

// GetPortfolioWithAssets loads a portfolio and its assets. The tenant
// id is read from ctx; RLS filters rows that do not belong to that
// tenant, so cross-tenant reads surface as ErrNotFound.
func (s *Store) GetPortfolioWithAssets(ctx context.Context, portfolioID string) (*Portfolio, error) {
	tenantID, err := tenant.FromContext(ctx)
	if err != nil {
		return nil, err
	}

	var p Portfolio
	err = s.withTenantTx(ctx, tenantID, func(tx pgx.Tx) error {
		row := tx.QueryRow(ctx,
			`SELECT id, tenant_id, name, created_at, updated_at
			   FROM portfolios
			  WHERE id = $1`,
			portfolioID,
		)
		if scanErr := row.Scan(&p.ID, &p.TenantID, &p.Name, &p.CreatedAt, &p.UpdatedAt); scanErr != nil {
			if errors.Is(scanErr, pgx.ErrNoRows) {
				return ErrNotFound
			}
			return fmt.Errorf("scan portfolio: %w", scanErr)
		}

		rows, queryErr := tx.Query(ctx,
			`SELECT symbol, weight
			   FROM portfolio_assets
			  WHERE portfolio_id = $1
			  ORDER BY symbol`,
			portfolioID,
		)
		if queryErr != nil {
			return fmt.Errorf("query assets: %w", queryErr)
		}
		defer rows.Close()

		for rows.Next() {
			var a Asset
			if scanErr := rows.Scan(&a.Symbol, &a.Weight); scanErr != nil {
				return fmt.Errorf("scan asset: %w", scanErr)
			}
			p.Assets = append(p.Assets, a)
		}
		return rows.Err()
	})
	if err != nil {
		return nil, err
	}
	return &p, nil
}
