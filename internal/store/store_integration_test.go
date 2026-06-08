//go:build integration

// Integration tests for the store package. They spin up a real
// Postgres via testcontainers-go, apply the migrations under
// migrations/, and exercise the tenant-scoped repository methods.
// Run with:
//
//	go test -race -tags=integration ./internal/store/...
//
// Requires a working Docker daemon. CI runs this in the
// `integration` job; local devs can run it via `make test-integration`.
package store_test

import (
	"context"
	"embed"
	"errors"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	tc "github.com/testcontainers/testcontainers-go"
	tcpostgres "github.com/testcontainers/testcontainers-go/modules/postgres"
	"github.com/testcontainers/testcontainers-go/wait"

	"github.com/SatyaChamana/FinSight/internal/store"
	"github.com/SatyaChamana/FinSight/internal/tenant"
)

//go:embed migrations/*.up.sql
var migrationsFS embed.FS

const (
	pgImage    = "postgres:15-alpine"
	pgUser     = "finsight"
	pgPassword = "finsight_test"
	pgDatabase = "finsight"
)

// runMigrations applies every *.up.sql under embedded migrations in
// lexicographic order against the connected pool.
func runMigrations(ctx context.Context, t *testing.T, pool *pgxpool.Pool) {
	t.Helper()

	entries, err := migrationsFS.ReadDir("migrations")
	if err != nil {
		t.Fatalf("read embedded migrations: %v", err)
	}
	var ups []string
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".up.sql") {
			ups = append(ups, e.Name())
		}
	}
	sort.Strings(ups)
	if len(ups) == 0 {
		t.Fatal("no *.up.sql files embedded")
	}

	for _, name := range ups {
		t.Logf("applying %s", name)
		sql, err := migrationsFS.ReadFile("migrations/" + name)
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		if _, err := pool.Exec(ctx, string(sql)); err != nil {
			t.Fatalf("apply %s: %v", name, err)
		}
	}
}

// bootstrapPostgres starts a Postgres container, opens a pool, and
// applies the migrations. Returns the pool and a cleanup func.
func bootstrapPostgres(t *testing.T) (*pgxpool.Pool, func()) {
	t.Helper()
	ctx := context.Background()

	pg, err := tcpostgres.Run(ctx, pgImage,
		tcpostgres.WithDatabase(pgDatabase),
		tcpostgres.WithUsername(pgUser),
		tcpostgres.WithPassword(pgPassword),
		tc.WithWaitStrategy(
			wait.ForLog("database system is ready to accept connections").
				WithOccurrence(2).
				WithStartupTimeout(60*time.Second),
		),
	)
	if err != nil {
		t.Fatalf("start postgres container: %v", err)
	}

	dsn, err := pg.ConnectionString(ctx, "sslmode=disable")
	if err != nil {
		t.Fatalf("connection string: %v", err)
	}

	openCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	pool, err := store.NewPool(openCtx, dsn)
	if err != nil {
		t.Fatalf("open pool: %v", err)
	}

	runMigrations(ctx, t, pool)

	cleanup := func() {
		pool.Close()
		// Terminate is best-effort.
		termCtx, termCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer termCancel()
		_ = pg.Terminate(termCtx)
	}
	return pool, cleanup
}

func TestRLS_TenantIsolation(t *testing.T) {
	pool, cleanup := bootstrapPostgres(t)
	defer cleanup()

	s := store.New(pool)
	ctx := context.Background()

	// 1. Create two tenants via the admin path.
	tenantA, err := s.CreateTenant(ctx, "Tenant Alpha")
	if err != nil {
		t.Fatalf("create tenant A: %v", err)
	}
	tenantB, err := s.CreateTenant(ctx, "Tenant Beta")
	if err != nil {
		t.Fatalf("create tenant B: %v", err)
	}
	if tenantA == tenantB {
		t.Fatal("tenant ids collided")
	}

	// 2. Each tenant creates a portfolio.
	ctxA := tenant.WithTenant(ctx, tenantA)
	ctxB := tenant.WithTenant(ctx, tenantB)

	pidA, err := s.CreatePortfolio(ctxA, "alpha-core", []store.Asset{
		{Symbol: "AAPL", Weight: 0.5},
		{Symbol: "MSFT", Weight: 0.5},
	})
	if err != nil {
		t.Fatalf("A creates portfolio: %v", err)
	}
	pidB, err := s.CreatePortfolio(ctxB, "beta-core", []store.Asset{
		{Symbol: "GOOG", Weight: 0.6},
		{Symbol: "META", Weight: 0.4},
	})
	if err != nil {
		t.Fatalf("B creates portfolio: %v", err)
	}

	// 3. Same-tenant read works.
	pA, err := s.GetPortfolioWithAssets(ctxA, pidA)
	if err != nil {
		t.Fatalf("A reads own portfolio: %v", err)
	}
	if pA.Name != "alpha-core" {
		t.Errorf("A own name: got %q, want alpha-core", pA.Name)
	}
	if len(pA.Assets) != 2 {
		t.Errorf("A own assets: got %d, want 2", len(pA.Assets))
	}

	// 4. Cross-tenant read is filtered by RLS. As A reading B's id,
	//    the policy hides the row and the store reports ErrNotFound.
	_, err = s.GetPortfolioWithAssets(ctxA, pidB)
	if !errors.Is(err, store.ErrNotFound) {
		t.Errorf("A reading B's portfolio: got err=%v, want ErrNotFound", err)
	}

	// 5. And the reverse: B cannot see A's.
	_, err = s.GetPortfolioWithAssets(ctxB, pidA)
	if !errors.Is(err, store.ErrNotFound) {
		t.Errorf("B reading A's portfolio: got err=%v, want ErrNotFound", err)
	}
}

func TestRLS_MissingTenantInContext(t *testing.T) {
	pool, cleanup := bootstrapPostgres(t)
	defer cleanup()

	s := store.New(pool)

	// Call GetPortfolioWithAssets with a bare context. Should fail
	// at the tenant.FromContext stage, well before any DB roundtrip.
	_, err := s.GetPortfolioWithAssets(context.Background(), "00000000-0000-0000-0000-000000000000")
	if !errors.Is(err, tenant.ErrMissing) {
		t.Errorf("got err=%v, want tenant.ErrMissing", err)
	}
}

func TestRLS_GUCFailsClosedWithoutSet(t *testing.T) {
	pool, cleanup := bootstrapPostgres(t)
	defer cleanup()

	ctx := context.Background()

	// Run a raw query under the finsight_app role without setting
	// app.tenant_id. With current_setting('app.tenant_id', true)
	// returning NULL, the policies must hide all rows.
	tx, err := pool.Begin(ctx)
	if err != nil {
		t.Fatalf("begin: %v", err)
	}
	defer tx.Rollback(ctx)

	if _, err := tx.Exec(ctx, "SET LOCAL ROLE finsight_app"); err != nil {
		t.Fatalf("set role: %v", err)
	}

	var count int
	if err := tx.QueryRow(ctx, "SELECT COUNT(*) FROM tenants").Scan(&count); err != nil {
		t.Fatalf("count tenants: %v", err)
	}
	if count != 0 {
		t.Errorf("with no GUC, expected 0 visible tenants, got %d", count)
	}

	// Likewise for portfolios and predictions.
	if err := tx.QueryRow(ctx, "SELECT COUNT(*) FROM portfolios").Scan(&count); err != nil {
		t.Fatalf("count portfolios: %v", err)
	}
	if count != 0 {
		t.Errorf("with no GUC, expected 0 visible portfolios, got %d", count)
	}
}

// Sanity: keep import alive without an actual use.
var _ pgx.Tx = (pgx.Tx)(nil)
