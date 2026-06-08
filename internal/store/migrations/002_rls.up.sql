-- 002_rls.up.sql
-- Layer 3 of FinSight's three-layer tenant isolation: Postgres Row-Level
-- Security. The application sets the active tenant per transaction with
--   SET LOCAL app.tenant_id = '<uuid>'
-- and every policy below filters rows on that GUC.
--
-- NULL safety: current_setting('app.tenant_id', true) returns NULL when the
-- GUC has never been set in this session or transaction. In SQL,
-- NULL = anything evaluates to NULL, not TRUE, so RLS treats the row as
-- not visible. The effect is that a connection that forgets to set
-- app.tenant_id sees zero rows and can insert zero rows, which is the
-- correct fail-closed behavior.
--
-- FORCE ROW LEVEL SECURITY is enabled so that even the table owner is
-- subject to the policies. Integration tests run as the table owner and
-- then SET ROLE finsight_app inside a transaction; FORCE keeps the floor
-- in place regardless of which role is active.

-- Create the application role if it does not already exist.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'finsight_app') THEN
        CREATE ROLE finsight_app NOLOGIN;
    END IF;
END
$$;

GRANT USAGE ON SCHEMA public TO finsight_app;

GRANT SELECT, INSERT, UPDATE, DELETE ON tenants          TO finsight_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON api_keys         TO finsight_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON portfolios       TO finsight_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON portfolio_assets TO finsight_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON predictions      TO finsight_app;

-- Enable and force RLS on every tenant-scoped table.
ALTER TABLE tenants          ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_keys         ENABLE ROW LEVEL SECURITY;
ALTER TABLE portfolios       ENABLE ROW LEVEL SECURITY;
ALTER TABLE portfolio_assets ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions      ENABLE ROW LEVEL SECURITY;

ALTER TABLE tenants          FORCE ROW LEVEL SECURITY;
ALTER TABLE api_keys         FORCE ROW LEVEL SECURITY;
ALTER TABLE portfolios       FORCE ROW LEVEL SECURITY;
ALTER TABLE portfolio_assets FORCE ROW LEVEL SECURITY;
ALTER TABLE predictions      FORCE ROW LEVEL SECURITY;

-- Policies. One per table. Each policy gates both reads (USING) and writes
-- (WITH CHECK) on the same predicate so a tenant cannot read across the
-- boundary and cannot write a row tagged with another tenant's id.

CREATE POLICY tenants_tenant_isolation ON tenants
    USING      (id::text = current_setting('app.tenant_id', true))
    WITH CHECK (id::text = current_setting('app.tenant_id', true));

CREATE POLICY api_keys_tenant_isolation ON api_keys
    USING      (tenant_id::text = current_setting('app.tenant_id', true))
    WITH CHECK (tenant_id::text = current_setting('app.tenant_id', true));

CREATE POLICY portfolios_tenant_isolation ON portfolios
    USING      (tenant_id::text = current_setting('app.tenant_id', true))
    WITH CHECK (tenant_id::text = current_setting('app.tenant_id', true));

-- portfolio_assets does not carry tenant_id directly. It reaches the
-- tenant through its portfolio, so the policy joins back to portfolios.
CREATE POLICY portfolio_assets_tenant_isolation ON portfolio_assets
    USING (
        EXISTS (
            SELECT 1 FROM portfolios p
            WHERE p.id = portfolio_assets.portfolio_id
              AND p.tenant_id::text = current_setting('app.tenant_id', true)
        )
    )
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM portfolios p
            WHERE p.id = portfolio_assets.portfolio_id
              AND p.tenant_id::text = current_setting('app.tenant_id', true)
        )
    );

CREATE POLICY predictions_tenant_isolation ON predictions
    USING      (tenant_id::text = current_setting('app.tenant_id', true))
    WITH CHECK (tenant_id::text = current_setting('app.tenant_id', true));
