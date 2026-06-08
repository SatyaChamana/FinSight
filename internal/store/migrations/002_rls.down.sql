-- 002_rls.down.sql
-- Reverses 002_rls.up.sql. Drops the per-table tenant isolation policies,
-- disables RLS on each table, revokes grants from finsight_app, and drops
-- the role. Uses IF EXISTS so reruns are safe.

DROP POLICY IF EXISTS predictions_tenant_isolation      ON predictions;
DROP POLICY IF EXISTS portfolio_assets_tenant_isolation ON portfolio_assets;
DROP POLICY IF EXISTS portfolios_tenant_isolation       ON portfolios;
DROP POLICY IF EXISTS api_keys_tenant_isolation         ON api_keys;
DROP POLICY IF EXISTS tenants_tenant_isolation          ON tenants;

ALTER TABLE IF EXISTS predictions      NO FORCE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS portfolio_assets NO FORCE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS portfolios       NO FORCE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS api_keys         NO FORCE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS tenants          NO FORCE ROW LEVEL SECURITY;

ALTER TABLE IF EXISTS predictions      DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS portfolio_assets DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS portfolios       DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS api_keys         DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS tenants          DISABLE ROW LEVEL SECURITY;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'finsight_app') THEN
        REVOKE SELECT, INSERT, UPDATE, DELETE ON predictions      FROM finsight_app;
        REVOKE SELECT, INSERT, UPDATE, DELETE ON portfolio_assets FROM finsight_app;
        REVOKE SELECT, INSERT, UPDATE, DELETE ON portfolios       FROM finsight_app;
        REVOKE SELECT, INSERT, UPDATE, DELETE ON api_keys         FROM finsight_app;
        REVOKE SELECT, INSERT, UPDATE, DELETE ON tenants          FROM finsight_app;
        REVOKE USAGE ON SCHEMA public FROM finsight_app;
    END IF;
END
$$;

DROP ROLE IF EXISTS finsight_app;
