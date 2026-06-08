-- 001_init.down.sql
-- Reverses 001_init.up.sql. Drops triggers, the shared trigger function,
-- the tables (children before parents), and finally the pgcrypto
-- extension. Every statement uses IF EXISTS so the script is idempotent.

DROP TRIGGER IF EXISTS portfolios_set_updated_at ON portfolios;
DROP TRIGGER IF EXISTS tenants_set_updated_at ON tenants;

DROP FUNCTION IF EXISTS set_updated_at();

DROP INDEX IF EXISTS predictions_tenant_portfolio_time_idx;
DROP INDEX IF EXISTS portfolios_tenant_id_idx;
DROP INDEX IF EXISTS api_keys_tenant_id_idx;

DROP TABLE IF EXISTS predictions;
DROP TABLE IF EXISTS portfolio_assets;
DROP TABLE IF EXISTS portfolios;
DROP TABLE IF EXISTS api_keys;
DROP TABLE IF EXISTS tenants;

DROP EXTENSION IF EXISTS pgcrypto;
