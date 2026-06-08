-- 001_init.up.sql
-- Initial FinSight schema. Creates the core multi-tenant tables (tenants,
-- api_keys, portfolios, portfolio_assets, predictions) plus an updated_at
-- trigger. Row-Level Security policies are added in a separate migration.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- tenants: the root of every tenant-scoped row in the system. Every other
-- domain row either has a tenant_id column or chains to one through a
-- foreign key. Deleting a tenant cascades to all of their data.
CREATE TABLE tenants (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name       TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- api_keys: one tenant can have many API keys (rotation, per-environment
-- keys, per-service keys). We store only the sha256 hex of the raw key so a
-- database leak does not yield usable credentials. revoked_at lets us keep
-- the audit trail of an old key while disabling it.
CREATE TABLE api_keys (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id  UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    key_hash   TEXT NOT NULL UNIQUE,
    label      TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    revoked_at TIMESTAMPTZ
);

CREATE INDEX api_keys_tenant_id_idx ON api_keys (tenant_id);

-- portfolios: a named collection of assets owned by a tenant. Portfolio
-- names are unique per tenant (not globally) so two tenants can both have a
-- portfolio called "core" without colliding.
CREATE TABLE portfolios (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id  UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    name       TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, name)
);

CREATE INDEX portfolios_tenant_id_idx ON portfolios (tenant_id);

-- portfolio_assets: the rows that actually describe what a portfolio holds.
-- Weights are normalized to [0, 1]; the composite primary key prevents the
-- same symbol from appearing twice in one portfolio. Asset rows reach the
-- tenant only through their portfolio, which the RLS policy traverses.
CREATE TABLE portfolio_assets (
    portfolio_id UUID NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    symbol       TEXT NOT NULL,
    weight       DOUBLE PRECISION NOT NULL CHECK (weight >= 0 AND weight <= 1),
    PRIMARY KEY (portfolio_id, symbol)
);

-- predictions: append-only ledger of model outputs for a portfolio. We keep
-- every prediction (no upsert) so we can audit drift, compare model
-- versions, and serve the most recent value cheaply via the composite
-- index. tenant_id is denormalized onto this table so RLS and most read
-- paths do not need a join.
CREATE TABLE predictions (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id     UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    portfolio_id  UUID NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    model_version TEXT NOT NULL,
    var_95        DOUBLE PRECISION NOT NULL,
    cvar_95       DOUBLE PRECISION NOT NULL,
    volatility    DOUBLE PRECISION NOT NULL,
    predicted_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX predictions_tenant_portfolio_time_idx
    ON predictions (tenant_id, portfolio_id, predicted_at DESC);

-- set_updated_at: trigger function used on every table that carries an
-- updated_at column. Pushes the timestamp bump into the database so
-- application code cannot forget to set it.
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER tenants_set_updated_at
    BEFORE UPDATE ON tenants
    FOR EACH ROW
    EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER portfolios_set_updated_at
    BEFORE UPDATE ON portfolios
    FOR EACH ROW
    EXECUTE FUNCTION set_updated_at();
