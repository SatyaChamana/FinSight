# Migrations

Plain SQL migrations for the FinSight Postgres database.

## Naming

Each migration is a pair of files:

- `NNN_name.up.sql` applies the change.
- `NNN_name.down.sql` reverses it.

`NNN` is a zero-padded sequence number (`001`, `002`, ...). Files are applied
in lexicographic order, so the prefix is what determines run order.

## Running

Use `scripts/migrate.sh` to apply or roll back migrations against the target
database. The script is intentionally thin: it pipes each `*.up.sql` (or
`*.down.sql`) into `psql` in order.

## Tooling

We are deliberately not adopting `golang-migrate` or a similar tool yet.
Plain SQL plus a shell wrapper is enough for the current phase. A real
migration tool with versioning, dirty-state tracking, and CI integration is
deferred to a later phase once the schema stabilizes.
