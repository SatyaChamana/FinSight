# Kubernetes Deployment -- deploy/

## Directory Structure

- `base/` -- Base Kustomize manifests (Deployment, Service, ConfigMap, etc.)
- `overlays/dev/` -- Dev environment patches (lower resources, debug logging)
- `overlays/prod/` -- Prod environment patches (higher resources, stricter security)

## Conventions

- Use Kustomize overlays, not Helm.
- Base manifests define the common structure; overlays patch environment-specific values.
- Resource requests and limits must be set on all containers.
- All secrets come from Kubernetes Secrets or external secret managers, never hardcoded.
- Health checks (liveness + readiness) defined in base Deployment.
- Image tags use SHA digests in prod, `:dev` tag in dev.
