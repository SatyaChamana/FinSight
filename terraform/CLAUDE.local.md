# Terraform IaC -- terraform/

## Directory Structure

- `modules/` -- Reusable Terraform modules (EKS, RDS, ElastiCache, S3, ECR, VPC, IAM)
- `environments/dev/` -- Dev environment root module
- `environments/prod/` -- Prod environment root module

## Conventions

- AWS provider. Region parameterized via variable.
- Remote state in S3 with DynamoDB locking.
- All resources tagged with `project=finsight`, `environment=<env>`, `managed-by=terraform`.
- Modules are generic and reusable; environments compose modules with env-specific values.
- No hardcoded ARNs, account IDs, or region strings.
- Use `terraform plan` before every `terraform apply`.
- Sensitive values (DB passwords, API keys) come from AWS Secrets Manager or SSM Parameter Store.
