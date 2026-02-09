# RAIA Infrastructure - Terraform

Production-grade AWS infrastructure for the RAIA (Responsible AI Analytics) platform.

## 🏗️ Architecture

```
terraform/
├── modules/                    # Reusable components
│   ├── vpc/                    # VPC, subnets, route tables
│   ├── ecr/                    # Container repositories
│   ├── rds/                    # PostgreSQL database
│   └── s3/                     # Object storage
│
├── layers/                     # Deployment layers (apply in order)
│   ├── 01-foundation/          # VPC, subnets, security groups
│   ├── 02-data/                # RDS, S3
│   ├── 03-ecr/                 # ECR repositories
│   ├── 04-compute/             # ECS Fargate, ALB
│   └── 05-monitoring/          # CloudWatch dashboard, alarms
│
└── envs/
    └── dev.tfvars              # Dev environment variables
```

## 🚀 Quick Start

### Prerequisites
- Terraform >= 1.14
- AWS CLI configured with credentials
- Docker (for building/pushing images)

### Deploy Infrastructure

```bash
# Set database password
export TF_VAR_db_password="YourSecurePassword123!"

# 1. Foundation (VPC, subnets)
cd terraform/layers/01-foundation
terraform init
terraform apply -var-file=../../envs/dev.tfvars

# 2. Data (RDS, S3)
cd ../02-data
terraform init
terraform apply -var-file=../../envs/dev.tfvars \
  -var="vpc_id=$(cd ../01-foundation && terraform output -raw vpc_id)" \
  -var='private_subnet_ids=["subnet-xxx","subnet-yyy"]' \
  -var="ecs_tasks_security_group_id=sg-xxx"

# 3. ECR (Container repositories)
cd ../03-ecr
terraform init
terraform apply -var-file=../../envs/dev.tfvars

# 4. Compute (ECS, ALB)
cd ../04-compute
terraform init
terraform apply -var-file=../../envs/dev.tfvars \
  -var="vpc_id=vpc-xxx" \
  -var='public_subnet_ids=["subnet-xxx"]' \
  # ... (see layer variables)

# 5. Monitoring (CloudWatch)
cd ../05-monitoring
terraform init
terraform apply -var-file=../../envs/dev.tfvars \
  -var="ecs_cluster_name=raia-dev-cluster"
```

## 📦 Microservices

| Service | Path | Port | Description |
|---------|------|------|-------------|
| api | /api/* | 8000 | Auth & user management |
| gateway | /gateway/* | 8000 | API routing, docs |
| classification | /classification/* | 8000 | ML classification |
| regression | /regression/* | 8000 | Statistical regression |
| fairness | /fairness/* | 8000 | Bias detection |
| data-drift | /data_drift/* | 8000 | Data monitoring |
| what-if | /what_if/* | 8000 | Counterfactual analysis |
| mainflow | /mainflow/* | 8000 | Workflow orchestration |

## 🔒 Security

- All traffic through ALB (no direct container access)
- RDS in private subnets (not publicly accessible)
- S3 with encryption and public access blocked
- Security groups with least-privilege rules
