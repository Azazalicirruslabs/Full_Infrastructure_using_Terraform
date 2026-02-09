# DATA_SOURCES.TF - Look up existing resources from task 1 and task 2


# Get current AWS Account ID and Region

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# Look up VPC from Phase 1 (by tag)

data "aws_vpc" "main" {
  filter {
    name   = "tag:Name"
    values = ["${var.project_name}-${var.environment}-vpc"]
  }
}

# Look up Public Subnets from task 1

data "aws_subnets" "public" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.main.id]
  }
  filter {
    name   = "tag:Type"
    values = ["public"]
  }
}

# Look up Private Subnets from task 1

data "aws_subnets" "private" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.main.id]
  }
  filter {
    name   = "tag:Type"
    values = ["private"]
  }
}


# Look up ECR Repositories from task-2 for each service

data "aws_ecr_repository" "services" {
  for_each = var.services                      # Loop through all 8 services
  name     = "${var.project_name}-${each.key}" # raia-api, raia-gateway, etc.
}

# for_each-> will give service keys like "api", "gateway", etc.
# each.key-> refers to the current service key in the loop, like "api", "gateway", etc.
# So, the repository names will be like "raia-api", "raia-gateway", etc.
## To get API service ECR URL:
# data.aws_ecr_repository.services["api"].repository_url
