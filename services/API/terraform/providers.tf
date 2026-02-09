# Terraform and Provider Configuration for API Service

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.30"
    }
  }
  required_version = ">= 1.14"
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "RAIA"
      Service     = "api"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}
