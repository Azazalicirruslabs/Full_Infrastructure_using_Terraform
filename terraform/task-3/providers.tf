terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.30"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }
  required_version = ">= 1.14"
}

provider "aws" {
  region = var.aws_region


  default_tags {
    tags = {
      project     = var.project_name
      Environment = var.environment
      Task        = "Task-3 ECS Deployment"
      ManagedBy   = "Terraform"
    }
  }
}
