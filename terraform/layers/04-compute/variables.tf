# Layer 04: Compute Variables

variable "aws_region" {
  description = "AWS region"
  type        = string
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
}

# NOTE: VPC, Subnet, Security Group, S3, RDS, and ECR values are now 
# automatically read from previous layer states via terraform_remote_state


# Service configurations
variable "services" {
  description = "Configuration for each microservice"
  type = map(object({
    cpu            = number
    memory         = number
    container_port = number
    desired_count  = number
    health_path    = string
  }))
  default = {
    "frontend" = {
      cpu            = 256
      memory         = 512
      container_port = 80
      desired_count  = 1
      health_path    = "/api-ui/health"
    }
    "api" = {
      cpu            = 256
      memory         = 512
      container_port = 8000
      desired_count  = 1
      health_path    = "/api/health"
    }
    "gateway" = {
      cpu            = 256
      memory         = 512
      container_port = 8000
      desired_count  = 1
      health_path    = "/gateway/health"
    }
    "classification" = {
      cpu            = 512
      memory         = 1024
      container_port = 8000
      desired_count  = 1
      health_path    = "/classification/health"
    }
    "regression" = {
      cpu            = 512
      memory         = 1024
      container_port = 8000
      desired_count  = 1
      health_path    = "/regression/health"
    }
    "fairness" = {
      cpu            = 512
      memory         = 1024
      container_port = 8000
      desired_count  = 1
      health_path    = "/fairness/health"
    }
    "data-drift" = {
      cpu            = 512
      memory         = 1024
      container_port = 8000
      desired_count  = 1
      health_path    = "/data_drift/health"
    }
    "what-if" = {
      cpu            = 512
      memory         = 1024
      container_port = 8000
      desired_count  = 1
      health_path    = "/what_if/health"
    }
    "mainflow" = {
      cpu            = 256
      memory         = 512
      container_port = 8000
      desired_count  = 1
      health_path    = "/mainflow/health"
    }
  }
}
