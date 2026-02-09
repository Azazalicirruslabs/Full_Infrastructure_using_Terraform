
# VARIABLES.TF - Configuration for ECS & Load Balancer Deployment

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "ap-south-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "raia"
}

# ECS Cluster Configuration

variable "ecs_cluster_name" {
  description = "ECS Cluster Name"
  type        = string
  default     = "raia-ecs-cluster"
}


# VPC Configuration (From task 1 - auto-detected via data source)

variable "vpc_id" {
  description = "VPC ID from task 1 "
  type        = string
  default     = ""
}

# Health Check (Global setting - same interval for all services)

variable "health_check_interval" {
  description = "Seconds between health checks"
  type        = number
  default     = 30
}

# Service Configurations (All services defined in ONE place)
# NOTE: Only deploying API for now - uncomment others after their ECR repos are created

variable "services" {
  description = "Configuration for each microservice"
  type = map(object({
    cpu            = number
    memory         = number
    container_port = number
    desired_count  = number
    health_path    = string
    dockerfile     = string
  }))
  default = {
    # API Service - Currently deployed
    "api" = {
      cpu            = 256 # 0.25 vCPU
      memory         = 512 # 512 MB RAM
      container_port = 8000
      desired_count  = 1
      health_path    = "/api/health"
      dockerfile     = "services/API/Dockerfile"
    }

    # ─────────────────────────────────────────────────────────────
    # UNCOMMENT BELOW SERVICES AFTER DEPLOYING THEIR ECR REPOS
    # ─────────────────────────────────────────────────────────────

    # "classification" = {
    #   cpu            = 512
    #   memory         = 1024
    #   container_port = 8000
    #   desired_count  = 1
    #   health_path    = "/classification/health"
    #   dockerfile     = "services/classification/Dockerfile"
    # }
    # "data-drift" = {  # Changed from data_drift (no underscores in target group names)
    #   cpu            = 512
    #   memory         = 1024
    #   container_port = 8000
    #   desired_count  = 1
    #   health_path    = "/data-drift/health"
    #   dockerfile     = "services/data_drift/Dockerfile"
    # }
    # "fairness" = {
    #   cpu            = 512
    #   memory         = 1024
    #   container_port = 8000
    #   desired_count  = 1
    #   health_path    = "/fairness/health"
    #   dockerfile     = "services/fairness/Dockerfile"
    # }
    # "gateway" = {
    #   cpu            = 256
    #   memory         = 512
    #   container_port = 8000
    #   desired_count  = 1
    #   health_path    = "/gateway/health"
    #   dockerfile     = "services/gateway/Dockerfile"
    # }
    # "mainflow" = {
    #   cpu            = 256
    #   memory         = 512
    #   container_port = 8000
    #   desired_count  = 1
    #   health_path    = "/mainflow/health"
    #   dockerfile     = "services/mainflow/Dockerfile"
    # }
    # "regression" = {
    #   cpu            = 512
    #   memory         = 1024
    #   container_port = 8000
    #   desired_count  = 1
    #   health_path    = "/regression/health"
    #   dockerfile     = "services/regression/Dockerfile"
    # }
    # "what-if" = {  # Changed from what_if (no underscores in target group names)
    #   cpu            = 512
    #   memory         = 1024
    #   container_port = 8000
    #   desired_count  = 1
    #   health_path    = "/what-if/health"
    #   dockerfile     = "services/what_if/Dockerfile"
    # }
  }
}
