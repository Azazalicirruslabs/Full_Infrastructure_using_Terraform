# Layer 03: ECR Variables

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

variable "services" {
  description = "List of microservice names"
  type        = list(string)
  default = [
    "api",
    "gateway",
    "classification",
    "regression",
    "fairness",
    "data-drift",
    "what-if",
    "mainflow"
  ]
}

variable "max_image_count" {
  description = "Maximum images to keep per repository"
  type        = number
  default     = 5
}
