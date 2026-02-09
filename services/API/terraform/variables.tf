# Variables for API Service Terraform

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "ap-south-1"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Project name prefix for resources"
  type        = string
  default     = "raia"
}

variable "max_image_count" {
  description = "Maximum number of images to keep in ECR"
  type        = number
  default     = 3
}

variable "image_tag" {
  description = "Tag for the Docker image"
  type        = string
  default     = "v1.0.0"
}
