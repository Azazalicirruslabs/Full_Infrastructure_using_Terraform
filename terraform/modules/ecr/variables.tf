# ECR Module Variables

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
}

variable "services" {
  description = "List of service names to create ECR repos for"
  type        = list(string)
}

variable "max_image_count" {
  description = "Maximum number of images to keep per repository"
  type        = number
  default     = 5
}

variable "scan_on_push" {
  description = "Enable image scanning on push"
  type        = bool
  default     = true
}
