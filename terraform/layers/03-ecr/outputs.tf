# Layer 03: ECR Outputs

output "repository_urls" {
  description = "Map of service name to ECR repository URL"
  value       = module.ecr.repository_urls
}

output "repository_arns" {
  description = "Map of service name to ECR repository ARN"
  value       = module.ecr.repository_arns
}
