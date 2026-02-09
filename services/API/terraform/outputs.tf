# Outputs for API Service

output "repository_url" {
  description = "ECR Repository URL for API service"
  value       = aws_ecr_repository.this.repository_url
}

output "repository_arn" {
  description = "ECR Repository ARN"
  value       = aws_ecr_repository.this.arn
}

output "repository_name" {
  description = "ECR Repository Name"
  value       = aws_ecr_repository.this.name
}

output "image_uri" {
  description = "Full image URI with tag"
  value       = "${aws_ecr_repository.this.repository_url}:latest"
}

output "aws_account_id" {
  description = "AWS Account ID"
  value       = data.aws_caller_identity.current.account_id
}

output "aws_region" {
  description = "AWS Region"
  value       = data.aws_region.current.name
}
