# Layer 04: Compute Outputs

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.main.name
}

output "ecs_cluster_arn" {
  description = "ECS cluster ARN"
  value       = aws_ecs_cluster.main.arn
}

output "alb_dns_name" {
  description = "ALB DNS name"
  value       = aws_lb.main.dns_name
}

output "alb_url" {
  description = "Full ALB URL"
  value       = "http://${aws_lb.main.dns_name}"
}

output "service_endpoints" {
  description = "Endpoints for each service"
  value = {
    for service, config in var.services :
    service => "http://${aws_lb.main.dns_name}${trimsuffix(local.service_paths[service], "/*")}"
  }
}
