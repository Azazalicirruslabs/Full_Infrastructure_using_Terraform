
# Load Balancer URL

output "alb_dns_name" {
  description = "DNS name of the Application Load Balancer"
  value       = aws_lb.main.dns_name
}

output "alb_url" {
  description = "Full URL of the Application Load Balancer"
  value       = "http://${aws_lb.main.dns_name}"
}

# Service Endpoints

output "service_endpoints" {
  description = "Endpoints for each service"
  value = {
    for service, config in var.services :
    service => "http://${aws_lb.main.dns_name}${replace(local.service_paths[service], "/*", "")}"
  }
}

# ECS Cluster Info

output "ecs_cluster_name" {
  description = "Name of the ECS cluster"
  value       = aws_ecs_cluster.main.name
}

output "ecs_cluster_arn" {
  description = "ARN of the ECS cluster"
  value       = aws_ecs_cluster.main.arn
}

