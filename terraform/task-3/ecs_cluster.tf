
# ECS_CLUSTER.TF - ECS Cluster and CloudWatch Logs

# ECS CLUSTER


resource "aws_ecs_cluster" "main" {
  name = var.ecs_cluster_name

  # Enable Container Insights for monitoring
  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = {
    Name = "${var.project_name}-cluster"
  }
}

# CLOUDWATCH LOG GROUPS

resource "aws_cloudwatch_log_group" "services" {
  for_each = var.services

  name              = "/ecs/${var.project_name}/${each.key}"
  retention_in_days = 7 # Keep logs for 7 days (saves money)

  tags = {
    Name    = "${var.project_name}-${each.key}-logs"
    Service = each.key
  }
}


# create 8 log groups
# /ecs/raia/api
# /ecs/raia/classification
# /ecs/raia/data_drift
# /ecs/raia/fairness
# /ecs/raia/gateway
# /ecs/raia/mainflow
# /ecs/raia/regression
# /ecs/raia/what_if
