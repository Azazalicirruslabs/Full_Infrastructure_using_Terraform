# Layer 05: Monitoring - CloudWatch Dashboard and Alarms

#------------------------------------------------------------------------------
# Read Layer 04 (Compute) Outputs
#------------------------------------------------------------------------------
data "terraform_remote_state" "compute" {
  backend = "local"
  config = {
    path = "${path.module}/../04-compute/terraform.tfstate"
  }
}

locals {
  ecs_cluster_name = data.terraform_remote_state.compute.outputs.ecs_cluster_name
}

#------------------------------------------------------------------------------
# SNS Topic for Alarms (optional)
#------------------------------------------------------------------------------
resource "aws_sns_topic" "alarms" {
  count = var.alarm_email != "" ? 1 : 0
  name  = "${var.project_name}-${var.environment}-alarms"
}

resource "aws_sns_topic_subscription" "email" {
  count     = var.alarm_email != "" ? 1 : 0
  topic_arn = aws_sns_topic.alarms[0].arn
  protocol  = "email"
  endpoint  = var.alarm_email
}

#------------------------------------------------------------------------------
# CloudWatch Dashboard
#------------------------------------------------------------------------------
resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = "${var.project_name}-${var.environment}-dashboard"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6
        properties = {
          title  = "ECS CPU Utilization"
          region = var.aws_region
          metrics = [
            ["AWS/ECS", "CPUUtilization", "ClusterName", local.ecs_cluster_name]
          ]
          stat   = "Average"
          period = 300
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 0
        width  = 12
        height = 6
        properties = {
          title  = "ECS Memory Utilization"
          region = var.aws_region
          metrics = [
            ["AWS/ECS", "MemoryUtilization", "ClusterName", local.ecs_cluster_name]
          ]
          stat   = "Average"
          period = 300
        }
      },
      {
        type   = "text"
        x      = 0
        y      = 6
        width  = 24
        height = 2
        properties = {
          markdown = "## RAIA Platform - ${var.environment} Environment\n\nMonitor CPU, Memory, and service health across all microservices."
        }
      }
    ]
  })
}

#------------------------------------------------------------------------------
# CloudWatch Alarm: High CPU
#------------------------------------------------------------------------------
resource "aws_cloudwatch_metric_alarm" "cpu_high" {
  alarm_name          = "${var.project_name}-${var.environment}-cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/ECS"
  period              = 300
  statistic           = "Average"
  threshold           = 80
  alarm_description   = "ECS cluster CPU utilization is above 80%"

  dimensions = {
    ClusterName = local.ecs_cluster_name
  }

  alarm_actions = var.alarm_email != "" ? [aws_sns_topic.alarms[0].arn] : []
  ok_actions    = var.alarm_email != "" ? [aws_sns_topic.alarms[0].arn] : []

  tags = {
    Name = "${var.project_name}-${var.environment}-cpu-alarm"
  }
}

