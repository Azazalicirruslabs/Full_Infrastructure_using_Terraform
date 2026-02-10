# Layer 04: Compute - ECS Cluster, Services, ALB
# This is the main compute layer that orchestrates all microservices

#------------------------------------------------------------------------------
# Read Previous Layer Outputs
#------------------------------------------------------------------------------
data "terraform_remote_state" "foundation" {
  backend = "local"
  config = {
    path = "${path.module}/../01-foundation/terraform.tfstate"
  }
}

data "terraform_remote_state" "data" {
  backend = "local"
  config = {
    path = "${path.module}/../02-data/terraform.tfstate"
  }
}

data "terraform_remote_state" "ecr" {
  backend = "local"
  config = {
    path = "${path.module}/../03-ecr/terraform.tfstate"
  }
}

locals {
  # From Layer 01
  vpc_id                      = data.terraform_remote_state.foundation.outputs.vpc_id
  public_subnet_ids           = data.terraform_remote_state.foundation.outputs.public_subnet_ids
  alb_security_group_id       = data.terraform_remote_state.foundation.outputs.alb_security_group_id
  ecs_tasks_security_group_id = data.terraform_remote_state.foundation.outputs.ecs_tasks_security_group_id

  # From Layer 02
  s3_bucket_id      = data.terraform_remote_state.data.outputs.s3_bucket_id
  s3_bucket_arn     = data.terraform_remote_state.data.outputs.s3_bucket_arn
  rds_endpoint      = data.terraform_remote_state.data.outputs.rds_endpoint
  rds_database_name = data.terraform_remote_state.data.outputs.rds_database_name

  # From Layer 03
  ecr_repository_urls = data.terraform_remote_state.ecr.outputs.repository_urls
}

#------------------------------------------------------------------------------
# Data Sources
#------------------------------------------------------------------------------
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

#------------------------------------------------------------------------------
# ECS Cluster
#------------------------------------------------------------------------------
resource "aws_ecs_cluster" "main" {
  name = "${var.project_name}-${var.environment}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-cluster"
  }
}

#------------------------------------------------------------------------------
# CloudWatch Log Groups (one per service)
#------------------------------------------------------------------------------
resource "aws_cloudwatch_log_group" "services" {
  for_each = var.services

  name              = "/ecs/${var.project_name}/${var.environment}/${each.key}"
  retention_in_days = 7

  tags = {
    Name    = "${var.project_name}-${each.key}-logs"
    Service = each.key
  }
}

#------------------------------------------------------------------------------
# IAM Roles
#------------------------------------------------------------------------------
resource "aws_iam_role" "ecs_execution_role" {
  name = "${var.project_name}-${var.environment}-ecs-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution_role_policy" {
  role       = aws_iam_role.ecs_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "ecs_task_role" {
  name = "${var.project_name}-${var.environment}-ecs-task-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

#------------------------------------------------------------------------------
# S3 Access Policy for ECS Tasks (Least Privilege)
#------------------------------------------------------------------------------
resource "aws_iam_role_policy" "ecs_task_s3_access" {
  count = local.s3_bucket_arn != "" ? 1 : 0

  name = "${var.project_name}-${var.environment}-s3-access"
  role = aws_iam_role.ecs_task_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3BucketAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          local.s3_bucket_arn,
          "${local.s3_bucket_arn}/*"
        ]
      }
    ]
  })
}

#------------------------------------------------------------------------------
# Task Definitions (one per service)
# Best Practices (2024+):
#   - skip_destroy = true: Don't deregister old revisions on destroy
#   - track_latest = true: Track latest ACTIVE revision (for CI/CD updates)
#------------------------------------------------------------------------------
resource "aws_ecs_task_definition" "services" {
  for_each = var.services

  family                   = "${var.project_name}-${var.environment}-${each.key}"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = each.value.cpu
  memory                   = each.value.memory
  execution_role_arn       = aws_iam_role.ecs_execution_role.arn
  task_role_arn            = aws_iam_role.ecs_task_role.arn

  # Lifecycle management (Terraform AWS Provider v5.37.0+)
  skip_destroy = true   # Don't deregister old revisions when Terraform destroys/replaces
  track_latest = true   # Sync with latest ACTIVE revision (allows CI/CD to update)

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  container_definitions = jsonencode([
    {
      name      = each.key
      image     = "${local.ecr_repository_urls[each.key]}:latest"
      essential = true

      portMappings = [
        {
          containerPort = each.value.container_port
          hostPort      = each.value.container_port
          protocol      = "tcp"
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.services[each.key].name
          "awslogs-region"        = data.aws_region.current.name
          "awslogs-stream-prefix" = "ecs"
        }
      }

      environment = concat(
        # Common environment variables
        [
          { name = "ENVIRONMENT", value = var.environment },
          { name = "SERVICE_NAME", value = each.key },
          { name = "AWS_REGION", value = data.aws_region.current.name }
        ],
        # Frontend-specific: NEXT_PUBLIC_BASE_URL points to ALB for API calls
        each.key == "frontend" ? [
          { name = "NEXT_PUBLIC_BASE_URL", value = "http://${aws_lb.main.dns_name}" },
          { name = "NODE_ENV", value = "production" }
        ] : [],
        # Backend services: S3 and RDS config
        local.s3_bucket_id != "" ? [
          { name = "S3_BUCKET", value = local.s3_bucket_id }
        ] : [],
        local.rds_endpoint != "" ? [
          { name = "DATABASE_HOST", value = split(":", local.rds_endpoint)[0] },
          { name = "DATABASE_PORT", value = "5432" },
          { name = "DATABASE_NAME", value = local.rds_database_name }
        ] : []
      )
    }
  ])

  tags = {
    Name    = "${var.project_name}-${each.key}-task"
    Service = each.key
  }
}

#------------------------------------------------------------------------------
# Application Load Balancer
#------------------------------------------------------------------------------
resource "aws_lb" "main" {
  name               = "${var.project_name}-${var.environment}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [local.alb_security_group_id]
  subnets            = local.public_subnet_ids

  enable_deletion_protection = false
  idle_timeout               = 60

  tags = {
    Name = "${var.project_name}-${var.environment}-alb"
  }
}

#------------------------------------------------------------------------------
# ALB Listener
#------------------------------------------------------------------------------
resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = "80"
  protocol          = "HTTP"

  # Default action: Forward to Frontend (catches all non-API traffic)
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.services["frontend"].arn
  }
}

#------------------------------------------------------------------------------
# Target Groups (one per service)
#------------------------------------------------------------------------------
resource "aws_lb_target_group" "services" {
  for_each = var.services

  name        = "${var.project_name}-${each.key}-tg"
  port        = each.value.container_port
  protocol    = "HTTP"
  vpc_id      = local.vpc_id
  target_type = "ip"

  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = each.value.health_path
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 3
  }

  tags = {
    Name    = "${var.project_name}-${each.key}-tg"
    Service = each.key
  }
}

#------------------------------------------------------------------------------
# Listener Rules (path-based routing for backend services only)
#------------------------------------------------------------------------------
locals {
  # Backend services with their path patterns (excludes frontend)
  backend_service_paths = {
    "api"            = "/api/*"
    "gateway"        = "/gateway/*"
    "classification" = "/classification/*"
    "regression"     = "/regression/*"
    "fairness"       = "/fairness/*"
    "data-drift"     = "/data_drift/*"
    "what-if"        = "/what_if/*"
    "mainflow"       = "/mainflow/*"
  }
}

resource "aws_lb_listener_rule" "backend_services" {
  for_each = local.backend_service_paths

  listener_arn = aws_lb_listener.http.arn
  priority     = index(keys(local.backend_service_paths), each.key) + 1

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.services[each.key].arn
  }

  condition {
    path_pattern {
      values = [each.value]
    }
  }
}

#------------------------------------------------------------------------------
# ECS Services (one per microservice)
#------------------------------------------------------------------------------
resource "aws_ecs_service" "services" {
  for_each = var.services

  name            = "${var.project_name}-${each.key}-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.services[each.key].arn
  desired_count   = each.value.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = local.public_subnet_ids
    security_groups  = [local.ecs_tasks_security_group_id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.services[each.key].arn
    container_name   = each.key
    container_port   = each.value.container_port
  }

  deployment_minimum_healthy_percent = 50
  deployment_maximum_percent         = 200

  depends_on = [
    aws_lb_listener.http,
    aws_lb_listener_rule.backend_services
  ]

  lifecycle {
    ignore_changes = [desired_count]
  }

  tags = {
    Name    = "${var.project_name}-${each.key}-service"
    Service = each.key
  }
}
