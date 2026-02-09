
# TASK DEFINITIONS (One per service)
# 8 Task Definitions (one per service)
# 8 ECS Services (one per service)

resource "aws_ecs_task_definition" "services" {
  for_each = var.services

  family                   = "${var.project_name}-${each.key}" # raia-api, raia-gateway, etc.
  network_mode             = "awsvpc"                          # Each task gets its own IP address
  requires_compatibilities = ["FARGATE"]
  cpu                      = each.value.cpu
  memory                   = each.value.memory

  # IAM roles
  execution_role_arn = aws_iam_role.ecs_execution_role.arn
  task_role_arn      = aws_iam_role.ecs_task_role.arn

  # Runtime platform for consistent builds
  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  # Container definition (JSON format)
  container_definitions = jsonencode([
    {
      name      = each.key
      image     = "${data.aws_ecr_repository.services[each.key].repository_url}:latest"
      essential = true

      # Port mapping
      portMappings = [
        {
          containerPort = each.value.container_port
          hostPort      = each.value.container_port
          protocol      = "tcp"
        }
      ]

      # CloudWatch Logs configuration
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.services[each.key].name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }

      # Environment variables 
      environment = [
        {
          name  = "ENVIRONMENT"
          value = var.environment
        },
        {
          name  = "SERVICE_NAME"
          value = each.key
        }
      ]

      # Secrets from AWS Secrets Manager (uncomment and configure as needed)
      # secrets = [
      #   {
      #     name      = "DATABASE_URL"
      #     valueFrom = "arn:aws:secretsmanager:${var.aws_region}:${data.aws_caller_identity.current.account_id}:secret:${var.project_name}/${var.environment}/db-url"
      #   }
      # ]
    }
  ])

  tags = {
    Name    = "${var.project_name}-${each.key}-task"
    Service = each.key
  }
}

# ECS SERVICES (One per microservice)

resource "aws_ecs_service" "services" {
  for_each = var.services

  name            = "${var.project_name}-${each.key}-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.services[each.key].arn
  desired_count   = each.value.desired_count # How many containers to run
  launch_type     = "FARGATE"

  # Network configuration
  network_configuration {
    subnets          = data.aws_subnets.public.ids # Using public for now (simpler)
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = true # Needed to pull images from ECR
  }

  # Connect to Load Balancer
  load_balancer {
    target_group_arn = aws_lb_target_group.services[each.key].arn
    container_name   = each.key
    container_port   = each.value.container_port
  }

  # Deployment settings
  deployment_minimum_healthy_percent = 50  # Keep 50% running during deploy
  deployment_maximum_percent         = 200 # Can temporarily have 200% during deploy

  # Wait for Load Balancer to be ready
  depends_on = [
    aws_lb_listener.http,
    aws_lb_listener_rule.services
  ]

  tags = {
    Name    = "${var.project_name}-${each.key}-service"
    Service = each.key
  }

  # Ignore changes to desired_count if auto-scaling is used later
  lifecycle {
    ignore_changes = [desired_count] #Don't overwrite auto-scaling changes.
  }
}

