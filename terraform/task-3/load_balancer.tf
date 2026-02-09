
# COMPONENTS:
# 1. ALB (Application Load Balancer) - The main entrance
# 2. Listener - Listens for incoming requests (on port 80)
# 3. Target Groups - Groups of containers for each service
# 4. Listener Rules - Routes requests to correct target group

# APPLICATION LOAD BALANCER

resource "aws_lb" "main" {
  name               = "${var.project_name}-alb"
  internal           = false # false->internet-facing (public)
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = data.aws_subnets.public.ids # Public subnets

  enable_deletion_protection = false
  idle_timeout               = 60 # Connection idle timeout in seconds

  tags = {
    Name = "${var.project_name}-alb"
  }
}

# HTTP LISTENER (Port 80)

# This listens for HTTP requests on port 80.
# Default action: Return 404 (unless matched by a rule)

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = "80"
  protocol          = "HTTP"

  # Default action if no rules match
  default_action {
    type = "fixed-response"
    fixed_response {
      content_type = "application/json"
      message_body = "{\"error\": \"Not Found\", \"message\": \"Use /api, /classification, /fairness, etc.\"}"
      status_code  = "404"
    }
  }
}

# HTTPS LISTENER (Port 443) - Uncomment when SSL certificate is available
# resource "aws_lb_listener" "https" {
#   load_balancer_arn = aws_lb.main.arn
#   port              = "443"
#   protocol          = "HTTPS"
#   ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
#   certificate_arn   = var.ssl_certificate_arn  # Add this variable when ready
#
#   default_action {
#     type = "fixed-response"
#     fixed_response {
#       content_type = "application/json"
#       message_body = "{\"error\": \"Not Found\", \"message\": \"Use /api, /classification, /fairness, etc.\"}"
#       status_code  = "404"
#     }
#   }
# }

# TARGET GROUPS (One per service)

resource "aws_lb_target_group" "services" {
  for_each = var.services

  name        = "${var.project_name}-${each.key}-tg"
  port        = each.value.container_port # 8000
  protocol    = "HTTP"
  vpc_id      = data.aws_vpc.main.id
  target_type = "ip" # Fargate 

  # Health check configuration
  health_check {
    enabled             = true
    healthy_threshold   = 2                      # 2 successful checks = healthy
    interval            = 30                     # Check every 30 seconds
    matcher             = "200"                  # Expect HTTP 200 OK
    path                = each.value.health_path # /api/health, etc.
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5 # 5 second timeout
    unhealthy_threshold = 3 # 3 failed checks = unhealthy
  }

  tags = {
    Name    = "${var.project_name}-${each.key}-tg"
    Service = each.key
  }
}

# LISTENER RULES (Route based on URL path)

# These rules tell the Load Balancer:
# - If URL starts with /api → Send to API target group
# - If URL starts with /classification → Send to Classification target group
# - And so on...

# Create a mapping for service path prefixes
locals {
  service_paths = {
    "api"            = "/api/*"
    "classification" = "/classification/*"
    "data_drift"     = "/data_drift/*"
    "fairness"       = "/fairness/*"
    "gateway"        = "/gateway/*"
    "mainflow"       = "/mainflow/*"
    "regression"     = "/regression/*"
    "what_if"        = "/what_if/*"
  }
}

resource "aws_lb_listener_rule" "services" {
  for_each = var.services

  listener_arn = aws_lb_listener.http.arn # Attach this rule to the HTTP listener (port 80)

  # Priority determines rule order (lower = checked first)
  # We use index to generate unique priorities
  # keys(var.services) = ["api", "classification", "data_drift", "fairness", ...]
  # index(keys(var.services), each.key) gives the position of the current service

  priority = index(keys(var.services), each.key) + 1

  action {
    type             = "forward" # Forward to target group- matching one
    target_group_arn = aws_lb_target_group.services[each.key].arn
  }

  #   "Send request to the matching target group"
  # If each.key = "api":
  #   → Forward to aws_lb_target_group.services["api"]
  #   → Which is "raia-api-tg" target group

  condition {
    path_pattern {
      values = [local.service_paths[each.key]] # ["/api/*"]  ->Match any URL starting with /api/
    }
  }
}


