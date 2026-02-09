
# SECURITY_GROUPS- Network Security Rules

# TWO SECURITY GROUPS:
# 1. For Load Balancer: Allow internet traffic (port 80)
# 2. For ECS Tasks: Allow only traffic from Load Balancer

# SECURITY GROUP 1: Application Load Balancer

# This security group allows:
# - INBOUND: Anyone from internet on port 80 (HTTP)
# - OUTBOUND: All traffic (to reach ECS tasks)


resource "aws_security_group" "alb" {
  name        = "${var.project_name}-alb-sg"
  description = "Security group for Application Load Balancer"
  vpc_id      = data.aws_vpc.main.id

  # INGRESS (Inbound) Rules

  # Allow HTTP from anywhere (internet)
  ingress {
    description = "HTTP from internet"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # 0.0.0.0/0 = anywhere
  }

  # Allow HTTPS from anywhere (for future SSL setup)
  ingress {
    description = "HTTPS from internet"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # EGRESS (Outbound) Rules 

  # Allow all outbound traffic (to reach ECS tasks)
  egress {
    description = "Allow all outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1" # -1 = all protocols
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-alb-sg"
  }
}


# SECURITY GROUP 2: ECS Tasks (Containers)

# This security group allows:
# - INBOUND: Only traffic from Load Balancer on port 8000
# - OUTBOUND: All traffic (to access internet, ECR, etc.)


resource "aws_security_group" "ecs_tasks" {
  name        = "${var.project_name}-ecs-tasks-sg"
  description = "Security group for ECS tasks"
  vpc_id      = data.aws_vpc.main.id

  # INGRESS: Allow traffic from Load Balancer only
  ingress {
    description     = "Traffic from ALB"
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id] # Only from ALB!
  }

  # EGRESS: Allow all outbound (for ECR pull, external APIs, etc.)
  egress {
    description = "Allow all outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-ecs-tasks-sg"
  }
}

