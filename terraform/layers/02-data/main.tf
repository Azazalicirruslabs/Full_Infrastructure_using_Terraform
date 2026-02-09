# Layer 02: Data - Main Configuration
# Creates: RDS PostgreSQL, S3 Bucket

#------------------------------------------------------------------------------
# Read Layer 01 (Foundation) Outputs
#------------------------------------------------------------------------------
data "terraform_remote_state" "foundation" {
  backend = "local"

  config = {
    path = "${path.module}/../01-foundation/terraform.tfstate"
  }
}

locals {
  vpc_id                      = data.terraform_remote_state.foundation.outputs.vpc_id
  private_subnet_ids          = data.terraform_remote_state.foundation.outputs.private_subnet_ids
  ecs_tasks_security_group_id = data.terraform_remote_state.foundation.outputs.ecs_tasks_security_group_id
}

#------------------------------------------------------------------------------
# RDS Module
#------------------------------------------------------------------------------
module "rds" {
  source = "../../modules/rds"

  project_name = var.project_name
  environment  = var.environment
  vpc_id       = local.vpc_id
  subnet_ids   = local.private_subnet_ids

  # Allow access from ECS tasks
  allowed_security_group_ids = [local.ecs_tasks_security_group_id]

  # Database config
  db_name     = var.db_name
  db_username = var.db_username
  db_password = var.db_password

  # Dev settings
  instance_class      = var.db_instance_class
  multi_az            = false
  deletion_protection = false
}

#------------------------------------------------------------------------------
# S3 Module
#------------------------------------------------------------------------------
module "s3" {
  source = "../../modules/s3"

  project_name  = var.project_name
  environment   = var.environment
  force_destroy = var.s3_force_destroy
}

