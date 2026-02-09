# ECR Module - Creates repositories for all microservices

#------------------------------------------------------------------------------
# ECR Repositories (one per service)
#------------------------------------------------------------------------------
resource "aws_ecr_repository" "this" {
  for_each = toset(var.services)

  name                 = "${var.project_name}-${each.key}"
  force_delete         = true
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = var.scan_on_push
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = {
    Name    = "${var.project_name}-${each.key}"
    Service = each.key
  }
}

#------------------------------------------------------------------------------
# Lifecycle Policy - Keep only latest N images
#------------------------------------------------------------------------------
resource "aws_ecr_lifecycle_policy" "this" {
  for_each = toset(var.services)

  repository = aws_ecr_repository.this[each.key].name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep only the last ${var.max_image_count} images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = var.max_image_count
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}
