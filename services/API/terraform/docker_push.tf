# Docker Build and Push for API Service

# ECR Login
resource "null_resource" "ecr_login" {
  provisioner "local-exec" {
    command = "aws ecr get-login-password --region ${data.aws_region.current.name} | docker login --username AWS --password-stdin ${data.aws_caller_identity.current.account_id}.dkr.ecr.${data.aws_region.current.name}.amazonaws.com"
  }

  triggers = {
    always_run = timestamp()
  }

  depends_on = [aws_ecr_repository.this]
}

# Build and Push Docker Image
resource "null_resource" "docker_build_push" {
  provisioner "local-exec" {
    # Go to RAIA-BACKEND root (up 3 levels from terraform folder)
    working_dir = "${path.module}/../../.."

    command = <<-EOT
      echo "Building Docker image for API service..."
      docker build -t ${var.project_name}-api:latest -f services/API/Dockerfile .

      echo "Tagging image for ECR..."
      docker tag ${var.project_name}-api:latest ${aws_ecr_repository.this.repository_url}:latest
      docker tag ${var.project_name}-api:latest ${aws_ecr_repository.this.repository_url}:${var.image_tag}

      echo "Pushing to ECR..."
      docker push ${aws_ecr_repository.this.repository_url}:latest
      docker push ${aws_ecr_repository.this.repository_url}:${var.image_tag}

      echo "Successfully pushed API service to ECR!"
    EOT

    interpreter = ["bash", "-c"]
  }

  triggers = {
    dockerfile_hash = filemd5("${path.module}/../Dockerfile")
    always_rebuild  = timestamp()
  }

  depends_on = [null_resource.ecr_login]
}
