# Layer 03: ECR - Main Configuration

module "ecr" {
  source = "../../modules/ecr"

  project_name    = var.project_name
  services        = var.services
  max_image_count = var.max_image_count
}
