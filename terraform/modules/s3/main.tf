# S3 Bucket Module

#------------------------------------------------------------------------------
# Random suffix for globally unique bucket name
#------------------------------------------------------------------------------
resource "random_id" "suffix" {
  byte_length = 4
}

#------------------------------------------------------------------------------
# S3 Bucket
#------------------------------------------------------------------------------
resource "aws_s3_bucket" "this" {
  bucket        = "${var.project_name}-${var.environment}-${random_id.suffix.hex}"
  force_destroy = var.force_destroy

  tags = {
    Name = "${var.project_name}-${var.environment}-bucket"
  }
}

#------------------------------------------------------------------------------
# Block Public Access
#------------------------------------------------------------------------------
resource "aws_s3_bucket_public_access_block" "this" {
  bucket = aws_s3_bucket.this.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

#------------------------------------------------------------------------------
# Versioning
#------------------------------------------------------------------------------
resource "aws_s3_bucket_versioning" "this" {
  bucket = aws_s3_bucket.this.id

  versioning_configuration {
    status = var.versioning_enabled ? "Enabled" : "Suspended"
  }
}

#------------------------------------------------------------------------------
# Server-Side Encryption
#------------------------------------------------------------------------------
resource "aws_s3_bucket_server_side_encryption_configuration" "this" {
  bucket = aws_s3_bucket.this.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

#------------------------------------------------------------------------------
# Lifecycle Rules
#------------------------------------------------------------------------------
resource "aws_s3_bucket_lifecycle_configuration" "this" {
  bucket = aws_s3_bucket.this.id

  rule {
    id     = "expire-old-versions"
    status = "Enabled"

    filter {
      prefix = ""
    }

    noncurrent_version_expiration {
      noncurrent_days = var.noncurrent_version_expiration_days
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}
