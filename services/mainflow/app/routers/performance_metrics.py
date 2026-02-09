"""
Performance Metrics API Router

This module provides RESTful API endpoints for performance summary analysis
of ML models (both Regression and Classification), including automatic model
type detection and LLM-based interpretation.

Endpoints:
    POST /mainflow/performance/summary - Generate performance summary with interpretation
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

from services.mainflow.app.core.performance_summary import PerformanceSummaryAnalyzer
from services.mainflow.app.schemas.performance_schema import (
    PerformanceSummaryRequest,
    PerformanceSummaryResponse,
)
from shared.auth import get_current_user

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/mainflow/performance", tags=["Performance Metrics"])


# ==================== API Endpoints ====================


@router.post("/summary", response_model=PerformanceSummaryResponse, status_code=200)
async def generate_performance_summary(
    request: PerformanceSummaryRequest, current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Generate comprehensive performance summary for a machine learning model.

    **Supports both Regression and Classification models with automatic detection.**

    **Authentication:** Requires Bearer token in Authorization header.

    This endpoint performs:
    - Secure model loading from provided URL (supports .pkl, .joblib, .onnx)
    - Automatic model type detection (Regression/Classification) or user override
    - Threshold-based metric evaluation with color-coding
    - LLM-generated interpretation using Claude via AWS Bedrock

    **Workflow:**
    1. Load model from URL (with security validation)
    2. Detect model type or use user-specified override
    3. Auto-detect target column (last column) if not provided
    4. Calculate appropriate metrics based on model type
    5. Evaluate each metric against user-defined or default thresholds
    6. Generate AI-powered interpretation of results
    7. Return enriched summary with status indicators

    **Regression Metrics:**
    - MAE, MSE, RMSE, R², MAPE, SMAPE, Adjusted R², Explained Variance

    **Classification Metrics:**
    - Accuracy, Precision, Recall, F1 Score, ROC-AUC

    Args:
        request: PerformanceSummaryRequest containing:
            - model_url: URL to model file (.pkl, .joblib, .onnx)
            - train_dataset_url: URL to training dataset (.csv, .parquet)
            - test_dataset_url: URL to test dataset (.csv, .parquet)
            - target_column: (Optional) Target column name, auto-detects last column if not provided
            - model_type: (Optional) "regression" or "classification" - overrides auto-detection
            - averaging_strategy: "macro" or "weighted" for classification metrics
            - regression_thresholds: (Optional) Custom thresholds for regression
            - classification_thresholds: (Optional) Custom thresholds for classification

    Returns:
        PerformanceSummaryResponse with:
            - model_type: Detected or specified model type
            - model_detection_confidence: Confidence level (high/medium/low/user_specified)
            - metrics: Train and test metrics with status indicators
            - interpretation: LLM-generated analysis (what_this_means, why_it_matters, risk_signal)
            - averaging_strategy: Averaging used for classification metrics (if applicable)
            - metadata: Model and evaluation context

    Raises:
        HTTPException 400: Invalid input data or configuration
        HTTPException 401: Unauthorized - invalid or missing token
        HTTPException 422: Validation error
        HTTPException 500: Internal processing error

    Example Request (Regression):
        ```json
        {
            "model_url": "https://s3.amazonaws.com/bucket/models/regression_model.pkl",
            "train_dataset_url": "https://s3.amazonaws.com/bucket/data/train.csv",
            "test_dataset_url": "https://s3.amazonaws.com/bucket/data/test.csv",
            "target_column": "price"
        }
        ```

    Example Request (Classification):
        ```json
        {
            "model_url": "https://s3.amazonaws.com/bucket/models/classifier.pkl",
            "train_dataset_url": "https://s3.amazonaws.com/bucket/data/train.csv",
            "test_dataset_url": "https://s3.amazonaws.com/bucket/data/test.csv",
            "model_type": "classification",
            "averaging_strategy": "weighted",
            "classification_thresholds": {
                "accuracy": {"good": 0.95, "acceptable": 0.85},
                "precision": {"good": 0.9, "acceptable": 0.8},
                "recall": {"good": 0.9, "acceptable": 0.8},
                "f1_score": {"good": 0.9, "acceptable": 0.8},
                "roc_auc": {"good": 0.95, "acceptable": 0.85}
            }
        }
        ```

    **Security Notes:**
    - Model URLs are validated against allowed S3 buckets/directories
    - Only whitelisted ML libraries are permitted during model deserialization
    - All model loading uses restricted unpickler to prevent code execution
    - ONNX models are loaded using onnxruntime InferenceSession
    """
    try:
        logger.info(
            "Performance summary request received for model: %s",
            request.model_url,
        )

        # Initialize analyzer
        analyzer = PerformanceSummaryAnalyzer()

        # Calculate performance summary
        response = analyzer.calculate_performance_summary(request)

        logger.info(
            "Performance summary generated successfully. Model type: %s (confidence: %s)",
            response.model_type,
            response.model_detection_confidence,
        )

        return response.model_dump()

    except ValueError as val_error:
        logger.error("Validation error: %s", val_error)
        raise HTTPException(status_code=422, detail=str(val_error)) from val_error

    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Error generating performance summary: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate performance summary: {str(exc)}") from exc
