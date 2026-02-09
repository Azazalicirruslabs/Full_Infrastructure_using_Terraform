"""
Performance Summary Schemas

This module defines Pydantic schemas for performance summary analysis,
including request/response models for both regression and classification model metrics evaluation.
"""

from typing import Any, Dict, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

# ==================== Threshold Models ====================


class MetricThreshold(BaseModel):
    """Threshold configuration for a single metric."""

    good: float = Field(..., description="Threshold for 'good' performance (green)")
    acceptable: float = Field(..., description="Threshold for 'acceptable' performance (yellow)")

    class Config:
        json_schema_extra = {"example": {"good": 0.1, "acceptable": 0.2}}


class RegressionThresholds(BaseModel):
    """Threshold configuration for all regression metrics."""

    mae: MetricThreshold = Field(
        default_factory=lambda: MetricThreshold(good=0.1, acceptable=0.2), description="Mean Absolute Error thresholds"
    )
    mse: MetricThreshold = Field(
        default_factory=lambda: MetricThreshold(good=0.05, acceptable=0.1), description="Mean Squared Error thresholds"
    )
    rmse: MetricThreshold = Field(
        default_factory=lambda: MetricThreshold(good=0.15, acceptable=0.25),
        description="Root Mean Squared Error thresholds",
    )
    r_squared: MetricThreshold = Field(
        default_factory=lambda: MetricThreshold(good=0.9, acceptable=0.8), description="R² Score thresholds"
    )
    mape: MetricThreshold = Field(
        default_factory=lambda: MetricThreshold(good=0.1, acceptable=0.2),
        description="Mean Absolute Percentage Error thresholds",
    )
    smape: MetricThreshold = Field(
        default_factory=lambda: MetricThreshold(good=10.0, acceptable=20.0),
        description="Symmetric Mean Absolute Percentage Error thresholds",
    )
    adjusted_r2: MetricThreshold = Field(
        default_factory=lambda: MetricThreshold(good=0.85, acceptable=0.75), description="Adjusted R² Score thresholds"
    )
    explained_variance: MetricThreshold = Field(
        default_factory=lambda: MetricThreshold(good=0.9, acceptable=0.8),
        description="Explained Variance Score thresholds",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "mae": {"good": 0.1, "acceptable": 0.2},
                "mse": {"good": 0.05, "acceptable": 0.1},
                "rmse": {"good": 0.15, "acceptable": 0.25},
                "r_squared": {"good": 0.9, "acceptable": 0.8},
                "mape": {"good": 0.1, "acceptable": 0.2},
                "smape": {"good": 10.0, "acceptable": 20.0},
                "adjusted_r2": {"good": 0.85, "acceptable": 0.75},
                "explained_variance": {"good": 0.9, "acceptable": 0.8},
            }
        }


class ClassificationThresholds(BaseModel):
    """Threshold configuration for all classification metrics."""

    accuracy: MetricThreshold = Field(
        default_factory=lambda: MetricThreshold(good=0.9, acceptable=0.8), description="Accuracy thresholds"
    )
    precision: MetricThreshold = Field(
        default_factory=lambda: MetricThreshold(good=0.85, acceptable=0.7), description="Precision thresholds"
    )
    recall: MetricThreshold = Field(
        default_factory=lambda: MetricThreshold(good=0.85, acceptable=0.7), description="Recall thresholds"
    )
    f1_score: MetricThreshold = Field(
        default_factory=lambda: MetricThreshold(good=0.85, acceptable=0.7), description="F1 Score thresholds"
    )
    roc_auc: MetricThreshold = Field(
        default_factory=lambda: MetricThreshold(good=0.9, acceptable=0.8), description="ROC-AUC thresholds"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "accuracy": {"good": 0.9, "acceptable": 0.8},
                "precision": {"good": 0.85, "acceptable": 0.7},
                "recall": {"good": 0.85, "acceptable": 0.7},
                "f1_score": {"good": 0.85, "acceptable": 0.7},
                "roc_auc": {"good": 0.9, "acceptable": 0.8},
            }
        }


# ==================== Metadata Models ====================


class PerformanceMetadata(BaseModel):
    """Metadata about the model and evaluation context."""

    asset_id: str = Field(..., description="Unique identifier for the model asset")
    asset_version: str = Field(..., description="Version of the model asset")
    model_version: Optional[str] = Field(None, description="Model version identifier")
    dataset_source: str = Field(..., description="Source dataset used for evaluation")
    dataset_version: Optional[str] = Field(None, description="Version or checksum of dataset")
    evaluation_date: str = Field(..., description="Date when evaluation was performed (ISO 8601)")
    project_id: Optional[str] = Field(None, description="Associated project identifier")

    class Config:
        json_schema_extra = {
            "example": {
                "asset_id": "model-regression-001",
                "asset_version": "v1.2.3",
                "model_version": "1.2.3",
                "dataset_source": "test_data_2026.csv",
                "dataset_version": "a1b2c3d4",
                "evaluation_date": "2026-01-21T10:30:00Z",
                "project_id": "project-123",
            }
        }


# ==================== Request Model ====================


class PerformanceSummaryRequest(BaseModel):
    """Request payload for performance summary analysis (supports both Regression and Classification)."""

    model_url: str = Field(..., description="URL to the model file (.pkl, .joblib, .onnx)")
    train_dataset_url: str = Field(..., description="URL to the training dataset (.csv, .parquet)")
    test_dataset_url: str = Field(..., description="URL to the test/evaluation dataset (.csv, .parquet)")
    target_column: Optional[str] = Field(
        None, description="Name of the target column (auto-detects last column if not provided)"
    )
    model_type: Optional[Literal["regression", "classification"]] = Field(
        None, description="Model type override. If not provided, auto-detection is used."
    )
    averaging_strategy: Literal["macro", "weighted"] = Field(
        "macro",
        description="Averaging strategy for classification metrics (Precision, Recall, F1). Options: 'macro' or 'weighted'",
    )
    regression_thresholds: Optional[RegressionThresholds] = Field(
        None, description="Threshold configuration for regression metrics. Uses defaults if not provided."
    )
    classification_thresholds: Optional[ClassificationThresholds] = Field(
        None, description="Threshold configuration for classification metrics. Uses defaults if not provided."
    )

    @field_validator("model_type", "target_column", mode="before")
    @classmethod
    def empty_string_to_none(cls, v):
        """Convert empty strings to None for optional fields."""
        if v == "" or v is None:
            return None
        return v

    @model_validator(mode="after")
    def set_default_thresholds(self):
        """Set default thresholds if not provided."""
        model_type = getattr(self, "model_type", None)
        if model_type == "regression":
            if self.regression_thresholds is None:
                self.regression_thresholds = RegressionThresholds()
        elif model_type == "classification":
            if self.classification_thresholds is None:
                self.classification_thresholds = ClassificationThresholds()
        else:
            # Preserve previous behavior when model_type is not specified or is unexpected
            if self.regression_thresholds is None:
                self.regression_thresholds = RegressionThresholds()
            if self.classification_thresholds is None:
                self.classification_thresholds = ClassificationThresholds()
        return self

    # class Config:
    #     json_schema_extra = {
    #         "example": {
    #             "model_url": "https://s3.amazonaws.com/my-bucket/models/regression_model.pkl",
    #             "train_dataset_url": "https://s3.amazonaws.com/my-bucket/data/train_data.csv",
    #             "test_dataset_url": "https://s3.amazonaws.com/my-bucket/data/test_data.csv",
    #             "target_column": "price",
    #             "thresholds": {
    #                 "mae": {"good": 0.1, "acceptable": 0.2},
    #                 "mse": {"good": 0.05, "acceptable": 0.1},
    #                 "rmse": {"good": 0.15, "acceptable": 0.25},
    #                 "r_squared": {"good": 0.9, "acceptable": 0.8}
    #             }
    #         }
    #     }


# ==================== Response Models ====================


class MetricDetail(BaseModel):
    """Detailed evaluation of a single metric."""

    value: float = Field(..., description="Actual metric value")
    threshold_good: float = Field(..., description="Threshold for good performance")
    threshold_acceptable: float = Field(..., description="Threshold for acceptable performance")
    status: Literal["good", "acceptable", "poor"] = Field(..., description="Performance status")
    color: Literal["green", "yellow", "red"] = Field(..., description="Visual indicator color")

    class Config:
        json_schema_extra = {
            "example": {
                "value": 0.15,
                "threshold_good": 0.1,
                "threshold_acceptable": 0.2,
                "status": "acceptable",
                "color": "yellow",
            }
        }


class RegressionMetricsOutput(BaseModel):
    """Evaluated regression metrics with status indicators."""

    mae: MetricDetail = Field(..., description="Mean Absolute Error evaluation")
    mse: MetricDetail = Field(..., description="Mean Squared Error evaluation")
    rmse: MetricDetail = Field(..., description="Root Mean Squared Error evaluation")
    r_squared: MetricDetail = Field(..., description="R² Score evaluation")
    mape: MetricDetail = Field(..., description="Mean Absolute Percentage Error evaluation")
    smape: MetricDetail = Field(..., description="Symmetric Mean Absolute Percentage Error evaluation")
    adjusted_r2: MetricDetail = Field(..., description="Adjusted R² Score evaluation")
    explained_variance: MetricDetail = Field(..., description="Explained Variance Score evaluation")


class ClassificationMetricsOutput(BaseModel):
    """Evaluated classification metrics with status indicators."""

    accuracy: MetricDetail = Field(..., description="Accuracy evaluation")
    precision: MetricDetail = Field(..., description="Precision evaluation")
    recall: MetricDetail = Field(..., description="Recall evaluation")
    f1_score: MetricDetail = Field(..., description="F1 Score evaluation")
    roc_auc: MetricDetail = Field(..., description="ROC-AUC evaluation")


class RegressionPerformanceMetrics(BaseModel):
    """Performance metrics for train and test sets (Regression)."""

    train: RegressionMetricsOutput = Field(..., description="Training set performance metrics")
    test: RegressionMetricsOutput = Field(..., description="Test set performance metrics")
    overfitting_score: float = Field(..., description="Overfitting indicator (0-1, lower is better)", ge=0, le=1)


class ClassificationPerformanceMetrics(BaseModel):
    """Performance metrics for train and test sets (Classification)."""

    train: ClassificationMetricsOutput = Field(..., description="Training set performance metrics")
    test: ClassificationMetricsOutput = Field(..., description="Test set performance metrics")
    overfitting_score: float = Field(..., description="Overfitting indicator (0-1, lower is better)", ge=0, le=1)


class PerformanceMetrics(BaseModel):
    """Performance metrics for train and test sets."""

    train: RegressionMetricsOutput = Field(..., description="Training set performance metrics")
    test: RegressionMetricsOutput = Field(..., description="Test set performance metrics")
    overfitting_score: float = Field(..., description="Overfitting indicator (0-1, lower is better)", ge=0, le=1)


class PerformanceInterpretation(BaseModel):
    """LLM-generated interpretation of performance results (Fairness-aligned format)."""

    what_this_means: str = Field(..., description="Plain-language explanation of key findings")
    why_it_matters: str = Field(..., description="Real-world implications and business impact")
    risk_signal: str = Field(..., description="Overall performance status and risk level")

    class Config:
        json_schema_extra = {
            "example": {
                "what_this_means": "The model demonstrates strong predictive performance with an R² score of 0.92, indicating it explains 92% of variance in the target variable. Error metrics (MAE, RMSE) are within acceptable ranges.",
                "why_it_matters": "High R² and low error metrics suggest the model can reliably predict outcomes. Consistent train/test performance indicates good generalization to unseen data.",
                "risk_signal": "Low Risk - All metrics are within acceptable thresholds with minimal overfitting detected.",
            }
        }


class PerformanceSummaryResponse(BaseModel):
    """Complete performance summary response (supports both Regression and Classification)."""

    model_type: Literal["Regression", "Classification", "Unknown"] = Field(
        ..., description="Automatically detected or user-specified model type"
    )
    model_detection_confidence: Literal["high", "medium", "low", "user_specified"] = Field(
        ..., description="Confidence level of model type detection"
    )
    metrics: Union[RegressionPerformanceMetrics, ClassificationPerformanceMetrics, Dict[str, Any]] = Field(
        ..., description="Performance metrics for train and test sets"
    )
    interpretation: PerformanceInterpretation = Field(..., description="LLM-generated performance interpretation")
    averaging_strategy: Optional[str] = Field(
        None, description="Averaging strategy used for classification metrics (macro/weighted)"
    )
    metadata: Optional[PerformanceMetadata] = Field(None, description="Model and evaluation metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "model_type": "Regression",
                "model_detection_confidence": "high",
                "metrics": {
                    "train": {
                        "mae": {
                            "value": 0.12,
                            "threshold_good": 0.1,
                            "threshold_acceptable": 0.2,
                            "status": "acceptable",
                            "color": "yellow",
                        },
                        "mse": {
                            "value": 0.038,
                            "threshold_good": 0.05,
                            "threshold_acceptable": 0.1,
                            "status": "good",
                            "color": "green",
                        },
                        "rmse": {
                            "value": 0.195,
                            "threshold_good": 0.15,
                            "threshold_acceptable": 0.25,
                            "status": "acceptable",
                            "color": "yellow",
                        },
                        "r_squared": {
                            "value": 0.95,
                            "threshold_good": 0.9,
                            "threshold_acceptable": 0.8,
                            "status": "good",
                            "color": "green",
                        },
                    },
                    "test": {
                        "mae": {
                            "value": 0.15,
                            "threshold_good": 0.1,
                            "threshold_acceptable": 0.2,
                            "status": "acceptable",
                            "color": "yellow",
                        },
                        "mse": {
                            "value": 0.045,
                            "threshold_good": 0.05,
                            "threshold_acceptable": 0.1,
                            "status": "good",
                            "color": "green",
                        },
                        "rmse": {
                            "value": 0.212,
                            "threshold_good": 0.15,
                            "threshold_acceptable": 0.25,
                            "status": "acceptable",
                            "color": "yellow",
                        },
                        "r_squared": {
                            "value": 0.92,
                            "threshold_good": 0.9,
                            "threshold_acceptable": 0.8,
                            "status": "good",
                            "color": "green",
                        },
                    },
                    "overfitting_score": 0.15,
                },
                "interpretation": {
                    "what_this_means": "The model shows strong performance on both train and test sets with minimal overfitting.",
                    "why_it_matters": "Excellent R² on both sets indicates the model generalizes well to unseen data.",
                    "risk_signal": "Low Risk - MAE slightly higher on test set, monitor for degradation over time.",
                },
                "metadata": {
                    "asset_id": "model-regression-001",
                    "asset_version": "v1.2.3",
                    "model_version": "1.2.3",
                    "dataset_source": "test_data_2026.csv",
                    "evaluation_date": "2026-01-21T10:30:00Z",
                },
            }
        }
