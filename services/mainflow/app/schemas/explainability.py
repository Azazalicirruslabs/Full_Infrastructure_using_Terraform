"""
Pydantic schemas for explainability analysis requests and responses.
"""

from typing import List

from pydantic import BaseModel, Field, field_validator


class ExplainabilityRequest(BaseModel):
    """Request schema for explainability analysis."""

    ref_dataset: str = Field(..., description="Path or URL to the training/reference dataset (.csv or .parquet)")
    cur_dataset: str = Field(..., description="Path or URL to the test/current dataset (.csv or .parquet)")
    model: str = Field(..., description="Path or URL to the model file (.joblib, .pkl, or .pickle)")
    target_column: str = Field(..., description="Name of the target column in the datasets")

    # Feature importance method selection
    feature_importance: bool = Field(default=True, description="Whether to compute feature importance")
    feature_importance_method: str = Field(
        default="shap", description="Method for feature importance: 'shap', 'gain', or 'permutation'"
    )

    @field_validator("feature_importance_method")
    @classmethod
    def validate_method(cls, v):
        """Ensure method is one of the supported values."""
        allowed_methods = ["shap", "gain", "permutation"]
        if v.lower() not in allowed_methods:
            raise ValueError(f"feature_importance_method must be one of {allowed_methods}, got '{v}'")
        return v.lower()


class FeatureImportanceItem(BaseModel):
    """Individual feature importance item."""

    name: str = Field(..., description="Feature name")
    importance: float = Field(..., description="Importance value")
    impact_direction: str = Field(..., description="Impact direction: 'positive' or 'negative'")
    rank: int = Field(..., description="Rank based on importance")


class LLMAnalysisResponse(BaseModel):
    """LLM-generated analysis insights."""

    what_this_means: str = Field(..., description="Plain language explanation of results")
    why_it_matters: str = Field(..., description="Business impact and implications")
    risk_signal: str = Field(..., description="Risk level and key concerns")


class ThresholdEvaluation(BaseModel):
    """Threshold evaluation result for a metric."""

    metric_name: str = Field(..., description="Name of the evaluated metric")
    metric_value: float = Field(..., description="Current value of the metric")
    status: str = Field(..., description="Status: 'acceptable', 'warning', or 'breach'")
    threshold_used: float = Field(..., description="The threshold value used for evaluation")
    message: str = Field(..., description="Human-readable status message")


class ExplainabilityResponse(BaseModel):
    """Response schema for explainability analysis."""

    model_type: str = Field(..., description="Type of model: 'Classification' or 'Regression'")
    computation_method: str = Field(..., description="Method used for feature importance computation")
    computed_at: str = Field(..., description="ISO timestamp when analysis was computed")
    llm_analysis: LLMAnalysisResponse = Field(..., description="LLM-generated insights and explanations")
    shap_available: bool = Field(..., description="Whether SHAP analysis is available")
    target_column: str = Field(..., description="Name of the target column used in analysis")
    total_features: int = Field(..., description="Total number of features")
    positive_impact_count: int = Field(..., description="Number of features with positive impact")
    negative_impact_count: int = Field(..., description="Number of features with negative impact")
    features: List[FeatureImportanceItem] = Field(..., description="List of feature importance items")

    class Config:
        json_schema_extra = {
            "example": {
                "model_type": "Classification",
                "computation_method": "shap",
                "computed_at": "2026-01-20T10:51:32.655219+00:00",
                "llm_analysis": {
                    "what_this_means": "The model shows moderate performance with 60.4% accuracy on test data...",
                    "why_it_matters": "This indicates potential overfitting that could impact production reliability...",
                    "risk_signal": "Medium risk - significant performance gap between training and test sets.",
                },
                "shap_available": True,
                "target_column": "budget_overrun",
                "total_features": 15,
                "positive_impact_count": 10,
                "negative_impact_count": 5,
                "features": [
                    {
                        "name": "memory_utilization",
                        "importance": 0.09162541047687169,
                        "impact_direction": "negative",
                        "rank": 1,
                    },
                    {
                        "name": "instance_count",
                        "importance": 0.08344094298569608,
                        "impact_direction": "positive",
                        "rank": 2,
                    },
                    {
                        "name": "project_priority_score",
                        "importance": 0.06532031848197074,
                        "impact_direction": "negative",
                        "rank": 3,
                    },
                ],
            }
        }
