import uuid

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from shared_migrations.models.base import Base


class ModelAnalysisResult(Base):
    __tablename__ = "model_analysis_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    request_id = Column(
        UUID(as_uuid=True), ForeignKey("model_analysis_requests.id", ondelete="CASCADE"), nullable=False, index=True
    )

    analysis_type = Column(String, nullable=False)

    overall_status = Column(String)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False, index=True)

    performance_metrics = Column(JSON)
    fairness_metrics = Column(JSON)
    feature_importance = Column(JSON)
    threshold_evaluations = Column(JSON)
    llm_analysis = Column(JSON)
    metadata_info = Column(JSON)
    warnings = Column(JSON)

    raw_response = Column(JSON, nullable=False)
    discover_id = Column(Integer, ForeignKey("discover.id", ondelete="CASCADE"), nullable=False, index=True)
    discover = relationship("Discover", back_populates="model_analysis_results")

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    request = relationship("ModelAnalysisRequest", back_populates="results")

    user = relationship("User", back_populates="model_analysis_results")
    tenant = relationship("Tenant", back_populates="model_analysis_results")
