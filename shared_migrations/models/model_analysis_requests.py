import uuid

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from shared_migrations.models.base import Base


class ModelAnalysisRequest(Base):
    __tablename__ = "model_analysis_requests"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    request_hash = Column(String, nullable=False, unique=True, index=True)

    analysis_type = Column(String, nullable=False)
    # regression | classification | fairness | drift

    request_payload = Column(JSON, nullable=False)

    reference_url = Column(Text)
    current_url = Column(Text)
    model_url = Column(Text)

    target_column = Column(String)
    sensitive_feature = Column(String)

    status = Column(String, nullable=False, default="PENDING")
    error_message = Column(Text)

    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False)
    discover_id = Column(Integer, ForeignKey("discover.id", ondelete="CASCADE"), nullable=False, index=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    user = relationship("User", back_populates="model_analysis_requests")
    tenant = relationship("Tenant", back_populates="model_analysis_requests")
    discover = relationship("Discover", back_populates="model_analysis_requests")
    results = relationship("ModelAnalysisResult", back_populates="request", cascade="all, delete-orphan")
