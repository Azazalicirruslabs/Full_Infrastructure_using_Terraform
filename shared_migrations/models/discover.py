from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import relationship

from shared_migrations.models.base import Base


class Discover(Base):

    __tablename__ = "discover"

    __table_args__ = (UniqueConstraint("tenant_id", "project_name", name="uq_discover_tenant_project"),)

    id = Column(Integer, primary_key=True, index=True)
    asset_type = Column(String)
    analysis_type = Column(String, nullable=False, index=True)
    project_name = Column(String, nullable=False)
    version = Column(String, nullable=True)
    model_type = Column(String, nullable=True)
    lifecycle_state = Column(String, nullable=True)
    uri = Column(String, nullable=True)
    owner = Column(String, nullable=True)
    tags = Column(ARRAY(String), nullable=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    description = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=func.now())

    model_analysis_requests = relationship(
        "ModelAnalysisRequest", back_populates="discover", cascade="all, delete-orphan"
    )

    model_analysis_results = relationship(
        "ModelAnalysisResult", back_populates="discover", cascade="all, delete-orphan"
    )
