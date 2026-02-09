from typing import List, Optional

from pydantic import BaseModel, Field


class DiscoverSearchRequest(BaseModel):
    registration_id: Optional[int] = None
    asset_type: Optional[str] = Field(None, description="Filter by asset type")
    project_name: Optional[str] = Field(None, description="Search by project name")
    page: int = Field(1, ge=1, description="Page number")
    limit: int = Field(10, ge=1, le=100, description="Records per page")


class AssetInfo(BaseModel):
    registration_id: Optional[int]
    asset_type: Optional[str]
    analysis_type: Optional[str]
    asset_name: Optional[str]
    version: Optional[str]
    framework: Optional[str]
    lifecycle_state: Optional[str]
    artifact_uri: Optional[str]
    owner_team: Optional[str]
    description: Optional[str]
    tags: Optional[List[str]]
    created_at: Optional[str]


class SearchDiscoverResponse(BaseModel):
    count: int
    data: List[AssetInfo]
