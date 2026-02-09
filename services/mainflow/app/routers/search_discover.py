from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import and_
from sqlalchemy.orm import Session

from services.mainflow.app.database.connections import get_db
from services.mainflow.app.schemas.search_discover_schema import (
    AssetInfo,
    DiscoverSearchRequest,
    SearchDiscoverResponse,
)
from shared.auth import get_current_user
from shared_migrations.models.discover import Discover

router = APIRouter(prefix="/mainflow", tags=["Assessment Search & Discover"])


@router.post("/assess", response_model=SearchDiscoverResponse)
def search_discover_projects(
    payload: DiscoverSearchRequest,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user_id = user.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID missing")

    base_filters = [Discover.user_id == user_id]

    if payload.registration_id:
        exists = (
            db.query(Discover.id)
            .filter(
                Discover.user_id == user_id,
                Discover.id == payload.registration_id,
            )
            .first()
        )

        if not exists:
            raise HTTPException(status_code=404, detail="No records found")

        base_filters.append(Discover.id == payload.registration_id)

    if payload.asset_type:
        asset_type_exists = (
            db.query(Discover.id).filter(Discover.user_id == user_id, Discover.asset_type == payload.asset_type).first()
        )

        if not asset_type_exists:
            raise HTTPException(status_code=404, detail="No such asset type found")

        base_filters.append(Discover.asset_type == payload.asset_type)

    if payload.project_name:
        project_exists = (
            db.query(Discover.id)
            .filter(and_(*base_filters, Discover.project_name.ilike(f"%{payload.project_name}%")))
            .first()
        )

        if not project_exists:
            raise HTTPException(status_code=404, detail="No such asset name found")

        base_filters.append(Discover.project_name.ilike(f"%{payload.project_name}%"))

    query = db.query(Discover).filter(and_(*base_filters)).order_by(Discover.created_at.desc())

    total_count = query.count()

    results = query.offset((payload.page - 1) * payload.limit).limit(payload.limit).all()

    if not results:
        raise HTTPException(status_code=404, detail="No discover records found")

    assets = [
        AssetInfo(
            registration_id=asset.id,
            asset_type=asset.asset_type,
            analysis_type=asset.analysis_type,
            asset_name=asset.project_name,
            version=getattr(asset, "version", None),
            framework=getattr(asset, "model_type", None),
            lifecycle_state=getattr(asset, "lifecycle_state", None),
            artifact_uri=getattr(asset, "uri", None),
            owner_team=getattr(asset, "owner", None),
            description=getattr(asset, "description", None),
            tags=getattr(asset, "tags", None),
            created_at=asset.created_at.isoformat() if isinstance(asset.created_at, datetime) else None,
        )
        for asset in results
    ]

    return SearchDiscoverResponse(count=total_count, data=assets)
