from fastapi import APIRouter

router = APIRouter(prefix="/agentic", tags=["health"])


@router.get("/health", status_code=200)
def health_check():
    """Health check endpoint for Agentic AI service."""
    return {"status": "Agentic AI-healthy"}
