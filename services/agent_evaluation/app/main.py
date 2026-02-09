import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.agent_evaluation.app.routers import health
from services.agent_evaluation.app.utils import logging_config

logging_config.setup_logging()

logger = logging.getLogger(__name__)
logger.info("Logging is configured for Agentic AI service.")

app = FastAPI(
    title="Welcome to Agentic AI Service",
    description="Service for managing Agentic AI requests.",
    version="3.1.0",
    docs_url="/agentic/docs",
    openapi_url="/agentic/openapi.json",
    redoc_url="/agentic/redocs",
)

# Allow CORS for all origins (you can customize this for your needs)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
