"""
FastAPI application factory for the Seldon Sandbox API.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from seldon.api.sessions import SessionManager
from seldon.api.routers import simulation, metrics, agents, experiments, settlements, network, advanced, llm, social, communities, economics, environment, genetics, beliefs, inner_life

# Load .env — try project root first, then CWD (handles Docker volume mount)
_project_root = Path(__file__).resolve().parents[3]  # src/seldon/api/app.py → project root
load_dotenv(_project_root / ".env")
load_dotenv(Path.cwd() / ".env")  # fallback for Docker /app/.env


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="Seldon Sandbox API",
        description="REST API for the Seldon Sandbox simulation engine",
        version="0.6.0",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    db_path = os.environ.get("SELDON_DB_PATH", "data/seldon.db")
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    application.state.session_manager = SessionManager(db_path=db_path)

    application.include_router(simulation.router, prefix="/api/simulation", tags=["simulation"])
    application.include_router(metrics.router, prefix="/api/metrics", tags=["metrics"])
    application.include_router(agents.router, prefix="/api/agents", tags=["agents"])
    application.include_router(experiments.router, prefix="/api/experiments", tags=["experiments"])
    application.include_router(settlements.router, prefix="/api/settlements", tags=["settlements"])
    application.include_router(network.router, prefix="/api/network", tags=["network"])
    application.include_router(advanced.router, prefix="/api/advanced", tags=["advanced"])
    application.include_router(llm.router, prefix="/api/llm", tags=["llm"])
    application.include_router(social.router, prefix="/api/social", tags=["social"])
    application.include_router(communities.router, prefix="/api/communities", tags=["communities"])
    application.include_router(economics.router, prefix="/api/economics", tags=["economics"])
    application.include_router(environment.router, prefix="/api/environment", tags=["environment"])
    application.include_router(genetics.router, prefix="/api/genetics", tags=["genetics"])
    application.include_router(beliefs.router, prefix="/api/beliefs", tags=["beliefs"])
    application.include_router(inner_life.router, prefix="/api/inner-life", tags=["inner-life"])

    @application.get("/api/health")
    def health_check():
        return {"status": "ok"}

    return application


app = create_app()
