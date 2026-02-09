"""
FastAPI application factory for the Seldon Sandbox API.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from seldon.api.sessions import SessionManager
from seldon.api.routers import simulation, metrics, agents, experiments, settlements, network, advanced, llm


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="Seldon Sandbox API",
        description="REST API for the Seldon Sandbox simulation engine",
        version="0.5.0",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.state.session_manager = SessionManager()

    application.include_router(simulation.router, prefix="/api/simulation", tags=["simulation"])
    application.include_router(metrics.router, prefix="/api/metrics", tags=["metrics"])
    application.include_router(agents.router, prefix="/api/agents", tags=["agents"])
    application.include_router(experiments.router, prefix="/api/experiments", tags=["experiments"])
    application.include_router(settlements.router, prefix="/api/settlements", tags=["settlements"])
    application.include_router(network.router, prefix="/api/network", tags=["network"])
    application.include_router(advanced.router, prefix="/api/advanced", tags=["advanced"])
    application.include_router(llm.router, prefix="/api/llm", tags=["llm"])

    @application.get("/api/health")
    def health_check():
        return {"status": "ok"}

    return application


app = create_app()
