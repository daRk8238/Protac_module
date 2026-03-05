"""
PROTAC Designer — FastAPI Webserver
------------------------------------
REST API wrapping the GNN prediction engine and data pipeline.

Run:
    uvicorn api.main:app --reload --port 8000

Endpoints:
    GET  /                        Health + version info
    GET  /api/health              Health check
    GET  /api/targets             List all targets in DB
    GET  /api/targets/{uniprot}   Get single target details
    POST /api/predict             Predict affinity for SMILES list
    POST /api/screen              Screen compound library against target
    POST /api/pipeline/run        Trigger data pipeline for new target
    GET  /api/jobs/{job_id}       Poll async job status
    GET  /api/results/{job_id}    Get job results
"""

import sys
import uuid
import logging
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routers import targets, predict, pipeline, jobs, search, generate, protac, score
from api.state import AppState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App Lifespan (startup/shutdown) ───────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared resources on startup."""
    logger.info("[Server] Starting PROTAC Designer API...")
    AppState.initialize()
    logger.info(f"[Server] Model loaded: {AppState.model_loaded}")
    logger.info(f"[Server] DB compounds: {AppState.compound_count}")
    yield
    logger.info("[Server] Shutting down...")


# ── App Instance ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="PROTAC Designer API",
    description="""
AI-driven modulator and PROTAC design pipeline.

Predicts binding affinity for small molecules against protein targets,
suggests PROTAC candidates with E3 ligase warheads, and ranks results
by predicted activity, drug-likeness, and synthetic accessibility.
    """,
    version="0.1.0",
    lifespan=lifespan,
)

# ── CORS (allow frontend dev server) ─────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(targets.router,  prefix="/api/targets",  tags=["Targets"])
app.include_router(predict.router,  prefix="/api",          tags=["Prediction"])
app.include_router(pipeline.router, prefix="/api/pipeline", tags=["Pipeline"])
app.include_router(jobs.router,     prefix="/api/jobs",     tags=["Jobs"])
app.include_router(search.router,   prefix="/api/search",   tags=["Search"])
app.include_router(generate.router, prefix="/api/generate",      tags=["Generation"])
app.include_router(protac.router,   prefix="/api/generate/protac", tags=["PROTAC Design"])
app.include_router(score.router,    prefix="/api/score",           tags=["Scoring"])


# ── Root & Health ─────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
async def root():
    return {
        "name": "PROTAC Designer API",
        "version": "0.1.0",
        "status": "running",
        "model_loaded": AppState.model_loaded,
        "docs": "/docs",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/health", tags=["Info"])
async def health():
    return {
        "status": "healthy",
        "model_loaded": AppState.model_loaded,
        "db_compounds": AppState.compound_count,
        "db_targets": AppState.target_count,
        "db_activities": AppState.activity_count,
    }
