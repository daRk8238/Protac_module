"""
PROTAC Design Router (Phase 3B)
POST /api/generate/protac       — design PROTACs for a target (async)
POST /api/generate/protac/sync  — design PROTACs synchronously (small runs)
GET  /api/generate/protac/e3-ligases — list available E3 ligases
"""

import sys
import uuid
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional

from api.job_store import job_store
from api.state import AppState, DEVICE

logger = logging.getLogger(__name__)

# ── Register under existing generate router ───────────────────────────────────
# This file's router is mounted at /api/generate/protac in main.py
router = APIRouter()


# ── Schemas ───────────────────────────────────────────────────────────────────

class PROTACRequest(BaseModel):
    target_uniprot: str = Field(
        ...,
        description="UniProt ID of the target protein to degrade",
        examples=["P00533"]
    )
    e3_ligase: str = Field(
        "CRBN",
        description="E3 ligase to recruit (CRBN, VHL, MDM2, IAP)",
        examples=["CRBN"]
    )
    warhead_smiles: Optional[str] = Field(
        None,
        description="Known binder SMILES for target (auto-selected from DB if not provided)",
        examples=["CCOc1cc2ncnc(Nc3cccc(Cl)c3F)c2cc1OCC"]
    )
    n_designs: int = Field(20, ge=5, le=100)
    linker_types: list[str] = Field(
        default=["peg_short", "peg_long", "alkyl_short", "mixed"],
        description="Linker categories: peg_short, peg_long, alkyl_short, alkyl_long, rigid, mixed"
    )
    max_mw: float = Field(900.0, ge=500, le=1200, description="Max PROTAC molecular weight")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/e3-ligases")
async def list_e3_ligases():
    """List available E3 ligases and their known binders."""
    from models.protac_designer import E3_WARHEADS, LINKER_LIBRARY
    return {
        "e3_ligases": {
            name: [{"name": w["name"], "smiles": w["smiles"]} for w in warheads]
            for name, warheads in E3_WARHEADS.items()
        },
        "linker_types": list(LINKER_LIBRARY.keys()),
        "linker_counts": {k: len(v) for k, v in LINKER_LIBRARY.items()},
    }


@router.post("")
async def design_protac(request: PROTACRequest, background_tasks: BackgroundTasks):
    """
    Design PROTAC molecules for a target protein.

    Assembles warhead (target binder) + linker + E3 ligase binder
    into full PROTAC candidates, scored and ranked by predicted
    degradation potential.

    Runs asynchronously — poll /api/jobs/{job_id} for results.
    """
    if not AppState.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    # Validate E3 ligase
    from models.protac_designer import E3_WARHEADS
    if request.e3_ligase not in E3_WARHEADS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown E3 ligase '{request.e3_ligase}'. "
                   f"Available: {list(E3_WARHEADS.keys())}"
        )

    job_id = str(uuid.uuid4())
    job_store.create(job_id, job_type="protac_design", meta=request.model_dump())
    background_tasks.add_task(_run_protac_design, job_id, request)

    return {
        "job_id": job_id,
        "status": "pending",
        "message": f"PROTAC design started. Poll /api/jobs/{job_id} for results.",
        "target_uniprot": request.target_uniprot,
        "e3_ligase": request.e3_ligase,
    }


@router.post("/sync")
async def design_protac_sync(request: PROTACRequest):
    """
    Synchronous PROTAC design — returns results directly.
    Best for small runs (n_designs <= 20).
    """
    if not AppState.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    from models.protac_designer import PROTACDesigner
    session = AppState.get_session()
    try:
        designer = PROTACDesigner(
            db_session=session,
            model=AppState.model,
            device=DEVICE,
        )
        results = designer.design(
            target_uniprot=request.target_uniprot,
            e3_ligase=request.e3_ligase,
            warhead_smiles=request.warhead_smiles,
            n_designs=min(request.n_designs, 20),
            linker_types=request.linker_types,
            max_mw=request.max_mw,
        )
        return results
    finally:
        session.close()


# ── Background Task ───────────────────────────────────────────────────────────

def _run_protac_design(job_id: str, request: PROTACRequest):
    job_store.update(job_id, status="running", progress="Selecting warhead and E3 binders...")
    try:
        from models.protac_designer import PROTACDesigner
        session = AppState.get_session()
        designer = PROTACDesigner(
            db_session=session,
            model=AppState.model,
            device=DEVICE,
        )

        job_store.update(job_id, progress="Assembling PROTAC candidates...")
        results = designer.design(
            target_uniprot=request.target_uniprot,
            e3_ligase=request.e3_ligase,
            warhead_smiles=request.warhead_smiles,
            n_designs=request.n_designs,
            linker_types=request.linker_types,
            max_mw=request.max_mw,
        )
        session.close()

        if "error" in results:
            job_store.fail(job_id, error=results["error"])
            return

        job_store.complete(job_id, result=results)
        logger.info(f"[PROTAC] Job {job_id} complete: "
                    f"{len(results.get('designs', []))} designs")

    except Exception as e:
        logger.error(f"[PROTAC] Job {job_id} failed: {e}", exc_info=True)
        job_store.fail(job_id, error=str(e))
