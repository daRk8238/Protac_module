"""
Generation Router (Phase 3A)
POST /api/generate/scaffold-hop   — generate analogs of a known active
GET  /api/generate/jobs/{job_id}  — poll generation job
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
router = APIRouter()


# ── Schemas ───────────────────────────────────────────────────────────────────

class ScaffoldHopRequest(BaseModel):
    smiles: str = Field(
        ...,
        description="SMILES of the reference active compound",
        examples=["CCOc1cc2ncnc(Nc3cccc(Cl)c3F)c2cc1OCC"]
    )
    target_uniprot: Optional[str] = Field(
        None,
        description="UniProt ID for novelty checking against DB",
        examples=["P00533"]
    )
    n_analogs: int = Field(50, ge=5, le=200, description="Number of analogs to return")
    min_similarity: float = Field(0.3, ge=0.1, le=0.9, description="Min Tanimoto to parent")
    max_similarity: float = Field(0.95, ge=0.5, le=1.0, description="Max Tanimoto to parent")
    methods: list[str] = Field(
        default=["brics", "bioisostere", "rgroup"],
        description="Generation methods to use"
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/scaffold-hop")
async def scaffold_hop(request: ScaffoldHopRequest, background_tasks: BackgroundTasks):
    """
    Generate analogs of a known active compound via scaffold hopping.

    Uses three complementary methods:
    - **BRICS**: Fragment + recombine at synthesizable bonds
    - **Bioisostere**: Swap functional groups for known equivalents
    - **R-group**: Enumerate substituents on the Murcko scaffold

    All candidates are scored by the GNN and ranked by predicted pChEMBL.
    Runs asynchronously — poll /api/jobs/{job_id} for results.
    """
    if not AppState.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python models/train.py` first."
        )

    # Validate SMILES immediately (fast check before queuing)
    from utils.mol_utils import standardize_smiles
    if standardize_smiles(request.smiles) is None:
        raise HTTPException(status_code=400, detail=f"Invalid SMILES: {request.smiles}")

    job_id = str(uuid.uuid4())
    job_store.create(job_id, job_type="scaffold_hop", meta=request.model_dump())
    background_tasks.add_task(_run_scaffold_hop, job_id, request)

    return {
        "job_id": job_id,
        "status": "pending",
        "message": f"Scaffold hopping started. Poll /api/jobs/{job_id} for results.",
        "input_smiles": request.smiles,
        "target_uniprot": request.target_uniprot,
    }


@router.post("/scaffold-hop/sync")
async def scaffold_hop_sync(request: ScaffoldHopRequest):
    """
    Synchronous version of scaffold hopping — waits and returns results directly.
    Best for small requests (n_analogs <= 30). Use async version for larger runs.
    """
    if not AppState.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    from utils.mol_utils import standardize_smiles
    if standardize_smiles(request.smiles) is None:
        raise HTTPException(status_code=400, detail=f"Invalid SMILES: {request.smiles}")

    from models.scaffold_hopper import ScaffoldHopper
    session = AppState.get_session()
    try:
        hopper = ScaffoldHopper(
            db_session=session,
            model=AppState.model,
            device=DEVICE,
        )
        results = hopper.generate(
            smiles=request.smiles,
            target_uniprot=request.target_uniprot,
            n_analogs=min(request.n_analogs, 30),  # cap sync at 30
            min_similarity=request.min_similarity,
            max_similarity=request.max_similarity,
            methods=request.methods,
        )
        return results
    finally:
        session.close()


# ── Background Task ───────────────────────────────────────────────────────────

def _run_scaffold_hop(job_id: str, request: ScaffoldHopRequest):
    """Background task: run scaffold hopping and store results."""
    job_store.update(job_id, status="running", progress="Initializing scaffold hopper...")

    try:
        from models.scaffold_hopper import ScaffoldHopper

        session = AppState.get_session()
        hopper = ScaffoldHopper(
            db_session=session,
            model=AppState.model,
            device=DEVICE,
        )

        job_store.update(job_id, progress="Generating analogs (BRICS + bioisostere + R-group)...")
        results = hopper.generate(
            smiles=request.smiles,
            target_uniprot=request.target_uniprot,
            n_analogs=request.n_analogs,
            min_similarity=request.min_similarity,
            max_similarity=request.max_similarity,
            methods=request.methods,
        )
        session.close()

        if "error" in results:
            job_store.fail(job_id, error=results["error"])
            return

        job_store.update(job_id, progress=f"Scoring {results['total_after_filter']} candidates...")
        job_store.complete(job_id, result=results)
        logger.info(f"[ScaffoldHop] Job {job_id} complete: "
                    f"{len(results.get('candidates', []))} candidates")

    except Exception as e:
        logger.error(f"[ScaffoldHop] Job {job_id} failed: {e}", exc_info=True)
        job_store.fail(job_id, error=str(e))
