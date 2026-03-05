"""
Prediction Router
POST /api/predict  — run GNN inference on submitted SMILES
POST /api/screen   — screen full compound DB against a target (async)
"""

import sys
import math
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from fastapi import APIRouter, HTTPException, BackgroundTasks

from api.schemas import PredictRequest, PredictResponse, MoleculeResult, ScreenRequest, ScreenResponse
from api.state import AppState, DEVICE
from api.job_store import job_store
from models.featurizer import smiles_to_graph
from utils.mol_utils import compute_properties, check_pains, standardize_smiles

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Core prediction function ──────────────────────────────────────────────────

def predict_single(smiles: str) -> dict:
    """
    Run GNN inference on a single SMILES string.
    Returns prediction dict with affinity + properties.
    """
    # Standardize
    canon = standardize_smiles(smiles)
    if canon is None:
        return {"valid": False, "smiles": smiles, "error": "Invalid SMILES"}

    # Compute properties
    props = compute_properties(canon)
    if props is None:
        return {"valid": False, "smiles": smiles, "error": "Property computation failed"}

    is_pains = check_pains(canon)

    result = {
        "smiles": canon,
        "valid": True,
        "mol_weight": props.get("mol_weight"),
        "logp": props.get("logp"),
        "passes_lipinski": props.get("passes_lipinski"),
        "is_pains": is_pains,
        "predicted_pchembl": None,
        "predicted_ic50_nM": None,
        "error": None,
    }

    # GNN prediction
    if AppState.model_loaded and AppState.model is not None:
        try:
            graph = smiles_to_graph(canon, label=None)
            if graph is None:
                result["error"] = "Graph conversion failed"
                return result

            graph = graph.to(DEVICE)

            # Add batch dimension
            from torch_geometric.data import Batch
            batch = Batch.from_data_list([graph])

            with torch.no_grad():
                fp = batch.fingerprint.view(1, -1)
                pred = AppState.model(batch, fp)
                pchembl = float(pred.item())

            # Convert pChEMBL to IC50 in nM: IC50(M) = 10^(-pChEMBL), then × 1e9 for nM
            ic50_nM = (10 ** (-pchembl)) * 1e9

            result["predicted_pchembl"] = round(pchembl, 3)
            result["predicted_ic50_nM"] = round(ic50_nM, 2)

        except Exception as e:
            logger.error(f"Prediction error for {canon}: {e}")
            result["error"] = f"Prediction failed: {str(e)}"

    return result


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Predict binding affinity for a list of SMILES strings.

    Returns predicted pChEMBL values and physicochemical properties
    for each molecule, sorted by predicted affinity (highest first).
    """
    if not AppState.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python models/train.py` first."
        )

    results = []
    for smi in request.smiles:
        pred = predict_single(smi)
        results.append(MoleculeResult(**pred))

    # Sort by predicted pChEMBL descending (None values last)
    results.sort(
        key=lambda r: r.predicted_pchembl if r.predicted_pchembl is not None else -999,
        reverse=True
    )

    valid_count = sum(1 for r in results if r.valid)

    return PredictResponse(
        target_uniprot=request.target_uniprot,
        total_submitted=len(request.smiles),
        total_valid=valid_count,
        results=results,
    )


@router.post("/screen", response_model=ScreenResponse)
async def screen_library(request: ScreenRequest, background_tasks: BackgroundTasks):
    """
    Screen the full compound library in the database against a target.
    Runs asynchronously — returns a job_id to poll for results.
    """
    from database.schema import TargetProtein
    session = AppState.get_session()
    try:
        target = session.query(TargetProtein).filter_by(
            uniprot_id=request.target_uniprot
        ).first()
        if not target:
            raise HTTPException(
                status_code=404,
                detail=f"Target {request.target_uniprot} not found. Run pipeline first."
            )
    finally:
        session.close()

    import uuid
    job_id = str(uuid.uuid4())
    job_store.create(job_id, job_type="screen", meta=request.model_dump())
    background_tasks.add_task(_run_screen, job_id, request)

    return ScreenResponse(
        job_id=job_id,
        target_uniprot=request.target_uniprot,
        status="pending",
        message=f"Screening job started. Poll /api/jobs/{job_id} for status.",
    )


# ── Background screening task ─────────────────────────────────────────────────

def _run_screen(job_id: str, request: ScreenRequest):
    """Background task: screen all DB compounds and rank by predicted affinity."""
    job_store.update(job_id, status="running", progress="Loading compounds...")

    try:
        from database.schema import Compound, Bioactivity, TargetProtein

        session = AppState.get_session()
        compounds = session.query(Compound).all()
        job_store.update(job_id, progress=f"Scoring {len(compounds)} compounds...")

        results = []
        for compound in compounds:
            if request.exclude_pains and compound.is_pains:
                continue

            pred = predict_single(compound.smiles)
            if not pred["valid"]:
                continue

            pchembl = pred.get("predicted_pchembl")
            if pchembl is None or pchembl < request.min_pchembl:
                continue

            results.append({
                "smiles": compound.smiles,
                "source": compound.source,
                "source_id": compound.source_id,
                "compound_role": compound.compound_role,
                "predicted_pchembl": pchembl,
                "predicted_ic50_nM": pred.get("predicted_ic50_nM"),
                "mol_weight": compound.mol_weight,
                "logp": compound.logp,
                "passes_lipinski": compound.passes_lipinski,
                "is_pains": compound.is_pains,
            })

        session.close()

        # Sort and take top N
        results.sort(key=lambda r: r["predicted_pchembl"], reverse=True)
        top_results = results[:request.top_n]

        job_store.complete(job_id, result={
            "target_uniprot": request.target_uniprot,
            "total_screened": len(compounds),
            "total_passing": len(results),
            "top_n": len(top_results),
            "candidates": top_results,
        })

        logger.info(f"[Screen] Job {job_id} complete: {len(top_results)} candidates")

    except Exception as e:
        logger.error(f"[Screen] Job {job_id} failed: {e}")
        job_store.fail(job_id, error=str(e))
