"""
Scoring Router (Phase 4)
POST /api/score          — score a list of SMILES
POST /api/score/rank     — score + rank with full scorecard
GET  /api/score/explain  — explain scoring methodology
"""

import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from api.state import AppState

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Schemas ───────────────────────────────────────────────────────────────────

class ScoreRequest(BaseModel):
    smiles: list[str] = Field(
        ..., min_length=1, max_length=200,
        description="List of SMILES to score",
        examples=[["CCOc1cc2ncnc(Nc3cccc(Cl)c3F)c2cc1OCC"]]
    )
    predicted_pchembl: Optional[list[float]] = Field(
        None,
        description="Optional predicted pChEMBL values (from /api/predict)"
    )
    rank_by: str = Field(
        "composite_score",
        description="Field to rank by: composite_score, qed_score, sa_score, admet_score, novelty_score"
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("")
async def score_molecules(request: ScoreRequest):
    """
    Score molecules across multiple dimensions:
    - **QED** — drug-likeness (0-1, higher=better)
    - **SA Score** — synthetic accessibility (1-10, lower=better)
    - **ADMET** — absorption, distribution, metabolism, excretion, toxicity
    - **Novelty** — distance from known compounds in DB
    - **Composite** — weighted combination of all above

    Also performs **desalting** — strips counterions and salts automatically.
    """
    from models.scorer import MoleculeScorer

    session = AppState.get_session()
    try:
        scorer = MoleculeScorer(db_session=session)
        pchembl_list = request.predicted_pchembl

        results = scorer.score_batch(request.smiles, pchembl_list)
        ranked = scorer.rank(results, by=request.rank_by)

        valid = [r for r in ranked if r.get("valid")]
        invalid = [r for r in ranked if not r.get("valid")]

        return {
            "total_submitted": len(request.smiles),
            "total_valid": len(valid),
            "total_invalid": len(invalid),
            "ranked_by": request.rank_by,
            "results": ranked,
        }
    finally:
        session.close()


@router.post("/single")
async def score_single(smiles: str, predicted_pchembl: Optional[float] = None):
    """Score a single molecule and return its full scorecard."""
    from models.scorer import MoleculeScorer

    session = AppState.get_session()
    try:
        scorer = MoleculeScorer(db_session=session)
        result = scorer.score(smiles, predicted_pchembl=predicted_pchembl)
        return result
    finally:
        session.close()


@router.get("/explain")
async def explain_scoring():
    """Explain the scoring methodology and weights."""
    from models.scorer import SCORE_WEIGHTS
    return {
        "methodology": "Weighted composite score combining drug-likeness, synthesizability, ADMET profile, novelty, and predicted affinity",
        "score_range": "0-1 (higher = better candidate)",
        "weights": SCORE_WEIGHTS,
        "components": {
            "qed_score": {
                "description": "Quantitative Estimate of Drug-likeness (Bickerton et al. 2012)",
                "range": "0-1",
                "higher_is_better": True,
            },
            "sa_score": {
                "description": "Synthetic Accessibility Score",
                "range": "1-10 (1=trivial, 10=impossible)",
                "higher_is_better": False,
                "normalized_field": "sa_score_normalized (0-1, higher=easier)"
            },
            "admet_score": {
                "description": "ADMET profile score based on rule-based filters",
                "range": "0-1",
                "higher_is_better": True,
                "flags": [
                    "oral_bioavailability", "gi_absorption", "bbb_penetrant",
                    "cyp3a4_inhibitor_risk", "herg_risk",
                    "mutagenicity_risk", "solubility_class"
                ]
            },
            "novelty_score": {
                "description": "1 - max Tanimoto similarity to DB compounds",
                "range": "0-1",
                "higher_is_better": True,
            },
            "predicted_pchembl": {
                "description": "GNN-predicted binding affinity",
                "range": "typically 5-10",
                "higher_is_better": True,
            },
        }
    }
