"""
Targets Router
GET /api/targets           — list all targets
GET /api/targets/{uniprot} — single target detail
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import APIRouter, HTTPException
from database.schema import TargetProtein, Bioactivity, Compound
from api.schemas import TargetResponse, TargetListResponse
from api.state import AppState
from sqlalchemy import func

router = APIRouter()


@router.get("", response_model=TargetListResponse)
async def list_targets():
    """List all protein targets currently in the database."""
    session = AppState.get_session()
    try:
        targets = session.query(TargetProtein).all()
        result = []
        for t in targets:
            activity_count = session.query(Bioactivity).filter_by(target_id=t.id).count()
            compound_count = session.query(Bioactivity).filter_by(
                target_id=t.id
            ).distinct(Bioactivity.compound_id).count()

            result.append(TargetResponse(
                uniprot_id=t.uniprot_id,
                gene_name=t.gene_name,
                protein_name=t.protein_name,
                organism=t.organism,
                pdb_id=t.pdb_id,
                binding_site_residue_count=len(t.binding_site_residues or []),
                compound_count=compound_count,
                activity_count=activity_count,
            ))
        return TargetListResponse(total=len(result), targets=result)
    finally:
        session.close()


@router.get("/{uniprot_id}", response_model=TargetResponse)
async def get_target(uniprot_id: str):
    """Get details for a specific target by UniProt ID."""
    session = AppState.get_session()
    try:
        target = session.query(TargetProtein).filter_by(uniprot_id=uniprot_id).first()
        if not target:
            raise HTTPException(status_code=404, detail=f"Target {uniprot_id} not found in database")

        activity_count = session.query(Bioactivity).filter_by(target_id=target.id).count()
        compound_count = session.query(Bioactivity).filter_by(
            target_id=target.id
        ).distinct(Bioactivity.compound_id).count()

        return TargetResponse(
            uniprot_id=target.uniprot_id,
            gene_name=target.gene_name,
            protein_name=target.protein_name,
            organism=target.organism,
            pdb_id=target.pdb_id,
            binding_site_residue_count=len(target.binding_site_residues or []),
            compound_count=compound_count,
            activity_count=activity_count,
        )
    finally:
        session.close()


@router.get("/{uniprot_id}/binding-site")
async def get_binding_site(uniprot_id: str):
    """Get binding site residues for a target."""
    session = AppState.get_session()
    try:
        target = session.query(TargetProtein).filter_by(uniprot_id=uniprot_id).first()
        if not target:
            raise HTTPException(status_code=404, detail=f"Target {uniprot_id} not found")
        return {
            "uniprot_id": uniprot_id,
            "pdb_id": target.pdb_id,
            "binding_site_residues": target.binding_site_residues or [],
            "residue_count": len(target.binding_site_residues or []),
        }
    finally:
        session.close()


@router.get("/{uniprot_id}/top-compounds")
async def get_top_compounds(uniprot_id: str, limit: int = 20, min_pchembl: float = 6.0):
    """Get top known active compounds for a target from the database."""
    session = AppState.get_session()
    try:
        target = session.query(TargetProtein).filter_by(uniprot_id=uniprot_id).first()
        if not target:
            raise HTTPException(status_code=404, detail=f"Target {uniprot_id} not found")

        activities = (
            session.query(Bioactivity)
            .filter(
                Bioactivity.target_id == target.id,
                Bioactivity.pchembl_value >= min_pchembl,
            )
            .order_by(Bioactivity.pchembl_value.desc())
            .limit(limit)
            .all()
        )

        results = []
        for act in activities:
            c = act.compound
            results.append({
                "smiles": c.smiles,
                "source_id": c.source_id,
                "pchembl_value": act.pchembl_value,
                "activity_type": act.activity_type,
                "modulator_type": act.modulator_type,
                "mol_weight": c.mol_weight,
                "logp": c.logp,
                "passes_lipinski": c.passes_lipinski,
            })

        return {
            "uniprot_id": uniprot_id,
            "gene_name": target.gene_name,
            "total": len(results),
            "compounds": results,
        }
    finally:
        session.close()
