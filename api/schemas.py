"""
Pydantic schemas for all API request and response models.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ── Prediction ────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    smiles: list[str] = Field(
        ...,
        description="List of SMILES strings to predict affinity for",
        min_length=1,
        max_length=500,
        examples=[["CCc1nn(C)c2cnc(Nc3cc(N4CCN(C)CC4)c(OC)cc3OC)nc12"]]
    )
    target_uniprot: Optional[str] = Field(
        None,
        description="UniProt ID of target (optional, used for context)",
        examples=["P00533"]
    )


class MoleculeResult(BaseModel):
    smiles: str
    predicted_pchembl: Optional[float]
    predicted_ic50_nM: Optional[float]       # converted from pChEMBL
    mol_weight: Optional[float]
    logp: Optional[float]
    passes_lipinski: Optional[bool]
    is_pains: Optional[bool]
    valid: bool
    error: Optional[str] = None


class PredictResponse(BaseModel):
    target_uniprot: Optional[str]
    total_submitted: int
    total_valid: int
    results: list[MoleculeResult]
    model_version: str = "0.1.0"


# ── Screening ─────────────────────────────────────────────────────────────────

class ScreenRequest(BaseModel):
    target_uniprot: str = Field(..., description="UniProt ID of target to screen against")
    top_n: int = Field(50, ge=1, le=500, description="Number of top candidates to return")
    min_pchembl: float = Field(5.0, ge=0, le=15, description="Minimum predicted pChEMBL threshold")
    exclude_pains: bool = Field(True, description="Filter out PAINS compounds")
    modulator_type: Optional[str] = Field(
        None,
        description="Filter by modulator type: inhibitor, activator, degrader",
    )


class ScreenResponse(BaseModel):
    job_id: str
    target_uniprot: str
    status: str
    message: str


# ── Pipeline ──────────────────────────────────────────────────────────────────

class PipelineRequest(BaseModel):
    target: str = Field(
        ...,
        description="Gene name or UniProt ID",
        examples=["BRD4", "P00533"]
    )
    max_activities: int = Field(1000, ge=100, le=10000)
    max_structures: int = Field(5, ge=1, le=20)
    fetch_zinc: bool = Field(True)
    zinc_count: int = Field(200, ge=50, le=5000)
    fetch_e3_binders: bool = Field(True)


class PipelineResponse(BaseModel):
    job_id: str
    target: str
    status: str
    message: str


# ── Jobs ──────────────────────────────────────────────────────────────────────

class JobStatus(BaseModel):
    job_id: str
    status: str                              # pending, running, complete, failed
    job_type: str                            # screen, pipeline
    created_at: str
    updated_at: str
    progress: Optional[str] = None
    error: Optional[str] = None


class JobResult(BaseModel):
    job_id: str
    status: str
    job_type: str
    result: Optional[dict] = None
    error: Optional[str] = None


# ── Targets ───────────────────────────────────────────────────────────────────

class TargetResponse(BaseModel):
    uniprot_id: str
    gene_name: Optional[str]
    protein_name: Optional[str]
    organism: Optional[str]
    pdb_id: Optional[str]
    binding_site_residue_count: int
    compound_count: int
    activity_count: int


class TargetListResponse(BaseModel):
    total: int
    targets: list[TargetResponse]
