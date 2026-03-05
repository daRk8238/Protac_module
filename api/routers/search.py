"""
Search Router
GET /api/search?q=egfr          — autocomplete suggestions
GET /api/search/resolve?q=egfr  — fully resolve a target identifier
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import APIRouter, Query
from utils.target_resolver import TargetResolver

router = APIRouter()
resolver = TargetResolver(organism_filter="Homo sapiens")


@router.get("")
async def search_targets(q: str = Query(..., min_length=2, description="Gene name, protein name, or UniProt ID")):
    """
    Autocomplete search for protein targets.
    Returns up to 10 suggestions from UniProt.

    Examples: EGFR, BRD4, sonic hedgehog, epidermal growth factor, P00533
    """
    suggestions = resolver.search_suggestions(q, max_results=10)
    return {
        "query": q,
        "total": len(suggestions),
        "suggestions": suggestions,
    }


@router.get("/resolve")
async def resolve_target(q: str = Query(..., min_length=2, description="Any protein identifier to resolve")):
    """
    Fully resolve a protein identifier to UniProt + ChEMBL IDs.
    Use this before submitting a pipeline job to confirm the target.

    Returns resolved identifiers and alternative matches if ambiguous.
    """
    resolved = resolver.resolve(q)
    return {
        "query": q,
        "found": resolved.found,
        "result": resolved.to_dict() if resolved.found else None,
        "message": (
            f"Resolved to {resolved.gene_name} ({resolved.uniprot_id})"
            if resolved.found
            else f"Could not resolve '{q}'. Try a UniProt ID (e.g. P00533) or full gene name."
        )
    }
