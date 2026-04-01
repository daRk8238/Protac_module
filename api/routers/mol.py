"""
Molecule Router
GET  /api/mol/svg?smiles=...        — returns SVG string for a SMILES
POST /api/mol/svg/batch             — returns SVGs for a list of SMILES
"""

import logging
from fastapi import APIRouter, Query
from fastapi.responses import Response
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


def smiles_to_svg(smiles: str, width: int = 300, height: int = 200) -> str | None:
    """Convert SMILES to SVG string using RDKit."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        from rdkit.Chem.Draw import rdMolDraw2D

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Clean up the molecule for display
        from rdkit.Chem import rdDepictor
        rdDepictor.Compute2DCoords(mol)

        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.drawOptions().addStereoAnnotation = True
        drawer.drawOptions().backgroundColour = (0.05, 0.08, 0.1, 1.0)  # dark bg

        # Atom colors for dark theme
        drawer.drawOptions().updateAtomPalette({
            6:  (0.82, 0.91, 1.0),    # C → light blue
            7:  (0.0,  0.9,  0.6),    # N → green
            8:  (1.0,  0.3,  0.4),    # O → red
            9:  (0.0,  0.9,  1.0),    # F → cyan
            16: (1.0,  0.84, 0.04),   # S → yellow
            17: (0.0,  0.9,  1.0),    # Cl → cyan
            35: (0.6,  0.2,  0.8),    # Br → purple
        })

        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        return svg

    except Exception as e:
        logger.debug(f"SVG generation failed for {smiles}: {e}")
        return None


@router.get("/svg")
async def get_mol_svg(
    smiles: str = Query(..., description="SMILES string"),
    width: int = Query(300, description="Image width in pixels"),
    height: int = Query(200, description="Image height in pixels"),
):
    """Return SVG image for a SMILES string."""
    svg = smiles_to_svg(smiles, width=width, height=height)
    if svg is None:
        return Response(
            content=_error_svg(width, height, "Invalid SMILES"),
            media_type="image/svg+xml"
        )
    return Response(content=svg, media_type="image/svg+xml")


class BatchSVGRequest(BaseModel):
    smiles: list[str]
    width: int = 280
    height: int = 180


@router.post("/svg/batch")
async def get_mol_svg_batch(request: BatchSVGRequest):
    """Return SVGs for a list of SMILES strings."""
    results = []
    for smi in request.smiles[:50]:  # limit to 50
        svg = smiles_to_svg(smi, width=request.width, height=request.height)
        results.append({
            "smiles": smi,
            "svg": svg,
            "valid": svg is not None,
        })
    return {"results": results}


def _error_svg(width: int, height: int, msg: str) -> str:
    """Return a placeholder SVG for invalid molecules."""
    return f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="#081425"/>
  <text x="50%" y="50%" text-anchor="middle" fill="#3d6e8a" 
        font-family="monospace" font-size="12">{msg}</text>
</svg>'''
