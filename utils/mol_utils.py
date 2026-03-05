"""
Molecule standardization, filtering, and featurization using RDKit.
All compounds pass through here before entering the database.
"""

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, inchi
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ─── Standardization ──────────────────────────────────────────────────────────

def standardize_smiles(smiles: str) -> str | None:
    """
    Canonicalize a SMILES string using RDKit.
    Returns None if the SMILES is invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception as e:
        logger.warning(f"Failed to standardize SMILES '{smiles}': {e}")
        return None


def smiles_to_inchikey(smiles: str) -> str | None:
    """Convert SMILES to InChIKey for deduplication."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        inchi_str = inchi.MolToInchi(mol)
        if inchi_str is None:
            return None
        return inchi.InchiToInchiKey(inchi_str)
    except Exception:
        return None


def smiles_to_inchi(smiles: str) -> str | None:
    """Convert SMILES to InChI string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return inchi.MolToInchi(mol)
    except Exception:
        return None


# ─── Physicochemical Properties ───────────────────────────────────────────────

def compute_properties(smiles: str) -> dict | None:
    """
    Compute all physicochemical properties for a molecule.
    Returns a dict ready to be stored in the Compound table.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        props = {
            "mol_weight": round(Descriptors.ExactMolWt(mol), 3),
            "logp": round(Descriptors.MolLogP(mol), 3),
            "hbd": rdMolDescriptors.CalcNumHBD(mol),
            "hba": rdMolDescriptors.CalcNumHBA(mol),
            "tpsa": round(Descriptors.TPSA(mol), 3),
            "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "num_stereocenters": len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        }
        props["passes_lipinski"] = check_lipinski(props)
        props["passes_veber"] = check_veber(props)
        return props
    except Exception as e:
        logger.warning(f"Property computation failed: {e}")
        return None


# ─── Drug-likeness Filters ────────────────────────────────────────────────────

def check_lipinski(props: dict) -> bool:
    """
    Lipinski Rule of Five — oral bioavailability filter for small molecules.
    PROTACs intentionally violate this; flag but don't exclude.
    """
    return (
        props["mol_weight"] <= 500 and
        props["logp"] <= 5 and
        props["hbd"] <= 5 and
        props["hba"] <= 10
    )


def check_veber(props: dict) -> bool:
    """
    Veber rules for oral bioavailability.
    Rotatable bonds <= 10 AND TPSA <= 140.
    """
    return props["rotatable_bonds"] <= 10 and props["tpsa"] <= 140


def check_pains(smiles: str) -> bool:
    """
    Check if molecule contains PAINS (Pan-Assay Interference) substructures.
    Returns True if molecule IS a PAINS (i.e., problematic).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog(params)
    return catalog.HasMatch(mol)


def check_aggregator(smiles: str) -> bool:
    """
    Simple heuristic for colloidal aggregators.
    More sophisticated: use the Shoichet aggregator predictor.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    # Heuristic: highly hydrophobic, low MW, aromatic-heavy → likely aggregator
    props = compute_properties(smiles)
    if props is None:
        return False
    return (
        props["logp"] > 4.5 and
        props["mol_weight"] < 350 and
        props["aromatic_rings"] >= 3
    )


# ─── Fingerprints & Features ──────────────────────────────────────────────────

def morgan_fingerprint(smiles: str, radius: int = 2, nbits: int = 2048) -> np.ndarray | None:
    """
    Compute Morgan (circular) fingerprint as a numpy bit array.
    This is the standard input for ML models in drug discovery.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits)
    return np.array(fp)


def rdkit_fingerprint(smiles: str, nbits: int = 2048) -> np.ndarray | None:
    """RDKit topological fingerprint — good complement to Morgan."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = Chem.RDKFingerprint(mol, fpSize=nbits)
    return np.array(fp)


def maccs_fingerprint(smiles: str) -> np.ndarray | None:
    """166-bit MACCS keys — interpretable, commonly used."""
    from rdkit.Chem import MACCSkeys
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(fp)


# ─── Full Processing Pipeline ─────────────────────────────────────────────────

def process_compound(smiles: str, source: str, source_id: str = None) -> dict | None:
    """
    Full processing pipeline for a single compound.
    Returns a dict ready to be inserted into the Compound table.
    Returns None if the molecule is invalid.
    """
    # 1. Standardize
    canonical_smiles = standardize_smiles(smiles)
    if canonical_smiles is None:
        logger.debug(f"Invalid SMILES skipped: {smiles}")
        return None

    # 2. Compute properties
    props = compute_properties(canonical_smiles)
    if props is None:
        return None

    # 3. Filters
    is_pains = check_pains(canonical_smiles)
    is_aggregator = check_aggregator(canonical_smiles)

    # 4. Build record
    return {
        "source": source,
        "source_id": source_id,
        "smiles": canonical_smiles,
        "inchi": smiles_to_inchi(canonical_smiles),
        "inchikey": smiles_to_inchikey(canonical_smiles),
        "is_pains": is_pains,
        "is_aggregator": is_aggregator,
        **props,
    }


def tanimoto_similarity(smiles1: str, smiles2: str) -> float | None:
    """Compute Tanimoto similarity between two molecules using Morgan FP."""
    from rdkit import DataStructs
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return None
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, 2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, 2048)
    return DataStructs.TanimotoSimilarity(fp1, fp2)
