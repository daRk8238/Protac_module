"""
Scoring & Ranking Engine (Phase 4)
------------------------------------
Comprehensive molecular scoring pipeline:
  1. Desalting        — strip salts/counterions, keep largest fragment
  2. SAScore          — synthetic accessibility (RDKit contribution)
  3. QED              — quantitative estimate of drug-likeness
  4. ADMET filters    — hERG, BBB, CYP, solubility (rule-based)
  5. Novelty score    — Tanimoto distance from DB compounds
  6. Composite score  — weighted combination → final ranking

Usage:
    scorer = MoleculeScorer(db_session)
    result = scorer.score(smiles)
    results = scorer.score_batch(smiles_list)
    ranked = scorer.rank(results)
"""

import logging
import numpy as np
from typing import Optional

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED, RDConfig
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import AllChem

from utils.mol_utils import (
    standardize_smiles, compute_properties, check_pains,
    smiles_to_inchikey, morgan_fingerprint
)

logger = logging.getLogger(__name__)

# ── SAScore Setup ─────────────────────────────────────────────────────────────
# RDKit's SA_Score contribution
try:
    import sys, os
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer
    SA_SCORE_AVAILABLE = True
    logger.info("[Scorer] SAScore loaded successfully")
except ImportError:
    SA_SCORE_AVAILABLE = False
    logger.warning("[Scorer] SAScore not available — using heuristic fallback")


# ── Composite Score Weights ───────────────────────────────────────────────────
SCORE_WEIGHTS = {
    "qed":          0.25,   # drug-likeness
    "sa_score":     0.20,   # synthetic accessibility
    "admet":        0.25,   # ADMET profile
    "novelty":      0.15,   # novelty vs known compounds
    "affinity":     0.15,   # predicted pChEMBL (if available)
}


class MoleculeScorer:
    def __init__(self, db_session=None):
        self.session = db_session
        self._db_fingerprints = None  # lazy loaded

    # ── Main Entry Points ──────────────────────────────────────────────────────

    def score(self, smiles: str, predicted_pchembl: float = None) -> dict:
        """
        Score a single molecule across all dimensions.
        Returns a full scorecard dict.
        """
        # Step 1: Desalt and standardize
        clean_smiles = self.desalt(smiles)
        if clean_smiles is None:
            return {"smiles": smiles, "valid": False, "error": "Invalid or unsalvageable SMILES"}

        mol = Chem.MolFromSmiles(clean_smiles)
        if mol is None:
            return {"smiles": smiles, "valid": False, "error": "RDKit parse failed"}

        # Step 2: Properties
        props = compute_properties(clean_smiles) or {}

        # Step 3: Individual scores
        qed_score = self._compute_qed(mol)
        sa_score = self._compute_sa_score(mol)
        admet = self._compute_admet(mol, props)
        novelty = self._compute_novelty(clean_smiles)

        # Step 4: Composite score
        composite = self._compute_composite(
            qed_score, sa_score, admet["admet_score"],
            novelty["novelty_score"], predicted_pchembl
        )

        return {
            "smiles": clean_smiles,
            "original_smiles": smiles if smiles != clean_smiles else None,
            "was_desalted": smiles != clean_smiles,
            "valid": True,
            "inchikey": smiles_to_inchikey(clean_smiles),

            # Properties
            **props,

            # Individual scores
            "qed_score": qed_score,
            "sa_score": sa_score,
            "sa_score_normalized": self._normalize_sa(sa_score),

            # ADMET
            **admet,

            # Novelty
            **novelty,

            # Affinity
            "predicted_pchembl": predicted_pchembl,
            "predicted_ic50_nM": round((10 ** -predicted_pchembl) * 1e9, 2) if predicted_pchembl else None,

            # Final composite
            "composite_score": composite,
        }

    def score_batch(self, smiles_list: list, predicted_pchembl_list: list = None) -> list:
        """Score a list of molecules. Returns list of scorecard dicts."""
        results = []
        for i, smi in enumerate(smiles_list):
            pchembl = predicted_pchembl_list[i] if predicted_pchembl_list else None
            result = self.score(smi, predicted_pchembl=pchembl)
            results.append(result)
        return results

    def rank(self, scored_results: list, by: str = "composite_score") -> list:
        """Sort scored results by a given field, descending."""
        valid = [r for r in scored_results if r.get("valid", False)]
        invalid = [r for r in scored_results if not r.get("valid", False)]
        valid.sort(key=lambda x: x.get(by, 0) or 0, reverse=True)
        return valid + invalid

    # ── Step 1: Desalting ─────────────────────────────────────────────────────

    def desalt(self, smiles: str) -> Optional[str]:
        """
        Remove salts and counterions, keep the largest organic fragment.
        Handles common issues: .Cl, .Na, .HCl, .TFA, etc.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # Use RDKit's built-in fragment chooser
            # Keeps the largest fragment by atom count
            chooser = rdMolStandardize.LargestFragmentChooser()
            clean_mol = chooser.choose(mol)

            if clean_mol is None:
                return None

            # Also remove common counterions manually
            uncharger = rdMolStandardize.Uncharger()
            clean_mol = uncharger.uncharge(clean_mol)

            canon = Chem.MolToSmiles(clean_mol, canonical=True)
            return canon if canon else None

        except Exception as e:
            logger.debug(f"Desalting failed for {smiles}: {e}")
            # Fallback: split on '.' and take longest fragment
            try:
                fragments = smiles.split(".")
                if len(fragments) > 1:
                    longest = max(fragments, key=len)
                    return standardize_smiles(longest)
                return standardize_smiles(smiles)
            except Exception:
                return None

    # ── Step 2: QED ───────────────────────────────────────────────────────────

    def _compute_qed(self, mol) -> Optional[float]:
        """
        Quantitative Estimate of Drug-likeness (QED).
        Score 0-1 where 1 = most drug-like.
        Based on Bickerton et al. Nature Chemistry 2012.
        """
        try:
            return round(QED.qed(mol), 4)
        except Exception:
            return None

    # ── Step 3: SAScore ───────────────────────────────────────────────────────

    def _compute_sa_score(self, mol) -> Optional[float]:
        """
        Synthetic Accessibility Score (1-10).
        1 = trivially synthesizable
        10 = practically impossible to synthesize
        Lower is better.
        """
        if SA_SCORE_AVAILABLE:
            try:
                return round(sascorer.calculateScore(mol), 3)
            except Exception:
                pass

        # Heuristic fallback if sascorer not available
        return self._sa_score_heuristic(mol)

    def _sa_score_heuristic(self, mol) -> float:
        """
        Heuristic SA score based on molecular complexity.
        Not as accurate as RDKit's sascorer but good enough for ranking.
        """
        try:
            mw = Descriptors.ExactMolWt(mol)
            rings = rdMolDescriptors.CalcNumRings(mol)
            stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
            rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
            bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
            spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)

            # Penalty-based score
            score = 1.0
            score += min(mw / 500, 3.0)          # MW penalty
            score += rings * 0.3                  # ring complexity
            score += stereo * 0.5                 # stereochemistry
            score += bridgehead * 1.0             # bridgehead atoms
            score += spiro * 0.8                  # spiro atoms
            score += max(0, rot_bonds - 10) * 0.1 # excessive flexibility

            return round(min(score, 10.0), 3)
        except Exception:
            return 5.0  # neutral default

    def _normalize_sa(self, sa_score: float) -> Optional[float]:
        """Normalize SA score to 0-1 where 1 = most synthesizable."""
        if sa_score is None:
            return None
        return round(1.0 - (sa_score - 1.0) / 9.0, 4)

    # ── Step 4: ADMET ─────────────────────────────────────────────────────────

    def _compute_admet(self, mol, props: dict) -> dict:
        """
        Rule-based ADMET profiling.
        More accurate alternatives: SwissADME API, ADMETlab API.
        """
        mw = props.get("mol_weight", 0)
        logp = props.get("logp", 0)
        hbd = props.get("hbd", 0)
        hba = props.get("hba", 0)
        tpsa = props.get("tpsa", 0)
        rot_bonds = props.get("rotatable_bonds", 0)

        flags = {}
        penalties = 0

        # ── Absorption ────────────────────────────────────────────────────────
        # Oral bioavailability (Lipinski + Veber)
        flags["oral_bioavailability"] = (
            mw <= 500 and logp <= 5 and hbd <= 5 and
            hba <= 10 and tpsa <= 140 and rot_bonds <= 10
        )
        if not flags["oral_bioavailability"]:
            penalties += 1

        # GI absorption (Boiled-egg model approximation)
        flags["gi_absorption"] = tpsa < 140 and logp < 5
        if not flags["gi_absorption"]:
            penalties += 1

        # ── Distribution ──────────────────────────────────────────────────────
        # BBB penetration (CNS drugs need this, others may not)
        flags["bbb_penetrant"] = (
            mw < 450 and logp > 0 and logp < 6 and
            tpsa < 90 and hbd <= 3
        )

        # ── Metabolism ────────────────────────────────────────────────────────
        # CYP3A4 inhibition risk (Molar refractivity + logP heuristic)
        try:
            mr = Descriptors.MolMR(mol)
            flags["cyp3a4_inhibitor_risk"] = logp > 3.5 and mr > 40
        except Exception:
            flags["cyp3a4_inhibitor_risk"] = None

        # ── Toxicity ──────────────────────────────────────────────────────────
        # hERG cardiotoxicity risk (basic nitrogen + logP)
        basic_nitrogens = sum(
            1 for atom in mol.GetAtoms()
            if atom.GetAtomicNum() == 7 and
            atom.GetTotalNumHs() > 0 and
            not atom.GetIsAromatic()
        )
        flags["herg_risk"] = logp > 3.7 and basic_nitrogens >= 1
        if flags["herg_risk"]:
            penalties += 2  # higher penalty — cardiotoxicity is serious

        # Mutagenicity (Ames test — aromatic amines, nitro groups)
        ames_patterns = [
            Chem.MolFromSmarts("[NH2]c"),           # aromatic amine
            Chem.MolFromSmarts("[N+](=O)[O-]"),      # nitro group
            Chem.MolFromSmarts("c1ccc2ccccc2c1"),    # polycyclic aromatic
        ]
        flags["mutagenicity_risk"] = any(
            mol.HasSubstructMatch(p) for p in ames_patterns if p
        )
        if flags["mutagenicity_risk"]:
            penalties += 2

        # ── Solubility ────────────────────────────────────────────────────────
        # ESOL approximation (Delaney model)
        try:
            log_sw = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * hba - 0.74
            flags["solubility_class"] = (
                "high" if log_sw > -2 else
                "moderate" if log_sw > -4 else
                "low"
            )
            flags["log_sw"] = round(log_sw, 3)
            if flags["solubility_class"] == "low":
                penalties += 1
        except Exception:
            flags["solubility_class"] = "unknown"
            flags["log_sw"] = None

        # ── ADMET Score ───────────────────────────────────────────────────────
        # 0-1 where 1 = clean ADMET profile
        max_penalties = 7
        admet_score = round(max(0, 1.0 - penalties / max_penalties), 4)

        return {
            "admet_flags": flags,
            "admet_penalties": penalties,
            "admet_score": admet_score,
        }

    # ── Step 5: Novelty ───────────────────────────────────────────────────────

    def _compute_novelty(self, smiles: str) -> dict:
        """
        Novelty score based on maximum Tanimoto similarity to DB compounds.
        novelty_score = 1 - max_similarity (lower similarity = more novel)
        """
        try:
            fp = morgan_fingerprint(smiles)
            if fp is None:
                return {"novelty_score": 0.5, "max_similarity_to_db": None}

            db_fps = self._get_db_fingerprints()
            if not db_fps:
                return {"novelty_score": 1.0, "max_similarity_to_db": 0.0}

            from rdkit import DataStructs
            from rdkit.Chem import AllChem

            mol = Chem.MolFromSmiles(smiles)
            query_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)

            similarities = DataStructs.BulkTanimotoSimilarity(query_fp, db_fps)
            max_sim = max(similarities) if similarities else 0.0

            novelty_score = round(1.0 - max_sim, 4)
            return {
                "novelty_score": novelty_score,
                "max_similarity_to_db": round(max_sim, 4),
            }
        except Exception as e:
            logger.debug(f"Novelty computation failed: {e}")
            return {"novelty_score": 0.5, "max_similarity_to_db": None}

    def _get_db_fingerprints(self):
        """Lazy-load and cache DB fingerprints for novelty calculation."""
        if self._db_fingerprints is not None:
            return self._db_fingerprints

        if self.session is None:
            return []

        try:
            from database.schema import Compound
            from rdkit.Chem import AllChem

            compounds = self.session.query(Compound.smiles).limit(500).all()
            fps = []
            for (smi,) in compounds:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))

            self._db_fingerprints = fps
            logger.info(f"[Scorer] Loaded {len(fps)} DB fingerprints for novelty")
            return fps
        except Exception as e:
            logger.warning(f"[Scorer] DB fingerprint load failed: {e}")
            return []

    # ── Step 6: Composite Score ───────────────────────────────────────────────

    def _compute_composite(
        self,
        qed: float,
        sa_score: float,
        admet_score: float,
        novelty_score: float,
        predicted_pchembl: float = None,
    ) -> float:
        """
        Weighted composite score (0-1) combining all dimensions.
        Higher = better candidate.
        """
        scores = {}

        # QED already 0-1
        scores["qed"] = qed or 0.0

        # SA score: normalize 1-10 → 1-0
        scores["sa_score"] = self._normalize_sa(sa_score) or 0.5

        # ADMET already 0-1
        scores["admet"] = admet_score or 0.5

        # Novelty already 0-1
        scores["novelty"] = novelty_score or 0.5

        # Affinity: normalize pChEMBL 5-10 → 0-1
        if predicted_pchembl is not None:
            scores["affinity"] = max(0, min(1, (predicted_pchembl - 5.0) / 5.0))
        else:
            scores["affinity"] = 0.5  # neutral if no prediction

        # Weighted sum
        composite = sum(
            SCORE_WEIGHTS[k] * scores[k]
            for k in SCORE_WEIGHTS
        )

        return round(composite, 4)
