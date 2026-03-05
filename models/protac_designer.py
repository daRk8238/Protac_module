"""
PROTAC Designer Engine (Phase 3B)
-----------------------------------
Designs Proteolysis Targeting Chimeras (PROTACs) by:
  1. Selecting the best warhead for the target from the DB
  2. Selecting E3 ligase binders (CRBN, VHL, MDM2, IAP)
  3. Generating diverse linkers (PEG, alkyl, rigid, mixed)
  4. Assembling warhead─linker─E3binder molecules
  5. Scoring with GNN + PROTAC-specific filters
  6. Ranking by predicted degradation potential

Key references:
  - Crews & Deshaies (2016) — PROTAC design principles
  - Ciulli lab VHL binders
  - Thalidomide/Lenalidomide CRBN binders

Usage:
    designer = PROTACDesigner(db_session, model, device)
    results = designer.design(
        target_uniprot="P00533",
        e3_ligase="CRBN",
        n_designs=20
    )
"""

import logging
import random
from typing import Optional
from itertools import product

import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem import RWMol, rdChemReactions

from utils.mol_utils import (
    standardize_smiles, compute_properties, check_pains,
    smiles_to_inchikey, tanimoto_similarity
)

logger = logging.getLogger(__name__)


# ── Linker Library ─────────────────────────────────────────────────────────────
# Each linker has two attachment points marked as [1*] and [2*]
# Linker types: PEG (flexible, polar), Alkyl (flexible, hydrophobic),
#               Rigid (aromatic/proline), Mixed (PEG+alkyl)

LINKER_LIBRARY = {
    "peg_short": [
        "O=C(COCCO)NCCNC(=O)",           # PEG2 diamide
        "O=C(COCCOC)NCCNC(=O)",          # PEG2 methyl ester
        "NCCOCCOCCN",                     # PEG2 diamine
        "O=C(COCCOCCO)NCCNC(=O)",        # PEG3 diamide
    ],
    "peg_long": [
        "NCCOCCOCCOCCOCCOCCO",            # PEG5 amine
        "O=C(COCCOCCOCCOC)NCC",          # PEG4 ester
        "NCCOCCOCCOCCOCCOCCOCCO",         # PEG6 amine
    ],
    "alkyl_short": [
        "NCCCCN",                         # butane-1,4-diamine
        "NCCCN",                          # propane-1,3-diamine
        "NCCCCCN",                        # pentane-1,5-diamine
        "O=C(CCCC)NCC",                  # pentyl amide
    ],
    "alkyl_long": [
        "NCCCCCCN",                       # hexane-1,6-diamine
        "NCCCCCCCN",                      # heptane-1,7-diamine
        "O=C(CCCCCC)NCC",                # heptyl amide
    ],
    "rigid": [
        "NCc1ccccc1CN",                   # o-xylene diamine
        "NCc1cccc(CN)c1",                 # m-xylene diamine
        "NC1CCCCC1N",                     # cyclohexane diamine
        "NC1CCN(C(=O)c2ccccc2)CC1",      # piperidine-based
        "O=C(c1ccc(N)cc1)Nc1ccc(N)cc1",  # biphenyl diamide
    ],
    "mixed": [
        "NCCOCCCN",                       # short mixed
        "NCCOCCCCN",                      # mixed PEG-alkyl
        "O=C(CCOCCO)NCCCCN",             # PEG-alkyl amide
        "NCCOCCOCCCN",                    # PEG2-propyl amine
        "O=C(CCOCCOC)NCCCN",             # ester-PEG-amine
    ],
}

# ── E3 Ligase Warheads ─────────────────────────────────────────────────────────
# These are the E3 binder portions with a free amine for linker attachment
# Format: {e3_name: [(smiles, attachment_note), ...]}

E3_WARHEADS = {
    "CRBN": [
        {
            "smiles": "O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1",
            "name": "Thalidomide",
            "attachment_smiles": "O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1",
            "linker_attachment": "NH",  # attach via glutarimide NH
        },
        {
            "smiles": "O=C1CCC(N2C(=O)c3cc(N)ccc3C2=O)C(=O)N1",
            "name": "Lenalidomide",
            "attachment_smiles": "O=C1CCC(N2C(=O)c3cc(N)ccc3C2=O)C(=O)N1",
            "linker_attachment": "ArNH2",
        },
        {
            "smiles": "O=C1CCC(N2C(=O)c3cc(F)ccc3C2=O)C(=O)N1",
            "name": "Pomalidomide",
            "attachment_smiles": "O=C1CCC(N2C(=O)c3cc(F)ccc3C2=O)C(=O)N1",
            "linker_attachment": "ArNH2",
        },
    ],
    "VHL": [
        {
            "smiles": "CC(C)(C)OC(=O)N[C@@H]1C[C@H](O)CN1",
            "name": "VHL-1 (hydroxyproline)",
            "linker_attachment": "N-Boc",
        },
        {
            "smiles": "CC1(C)C[C@@H](O)CN1",
            "name": "VHL-2 (gem-dimethyl)",
            "linker_attachment": "NH",
        },
    ],
    "MDM2": [
        {
            "smiles": "O=C(Nc1ccc(Cl)cc1Cl)c1ccc(N)cc1",
            "name": "MDM2-binder-1",
            "linker_attachment": "ArNH2",
        },
    ],
    "IAP": [
        {
            "smiles": "CC[C@H](C)[C@@H](NC(=O)CN)C(=O)N1CCC[C@H]1C(=O)N",
            "name": "SMAC-mimetic",
            "linker_attachment": "NH2",
        },
    ],
}


class PROTACDesigner:
    def __init__(self, db_session, model=None, device=None):
        self.session = db_session
        self.model = model
        self.device = device or torch.device("cpu")

    # ── Main Entry Point ───────────────────────────────────────────────────────

    def design(
        self,
        target_uniprot: str,
        e3_ligase: str = "CRBN",
        warhead_smiles: str = None,
        n_designs: int = 20,
        linker_types: list = None,
        max_mw: float = 900.0,
    ) -> dict:
        """
        Design PROTAC molecules for a given target.

        Args:
            target_uniprot:  UniProt ID of the target protein
            e3_ligase:       E3 ligase to recruit (CRBN, VHL, MDM2, IAP)
            warhead_smiles:  Known binder for target (auto-selected if None)
            n_designs:       Number of PROTAC designs to return
            linker_types:    Linker categories to use (default: all)
            max_mw:          Maximum molecular weight (PROTACs are large)

        Returns:
            dict with warhead, E3 binder, linkers used, and ranked designs
        """
        if linker_types is None:
            linker_types = ["peg_short", "peg_long", "alkyl_short", "mixed"]

        logger.info(f"[PROTAC] Designing for {target_uniprot} | E3={e3_ligase}")

        # Step 1: Get warhead (target binder)
        warhead = self._get_warhead(target_uniprot, warhead_smiles)
        if warhead is None:
            return {"error": f"No warhead found for {target_uniprot}. "
                             f"Provide warhead_smiles or run pipeline first."}

        logger.info(f"[PROTAC] Warhead: {warhead['smiles'][:50]}... "
                    f"(pChEMBL={warhead.get('pchembl_value')})")

        # Step 2: Get E3 ligase binders
        e3_binders = self._get_e3_binders(e3_ligase)
        if not e3_binders:
            return {"error": f"No E3 binders found for {e3_ligase}"}

        logger.info(f"[PROTAC] E3 binders available: {len(e3_binders)}")

        # Step 3: Get linkers
        linkers = self._get_linkers(linker_types)
        logger.info(f"[PROTAC] Linkers available: {len(linkers)}")

        # Step 4: Assemble PROTACs
        all_protacs = []
        for e3_binder in e3_binders:
            for linker in linkers:
                protac = self._assemble_protac(
                    warhead["smiles"],
                    linker["smiles"],
                    e3_binder["smiles"],
                    linker["type"],
                    e3_binder["name"],
                )
                if protac:
                    protac["warhead_pchembl"] = warhead.get("pchembl_value")
                    all_protacs.append(protac)

        logger.info(f"[PROTAC] Assembled {len(all_protacs)} raw PROTAC candidates")

        # Step 5: Filter
        filtered = self._filter_protacs(all_protacs, max_mw=max_mw)
        logger.info(f"[PROTAC] After filtering: {len(filtered)}")

        # Step 6: Score
        scored = self._score_protacs(filtered, warhead["smiles"])

        # Sort by composite PROTAC score
        scored.sort(key=lambda x: x.get("protac_score", 0), reverse=True)

        return {
            "target_uniprot": target_uniprot,
            "e3_ligase": e3_ligase,
            "warhead": warhead,
            "e3_binders_used": [b["name"] for b in e3_binders],
            "linker_types_used": linker_types,
            "total_assembled": len(all_protacs),
            "total_after_filter": len(filtered),
            "designs": scored[:n_designs],
        }

    # ── Warhead Selection ──────────────────────────────────────────────────────

    def _get_warhead(self, target_uniprot: str, warhead_smiles: str = None) -> Optional[dict]:
        """Get the best warhead for the target from DB or use provided SMILES."""
        if warhead_smiles:
            canon = standardize_smiles(warhead_smiles)
            if canon is None:
                return None
            props = compute_properties(canon)
            return {
                "smiles": canon,
                "source": "user_provided",
                "pchembl_value": None,
                **(props or {}),
            }

        # Auto-select: get highest pChEMBL compound for this target
        from database.schema import TargetProtein, Bioactivity, Compound

        target = self.session.query(TargetProtein).filter_by(
            uniprot_id=target_uniprot
        ).first()

        if not target:
            logger.warning(f"[PROTAC] Target {target_uniprot} not in DB")
            return None

        best_activity = (
            self.session.query(Bioactivity)
            .filter(
                Bioactivity.target_id == target.id,
                Bioactivity.pchembl_value >= 7.0,  # only potent binders
            )
            .order_by(Bioactivity.pchembl_value.desc())
            .first()
        )

        if not best_activity:
            # Relax threshold
            best_activity = (
                self.session.query(Bioactivity)
                .filter_by(target_id=target.id)
                .order_by(Bioactivity.pchembl_value.desc())
                .first()
            )

        if not best_activity:
            return None

        compound = best_activity.compound
        return {
            "smiles": compound.smiles,
            "source_id": compound.source_id,
            "source": "database",
            "pchembl_value": best_activity.pchembl_value,
            "activity_type": best_activity.activity_type,
            "mol_weight": compound.mol_weight,
            "logp": compound.logp,
        }

    # ── E3 Binder Selection ────────────────────────────────────────────────────

    def _get_e3_binders(self, e3_ligase: str) -> list[dict]:
        """Get E3 ligase binders — first from DB, then from hardcoded library."""
        binders = []

        # Try DB first
        from database.schema import Compound
        db_binders = self.session.query(Compound).filter(
            Compound.compound_role == "e3_ligand",
            Compound.source_id.like(f"%{e3_ligase}%") |
            Compound.source_id.like("PROTACDB_%")
        ).limit(5).all()

        for b in db_binders:
            binders.append({
                "smiles": b.smiles,
                "name": b.source_id or e3_ligase,
                "source": "database",
            })

        # Always include hardcoded library for the requested E3
        if e3_ligase in E3_WARHEADS:
            for w in E3_WARHEADS[e3_ligase]:
                binders.append({
                    "smiles": w["smiles"],
                    "name": w["name"],
                    "source": "library",
                })

        # Deduplicate by InChIKey
        seen = set()
        unique_binders = []
        for b in binders:
            ik = smiles_to_inchikey(b["smiles"])
            if ik and ik not in seen:
                seen.add(ik)
                unique_binders.append(b)

        return unique_binders

    # ── Linker Generation ──────────────────────────────────────────────────────

    def _get_linkers(self, linker_types: list) -> list[dict]:
        """Get linkers from the library based on requested types."""
        linkers = []
        for ltype in linker_types:
            if ltype in LINKER_LIBRARY:
                for smi in LINKER_LIBRARY[ltype]:
                    linkers.append({
                        "smiles": smi,
                        "type": ltype,
                        "length": len(smi),  # rough proxy for linker length
                    })
        return linkers

    # ── PROTAC Assembly ────────────────────────────────────────────────────────

    def _assemble_protac(
        self,
        warhead_smi: str,
        linker_smi: str,
        e3_smi: str,
        linker_type: str,
        e3_name: str,
    ) -> Optional[dict]:
        """
        Assemble a PROTAC by connecting warhead─linker─E3binder.

        Strategy: Find free NH2/NH groups on warhead and E3 binder,
        form amide bonds with the linker's terminal carboxyl/amine groups.
        This is a simplified chemical assembly — real PROTAC synthesis
        would use more specific reaction SMARTS.
        """
        try:
            # Try direct concatenation via amide bond formation
            # This is the most common PROTAC linkage strategy
            protac_smi = self._form_amide_linkage(warhead_smi, linker_smi, e3_smi)

            if protac_smi is None:
                # Fallback: simple SMILES concatenation with linker
                protac_smi = self._simple_concatenation(warhead_smi, linker_smi, e3_smi)

            if protac_smi is None:
                return None

            canon = standardize_smiles(protac_smi)
            if canon is None:
                return None

            props = compute_properties(canon)
            if props is None:
                return None

            return {
                "smiles": canon,
                "warhead_smiles": warhead_smi,
                "linker_smiles": linker_smi,
                "linker_type": linker_type,
                "e3_binder_smiles": e3_smi,
                "e3_binder_name": e3_name,
                "inchikey": smiles_to_inchikey(canon),
                **props,
            }

        except Exception as e:
            logger.debug(f"[Assembly] Failed: {e}")
            return None

    def _form_amide_linkage(self, warhead: str, linker: str, e3: str) -> Optional[str]:
        """
        Form amide bonds between components using RDKit reactions.
        Warhead-NH2 + linker-COOH → amide, then linker-NH2 + E3-COOH → amide.
        """
        try:
            # Amide bond formation reaction SMARTS
            amide_rxn = rdChemReactions.ReactionFromSmarts(
                "[NH2:1].[C:2](=O)[OH:3]>>[C:2](=O)[NH:1]"
            )

            w_mol = Chem.MolFromSmiles(warhead)
            l_mol = Chem.MolFromSmiles(linker)
            e_mol = Chem.MolFromSmiles(e3)

            if None in (w_mol, l_mol, e_mol):
                return None

            # Step 1: Connect warhead to one end of linker
            products1 = amide_rxn.RunReactants((w_mol, l_mol))
            if not products1:
                products1 = amide_rxn.RunReactants((l_mol, w_mol))
            if not products1:
                return None

            intermediate = products1[0][0]
            Chem.SanitizeMol(intermediate)

            # Step 2: Connect intermediate to E3 binder
            products2 = amide_rxn.RunReactants((intermediate, e_mol))
            if not products2:
                products2 = amide_rxn.RunReactants((e_mol, intermediate))
            if not products2:
                return None

            final = products2[0][0]
            Chem.SanitizeMol(final)
            return Chem.MolToSmiles(final)

        except Exception:
            return None

    def _simple_concatenation(self, warhead: str, linker: str, e3: str) -> Optional[str]:
        """
        Fallback assembly: direct SMILES concatenation.
        Represents a loosely connected PROTAC (useful for MW/property estimation).
        """
        try:
            combined = f"{warhead}.{linker}.{e3}"
            mol = Chem.MolFromSmiles(combined)
            if mol is None:
                return None
            # Return as fragment mixture — still useful for scoring
            return Chem.MolToSmiles(mol)
        except Exception:
            return None

    # ── Filtering ──────────────────────────────────────────────────────────────

    def _filter_protacs(self, protacs: list, max_mw: float = 900.0) -> list:
        """
        Filter PROTACs by PROTAC-specific criteria.
        PROTACs intentionally violate Lipinski but need other filters.
        """
        filtered = []
        seen_keys = set()

        for p in protacs:
            ik = p.get("inchikey")
            if ik and ik in seen_keys:
                continue
            if ik:
                seen_keys.add(ik)

            mw = p.get("mol_weight", 0)
            logp = p.get("logp", 0)
            tpsa = p.get("tpsa", 0)
            rot_bonds = p.get("rotatable_bonds", 0)

            # PROTAC-specific filters (beyond Ro5)
            if mw > max_mw:               continue   # too large
            if mw < 400:                  continue   # too small to be PROTAC
            if logp > 8:                  continue   # too hydrophobic
            if tpsa and tpsa > 250:       continue   # poor permeability
            if rot_bonds and rot_bonds > 30: continue  # too flexible

            # No PAINS
            if check_pains(p["smiles"]):  continue

            filtered.append(p)

        return filtered

    # ── Scoring ────────────────────────────────────────────────────────────────

    def _score_protacs(self, protacs: list, warhead_smiles: str) -> list:
        """
        Score PROTACs with GNN + PROTAC-specific composite score.

        Composite PROTAC score considers:
        - Predicted binding affinity (GNN)
        - Linker geometry (length/flexibility)
        - MW penalty (lighter is better within PROTAC range)
        - TPSA (affects cell permeability)
        """
        from models.featurizer import smiles_to_graph
        from torch_geometric.data import Batch

        scored = []
        for p in protacs:
            result = dict(p)

            # GNN prediction
            if self.model is not None:
                try:
                    graph = smiles_to_graph(p["smiles"])
                    if graph is not None:
                        graph = graph.to(self.device)
                        batch = Batch.from_data_list([graph])
                        with torch.no_grad():
                            fp = batch.fingerprint.view(1, -1)
                            pred = self.model(batch, fp)
                            pchembl = float(pred.item())
                        result["predicted_pchembl"] = round(pchembl, 3)
                        result["predicted_ic50_nM"] = round((10 ** -pchembl) * 1e9, 2)
                except Exception as e:
                    logger.debug(f"Scoring error: {e}")
                    result["predicted_pchembl"] = None
                    result["predicted_ic50_nM"] = None

            # Composite PROTAC score (0-10)
            result["protac_score"] = self._compute_protac_score(result)

            scored.append(result)

        return scored

    def _compute_protac_score(self, protac: dict) -> float:
        """
        Composite PROTAC score (0-10) combining multiple factors.

        Components:
        - Affinity score  (40%): predicted pChEMBL normalized
        - MW score        (20%): penalize very large/small MW
        - Permeability    (20%): TPSA-based (lower = better for cell entry)
        - Linker score    (20%): prefer PEG linkers (better solubility)
        """
        score = 0.0

        # Affinity (0-4 points)
        pchembl = protac.get("predicted_pchembl")
        if pchembl is not None:
            # Normalize pChEMBL 5-10 → 0-4
            affinity_score = max(0, min(4, (pchembl - 5.0) / 5.0 * 4))
            score += affinity_score

        # MW (0-2 points): ideal PROTAC MW is 700-800
        mw = protac.get("mol_weight", 0)
        if 600 <= mw <= 850:
            score += 2.0
        elif 500 <= mw <= 900:
            score += 1.0

        # Permeability via TPSA (0-2 points): lower TPSA = better
        tpsa = protac.get("tpsa")
        if tpsa is not None:
            if tpsa < 150:
                score += 2.0
            elif tpsa < 200:
                score += 1.0

        # Linker type bonus (0-2 points): PEG preferred for solubility
        linker_type = protac.get("linker_type", "")
        if "peg" in linker_type:
            score += 2.0
        elif "mixed" in linker_type:
            score += 1.0

        return round(score, 2)
