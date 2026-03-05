"""
Scaffold Hopping Engine (Phase 3A)
------------------------------------
Generates structurally diverse analogs of a known active compound using:
  1. BRICS fragmentation + recombination
  2. Bioisosteric replacements (common medicinal chemistry transformations)
  3. R-group enumeration on Murcko scaffold
  4. GNN scoring + ranking of all candidates

Usage:
    hopper = ScaffoldHopper(db_session, model, device)
    results = hopper.generate(
        smiles="CCOc1cc2ncnc(Nc3cccc(Cl)c3F)c2cc1OCC",  # Erlotinib
        target_uniprot="P00533",
        n_analogs=50
    )
"""

import logging
import random
from typing import Optional
from itertools import combinations

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS, AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import RWMol, rdMolDescriptors

from utils.mol_utils import (
    standardize_smiles, compute_properties, check_pains,
    smiles_to_inchikey, morgan_fingerprint, tanimoto_similarity
)

logger = logging.getLogger(__name__)


# ── Bioisostere Replacement Rules ─────────────────────────────────────────────
# Format: (SMARTS_to_replace, replacement_SMILES_list)
# These are the most common medicinal chemistry transformations

BIOISOSTERE_RULES = [
    # Carboxylic acid → tetrazole, hydroxamic acid, sulfonamide
    ("[CX3](=O)[OH]", ["c1nnn[nH]1", "C(=O)NO", "S(=O)(=O)N"]),

    # Ester → amide
    ("[CX3](=O)O[CX4]", ["C(=O)N"]),

    # Phenyl → pyridine, pyrimidine, thiophene
    ("c1ccccc1", ["c1ccncc1", "c1ccncc1", "c1ccsc1"]),

    # F → Cl, Br → F (halogen swap)
    ("[F]", ["[Cl]", "[Br]"]),
    ("[Cl]", ["[F]", "[Br]"]),

    # NH → O, S (heteroatom swap)
    ("[NH]", ["[O]", "[S]"]),

    # CH2-CH2 → CH=CH (saturation change)
    ("[CX4H2][CX4H2]", ["[CH]=[CH]"]),

    # Methyl → ethyl, cyclopropyl
    ("[CH3]", ["[CH2][CH3]", "C1CC1"]),

    # OH → NH2, F (hydroxyl bioisosteres)
    ("[OH]", ["[NH2]", "[F]"]),

    # Amide NH → N-methyl
    ("[NH]C(=O)", ["[N](C)C(=O)"]),
]


class ScaffoldHopper:
    def __init__(self, db_session, model=None, device=None):
        self.session = db_session
        self.model = model
        self.device = device or torch.device("cpu")

    # ── Main Entry Point ───────────────────────────────────────────────────────

    def generate(
        self,
        smiles: str,
        target_uniprot: str = None,
        n_analogs: int = 50,
        min_similarity: float = 0.3,   # minimum Tanimoto to parent
        max_similarity: float = 0.95,  # avoid near-duplicates
        methods: list = None,
    ) -> dict:
        """
        Generate and score analogs of a known active compound.

        Args:
            smiles:          SMILES of the reference active compound
            target_uniprot:  UniProt ID for context (used to check novelty vs DB)
            n_analogs:       Number of analogs to generate before filtering
            min_similarity:  Minimum Tanimoto similarity to parent (keeps relevance)
            max_similarity:  Maximum Tanimoto similarity (avoids near-duplicates)
            methods:         List of methods to use (default: all)

        Returns:
            dict with scaffold info, analogs list, and scoring results
        """
        if methods is None:
            methods = ["brics", "bioisostere", "rgroup"]

        # Validate input
        canon = standardize_smiles(smiles)
        if canon is None:
            return {"error": f"Invalid SMILES: {smiles}"}

        mol = Chem.MolFromSmiles(canon)
        if mol is None:
            return {"error": "Could not parse molecule"}

        logger.info(f"[Hopper] Generating analogs for: {canon}")
        logger.info(f"[Hopper] Methods: {methods}, Target: {n_analogs} analogs")

        # Extract scaffold
        scaffold = self._get_murcko_scaffold(canon)
        logger.info(f"[Hopper] Murcko scaffold: {scaffold}")

        # Generate candidates from each method
        all_candidates = set()

        if "brics" in methods:
            brics_analogs = self._brics_analogs(canon, n=n_analogs)
            all_candidates.update(brics_analogs)
            logger.info(f"[Hopper] BRICS generated: {len(brics_analogs)}")

        if "bioisostere" in methods:
            bio_analogs = self._bioisostere_analogs(canon, n=n_analogs // 2)
            all_candidates.update(bio_analogs)
            logger.info(f"[Hopper] Bioisostere generated: {len(bio_analogs)}")

        if "rgroup" in methods:
            rgroup_analogs = self._rgroup_analogs(canon, n=n_analogs // 2)
            all_candidates.update(rgroup_analogs)
            logger.info(f"[Hopper] R-group generated: {len(rgroup_analogs)}")

        logger.info(f"[Hopper] Total raw candidates: {len(all_candidates)}")

        # Filter candidates
        filtered = self._filter_candidates(
            list(all_candidates),
            parent_smiles=canon,
            min_similarity=min_similarity,
            max_similarity=max_similarity,
            target_uniprot=target_uniprot,
        )
        logger.info(f"[Hopper] After filtering: {len(filtered)}")

        # Score with GNN
        scored = self._score_candidates(filtered, parent_smiles=canon)

        # Sort by predicted pChEMBL
        scored.sort(
            key=lambda x: x.get("predicted_pchembl") or 0,
            reverse=True
        )

        return {
            "parent_smiles": canon,
            "parent_scaffold": scaffold,
            "methods_used": methods,
            "total_generated": len(all_candidates),
            "total_after_filter": len(filtered),
            "candidates": scored[:n_analogs],
        }

    # ── Scaffold Extraction ────────────────────────────────────────────────────

    def _get_murcko_scaffold(self, smiles: str) -> Optional[str]:
        """Extract the Murcko scaffold (ring systems + linkers, no substituents)."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold)
        except Exception:
            return None

    # ── Generation Methods ─────────────────────────────────────────────────────

    def _brics_analogs(self, smiles: str, n: int = 100) -> set:
        """
        BRICS (Breaking Retrosynthetically Interesting Chemical Substructures)
        Fragments the molecule at synthesizable bonds, then recombines fragments
        from the database to generate novel but synthesizable analogs.
        """
        candidates = set()
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return candidates

        try:
            # Fragment the parent molecule
            fragments = list(BRICS.BRICSDecompose(mol))
            logger.debug(f"[BRICS] {len(fragments)} fragments from parent")

            if not fragments:
                return candidates

            # Get additional fragments from DB compounds
            db_fragments = self._get_db_fragments(max_frags=200)
            all_fragments = list(set(fragments) | db_fragments)

            # Recombine fragments to make new molecules
            # Try random pairwise combinations
            random.shuffle(all_fragments)
            frag_mols = []
            for frag_smi in all_fragments[:50]:
                frag_mol = Chem.MolFromSmiles(frag_smi)
                if frag_mol:
                    frag_mols.append(frag_mol)

            if len(frag_mols) >= 2:
                new_mols = BRICS.BRICSBuild(frag_mols)
                count = 0
                for new_mol in new_mols:
                    if count >= n:
                        break
                    try:
                        smi = Chem.MolToSmiles(new_mol)
                        canon = standardize_smiles(smi)
                        if canon and canon != smiles:
                            candidates.add(canon)
                            count += 1
                    except Exception:
                        continue

        except Exception as e:
            logger.warning(f"[BRICS] Failed: {e}")

        return candidates

    def _get_db_fragments(self, max_frags: int = 200) -> set:
        """Extract BRICS fragments from compounds already in the database."""
        from database.schema import Compound
        fragments = set()
        try:
            compounds = self.session.query(Compound).limit(100).all()
            for c in compounds:
                mol = Chem.MolFromSmiles(c.smiles)
                if mol is None:
                    continue
                try:
                    frags = BRICS.BRICSDecompose(mol)
                    fragments.update(frags)
                    if len(fragments) >= max_frags:
                        break
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"[DB Fragments] Failed: {e}")
        return fragments

    def _bioisostere_analogs(self, smiles: str, n: int = 50) -> set:
        """
        Apply bioisosteric replacement rules to generate analogs.
        Each rule swaps a functional group for a bioisosteric equivalent.
        """
        candidates = set()
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return candidates

        for smarts, replacements in BIOISOSTERE_RULES:
            if len(candidates) >= n:
                break
            try:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern is None:
                    continue
                if not mol.HasSubstructMatch(pattern):
                    continue

                for replacement in replacements:
                    try:
                        rep_mol = Chem.MolFromSmiles(replacement)
                        if rep_mol is None:
                            continue

                        # Use RDKit's ReplaceSubstructs
                        new_mols = AllChem.ReplaceSubstructs(mol, pattern, rep_mol)
                        for new_mol in new_mols:
                            try:
                                Chem.SanitizeMol(new_mol)
                                smi = Chem.MolToSmiles(new_mol)
                                canon = standardize_smiles(smi)
                                if canon and canon != smiles:
                                    candidates.add(canon)
                            except Exception:
                                continue
                    except Exception:
                        continue
            except Exception:
                continue

        return candidates

    def _rgroup_analogs(self, smiles: str, n: int = 50) -> set:
        """
        R-group enumeration: identify attachment points on the Murcko scaffold
        and enumerate common substituents at each position.
        """
        candidates = set()

        # Common R-groups to try at substituent positions
        r_groups = [
            "C", "CC", "CCC", "C(C)C",           # alkyl chains
            "CF", "CCl", "CBr",                    # halogenated
            "CO", "CN", "CS",                      # heteroatom-bearing
            "c1ccccc1", "c1ccncc1",                # aryl
            "C(=O)N", "C(=O)O", "S(=O)(=O)N",     # polar
            "OC", "NC", "SC",                      # ether/amine/thioether
            "C1CC1", "C1CCC1", "C1CCCC1",          # cycloalkyl
            "C#N", "N=[N+]=[N-]",                  # unusual groups
        ]

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return candidates

            # Get scaffold and find attachment points
            scaffold_smi = self._get_murcko_scaffold(smiles)
            if scaffold_smi is None:
                return candidates

            scaffold = Chem.MolFromSmiles(scaffold_smi)
            if scaffold is None:
                return candidates

            # Find atoms in mol NOT in scaffold (these are R-groups)
            match = mol.GetSubstructMatch(scaffold)
            if not match:
                return candidates

            scaffold_atom_indices = set(match)
            attachment_points = []

            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if (i in scaffold_atom_indices) != (j in scaffold_atom_indices):
                    # This bond connects scaffold to R-group
                    attachment_points.append(i if i in scaffold_atom_indices else j)

            if not attachment_points:
                return candidates

            # Try substituting at each attachment point
            random.shuffle(r_groups)
            for r_smi in r_groups[:20]:
                if len(candidates) >= n:
                    break
                try:
                    # Build a modified molecule by changing the substituent
                    # Simple approach: replace terminal atoms
                    rw_mol = RWMol(mol)
                    for ap in attachment_points[:2]:  # limit to 2 attachment points
                        atom = rw_mol.GetAtomWithIdx(ap)
                        if atom.GetDegree() == 1:  # terminal atom
                            # Replace with R-group
                            r_mol = Chem.MolFromSmiles(r_smi)
                            if r_mol:
                                new_mols = AllChem.ReplaceSubstructs(
                                    mol,
                                    Chem.MolFromSmarts(f"[#{atom.GetAtomicNum()}]"),
                                    r_mol,
                                    replaceAll=False
                                )
                                for nm in new_mols[:3]:
                                    try:
                                        Chem.SanitizeMol(nm)
                                        smi = Chem.MolToSmiles(nm)
                                        canon = standardize_smiles(smi)
                                        if canon and canon != smiles:
                                            candidates.add(canon)
                                    except Exception:
                                        continue
                except Exception:
                    continue

        except Exception as e:
            logger.warning(f"[R-group] Failed: {e}")

        return candidates

    # ── Filtering ──────────────────────────────────────────────────────────────

    def _filter_candidates(
        self,
        candidates: list,
        parent_smiles: str,
        min_similarity: float,
        max_similarity: float,
        target_uniprot: str = None,
    ) -> list:
        """
        Filter candidates by:
        - Valid SMILES
        - Tanimoto similarity range vs parent
        - Drug-likeness (MW < 700, not PAINS)
        - Not already in DB for this target
        """
        from database.schema import Compound

        # Get InChIKeys already in DB to avoid exact duplicates
        existing_keys = set()
        try:
            db_compounds = self.session.query(Compound.inchikey).all()
            existing_keys = {row[0] for row in db_compounds if row[0]}
        except Exception:
            pass

        filtered = []
        seen_inchikeys = set()

        for smi in candidates:
            try:
                # Validity
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue

                # Dedup
                ik = smiles_to_inchikey(smi)
                if ik in seen_inchikeys:
                    continue
                seen_inchikeys.add(ik)

                # Similarity to parent
                sim = tanimoto_similarity(smi, parent_smiles)
                if sim is None or sim < min_similarity or sim > max_similarity:
                    continue

                # Basic drug-likeness
                mw = Descriptors.ExactMolWt(mol)
                if mw > 700 or mw < 100:
                    continue

                # PAINS filter
                if check_pains(smi):
                    continue

                filtered.append({
                    "smiles": smi,
                    "similarity_to_parent": round(sim, 3),
                    "inchikey": ik,
                    "is_novel": ik not in existing_keys,
                })

            except Exception:
                continue

        return filtered

    # ── Scoring ────────────────────────────────────────────────────────────────

    def _score_candidates(self, candidates: list, parent_smiles: str) -> list:
        """Score all candidates with GNN + compute properties."""
        from models.featurizer import smiles_to_graph
        from torch_geometric.data import Batch

        scored = []

        for item in candidates:
            smi = item["smiles"]
            props = compute_properties(smi)
            if props is None:
                continue

            result = {
                **item,
                "mol_weight": props.get("mol_weight"),
                "logp": props.get("logp"),
                "hbd": props.get("hbd"),
                "hba": props.get("hba"),
                "tpsa": props.get("tpsa"),
                "passes_lipinski": props.get("passes_lipinski"),
                "predicted_pchembl": None,
                "predicted_ic50_nM": None,
            }

            # GNN prediction
            if self.model is not None:
                try:
                    graph = smiles_to_graph(smi)
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
                    logger.debug(f"Scoring failed for {smi}: {e}")

            scored.append(result)

        return scored
