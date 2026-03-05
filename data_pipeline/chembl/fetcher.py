"""
ChEMBL Data Pipeline (v2 — Universal Target Resolution)
---------------------------------------------------------
Uses TargetResolver to find any protein before fetching bioactivities.
Accepts gene names, protein names, UniProt IDs, or ChEMBL IDs.

Usage:
    fetcher = ChEMBLFetcher(session)
    fetcher.fetch_target("EGFR")
    fetcher.fetch_target("sonic hedgehog")
    fetcher.fetch_target("Q09864")
    fetcher.fetch_target("epidermal growth factor")
"""

import logging
from typing import Optional
from tqdm import tqdm

from chembl_webresource_client.new_client import new_client
from database.schema import TargetProtein, Compound, Bioactivity
from utils.mol_utils import process_compound
from utils.target_resolver import TargetResolver

logger = logging.getLogger(__name__)

RELEVANT_ACTIVITY_TYPES = {"IC50", "Ki", "Kd", "EC50", "AC50", "GI50", "Potency"}
MIN_CONFIDENCE = 6
MIN_PCHEMBL = 5.0


class ChEMBLFetcher:
    def __init__(self, db_session):
        self.session = db_session
        self.activity_api = new_client.activity
        self.resolver = TargetResolver(organism_filter="Homo sapiens")

    def fetch_target(self, query: str, max_activities: int = 5000) -> Optional[TargetProtein]:
        """
        Fetch target + bioactivities for ANY protein identifier.
        Accepts: gene name, protein name, UniProt ID, or ChEMBL ID.
        """
        logger.info(f"[ChEMBL] Resolving target: '{query}'")
        resolved = self.resolver.resolve(query)

        if not resolved.found:
            logger.error(f"[ChEMBL] Could not resolve '{query}'")
            return None

        logger.info(f"[ChEMBL] Resolved → {resolved.gene_name} | "
                    f"UniProt={resolved.uniprot_id} | ChEMBL={resolved.chembl_id}")

        if resolved.alternatives:
            logger.info("[ChEMBL] Other matches:")
            for alt in resolved.alternatives[:3]:
                logger.info(f"  - {alt.get('gene_name')} ({alt.get('uniprot_id')}) — {alt.get('organism')}")

        db_target = self._get_or_create_target(resolved)

        if resolved.chembl_id:
            self._fetch_bioactivities(resolved.chembl_id, db_target, max_activities)
        else:
            logger.info(f"[ChEMBL] No ChEMBL entry for {resolved.uniprot_id} — "
                        f"target saved without bioactivity data. PDB + ZINC still available.")

        return db_target

    def search_suggestions(self, query: str) -> list[dict]:
        """Return autocomplete suggestions for a partial protein name/gene."""
        return self.resolver.search_suggestions(query, max_results=10)

    def _get_or_create_target(self, resolved) -> TargetProtein:
        if resolved.uniprot_id:
            existing = self.session.query(TargetProtein).filter_by(
                uniprot_id=resolved.uniprot_id
            ).first()
            if existing:
                logger.info(f"[ChEMBL] Target already in DB: {existing}")
                return existing

        db_target = TargetProtein(
            uniprot_id=resolved.uniprot_id or resolved.chembl_id,
            gene_name=resolved.gene_name or resolved.chembl_name or resolved.query,
            protein_name=resolved.protein_name or resolved.chembl_name,
            organism=resolved.organism,
            sequence=resolved.sequence,
        )
        self.session.add(db_target)
        self.session.commit()
        logger.info(f"[ChEMBL] Created target: {db_target}")
        return db_target

    def _fetch_bioactivities(self, chembl_id: str, db_target: TargetProtein, max_activities: int):
        logger.info(f"[ChEMBL] Fetching bioactivities for {chembl_id}...")
        activities = self.activity_api.filter(
            target_chembl_id=chembl_id,
            standard_type__in=list(RELEVANT_ACTIVITY_TYPES),
            confidence_score__gte=MIN_CONFIDENCE,
            pchembl_value__isnull=False,
        ).only([
            "molecule_chembl_id", "canonical_smiles", "standard_type",
            "standard_value", "standard_units", "pchembl_value",
            "assay_type", "assay_description", "assay_chembl_id", "confidence_score",
        ])

        activities = list(activities)[:max_activities]
        logger.info(f"[ChEMBL] Retrieved {len(activities)} raw activity records")

        saved, skipped = 0, 0
        for act in tqdm(activities, desc=f"Processing {db_target.gene_name}"):
            try:
                if self._process_activity(act, db_target):
                    saved += 1
                else:
                    skipped += 1
            except Exception as e:
                logger.debug(f"Activity error: {e}")
                skipped += 1

        self.session.commit()
        logger.info(f"[ChEMBL] Done. Saved: {saved}, Skipped: {skipped}")

    def _process_activity(self, act: dict, db_target: TargetProtein) -> bool:
        smiles = act.get("canonical_smiles")
        pchembl = act.get("pchembl_value")
        if not smiles or not pchembl or float(pchembl) < MIN_PCHEMBL:
            return False

        db_compound = self._get_or_create_compound(smiles, act.get("molecule_chembl_id"))
        if db_compound is None:
            return False

        existing = self.session.query(Bioactivity).filter_by(
            target_id=db_target.id,
            compound_id=db_compound.id,
            activity_type=act.get("standard_type"),
            pchembl_value=float(pchembl),
        ).first()
        if existing:
            return False

        self.session.add(Bioactivity(
            target_id=db_target.id,
            compound_id=db_compound.id,
            activity_type=act.get("standard_type"),
            activity_value=float(act["standard_value"]) if act.get("standard_value") else None,
            activity_units=act.get("standard_units"),
            pchembl_value=float(pchembl),
            assay_type=act.get("assay_type"),
            assay_description=(act.get("assay_description") or "")[:500],
            chembl_assay_id=act.get("assay_chembl_id"),
            confidence_score=act.get("confidence_score"),
            modulator_type=self._infer_modulator_type(act),
        ))
        return True

    def _get_or_create_compound(self, smiles: str, chembl_id: str) -> Optional[Compound]:
        processed = process_compound(smiles, source="chembl", source_id=chembl_id)
        if processed is None:
            return None
        inchikey = processed.get("inchikey")
        if inchikey:
            existing = self.session.query(Compound).filter_by(inchikey=inchikey).first()
            if existing:
                return existing
        compound = Compound(**processed)
        self.session.add(compound)
        self.session.flush()
        return compound

    def _infer_modulator_type(self, act: dict) -> str:
        desc = (act.get("assay_description") or "").lower()
        atype = (act.get("standard_type") or "").upper()
        if any(w in desc for w in ["degrad", "protac", "dbtag"]):
            return "degrader"
        if any(w in desc for w in ["activat", "agonist", "potentiat"]):
            return "activator"
        if any(w in desc for w in ["inhibit", "antagon", "block"]):
            return "inhibitor"
        if atype in {"IC50", "Ki"}:
            return "inhibitor"
        if atype == "EC50":
            return "activator"
        return "unknown"
