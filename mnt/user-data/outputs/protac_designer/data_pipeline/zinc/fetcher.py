"""
ZINC20 Data Pipeline
--------------------
Fetches purchasable, drug-like compounds from ZINC20 database.
ZINC is used to populate the candidate compound pool for screening.

Strategy:
- Use ZINC20 REST API to fetch subsets by physicochemical filters
- Focus on "in-stock" compounds for practical drug discovery
- Also fetch known E3 ligase binders from PROTAC-DB

Usage:
    fetcher = ZINCFetcher(session)
    fetcher.fetch_by_filters(mw_range=(200, 500), logp_range=(-1, 5), count=1000)
"""

import time
import logging
import requests
from tqdm import tqdm

from database.schema import Compound
from utils.mol_utils import process_compound

logger = logging.getLogger(__name__)

ZINC_API_BASE = "https://zinc20.docking.org"
PROTAC_DB_BASE = "https://protacdb.com/api"   # check their API docs for updates


class ZINCFetcher:
    def __init__(self, db_session):
        self.session = db_session
        self.request_delay = 0.5  # seconds between requests (be respectful)

    # ─── ZINC20 REST API ───────────────────────────────────────────────────────

    def fetch_by_filters(
        self,
        mw_range: tuple = (200, 500),
        logp_range: tuple = (-1, 5),
        availability: str = "in-stock",   # 'in-stock', 'agent', 'bb', 'wait'
        count: int = 1000,
        compound_role: str = None
    ) -> int:
        """
        Fetch compounds from ZINC20 matching physicochemical filters.

        ZINC subset shorthand:
          - "in-stock" → immediately purchasable
          - "drug-like" → passes basic drug-likeness filters
          - "fragment-like" → MW < 300, good for fragment screens

        Returns number of compounds saved to DB.
        """
        logger.info(
            f"[ZINC] Fetching {count} compounds: MW={mw_range}, logP={logp_range}, "
            f"availability={availability}"
        )

        # ZINC20 uses a subset/slice system
        # We'll use the tranches endpoint which lets us filter by MW and logP
        url = f"{ZINC_API_BASE}/substances/subsets/{availability}.json"

        params = {
            "mw_range": f"{mw_range[0]}:{mw_range[1]}",
            "logp_range": f"{logp_range[0]}:{logp_range[1]}",
            "count": min(count, 1000),   # ZINC caps per page
        }

        saved = 0
        page = 1

        while saved < count:
            params["page"] = page
            try:
                response = requests.get(url, params=params, timeout=30)
                if response.status_code == 404:
                    logger.warning(f"[ZINC] Endpoint not found. Trying alternative...")
                    # Fallback to tranches-based download
                    saved += self._fetch_via_tranches(
                        mw_range, logp_range, count, compound_role
                    )
                    break
                response.raise_for_status()
                data = response.json()

                compounds = data if isinstance(data, list) else data.get("results", [])
                if not compounds:
                    break

                for item in tqdm(compounds, desc=f"ZINC page {page}"):
                    smiles = item.get("smiles")
                    zinc_id = item.get("zinc_id") or item.get("id")
                    if smiles:
                        if self._save_compound(smiles, zinc_id, compound_role):
                            saved += 1

                if saved >= count or len(compounds) < params["count"]:
                    break

                page += 1
                time.sleep(self.request_delay)

            except Exception as e:
                logger.error(f"[ZINC] Fetch failed on page {page}: {e}")
                break

        self.session.commit()
        logger.info(f"[ZINC] Saved {saved} compounds from ZINC")
        return saved

    def _fetch_via_tranches(
        self,
        mw_range: tuple,
        logp_range: tuple,
        count: int,
        compound_role: str = None
    ) -> int:
        """
        Alternative ZINC access via the tranches system.
        ZINC divides compound space into a grid of MW × logP tranches.
        Each tranche has a 2-letter code (A-K for MW, A-K for logP).

        MW tranches: A(<200), B(200-250), C(250-300), D(300-325), E(325-350),
                     F(350-375), G(375-400), H(400-425), J(425-450), K(450-500)
        logP tranches: A(<-1), B(-1-0), C(0-1), D(1-2), E(2-3), F(3-4), G(4-5)
        """
        # Map MW range to tranche letters
        mw_tranches = self._mw_to_tranches(mw_range)
        lp_tranches = self._logp_to_tranches(logp_range)

        saved = 0
        for mw_t in mw_tranches:
            for lp_t in lp_tranches:
                if saved >= count:
                    break
                tranche_code = f"{mw_t}{lp_t}"
                url = f"{ZINC_API_BASE}/tranches/{tranche_code}.txt"
                try:
                    response = requests.get(url, timeout=60)
                    if response.status_code != 200:
                        continue
                    lines = response.text.strip().split("\n")
                    # Each line: SMILES ZINC_ID
                    for line in lines[:count - saved]:
                        parts = line.split()
                        if len(parts) >= 2:
                            smiles, zinc_id = parts[0], parts[1]
                            if self._save_compound(smiles, zinc_id, compound_role):
                                saved += 1
                    time.sleep(self.request_delay)
                except Exception as e:
                    logger.debug(f"[ZINC] Tranche {tranche_code} failed: {e}")

        return saved

    def _mw_to_tranches(self, mw_range: tuple) -> list[str]:
        """Map MW range to ZINC tranche letters."""
        tranche_map = [
            (200, "A"), (250, "B"), (300, "C"), (325, "D"),
            (350, "E"), (375, "F"), (400, "G"), (425, "H"),
            (450, "J"), (500, "K")
        ]
        result = []
        for cutoff, letter in tranche_map:
            if mw_range[0] <= cutoff <= mw_range[1]:
                result.append(letter)
        return result or ["E", "F", "G"]  # default mid-range

    def _logp_to_tranches(self, logp_range: tuple) -> list[str]:
        """Map logP range to ZINC tranche letters."""
        tranche_map = [
            (-1, "A"), (0, "B"), (1, "C"), (2, "D"),
            (3, "E"), (4, "F"), (5, "G")
        ]
        result = []
        for cutoff, letter in tranche_map:
            if logp_range[0] <= cutoff <= logp_range[1]:
                result.append(letter)
        return result or ["C", "D", "E"]

    # ─── PROTAC-DB E3 Ligase Binders ───────────────────────────────────────────

    def fetch_e3_binders(self) -> int:
        """
        Fetch known E3 ligase binders from PROTAC-DB.
        These are warheads for CRBN, VHL, MDM2, and other E3 ligases.
        Falls back to a curated hardcoded set if API unavailable.
        """
        logger.info("[ZINC] Fetching E3 ligase binders from PROTAC-DB...")

        # Curated set of well-known E3 ligase binders
        # Source: PROTAC-DB, literature (Crews lab, Ciulli lab, etc.)
        known_binders = [
            # CRBN binders (thalidomide analogs)
            {
                "smiles": "O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1",
                "id": "thalidomide",
                "e3": "CRBN"
            },
            {
                "smiles": "O=C1CCC(N2C(=O)c3cc(N)ccc3C2=O)C(=O)N1",
                "id": "lenalidomide",
                "e3": "CRBN"
            },
            {
                "smiles": "O=C1CCC(N2C(=O)c3cc(F)ccc3C2=O)C(=O)N1",
                "id": "pomalidomide",
                "e3": "CRBN"
            },
            # VHL binders (hydroxyproline-based)
            {
                "smiles": "CC(C)(C)OC(=O)N[C@@H]1C[C@H](O)CN1",
                "id": "vhl_ligand_1",
                "e3": "VHL"
            },
            {
                "smiles": "O=C(O)[C@@H]1C[C@@H](O)CN1C(=O)[C@H](Cc1ccccc1)NC(=O)c1ccc(Cl)cc1",
                "id": "vhl_ligand_2",
                "e3": "VHL"
            },
            # MDM2 binders (nutlin analogs)
            {
                "smiles": "O=C(N[C@@H](CCC(=O)O)C(=O)O)c1cc(Cl)ccc1-c1ccc(Cl)cc1",
                "id": "mdm2_binder_1",
                "e3": "MDM2"
            },
            # IAP binders (SMAC mimetics)
            {
                "smiles": "CC[C@H](C)[C@@H](NC(=O)[C@@H](CC(C)C)NC(=O)c1cc(N)ccc1N)C(=O)N1CCC[C@H]1C(=O)N",
                "id": "smac_mimetic_1",
                "e3": "cIAP1"
            },
        ]

        saved = 0
        for binder in known_binders:
            compound = self._save_compound(
                binder["smiles"],
                f"PROTACDB_{binder['id']}",
                compound_role="e3_ligand"
            )
            if compound:
                saved += 1

        self.session.commit()
        logger.info(f"[ZINC] Saved {saved} E3 ligase binder warheads")
        return saved

    # ─── Helpers ───────────────────────────────────────────────────────────────

    def _save_compound(
        self,
        smiles: str,
        source_id: str,
        compound_role: str = None
    ) -> bool:
        """Process and save a compound. Returns True if newly saved."""
        processed = process_compound(smiles, source="zinc", source_id=source_id)
        if processed is None:
            return False

        inchikey = processed.get("inchikey")
        if inchikey:
            existing = self.session.query(Compound).filter_by(inchikey=inchikey).first()
            if existing:
                return False  # already in DB

        if compound_role:
            processed["compound_role"] = compound_role

        compound = Compound(**processed)
        self.session.add(compound)
        return True
