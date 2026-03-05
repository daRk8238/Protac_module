"""
PDB Structure Pipeline
----------------------
Fetches protein structures from the RCSB PDB for a given UniProt ID.
Selects the best structure (highest resolution X-ray or cryo-EM),
downloads the PDB file, and extracts key metadata.

Usage:
    fetcher = PDBFetcher(session, structures_dir="data/structures")
    fetcher.fetch_structures_for_target(db_target, max_structures=5)
"""

import os
import json
import logging
import requests
from pathlib import Path
from typing import Optional

from database.schema import TargetProtein, PDBStructure

logger = logging.getLogger(__name__)

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DATA_URL = "https://data.rcsb.org/rest/v1/core/entry"
RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download"


class PDBFetcher:
    def __init__(self, db_session, structures_dir: str = "data/structures"):
        self.session = db_session
        self.structures_dir = Path(structures_dir)
        self.structures_dir.mkdir(parents=True, exist_ok=True)

    # ─── Search ────────────────────────────────────────────────────────────────

    def search_by_uniprot(self, uniprot_id: str) -> list[dict]:
        """
        Search RCSB for all PDB entries containing a given UniProt accession.
        Returns list of result dicts with pdb_id and score.
        """
        query = {
            "query": {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                    "operator": "exact_match",
                    "value": uniprot_id
                }
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {"start": 0, "rows": 50},
                "sort": [{"sort_by": "score", "direction": "desc"}]
            }
        }

        try:
            response = requests.post(RCSB_SEARCH_URL, json=query, timeout=30)
            response.raise_for_status()
            data = response.json()
            results = data.get("result_set", [])
            logger.info(f"[PDB] Found {len(results)} structures for UniProt {uniprot_id}")
            return results
        except requests.RequestException as e:
            logger.error(f"[PDB] Search failed for {uniprot_id}: {e}")
            return []

    # ─── Metadata Fetching ─────────────────────────────────────────────────────

    def get_structure_metadata(self, pdb_id: str) -> Optional[dict]:
        """Fetch resolution, method, and ligand info for a PDB entry."""
        url = f"{RCSB_DATA_URL}/{pdb_id.upper()}"
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            data = response.json()

            # Extract resolution
            resolution = None
            refine = data.get("refine", [{}])
            if refine:
                resolution = refine[0].get("ls_d_res_high")  # highest resolution shell

            # If cryo-EM, resolution is stored differently
            if resolution is None:
                em_3d = data.get("em_3d_reconstruction", [{}])
                if em_3d:
                    resolution = em_3d[0].get("resolution")

            # Extract method
            method = data.get("exptl", [{}])[0].get("method", "UNKNOWN")

            # Check for small molecule ligands
            ligand_ids = []
            nonpoly = data.get("rcsb_entry_info", {}).get("nonpolymer_entity_count", 0)
            has_ligand = nonpoly > 0

            return {
                "pdb_id": pdb_id.upper(),
                "resolution": float(resolution) if resolution else None,
                "method": method,
                "has_ligand": has_ligand,
            }
        except Exception as e:
            logger.debug(f"[PDB] Metadata fetch failed for {pdb_id}: {e}")
            return None

    # ─── Download ──────────────────────────────────────────────────────────────

    def download_pdb_file(self, pdb_id: str) -> Optional[str]:
        """
        Download the PDB file to local disk.
        Returns the file path, or None if download failed.
        """
        pdb_id = pdb_id.upper()
        file_path = self.structures_dir / f"{pdb_id}.pdb"

        # Skip if already downloaded
        if file_path.exists():
            logger.debug(f"[PDB] Already downloaded: {pdb_id}")
            return str(file_path)

        url = f"{RCSB_DOWNLOAD_URL}/{pdb_id}.pdb"
        try:
            response = requests.get(url, timeout=60, stream=True)
            response.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"[PDB] Downloaded: {pdb_id} → {file_path}")
            return str(file_path)
        except Exception as e:
            logger.error(f"[PDB] Download failed for {pdb_id}: {e}")
            return None

    # ─── Main Entry Point ──────────────────────────────────────────────────────

    def fetch_structures_for_target(
        self,
        db_target: TargetProtein,
        max_structures: int = 5,
        methods: list = ["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"]
    ) -> list[PDBStructure]:
        """
        Full pipeline: search → filter → download → store in DB.

        Selects best structures by:
        1. Method preference (X-ray first, then cryo-EM)
        2. Resolution (lower = better)
        3. Has co-crystallized ligand (useful for binding site info)
        """
        uniprot_id = db_target.uniprot_id
        logger.info(f"[PDB] Fetching structures for {db_target.gene_name} ({uniprot_id})")

        # 1. Search
        results = self.search_by_uniprot(uniprot_id)
        if not results:
            logger.warning(f"[PDB] No structures found for {uniprot_id}")
            return []

        pdb_ids = [r["identifier"] for r in results]

        # 2. Fetch metadata for all hits (to sort by resolution)
        structures_meta = []
        for pdb_id in pdb_ids[:20]:  # limit API calls
            meta = self.get_structure_metadata(pdb_id)
            if meta and meta.get("method") in methods:
                structures_meta.append(meta)

        # 3. Sort: ligand-containing first, then by resolution
        structures_meta.sort(
            key=lambda x: (
                not x.get("has_ligand", False),   # True first
                x.get("resolution") or 99.0        # lower resolution first
            )
        )

        logger.info(f"[PDB] Ranked {len(structures_meta)} valid structures")

        # 4. Download top N and save to DB
        saved_structures = []
        for meta in structures_meta[:max_structures]:
            pdb_id = meta["pdb_id"]

            # Check if already in DB
            existing = self.session.query(PDBStructure).filter_by(pdb_id=pdb_id).first()
            if existing:
                saved_structures.append(existing)
                continue

            # Download
            file_path = self.download_pdb_file(pdb_id)

            # Save to DB
            db_structure = PDBStructure(
                pdb_id=pdb_id,
                target_id=db_target.id,
                resolution=meta.get("resolution"),
                method=meta.get("method"),
                has_ligand=meta.get("has_ligand", False),
                file_path=file_path,
            )
            self.session.add(db_structure)
            saved_structures.append(db_structure)

            # Set best structure on target if not set yet
            if db_target.pdb_id is None and file_path:
                db_target.pdb_id = pdb_id

        self.session.commit()
        logger.info(f"[PDB] Saved {len(saved_structures)} structures for {db_target.gene_name}")
        return saved_structures

    # ─── Binding Site Extraction (BioPython) ───────────────────────────────────

    def extract_binding_site(
        self,
        pdb_file: str,
        ligand_residue_name: str = None,
        radius_angstrom: float = 6.0
    ) -> list[dict]:
        """
        Extract residues within `radius_angstrom` of a ligand.
        If no ligand name given, uses the first non-water HETATM.

        Returns list of residue dicts: {chain, residue_id, residue_name, distance}
        """
        try:
            from Bio import PDB as BioPDB

            parser = BioPDB.PDBParser(QUIET=True)
            structure = parser.get_structure("protein", pdb_file)
            model = structure[0]

            # Find ligand atoms
            ligand_atoms = []
            for chain in model:
                for residue in chain:
                    if residue.id[0] == " ":  # skip standard amino acids
                        continue
                    if residue.resname in {"HOH", "WAT", "DOD"}:  # skip water
                        continue
                    if ligand_residue_name and residue.resname != ligand_residue_name:
                        continue
                    ligand_atoms.extend(residue.get_atoms())

            if not ligand_atoms:
                logger.warning(f"[PDB] No ligand found in {pdb_file}")
                return []

            # Find protein residues within radius
            ns = BioPDB.NeighborSearch(list(model.get_atoms()))
            nearby_residues = set()
            for atom in ligand_atoms:
                nearby = ns.search(atom.coord, radius_angstrom, "R")
                for residue in nearby:
                    if residue.id[0] == " ":  # only standard residues
                        nearby_residues.add(residue)

            # Format results
            site_residues = []
            for residue in sorted(nearby_residues, key=lambda r: r.id[1]):
                site_residues.append({
                    "chain": residue.get_parent().id,
                    "residue_id": residue.id[1],
                    "residue_name": residue.resname,
                })

            logger.info(f"[PDB] Found {len(site_residues)} binding site residues")
            return site_residues

        except ImportError:
            logger.error("[PDB] BioPython not installed. pip install biopython")
            return []
        except Exception as e:
            logger.error(f"[PDB] Binding site extraction failed: {e}")
            return []
