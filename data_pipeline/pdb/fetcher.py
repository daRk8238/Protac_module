import os, logging, requests
from pathlib import Path
from typing import Optional
from database.schema import TargetProtein, PDBStructure

logger = logging.getLogger(__name__)
RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DATA_URL = "https://data.rcsb.org/rest/v1/core/entry"
RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download"

class PDBFetcher:
    def __init__(self, db_session, structures_dir="data/structures"):
        self.session = db_session
        self.structures_dir = Path(structures_dir)
        self.structures_dir.mkdir(parents=True, exist_ok=True)

    def search_by_uniprot(self, uniprot_id):
        query = {
            "query": {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                "operator": "exact_match", "value": uniprot_id}},
            "return_type": "entry",
            "request_options": {"paginate": {"start": 0, "rows": 50}}
        }
        try:
            r = requests.post(RCSB_SEARCH_URL, json=query, timeout=30)
            r.raise_for_status()
            results = r.json().get("result_set", [])
            logger.info(f"[PDB] Found {len(results)} structures for {uniprot_id}")
            return results
        except Exception as e:
            logger.error(f"[PDB] Search failed: {e}")
            return []

    def get_structure_metadata(self, pdb_id):
        try:
            r = requests.get(f"{RCSB_DATA_URL}/{pdb_id.upper()}", timeout=20)
            r.raise_for_status()
            data = r.json()
            resolution = None
            refine = data.get("refine", [{}])
            if refine:
                resolution = refine[0].get("ls_d_res_high")
            if resolution is None:
                em = data.get("em_3d_reconstruction", [{}])
                if em:
                    resolution = em[0].get("resolution")
            method = data.get("exptl", [{}])[0].get("method", "UNKNOWN")
            has_ligand = data.get("rcsb_entry_info", {}).get("nonpolymer_entity_count", 0) > 0
            return {"pdb_id": pdb_id.upper(), "resolution": float(resolution) if resolution else None,
                    "method": method, "has_ligand": has_ligand}
        except Exception as e:
            logger.debug(f"[PDB] Metadata failed for {pdb_id}: {e}")
            return None

    def download_pdb_file(self, pdb_id):
        pdb_id = pdb_id.upper()
        file_path = self.structures_dir / f"{pdb_id}.pdb"
        if file_path.exists():
            return str(file_path)
        try:
            r = requests.get(f"{RCSB_DOWNLOAD_URL}/{pdb_id}.pdb", timeout=60, stream=True)
            r.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            logger.info(f"[PDB] Downloaded {pdb_id}")
            return str(file_path)
        except Exception as e:
            logger.error(f"[PDB] Download failed for {pdb_id}: {e}")
            return None

    def fetch_structures_for_target(self, db_target, max_structures=5,
                                     methods=["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"]):
        results = self.search_by_uniprot(db_target.uniprot_id)
        if not results:
            return []
        pdb_ids = [r["identifier"] for r in results]
        metas = []
        for pdb_id in pdb_ids[:20]:
            m = self.get_structure_metadata(pdb_id)
            if m and m.get("method") in methods:
                metas.append(m)
        metas.sort(key=lambda x: (not x.get("has_ligand", False), x.get("resolution") or 99.0))
        saved = []
        for meta in metas[:max_structures]:
            pdb_id = meta["pdb_id"]
            existing = self.session.query(PDBStructure).filter_by(pdb_id=pdb_id).first()
            if existing:
                saved.append(existing)
                continue
            file_path = self.download_pdb_file(pdb_id)
            s = PDBStructure(pdb_id=pdb_id, target_id=db_target.id,
                             resolution=meta.get("resolution"), method=meta.get("method"),
                             has_ligand=meta.get("has_ligand", False), file_path=file_path)
            self.session.add(s)
            if db_target.pdb_id is None and file_path:
                db_target.pdb_id = pdb_id
            saved.append(s)
        self.session.commit()
        logger.info(f"[PDB] Saved {len(saved)} structures")
        return saved

    def extract_binding_site(self, pdb_file, ligand_residue_name=None, radius_angstrom=6.0):
        try:
            from Bio import PDB as BioPDB
            parser = BioPDB.PDBParser(QUIET=True)
            structure = parser.get_structure("protein", pdb_file)
            model = structure[0]
            ligand_atoms = []
            for chain in model:
                for residue in chain:
                    if residue.id[0] == " ":
                        continue
                    if residue.resname in {"HOH", "WAT", "DOD"}:
                        continue
                    if ligand_residue_name and residue.resname != ligand_residue_name:
                        continue
                    ligand_atoms.extend(residue.get_atoms())
            if not ligand_atoms:
                return []
            ns = BioPDB.NeighborSearch(list(model.get_atoms()))
            nearby = set()
            for atom in ligand_atoms:
                for residue in ns.search(atom.coord, radius_angstrom, "R"):
                    if residue.id[0] == " ":
                        nearby.add(residue)
            return [{"chain": r.get_parent().id, "residue_id": r.id[1], "residue_name": r.resname}
                    for r in sorted(nearby, key=lambda r: r.id[1])]
        except Exception as e:
            logger.error(f"[PDB] Binding site extraction failed: {e}")
            return []