
import time, logging, requests
from tqdm import tqdm
from database.schema import Compound
from utils.mol_utils import process_compound

logger = logging.getLogger(__name__)

class ZINCFetcher:
    def __init__(self, db_session):
        self.session = db_session
        self.request_delay = 0.5

    def fetch_by_filters(self, mw_range=(200,500), logp_range=(-1,5),
                         availability="in-stock", count=1000, compound_role=None):
        logger.info(f"[ZINC] Fetching {count} compounds...")
        saved = self._fetch_via_tranches(mw_range, logp_range, count, compound_role)
        self.session.commit()
        logger.info(f"[ZINC] Saved {saved} compounds")
        return saved

    def _fetch_via_tranches(self, mw_range, logp_range, count, compound_role=None):
        mw_tranches = self._mw_to_tranches(mw_range)
        lp_tranches = self._logp_to_tranches(logp_range)
        saved = 0
        for mw_t in mw_tranches:
            for lp_t in lp_tranches:
                if saved >= count:
                    break
                url = f"https://zinc20.docking.org/tranches/{mw_t}{lp_t}.txt"
                try:
                    r = requests.get(url, timeout=60)
                    if r.status_code != 200:
                        continue
                    for line in r.text.strip().split("\n")[:count - saved]:
                        parts = line.split()
                        if len(parts) >= 2:
                            if self._save_compound(parts[0], parts[1], compound_role):
                                saved += 1
                    time.sleep(self.request_delay)
                except Exception as e:
                    logger.debug(f"[ZINC] Tranche {mw_t}{lp_t} failed: {e}")
        return saved

    def _mw_to_tranches(self, mw_range):
        tranche_map = [(200,"A"),(250,"B"),(300,"C"),(325,"D"),(350,"E"),
                       (375,"F"),(400,"G"),(425,"H"),(450,"J"),(500,"K")]
        return [l for cutoff, l in tranche_map if mw_range[0] <= cutoff <= mw_range[1]] or ["E","F","G"]

    def _logp_to_tranches(self, logp_range):
        tranche_map = [(-1,"A"),(0,"B"),(1,"C"),(2,"D"),(3,"E"),(4,"F"),(5,"G")]
        return [l for cutoff, l in tranche_map if logp_range[0] <= cutoff <= logp_range[1]] or ["C","D","E"]

    def fetch_e3_binders(self):
        logger.info("[ZINC] Loading E3 ligase binders...")
        binders = [
            {"smiles": "O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1", "id": "thalidomide", "e3": "CRBN"},
            {"smiles": "O=C1CCC(N2C(=O)c3cc(N)ccc3C2=O)C(=O)N1", "id": "lenalidomide", "e3": "CRBN"},
            {"smiles": "O=C1CCC(N2C(=O)c3cc(F)ccc3C2=O)C(=O)N1", "id": "pomalidomide", "e3": "CRBN"},
            {"smiles": "CC(C)(C)OC(=O)N[C@@H]1C[C@H](O)CN1", "id": "vhl_1", "e3": "VHL"},
            {"smiles": "O=C(O)[C@@H]1C[C@@H](O)CN1C(=O)[C@H](Cc1ccccc1)NC(=O)c1ccc(Cl)cc1", "id": "vhl_2", "e3": "VHL"},
        ]
        saved = sum(1 for b in binders
                    if self._save_compound(b["smiles"], f"PROTACDB_{b['id']}", "e3_ligand"))
        self.session.commit()
        logger.info(f"[ZINC] Saved {saved} E3 binders")
        return saved

    def _save_compound(self, smiles, source_id, compound_role=None):
        processed = process_compound(smiles, source="zinc", source_id=source_id)
        if not processed:
            return False
        inchikey = processed.get("inchikey")
        if inchikey and self.session.query(Compound).filter_by(inchikey=inchikey).first():
            return False
        if compound_role:
            processed["compound_role"] = compound_role
        self.session.add(Compound(**processed))
        return True
