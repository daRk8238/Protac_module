"""
PROTAC Designer — Data Pipeline Orchestrator
---------------------------------------------
Run this script to populate your local database with:
  1. Target protein data (from ChEMBL + PDB)
  2. Bioactivity data for the target
  3. Candidate compounds from ZINC
  4. E3 ligase binders for PROTAC design

Usage:
    python run_pipeline.py --target EGFR
    python run_pipeline.py --target P00533 --max-activities 2000
    python run_pipeline.py --target BRD4 --fetch-zinc --zinc-count 500
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from database.schema import init_db, get_session
from data_pipeline.chembl.fetcher import ChEMBLFetcher
from data_pipeline.pdb.fetcher import PDBFetcher
from data_pipeline.zinc.fetcher import ZINCFetcher

# ─── Logging Setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log"),
    ]
)
logger = logging.getLogger(__name__)


# ─── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(
    target_query: str,
    db_path: str = "protac_designer.db",
    structures_dir: str = "data/structures",
    max_activities: int = 3000,
    max_structures: int = 5,
    fetch_zinc: bool = True,
    zinc_count: int = 500,
    fetch_e3_binders: bool = True,
):
    """
    Full data pipeline for a given protein target.

    Args:
        target_query:    Gene name or UniProt ID (e.g., 'EGFR' or 'P00533')
        db_path:         SQLite database file path
        structures_dir:  Directory to store downloaded PDB files
        max_activities:  Max number of ChEMBL activities to fetch
        max_structures:  Max number of PDB structures to download
        fetch_zinc:      Whether to also fetch ZINC compounds
        zinc_count:      Number of ZINC compounds to fetch
        fetch_e3_binders: Whether to fetch E3 ligase binder warheads
    """
    logger.info("=" * 60)
    logger.info(f"PROTAC Designer — Data Pipeline")
    logger.info(f"Target: {target_query}")
    logger.info("=" * 60)

    # Initialize DB
    engine = init_db(db_path)
    session = get_session(engine)

    try:
        # ── Step 1: ChEMBL (Target + Bioactivities) ──────────────────────────
        logger.info("\n[Step 1/4] Fetching ChEMBL target + bioactivities...")
        chembl = ChEMBLFetcher(session)
        db_target = chembl.fetch_target(target_query, max_activities=max_activities)

        if db_target is None:
            logger.error(f"Could not find target: {target_query}. Exiting.")
            return

        logger.info(f"✓ Target saved: {db_target.gene_name} (UniProt: {db_target.uniprot_id})")

        # Count what we got
        from database.schema import Bioactivity
        activity_count = session.query(Bioactivity).filter_by(
            target_id=db_target.id
        ).count()
        logger.info(f"✓ Bioactivities in DB: {activity_count}")

        # ── Step 2: PDB Structures ────────────────────────────────────────────
        logger.info("\n[Step 2/4] Fetching PDB structures...")
        pdb = PDBFetcher(session, structures_dir=structures_dir)
        structures = pdb.fetch_structures_for_target(db_target, max_structures=max_structures)

        logger.info(f"✓ Structures downloaded: {len(structures)}")
        for s in structures:
            res_str = f"{s.resolution:.2f}Å" if s.resolution else "N/A"
            ligand_str = "with ligand" if s.has_ligand else "apo"
            logger.info(f"  - {s.pdb_id}: {res_str} ({s.method}) [{ligand_str}]")

        # Extract binding site from best structure
        if structures and structures[0].file_path and structures[0].has_ligand:
            best = structures[0]
            logger.info(f"\n[Step 2b] Extracting binding site from {best.pdb_id}...")
            site_residues = pdb.extract_binding_site(best.file_path)
            if site_residues:
                db_target.binding_site_residues = site_residues
                session.commit()
                logger.info(f"✓ Binding site: {len(site_residues)} residues")

        # ── Step 3: ZINC Compounds ────────────────────────────────────────────
        if fetch_zinc:
            logger.info("\n[Step 3/4] Fetching ZINC candidate compounds...")
            zinc = ZINCFetcher(session)
            n_saved = zinc.fetch_by_filters(
                mw_range=(200, 500),
                logp_range=(-1, 5),
                availability="in-stock",
                count=zinc_count,
            )
            logger.info(f"✓ ZINC compounds saved: {n_saved}")
        else:
            logger.info("\n[Step 3/4] Skipping ZINC fetch (--no-zinc flag)")

        # ── Step 4: E3 Ligase Binders ─────────────────────────────────────────
        if fetch_e3_binders:
            logger.info("\n[Step 4/4] Fetching E3 ligase binders for PROTAC design...")
            zinc = ZINCFetcher(session)
            n_e3 = zinc.fetch_e3_binders()
            logger.info(f"✓ E3 ligase binders saved: {n_e3}")
        else:
            logger.info("\n[Step 4/4] Skipping E3 binder fetch")

        # ── Summary ───────────────────────────────────────────────────────────
        from database.schema import Compound
        total_compounds = session.query(Compound).count()
        total_activities = session.query(Bioactivity).count()

        logger.info("\n" + "=" * 60)
        logger.info("Pipeline Complete! Database Summary:")
        logger.info(f"  Total compounds:    {total_compounds:,}")
        logger.info(f"  Total activities:   {total_activities:,}")
        logger.info(f"  Structures saved:   {len(structures)}")
        logger.info(f"  Database file:      {db_path}")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.info("\n[Pipeline] Interrupted by user. Saving progress...")
        session.commit()
    except Exception as e:
        logger.error(f"[Pipeline] Fatal error: {e}", exc_info=True)
        session.rollback()
    finally:
        session.close()


# ─── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PROTAC Designer — Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --target EGFR
  python run_pipeline.py --target BRD4 --max-activities 5000
  python run_pipeline.py --target P00533 --no-zinc
  python run_pipeline.py --target KRAS --zinc-count 2000
        """
    )
    parser.add_argument(
        "--target", required=True,
        help="Target gene name (e.g., EGFR, BRD4) or UniProt ID (e.g., P00533)"
    )
    parser.add_argument(
        "--db", default="protac_designer.db",
        help="SQLite database path (default: protac_designer.db)"
    )
    parser.add_argument(
        "--structures-dir", default="data/structures",
        help="Directory to save PDB files"
    )
    parser.add_argument(
        "--max-activities", type=int, default=3000,
        help="Max ChEMBL activities to fetch (default: 3000)"
    )
    parser.add_argument(
        "--max-structures", type=int, default=5,
        help="Max PDB structures to download (default: 5)"
    )
    parser.add_argument(
        "--no-zinc", action="store_true",
        help="Skip ZINC compound fetching"
    )
    parser.add_argument(
        "--zinc-count", type=int, default=500,
        help="Number of ZINC compounds to fetch (default: 500)"
    )
    parser.add_argument(
        "--no-e3", action="store_true",
        help="Skip E3 ligase binder fetching"
    )

    args = parser.parse_args()

    run_pipeline(
        target_query=args.target,
        db_path=args.db,
        structures_dir=args.structures_dir,
        max_activities=args.max_activities,
        max_structures=args.max_structures,
        fetch_zinc=not args.no_zinc,
        zinc_count=args.zinc_count,
        fetch_e3_binders=not args.no_e3,
    )


if __name__ == "__main__":
    main()
