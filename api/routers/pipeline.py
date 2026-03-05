"""
Pipeline Router
POST /api/pipeline/run — trigger data pipeline for a new target (async)
"""

import sys
import uuid
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import APIRouter, BackgroundTasks
from api.schemas import PipelineRequest, PipelineResponse
from api.job_store import job_store
from api.state import AppState

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/run", response_model=PipelineResponse)
async def run_pipeline(request: PipelineRequest, background_tasks: BackgroundTasks):
    """
    Trigger the data pipeline for a new protein target.
    Fetches ChEMBL bioactivities, PDB structures, and ZINC compounds.
    Runs asynchronously — poll /api/jobs/{job_id} for status.
    """
    job_id = str(uuid.uuid4())
    job_store.create(job_id, job_type="pipeline", meta=request.model_dump())
    background_tasks.add_task(_run_pipeline_task, job_id, request)

    return PipelineResponse(
        job_id=job_id,
        target=request.target,
        status="pending",
        message=f"Pipeline started for '{request.target}'. Poll /api/jobs/{job_id} for updates.",
    )


@router.get("/targets")
async def list_pipeline_targets():
    """List targets that have been fetched via the pipeline."""
    from database.schema import TargetProtein
    session = AppState.get_session()
    try:
        targets = session.query(TargetProtein).all()
        return {
            "targets": [
                {
                    "uniprot_id": t.uniprot_id,
                    "gene_name": t.gene_name,
                    "pdb_id": t.pdb_id,
                }
                for t in targets
            ]
        }
    finally:
        session.close()


def _run_pipeline_task(job_id: str, request: PipelineRequest):
    """Background task: run the full data pipeline."""
    job_store.update(job_id, status="running", progress=f"Starting pipeline for {request.target}...")
    try:
        from database.schema import init_db, get_session, Bioactivity, Compound
        from data_pipeline.chembl.fetcher import ChEMBLFetcher
        from data_pipeline.pdb.fetcher import PDBFetcher
        from data_pipeline.zinc.fetcher import ZINCFetcher

        engine = init_db("protac_designer.db")
        session = get_session(engine)

        # Step 1: ChEMBL
        job_store.update(job_id, progress="[1/4] Fetching ChEMBL bioactivities...")
        chembl = ChEMBLFetcher(session)
        db_target = chembl.fetch_target(request.target, max_activities=request.max_activities)

        if db_target is None:
            session.close()
            job_store.fail(job_id, error=f"Target '{request.target}' not found. Try a UniProt ID or well-known gene name.")
            return

        # Step 2: PDB
        job_store.update(job_id, progress="[2/4] Downloading PDB structures...")
        pdb = PDBFetcher(session, structures_dir="data/structures")
        structures = pdb.fetch_structures_for_target(db_target, max_structures=request.max_structures)

        if structures and structures[0].file_path and structures[0].has_ligand:
            site = pdb.extract_binding_site(structures[0].file_path)
            if site:
                db_target.binding_site_residues = site
                session.commit()

        # Step 3: ZINC
        if request.fetch_zinc:
            job_store.update(job_id, progress="[3/4] Fetching ZINC compounds...")
            zinc = ZINCFetcher(session)
            zinc.fetch_by_filters(count=request.zinc_count)
        else:
            job_store.update(job_id, progress="[3/4] Skipping ZINC...")

        # Step 4: E3 binders
        if request.fetch_e3_binders:
            job_store.update(job_id, progress="[4/4] Loading E3 ligase binders...")
            zinc = ZINCFetcher(session)
            zinc.fetch_e3_binders()

        # Save all values from db_target BEFORE closing the session
        target_id       = db_target.id
        target_uniprot  = db_target.uniprot_id
        target_gene     = db_target.gene_name
        target_pdb      = db_target.pdb_id
        n_structures    = len(structures) if structures else 0

        session.close()
        AppState.refresh_counts()

        # Open a fresh session for final counts
        s2 = AppState.get_session()
        try:
            activity_count = s2.query(Bioactivity).filter_by(target_id=target_id).count()
            compound_count = s2.query(Compound).count()
        finally:
            s2.close()

        job_store.complete(job_id, result={
            "target":                request.target,
            "uniprot_id":            target_uniprot,
            "gene_name":             target_gene,
            "pdb_id":                target_pdb,
            "structures_downloaded": n_structures,
            "activity_count":        activity_count,
            "total_compounds":       compound_count,
        })
        logger.info(f"[Pipeline] Job {job_id} complete for {target_gene}")

    except Exception as e:
        logger.error(f"[Pipeline] Job {job_id} failed: {e}", exc_info=True)
        job_store.fail(job_id, error=str(e))
