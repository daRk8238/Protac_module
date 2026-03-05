"""
Jobs Router
GET /api/jobs/           — list all jobs
GET /api/jobs/{job_id}   — poll job status
GET /api/results/{job_id} — get job result
"""

from fastapi import APIRouter, HTTPException
from api.job_store import job_store
from api.schemas import JobStatus, JobResult

router = APIRouter()


@router.get("")
async def list_jobs():
    """List all jobs (most recent first)."""
    jobs = job_store.list_all()
    jobs.sort(key=lambda j: j["created_at"], reverse=True)
    return {"total": len(jobs), "jobs": jobs}


@router.get("/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """
    Poll the status of an async job.
    Status values: pending → running → complete | failed
    """
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return JobStatus(
        job_id=job["job_id"],
        status=job["status"],
        job_type=job["job_type"],
        created_at=job["created_at"],
        updated_at=job["updated_at"],
        progress=job.get("progress"),
        error=job.get("error"),
    )


@router.get("/{job_id}/results", response_model=JobResult)
async def get_job_results(job_id: str):
    """
    Get the full results of a completed job.
    Returns 404 if job not found, 202 if still running.
    """
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job["status"] == "pending":
        raise HTTPException(status_code=202, detail="Job is pending")

    if job["status"] == "running":
        raise HTTPException(
            status_code=202,
            detail=f"Job is running: {job.get('progress', '...')}"
        )

    if job["status"] == "failed":
        return JobResult(
            job_id=job_id,
            status="failed",
            job_type=job["job_type"],
            error=job.get("error"),
        )

    return JobResult(
        job_id=job_id,
        status="complete",
        job_type=job["job_type"],
        result=job.get("result"),
    )
