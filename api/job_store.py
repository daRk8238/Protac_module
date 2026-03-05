"""
In-memory job store for tracking async background jobs.
For production, swap this with Redis.
"""

from datetime import datetime
from typing import Optional
import threading


class JobStore:
    """Thread-safe in-memory job store."""

    def __init__(self):
        self._jobs: dict = {}
        self._lock = threading.Lock()

    def create(self, job_id: str, job_type: str, meta: dict = None) -> dict:
        now = datetime.utcnow().isoformat()
        job = {
            "job_id": job_id,
            "job_type": job_type,
            "status": "pending",
            "created_at": now,
            "updated_at": now,
            "progress": None,
            "result": None,
            "error": None,
            "meta": meta or {},
        }
        with self._lock:
            self._jobs[job_id] = job
        return job

    def update(self, job_id: str, status: str = None, progress: str = None):
        with self._lock:
            if job_id not in self._jobs:
                return
            if status:
                self._jobs[job_id]["status"] = status
            if progress:
                self._jobs[job_id]["progress"] = progress
            self._jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()

    def complete(self, job_id: str, result: dict):
        with self._lock:
            if job_id not in self._jobs:
                return
            self._jobs[job_id]["status"] = "complete"
            self._jobs[job_id]["result"] = result
            self._jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()

    def fail(self, job_id: str, error: str):
        with self._lock:
            if job_id not in self._jobs:
                return
            self._jobs[job_id]["status"] = "failed"
            self._jobs[job_id]["error"] = error
            self._jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()

    def get(self, job_id: str) -> Optional[dict]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_all(self) -> list:
        with self._lock:
            return list(self._jobs.values())


# Singleton
job_store = JobStore()
