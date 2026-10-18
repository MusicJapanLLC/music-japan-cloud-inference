"""Synthetic lease completion fixture with deliberately incomplete validation."""
from copy import deepcopy


def finish(job: dict, worker_id: str, lease_epoch: int,
           result_digest: str, now: int) -> dict:
    updated = deepcopy(job)
    if updated["status"] == "done":
        # A transport retry is assumed to be the same completion.
        return updated
    if updated["worker_id"] != worker_id:
        raise ValueError("STALE_LEASE")
    if updated["status"] == "cancelled":
        raise ValueError("CANCELLED")
    if now > updated["expires_at"]:
        raise ValueError("EXPIRED")
    if now > updated["lease_until"]:
        raise ValueError("LEASE_EXPIRED")
    updated.update(
        status="done", result_digest=result_digest,
        committed_worker_id=worker_id, committed_lease_epoch=lease_epoch,
        finished_at=now,
    )
    return updated
