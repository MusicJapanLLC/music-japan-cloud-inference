"""Public acceptance tests; not a secret or held-out capability benchmark."""
from copy import deepcopy
import unittest

from job_receipt import finish


class LeaseFinalizationAcceptance(unittest.TestCase):
    def job(self, **changes):
        value = {
            "job_id": "synthetic-job-1", "status": "running",
            "worker_id": "worker-a", "lease_epoch": 4,
            "lease_until": 110, "expires_at": 200,
            "result_digest": None, "committed_worker_id": None,
            "committed_lease_epoch": None, "finished_at": None,
            "metadata": {"notes": ["preserve"], "attempt": 2},
        }
        value.update(changes)
        return value

    def done(self, **changes):
        value = self.job(
            status="done", result_digest="digest-a",
            committed_worker_id="worker-a", committed_lease_epoch=4,
            finished_at=90,
        )
        value.update(changes)
        return value

    def reject(self, job, code, *, worker="worker-a", epoch=4,
               digest="digest-a", now=100):
        before = deepcopy(job)
        with self.assertRaises(ValueError) as caught:
            finish(job, worker, epoch, digest, now)
        self.assertEqual(str(caught.exception), code)
        self.assertEqual(job, before, "rejection must not mutate input")

    def test_first_completion_commits_only_the_five_receipt_fields(self):
        job = self.job()
        before = deepcopy(job)
        expected = deepcopy(job)
        expected.update(status="done", result_digest="digest-a",
                        committed_worker_id="worker-a",
                        committed_lease_epoch=4, finished_at=100)
        actual = finish(job, "worker-a", 4, "digest-a", 100)
        self.assertEqual(actual, expected)
        self.assertIsNot(actual, job)
        self.assertEqual(job, before)

    def test_success_deep_copies_unknown_nested_metadata(self):
        job = self.job(extra={"items": [{"value": "original"}]})
        actual = finish(job, "worker-a", 4, "digest-a", 100)
        actual["metadata"]["notes"].append("changed")
        actual["extra"]["items"][0]["value"] = "changed"
        self.assertEqual(job["metadata"]["notes"], ["preserve"])
        self.assertEqual(job["extra"], {"items": [{"value": "original"}]})

    def test_running_wrong_worker_is_stale(self):
        self.reject(self.job(), "STALE_LEASE", worker="worker-b")

    def test_running_old_epoch_same_worker_is_stale(self):
        self.reject(self.job(), "STALE_LEASE", epoch=3)

    def test_running_future_epoch_cannot_claim_a_lease(self):
        self.reject(self.job(), "STALE_LEASE", epoch=5)

    def test_running_stale_identity_precedes_elapsed_deadlines(self):
        self.reject(self.job(lease_until=80, expires_at=90),
                    "STALE_LEASE", epoch=3, now=100)

    def test_job_expiry_exact_boundary_is_rejected(self):
        self.reject(self.job(lease_until=300), "EXPIRED", now=200)

    def test_lease_expiry_exact_boundary_is_rejected(self):
        self.reject(self.job(), "LEASE_EXPIRED", now=110)

    def test_both_deadlines_elapsed_reports_job_expiry(self):
        self.reject(self.job(), "EXPIRED", now=250)

    def test_one_tick_before_both_deadlines_can_complete(self):
        job = self.job(lease_until=101, expires_at=101)
        actual = finish(job, "worker-a", 4, "digest-a", 100)
        self.assertEqual(actual["status"], "done")
        self.assertEqual(actual["finished_at"], 100)
        self.assertEqual(actual["lease_until"], 101)
        self.assertEqual(actual["expires_at"], 101)

    def test_cancelled_state_precedes_wrong_identity_and_expiry(self):
        self.reject(self.job(status="cancelled", expires_at=50),
                    "CANCELLED", worker="worker-b", epoch=3, now=500)

    def test_expired_state_rejects_even_with_future_deadlines(self):
        self.reject(self.job(status="expired", lease_until=1000,
                             expires_at=2000), "EXPIRED", now=100)

    def test_expired_state_precedes_wrong_worker(self):
        self.reject(self.job(status="expired"), "EXPIRED", worker="worker-b")

    def test_unknown_status_precedes_identity_and_deadlines(self):
        self.reject(self.job(status="queued", expires_at=50),
                    "INVALID_STATUS", worker="worker-b", now=500)

    def test_exact_terminal_replay_preserves_original_receipt(self):
        job = self.done()
        before = deepcopy(job)
        actual = finish(job, "worker-a", 4, "digest-a", 100)
        self.assertEqual(actual, before)
        self.assertEqual(job, before)
        self.assertIsNot(actual, job)
        actual["metadata"]["notes"].append("changed")
        self.assertEqual(job["metadata"]["notes"], ["preserve"])

    def test_exact_terminal_replay_survives_expiry_and_assignment_change(self):
        job = self.done(worker_id="worker-b", lease_epoch=5,
                        lease_until=110, expires_at=200)
        before = deepcopy(job)
        actual = finish(job, "worker-a", 4, "digest-a", 500)
        self.assertEqual(actual, before)
        self.assertEqual(actual["finished_at"], 90)
        self.assertEqual(job, before)

    def test_terminal_different_digest_conflicts_even_after_expiry(self):
        self.reject(self.done(), "RESULT_CONFLICT", digest="digest-b", now=500)

    def test_terminal_wrong_worker_cannot_replay_matching_digest(self):
        self.reject(self.done(), "STALE_LEASE", worker="worker-b")

    def test_terminal_wrong_epoch_cannot_replay_matching_digest(self):
        self.reject(self.done(), "STALE_LEASE", epoch=3)

    def test_terminal_current_assignment_cannot_replace_committed_identity(self):
        self.reject(self.done(worker_id="worker-b", lease_epoch=5,
                              lease_until=1000, expires_at=2000),
                    "STALE_LEASE", worker="worker-b", epoch=5, now=100)

    def test_terminal_stale_identity_precedes_digest_conflict(self):
        self.reject(self.done(), "STALE_LEASE", worker="worker-b",
                    digest="digest-b", now=500)


if __name__ == "__main__":
    unittest.main()
