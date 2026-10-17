import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cloud_probe as m

class CloudProbeTests(unittest.TestCase):
    def test_integrity(self):
        a=m.signed_record({"ram":16,"local":True})
        self.assertTrue(m.verify(a))
        a["ram"]=64
        self.assertFalse(m.verify(a))
    def test_free_vm_not_mislabeled_high_gpu(self):
        with patch.object(m,"linux_memory_gib",return_value=16),patch.object(m,"nvidia_names",return_value=[]),patch.object(m.platform,"system",return_value="Linux"):
            record=m.collect()
            self.assertFalse(record["minimum_requirements"]["ram_64gib"])
            self.assertFalse(record["minimum_requirements"]["nvidia_gpu_observed"])
            self.assertEqual(record["paid_jobs_or_credits_used_by_script"],0)
            self.assertTrue(m.verify(record))
    def test_memory_stat(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"meminfo"
            p.write_text("MemTotal:       16777216 kB\nMemFree: 42 kB\n")
            self.assertEqual(m.linux_memory_gib(p),16)
    def test_missing_memory_fails_closed(self):
        self.assertIsNone(m.linux_memory_gib(Path("/definitely/not/here/meminfo")))
    def test_hash_stable(self):
        a={"one":1,"two":{"a":2,"z":0}}
        b={"two":{"z":0,"a":2},"one":1}
        self.assertEqual(m.normalized(a),m.normalized(b))
if __name__=="__main__": unittest.main()
