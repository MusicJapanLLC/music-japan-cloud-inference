import hashlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import experimental_gptoss120b as m
import official120b_chat as chat

class OSS120BTests(unittest.TestCase):
    def test_original_official_model_derivative(self):
        self.assertEqual(m.REPO,"ggml-org/gpt-oss-120b-GGUF")
        self.assertEqual(m.SIZE_BYTES,63_387_346_208)
        self.assertEqual(len(m.SHA256),64)
        self.assertTrue(m.SOURCE.startswith("https://huggingface.co/"))
        self.assertIn(m.REV,m.SOURCE)
    def test_fake_weight_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/"not-real.gguf"
            p.write_bytes(b"fake")
            with self.assertRaisesRegex(ValueError,"INTEGRITY"):
                m.verify_weights(p)
    def test_local_resource_guard_fails_if_disk_small(self):
        with tempfile.TemporaryDirectory() as t:
            with patch.object(m,"linux_memory_gib",return_value=16), patch.object(m.shutil,"disk_usage",return_value=type("D",(),{"free":10*2**30,"total":144*2**30})()):
                with self.assertRaisesRegex(RuntimeError,"DISK"):
                    m.local_resources(Path(t))
    def test_local_resource_guard_fails_if_ram_small(self):
        with tempfile.TemporaryDirectory() as t:
            with patch.object(m,"linux_memory_gib",return_value=8), patch.object(m.shutil,"disk_usage",return_value=type("D",(),{"free":100*2**30,"total":144*2**30})()):
                with self.assertRaisesRegex(RuntimeError,"RAM"):
                    m.local_resources(Path(t))
    def test_offline_default_never_downloads_model(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)/"report.json"
            res=subprocess.run([sys.executable,m.__file__,
                "--binary","missing","--weights",t,"--out",str(out)],
                capture_output=True,text=True,timeout=8)
            self.assertEqual(res.returncode,0)
            self.assertIn("DRY_RUN",res.stdout)
            self.assertFalse(out.exists())
    def test_url_allowlist_on_http(self):
        with self.assertRaises(ValueError):
            m.client("a-key","http://unsafe.tld/api",timeout=1)
    def test_exact_params_not_imitating_gpt6(self):
        self.assertEqual(m.ID,"gpt-oss-120b-MXFP4")
        self.assertNotIn("claude",m.ID.lower())
    def test_cgroup_oom_counters_only(self):
        with tempfile.TemporaryDirectory() as t:
            fp=Path(t)/"events"
            fp.write_text("low 1\nhigh 2\nmax 3\noom 4\noom_kill 5\n",encoding="utf8")
            self.assertEqual(m.cgroup_memory_events(fp),{"high":2,"max":3,"oom":4,"oom_kill":5})

    def test_missing_cgroup_is_empty(self):
        self.assertEqual(m.cgroup_memory_events(Path("/nonexistent/mj-cgroup-events")),{})

    def test_log_diagnostic_only_categories(self):
        with tempfile.TemporaryDirectory() as t:
            fp=Path(t)/"log"
            fp.write_text("failed to allocate buffer; OOM killer; mmap=true; account-key-not-for-logs",encoding="utf8")
            result=m.diagnose_runtime(fp)
            self.assertTrue(result["oom"])
            self.assertTrue(result["allocation"])
            self.assertTrue(result["mmap"])
            self.assertNotIn("account-key-not-for-logs",str(result))

    def test_limited_loading_grace(self):
        self.assertLessEqual(m.LOAD_LIMIT_SECONDS,600)
        self.assertGreaterEqual(m.LOAD_LIMIT_SECONDS,300)

    def test_compute_claim_not_baked_in(self):
        self.assertGreater(m.MIN_DISK_FREE_GIB,60)
        self.assertEqual(m.MIN_RAM_GIB,12)

class WeightDownloadTests(unittest.TestCase):
    """Use small real digest fixtures and a fake curl; never download weights."""
    def setUp(self):
        directory=tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.out=Path(directory.name)/"weights.gguf"
        self.partial=self.out.with_suffix(".part")
        self.data=b"synthetic weight fixture"
        self.clock=0.0
        for context in (
            patch.object(m,"SIZE_BYTES",len(self.data)),
            patch.object(m,"SHA256",hashlib.sha256(self.data).hexdigest()),
            patch.object(m.time,"monotonic",side_effect=lambda:self.clock),
            patch.object(m.time,"sleep",side_effect=self.advance),
            redirect_stdout(io.StringIO()),
        ):
            context.__enter__()
            self.addCleanup(context.__exit__,None,None,None)

    def advance(self,seconds):
        self.clock+=seconds

    def test_http2_cancel_resumes_current_partial_over_http1_and_verifies(self):
        offsets=[]
        budgets=[]
        def fake_curl(command,check,timeout):
            self.assertTrue(check)
            self.assertIn("--http1.1",command)
            self.assertEqual(command[1],"--disable")
            self.assertEqual(command[command.index("--continue-at")+1],"-")
            self.assertNotIn("--retry",command)
            self.assertEqual(command[-1],m.SOURCE)
            self.assertIn(m.REV,command[-1])
            self.assertEqual(float(command[command.index("--max-time")+1]),timeout)
            self.assertFalse(self.out.exists())
            offsets.append(self.partial.stat().st_size if self.partial.exists() else 0)
            budgets.append(timeout)
            if len(offsets)==1:
                self.partial.write_bytes(self.data[:7])
                self.advance(200)
                raise subprocess.CalledProcessError(92,command)
            with self.partial.open("ab") as fp:fp.write(self.data[7:])
            self.advance(100)
            return subprocess.CompletedProcess(command,0)
        with patch.object(m.subprocess,"run",side_effect=fake_curl):
            proof=m.retrieve_public_weights(self.out)
        self.assertEqual(offsets,[0,7])
        self.assertEqual(budgets,[1200,999])
        self.assertEqual(proof["sha256"],hashlib.sha256(self.data).hexdigest())
        self.assertEqual(self.out.read_bytes(),self.data)
        self.assertFalse(self.partial.exists())

    def test_transient_failures_stop_after_three_attempts(self):
        def fail(command,**kwargs):
            with self.partial.open("ab") as fp:fp.write(b"x")
            raise subprocess.CalledProcessError(56,command)
        with patch.object(m.subprocess,"run",side_effect=fail) as curl:
            with self.assertRaisesRegex(RuntimeError,"WEIGHTS_DOWNLOAD_CURL_56"):
                m.retrieve_public_weights(self.out)
        self.assertEqual(curl.call_count,3)
        self.assertFalse(self.out.exists())
        self.assertFalse(self.partial.exists())

    def test_http_rejection_write_failure_and_bad_certificate_do_not_retry(self):
        for code in (22,23,60):
            with self.subTest(curl_exit=code), patch.object(m.subprocess,"run",
                    side_effect=subprocess.CalledProcessError(code,["curl"])) as curl:
                with self.assertRaisesRegex(RuntimeError,f"WEIGHTS_DOWNLOAD_CURL_{code}"):
                    m.retrieve_public_weights(self.out)
                curl.assert_called_once()
                self.assertFalse(self.out.exists())

    def test_total_time_is_shared_across_attempts(self):
        def fail(command,**kwargs):
            self.partial.write_bytes(b"part")
            self.advance(1200)
            raise subprocess.CalledProcessError(18,command)
        with patch.object(m.subprocess,"run",side_effect=fail) as curl:
            with self.assertRaisesRegex(TimeoutError,"TIME_BUDGET"):
                m.retrieve_public_weights(self.out)
        curl.assert_called_once()
        self.assertFalse(self.partial.exists())
        self.assertFalse(self.out.exists())

    def test_python_timeout_is_bounded_and_not_retried(self):
        with patch.object(m.subprocess,"run",
                side_effect=subprocess.TimeoutExpired(["curl"],1200)) as curl:
            with self.assertRaisesRegex(TimeoutError,"TIME_BUDGET"):
                m.retrieve_public_weights(self.out)
        curl.assert_called_once()
        self.assertEqual(curl.call_args.kwargs["timeout"],1200)
        self.assertFalse(self.out.exists())

    def test_wrong_size_or_digest_never_publishes_or_retries(self):
        for data in (self.data[:-1],b"x"*len(self.data)):
            with self.subTest(bytes=len(data)):
                def fake_curl(command,**kwargs):
                    self.partial.write_bytes(data)
                    return subprocess.CompletedProcess(command,0)
                with patch.object(m.subprocess,"run",side_effect=fake_curl) as curl:
                    with self.assertRaisesRegex(ValueError,"INTEGRITY_MISMATCH"):
                        m.retrieve_public_weights(self.out)
                curl.assert_called_once()
                self.assertFalse(self.out.exists())
                self.assertFalse(self.partial.exists())

    def test_cached_file_is_still_verified_without_network(self):
        self.out.write_bytes(self.data)
        with patch.object(m.subprocess,"run") as curl:
            self.assertEqual(m.retrieve_public_weights(self.out)["bytes"],len(self.data))
            self.out.write_bytes(b"x"*len(self.data))
            with self.assertRaisesRegex(ValueError,"INTEGRITY_MISMATCH"):
                m.retrieve_public_weights(self.out)
        curl.assert_not_called()

    def test_download_error_or_invalid_bytes_prevents_model_start(self):
        for code in (22,None):
            with self.subTest(curl_exit=code):
                def fake_curl(command,**kwargs):
                    if code:raise subprocess.CalledProcessError(code,command)
                    Path(command[command.index("--output")+1]).write_bytes(b"invalid")
                    return subprocess.CompletedProcess(command,0)
                with patch.dict(os.environ,{"MJ_PUBLIC_RESEARCH_QUESTION":"Calculate 47 times 83."}), \
                     patch.object(m,"local_resources",return_value={"ram_gib":16,"disk_free_gib":86}), \
                     patch.object(m.subprocess,"run",side_effect=fake_curl), \
                     patch.object(chat,"run") as infer:
                    with self.assertRaises((RuntimeError,ValueError)):
                        chat.main(["--execute","--binary","missing",
                            "--weights-dir",str(self.out.parent),
                            "--out",str(self.out.parent/"answer.json")])
                infer.assert_not_called()
                self.assertFalse((self.out.parent/"answer.json").exists())
                self.assertFalse((self.out.parent/m.FILENAME).exists())

if __name__=="__main__":
    unittest.main()
