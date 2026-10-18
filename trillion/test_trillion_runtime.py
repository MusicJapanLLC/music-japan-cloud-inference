"""Offline execution-contract tests, not 1.6T inference or GPU benchmarks."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import trillion_runtime as r


def lock():
    return {
        'schema':'mj.trillion-model-lock/1.0',
        'repo':r.MODEL,'revision':r.REVISION,'license':'mit',
        'reported_parameters':1650497936906,
        'config':{'model_type':'deepseek_v4','dspark_block_size':8,
                  'dspark_noise_token_id':1,'dspark_target_layer_ids':[1,2],
                  'dspark_markov_rank':1},
        'config_sha256':'a'*64,'license_sha256':'b'*64,
        'auxiliary_files':[{'path':name,'bytes':4,'sha256':('b' if name=='LICENSE' else 'a')*64}
                           for name in ('LICENSE','config.json','generation_config.json',
                                        'model.safetensors.index.json','tokenizer.json','tokenizer_config.json')],
        'weights':[{'path':f'model-{i:05}-of-00066.safetensors',
                    'bytes':r.WEIGHT_BYTES//66+(i <= r.WEIGHT_BYTES%66),
                    'sha256':f'{i:064x}'} for i in range(1,67)],
        'total_weight_bytes':r.WEIGHT_BYTES,
    }


def gpus():
    return [{'uuid':'GPU-'+str(i),'name':'H200','total_bytes':141*2**30,
             'free_bytes':140*2**30,'compute_capability':9.0} for i in range(8)]


class ModelLockTests(unittest.TestCase):
    def test_real_fixed_model_is_over_550b_and_latest_0813(self):
        self.assertGreater(r.validate_lock(lock())['reported_parameters'],550_000_000_000)
        self.assertEqual(r.MODEL,'deepseek-ai/DeepSeek-V4-Pro-0813')
        self.assertEqual(r.REVISION,'72e1d3230f6c080a530b0a1d46f8eb4602340597')

    def test_smaller_checkpoint_and_model_alias_rejected(self):
        for key,value in [('repo','deepseek-ai/DeepSeek-V4-Flash'),
                          ('revision','main'),('reported_parameters',550_000_000_000)]:
            d=lock();d[key]=value
            with self.assertRaises(r.ContractError):r.validate_lock(d)

    def test_count_size_and_duplicate_shards_rejected(self):
        ds=[]
        d=lock();d['weights'].pop();ds.append(d)
        d=lock();d['weights'][1]=copy.deepcopy(d['weights'][0]);ds.append(d)
        d=lock();d['total_weight_bytes']-=1;ds.append(d)
        d=lock();d['weights'][0]['bytes']+=1;ds.append(d)
        for d in ds:
            with self.subTest(),self.assertRaises(r.ContractError):r.validate_lock(d)

    def test_unsafe_or_nonsafetensor_path_and_sha_rejected(self):
        for path in ('../x.safetensors','/x.safetensors','C:/x.safetensors',
                     'x\\y.safetensors','foo.py','a//b.safetensors'):
            d=lock();d['weights'][0]['path']=path
            with self.subTest(path=path),self.assertRaises(r.ContractError):r.validate_lock(d)
        d=lock();d['weights'][0]['sha256']='not-a-sha'
        with self.assertRaises(r.ContractError):r.validate_lock(d)

    def test_native_dspark_required_and_unreviewed_code_refused(self):
        d=lock();d['config'].pop('dspark_block_size')
        with self.assertRaises(r.ContractError):r.validate_lock(d)
        d=lock();d['config']['model_type']='different'
        with self.assertRaises(r.ContractError):r.validate_lock(d)

    def test_license_mismatch_and_bool_parameters_refused(self):
        for key,value in [('license','apache-2.0'),('reported_parameters',True),
                          ('total_weight_bytes',float('nan'))]:
            d=lock();d[key]=value
            with self.assertRaises(r.ContractError):r.validate_lock(d)

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaises(r.ContractError):r.loads('{"repo":"a","repo":"b"}')

    def test_lock_digest_is_anchored_not_just_self_reported(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'model.lock.json';raw=r.canonical(lock());path.write_bytes(raw)
            with mock.patch.object(r,'LOCK_SHA256',hashlib.sha256(raw).hexdigest()):
                self.assertEqual(r.read_lock(path)['repo'],r.MODEL)
                d=lock();d['weights'][0]['sha256']='f'*64;path.write_bytes(r.canonical(d))
                with self.assertRaises(r.ContractError):r.read_lock(path)

    def test_auxiliary_assets_must_cover_tokenizer_and_match_config(self):
        d=lock();d['auxiliary_files']=[x for x in d['auxiliary_files'] if x['path']!='tokenizer.json']
        with self.assertRaises(r.ContractError):r.validate_lock(d)
        d=lock();d['auxiliary_files'][1]['sha256']='d'*64
        with self.assertRaises(r.ContractError):r.validate_lock(d)

    def test_verified_files_require_real_bytes_and_not_manifest_only(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);p=root/'tiny.safetensors';p.write_bytes(b'owned-test')
            entries=[{'path':p.name,'bytes':len(b'owned-test'),
                      'sha256':hashlib.sha256(b'owned-test').hexdigest()}]
            result=r.verify_files(root,entries)
            self.assertEqual(result['verified_files'],1)
            p.write_bytes(b'wrong-test')
            with self.assertRaises(r.ContractError):r.verify_files(root,entries)

    def test_missing_special_and_symlink_files_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);entry={'path':'missing.safetensors','bytes':1,'sha256':'a'*64}
            with self.assertRaises(r.ContractError):r.verify_files(root,[entry])
            target=root/'real.safetensors';target.write_bytes(b'x');link=root/'link.safetensors'
            try:link.symlink_to(target)
            except OSError:self.skipTest('host cannot create symlinks; exercised on Linux')
            entry.update(path=link.name,sha256=hashlib.sha256(b'x').hexdigest())
            with self.assertRaises(r.ContractError):r.verify_files(root,[entry])


class PreflightTests(unittest.TestCase):
    def test_small_free_workers_are_rejected_before_launch(self):
        for hw in ([],gpus()[:1],[{**gpus()[0],'free_bytes':96*2**30}]):
            p=r.preflight(lock(),hw,'0.25.0')
            self.assertFalse(p['eligible_for_launch'])
            self.assertFalse(p['weights_verified'])
            self.assertFalse(p['model_inference_verified'])

    def test_eight_large_idle_cards_satisfy_planning_not_prove_running(self):
        p=r.preflight(lock(),gpus(),'0.25.0')
        self.assertTrue(p['eligible_for_launch'])
        self.assertFalse(p['model_inference_verified'])
        self.assertTrue(p['memory_requirement_is_estimate'])

    def test_occupied_or_old_cards_and_old_engine_rejected(self):
        cases=[([{**x,'free_bytes':32*2**30} for x in gpus()],'0.25.0'),
               ([{**x,'compute_capability':8.0} for x in gpus()],'0.25.0'),
               (gpus(),'0.20.0'),(gpus(),'unknown')]
        for hw,v in cases:
            self.assertFalse(r.preflight(lock(),hw,v)['eligible_for_launch'])

    def test_duplicate_gpu_identity_is_not_extra_memory(self):
        with self.assertRaises(r.ContractError):r.preflight(lock(),[gpus()[0]]*8,'0.25.0')

    def test_command_keeps_target_model_and_exact_optimized_verification(self):
        cmd=r.build_command(Path('/owned/weights'),32768,speculation=True)
        self.assertIn('--enable-prefix-caching',cmd)
        self.assertIn('--enable-chunked-prefill',cmd)
        self.assertEqual(cmd[cmd.index('--host')+1],'127.0.0.1')
        self.assertNotIn('--trust-remote-code',cmd)
        spec=json.loads(cmd[cmd.index('--speculative-config')+1])
        self.assertEqual(spec['method'],'dspark')
        self.assertEqual(spec['num_speculative_tokens'],7)
        self.assertNotIn('synthetic',json.dumps(spec))

    def test_baseline_arm_disables_only_speculation(self):
        a=r.build_command(Path('/owned/w'),8192,speculation=True)
        b=r.build_command(Path('/owned/w'),8192,speculation=False)
        i=a.index('--speculative-config');del a[i:i+2]
        self.assertEqual(a,b)

    def test_context_and_shell_injection_inputs_not_accepted(self):
        for context in (0,1024,1048576,True):
            with self.assertRaises(r.ContractError):r.build_command(Path('/x'),context,True)

    def test_gpu_csv_parsing(self):
        hw=r.parse_gpu_csv('GPU-a, NVIDIA H200, 143771, 143000, 9.0\n')
        self.assertEqual(hw[0]['free_bytes'],143000*2**20)
        for raw in ('wrong','GPU-a, H200, 2, 4, 9.0','GPU-a, H200, NaN, 1, 9.0'):
            with self.assertRaises(r.ContractError):r.parse_gpu_csv(raw)

    def test_cpu_only_execute_does_not_launch_or_read_credentials(self):
        with mock.patch.object(r,'read_lock',return_value=lock()), \
             mock.patch.object(r,'collect_gpus',return_value=[]), \
             mock.patch.object(r.importlib.metadata,'version',return_value='0.25.0'), \
             mock.patch.object(r.subprocess,'run') as run:
            result=r.main(['--lock','unused','--weights-dir','unused','--execute',
                           '--api-key-file','must-not-be-read'])
        self.assertEqual(result,2)
        run.assert_not_called()

    def test_environment_drops_provider_keys_and_requires_own_auth(self):
        env=r.runtime_environment({'OPENAI_API_KEY':'secret','HF_TOKEN':'other',
                                   'PATH':'/bin','HOME':'/owned'},'a'*40)
        self.assertNotIn('OPENAI_API_KEY',env);self.assertNotIn('HF_TOKEN',env)
        self.assertEqual(env['VLLM_API_KEY'],'a'*40)
        self.assertEqual(env['HF_HUB_OFFLINE'],'1')
        with self.assertRaises(r.ContractError):r.runtime_environment({},'weak')


if __name__=='__main__':unittest.main()
