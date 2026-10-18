import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
import current_frontier_eval as evaluator


class EvaluationTests(unittest.TestCase):
    def test_default_is_offline_dry_run(self):
        with patch.object(evaluator,'run') as run, contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(evaluator.main(['--profile','qwen38-27b-iq3s']),0)
        run.assert_not_called()
        self.assertFalse(json.loads(output.getvalue())['model_inference'])

    def test_exact_profiles_and_runtime_pins(self):
        self.assertEqual(len(evaluator.PROFILES),2)
        nemo=evaluator.PROFILES['nemotron35-120b-iq3xxs']
        self.assertEqual(sum(x[1] for x in nemo['files']),59764210568)
        self.assertEqual([x[0].split('-')[-1] for x in nemo['files']],['00003.gguf']*3)
        self.assertEqual(len(evaluator.LLAMA_COMMIT),40)
        for profile in evaluator.PROFILES.values():
            self.assertEqual(len(profile['revision']),40)
            for name,size,digest in profile['files']:
                self.assertFalse(Path(name).is_absolute())
                self.assertNotIn('..',Path(name).parts)
                self.assertGreater(size,0)
                self.assertEqual(len(digest),64)

    def test_hash_and_size_both_required(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'weight'
            path.write_bytes(b'abc')
            correct=('weight',3,evaluator.hashlib.sha256(b'abc').hexdigest())
            evaluator.verify_file(path,correct)
            for item in [('weight',4,correct[2]),('weight',3,'0'*64)]:
                with self.assertRaises(ValueError):evaluator.verify_file(path,item)

    def test_download_shards_verified_sequentially_before_return(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            files=[('shards/'+str(i)+'.gguf',1,evaluator.hashlib.sha256(bytes([i])).hexdigest()) for i in range(3)]
            events=[]
            def download(command,**kwargs):
                path=Path(command[command.index('--output')+1])
                index=int(path.name.split('.')[0]);events.append(index)
                path.write_bytes(bytes([index]))
            with patch.object(evaluator.subprocess,'run',side_effect=download):
                paths=evaluator.download_weights(folder,dict(repo='public/repo',revision='a'*40,files=files),evaluator.time.monotonic()+20)
            self.assertEqual(events,[0,1,2])
            self.assertEqual(len(paths),3)
            self.assertFalse(list(folder.rglob('*.part')))

    def test_bad_shard_never_activated_or_followed_by_download(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            profile=dict(repo='public/repo',revision='a'*40,files=[('one',1,'0'*64),('two',1,'0'*64)])
            def corrupt(command,**kwargs):Path(command[command.index('--output')+1]).write_bytes(b'x')
            with patch.object(evaluator.subprocess,'run',side_effect=corrupt) as download:
                with self.assertRaises(ValueError):evaluator.download_weights(folder,profile,evaluator.time.monotonic()+20)
            self.assertEqual(download.call_count,1)
            self.assertFalse((folder/'one').exists())
            self.assertFalse(list(folder.glob('*.part')))

    def test_strict_and_fence_scores_separate_without_rewriting(self):
        text=json.dumps(evaluator.EXPECTED)
        self.assertTrue(evaluator.grade_text(text)['strict_contract_passed'])
        fenced='```json\n'+text+'\n```'
        result=evaluator.grade_text(fenced)
        self.assertFalse(result['strict_contract_passed'])
        self.assertTrue(result['fence_contract_passed'])
        self.assertFalse(evaluator.grade_text(text+' explanation')['strict_json'])
        self.assertTrue(evaluator.grade_text(text.replace('50','50.0').replace('11','11.0'))['strict_contract_passed'])
        self.assertFalse(evaluator.grade_text(text.replace('50','true'))['strict_contract_passed'])

    def test_generation_body_is_fixed_and_bounded(self):
        body=evaluator.model_body('qwen38-27b-iq3s')
        self.assertEqual(body['reasoning_effort'],'low')
        self.assertIs(body['chat_template_kwargs']['enable_thinking'],True)
        self.assertIs(body['chat_template_kwargs']['preserve_thinking'],True)
        self.assertEqual(body['max_tokens'],1024)
        self.assertEqual(body['messages'],[dict(role='user',content=evaluator.PROMPT)])
        nemo=evaluator.model_body('nemotron35-120b-iq3xxs')
        self.assertNotIn('reasoning_effort',nemo)
        self.assertIs(nemo['chat_template_kwargs']['enable_thinking'],True)
        self.assertIs(nemo['chat_template_kwargs']['medium_effort'],True)

    def test_resource_failure_makes_no_network_or_process(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);binary=root/'llama-server';binary.write_text('synthetic')
            with patch.object(evaluator,'resources',side_effect=ValueError('SYNTHETIC_SECRET')), patch.object(evaluator,'download_weights') as download, patch.object(evaluator.subprocess,'Popen') as spawn, contextlib.redirect_stdout(io.StringIO()) as log:
                code=evaluator.run('qwen38-27b-iq3s',binary,root/'weights',root/'receipt.json')
            self.assertEqual(code,2);download.assert_not_called();spawn.assert_not_called()
            record=json.loads((root/'receipt.json').read_text())
            self.assertEqual(record['generation_attempts'],0)
            self.assertEqual(record['phase'],'resource_guard')
            self.assertEqual(record['error_code'],'OPERATION_FAILED')
            self.assertNotIn('SYNTHETIC_SECRET',log.getvalue())

    def lifecycle(self, fail_generation=False, finish='stop'):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        root=Path(temp.name);binary=root/'llama-server';binary.write_text('synthetic')
        child=Mock();child.poll.return_value=None
        requests=[]
        def local_request(port,token,route,timeout,body=None):
            requests.append(route)
            if route=='/health':return {'status':'ok'}
            if route=='/v1/models':return {'data':[{'id':'qwen38-27b-iq3s'}]}
            if fail_generation:raise TimeoutError('SYNTHETIC_SECRET')
            return dict(model='qwen38-27b-iq3s',choices=[dict(message=dict(content=json.dumps(evaluator.EXPECTED)),finish_reason=finish)],usage={'completion_tokens':100})
        with patch.object(evaluator,'resources',return_value={}), patch.object(evaluator,'download_weights',return_value=[root/'synthetic.gguf']), patch.object(evaluator.subprocess,'run',return_value=Mock(stdout=evaluator.LLAMA_COMMIT,stderr='')), patch.object(evaluator.subprocess,'Popen',return_value=child) as spawn, patch.object(evaluator,'request',side_effect=local_request), patch.object(evaluator.socket,'socket') as local_socket, contextlib.redirect_stdout(io.StringIO()):
            local_socket.return_value.__enter__.return_value.getsockname.return_value=('127.0.0.1',18895)
            code=evaluator.run('qwen38-27b-iq3s',binary,root/'weights',root/'receipt.json')
        record=json.loads((root/'receipt.json').read_text())
        return code,record,requests,child,spawn.call_args

    def test_success_preserves_response_one_generation_and_own_child_cleanup(self):
        code,record,requests,child,spawn=self.lifecycle()
        self.assertEqual(code,0)
        self.assertEqual(requests.count('/v1/chat/completions'),1)
        self.assertEqual(record['finish_reason'],'stop')
        self.assertEqual(record['usage'],{'completion_tokens':100})
        self.assertTrue(record['grading']['strict_contract_passed'])
        self.assertEqual(record['raw_response']['choices'][0]['message']['content'],record['final_content'])
        child.terminate.assert_called_once_with()
        self.assertEqual(set(spawn.kwargs['env'])-{'PATH','LD_LIBRARY_PATH','LANG','LC_ALL','TMPDIR'},{'LLAMA_API_KEY'})
        self.assertIn('--no-repack',spawn.args[0])

    def test_visible_cutoff_preserved_but_not_successful_completion(self):
        code,record,requests,child,_=self.lifecycle(finish='length')
        self.assertEqual(code,2)
        self.assertTrue(record['inference_confirmed'])
        self.assertTrue(record['output_truncated'])
        self.assertFalse(record['complete_final_answer'])
        self.assertTrue(record['final_content'])
        self.assertEqual(record['finish_reason'],'length')
        self.assertEqual(requests.count('/v1/chat/completions'),1)

    def test_generation_timeout_never_retries_and_cleans_up(self):
        code,record,requests,child,_=self.lifecycle(True)
        self.assertEqual(code,2)
        self.assertEqual(requests.count('/v1/chat/completions'),1)
        self.assertEqual(record['generation_attempts'],1)
        self.assertFalse(record['inference_confirmed'])
        self.assertEqual(record['error_type'],'TimeoutError')
        self.assertEqual(record['error_code'],'OPERATION_TIMEOUT')
        self.assertEqual(record['phase'],'generation')
        self.assertGreaterEqual(record['generation_seconds'],0)
        self.assertNotIn('SYNTHETIC_SECRET',json.dumps(record))
        child.terminate.assert_called_once_with()

    def test_deadline_expiry_prevents_more_work(self):
        with self.assertRaises(TimeoutError):evaluator.time_left(evaluator.time.monotonic()-1,900)


if __name__=='__main__':unittest.main()
