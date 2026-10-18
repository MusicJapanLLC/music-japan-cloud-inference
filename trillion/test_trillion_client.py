"""Synthetic transport and concurrency tests, NOT model-quality measurements."""
import copy
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import unittest
import trillion_client as c
from trillion_runtime import SERVED_MODEL


def body(text='Generate a public code example.'):
    return {'model':SERVED_MODEL,'messages':[{'role':'user','content':text}],
            'metadata':{'data_class':'public'},'temperature':0,'max_tokens':512,'stream':False}


def answer():
    return {'model':SERVED_MODEL,'choices':[{'message':{'role':'assistant','content':'example'},
            'finish_reason':'stop'}],'usage':{'prompt_tokens':12,'completion_tokens':5,'total_tokens':17}}


class OptimizerTests(unittest.TestCase):
    def test_eight_duplicate_requests_generate_once(self):
        gate=threading.Barrier(8);lock=threading.Lock();calls=[]
        def provider(request):
            with lock:calls.append(copy.deepcopy(request))
            time.sleep(.1)
            return answer()
        pool=c.SingleFlight(provider,ttl_seconds=30)
        def one(_):gate.wait();return pool.complete(body(),namespace='owner-test',reuse=True)
        with ThreadPoolExecutor(max_workers=8) as ex:results=list(ex.map(one,range(8)))
        self.assertEqual(len(calls),1)
        self.assertEqual(sum(x['mj_optimization']['inference_performed_by_this_request'] for x in results),1)
        self.assertTrue(all(x['model']==SERVED_MODEL for x in results))
        self.assertFalse(any(x['mj_optimization']['model_quality_uplift_claimed'] for x in results))

    def test_no_reuse_is_the_default(self):
        calls=[]
        pool=c.SingleFlight(lambda b:(calls.append(1) or answer()))
        pool.complete(body(),namespace='owner-test');pool.complete(body(),namespace='owner-test')
        self.assertEqual(len(calls),2)

    def test_namespaces_prompts_and_budgets_do_not_share(self):
        calls=[];pool=c.SingleFlight(lambda b:(calls.append(1) or answer()))
        pool.complete(body(),namespace='a',reuse=True)
        pool.complete(body(),namespace='b',reuse=True)
        pool.complete(body('Other public input'),namespace='a',reuse=True)
        r=body();r['max_tokens']=1024;pool.complete(r,namespace='a',reuse=True)
        self.assertEqual(len(calls),4)

    def test_mutating_returned_response_does_not_poison_cache(self):
        pool=c.SingleFlight(lambda b:answer())
        first=pool.complete(body(),namespace='a',reuse=True)
        first['choices'][0]['message']['content']='poisoned'
        second=pool.complete(body(),namespace='a',reuse=True)
        self.assertEqual(second['choices'][0]['message']['content'],'example')

    def test_cache_expires_and_is_bounded(self):
        clock=[1.0];calls=[]
        pool=c.SingleFlight(lambda b:(calls.append(1) or answer()),ttl_seconds=1,max_entries=1,clock=lambda:clock[0])
        pool.complete(body(),namespace='a',reuse=True);clock[0]=3
        pool.complete(body(),namespace='a',reuse=True)
        pool.complete(body('other'),namespace='a',reuse=True)
        pool.complete(body(),namespace='a',reuse=True)
        self.assertEqual(len(calls),4)

    def test_cache_ttl_does_not_slide_on_each_hit(self):
        clock=[1.0];calls=[]
        pool=c.SingleFlight(lambda b:(calls.append(1) or answer()),ttl_seconds=2,clock=lambda:clock[0])
        pool.complete(body(),namespace='a',reuse=True)
        clock[0]=2.9;pool.complete(body(),namespace='a',reuse=True)
        clock[0]=3.1;pool.complete(body(),namespace='a',reuse=True)
        self.assertEqual(len(calls),2)

    def test_capacity_also_bounds_nonreuse_requests(self):
        entered=threading.Event();release=threading.Event()
        def provider(request):entered.set();release.wait(timeout=3);return answer()
        pool=c.SingleFlight(provider,max_inflight=1)
        with ThreadPoolExecutor(max_workers=1) as ex:
            pending=ex.submit(pool.complete,body(),namespace='a')
            self.assertTrue(entered.wait(1))
            try:
                with self.assertRaises(c.InferenceError):pool.complete(body('other'),namespace='a')
            finally:release.set()
            self.assertEqual(pending.result()['model'],SERVED_MODEL)

    def test_failed_and_truncated_requests_are_never_cached(self):
        for kind in ('error','length','wrong-model'):
            calls=[]
            def provider(request):
                calls.append(1)
                if kind=='error':raise RuntimeError('private upstream error')
                result=answer()
                if kind=='length':result['choices'][0]['finish_reason']='length'
                else:result['model']='small-model'
                return result
            pool=c.SingleFlight(provider)
            for _ in range(2):
                with self.assertRaises(c.InferenceError):pool.complete(body(),namespace='a',reuse=True)
            self.assertEqual(len(calls),2)

    def test_validation_precedes_cache_or_network(self):
        calls=[];pool=c.SingleFlight(lambda b:(calls.append(1) or answer()))
        bads=[]
        b=body();b['metadata']['data_class']='private';bads.append(b)
        b=body();b['model']='gpt-6-astra';bads.append(b)
        b=body();b['api_key']='secret';bads.append(b)
        b=body();b['temperature']=float('nan');bads.append(b)
        for b in bads:
            with self.assertRaises(c.InferenceError):pool.complete(b,namespace='a',reuse=True)
        self.assertEqual(calls,[])

    def test_stochastic_reuse_requires_deterministic_parameters(self):
        b=body();b['temperature']=.8
        pool=c.SingleFlight(lambda x:answer())
        with self.assertRaises(c.InferenceError):pool.complete(b,namespace='a',reuse=True)

    def test_unknown_fields_and_empty_namespace_are_refused(self):
        pool=c.SingleFlight(lambda b:answer())
        with self.assertRaises(c.InferenceError):pool.complete(body(),namespace='')
        b=body();b['base_url']='https://remote.example'
        with self.assertRaises(c.InferenceError):pool.complete(b,namespace='a')

    def test_invalid_usage_does_not_appear_successful(self):
        result=answer();result['usage']['completion_tokens']=-1
        with self.assertRaises(c.InferenceError):c.validate_response(result)


if __name__=='__main__':unittest.main()
