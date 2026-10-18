#!/usr/bin/env python3
"""Exact-request optimization for the fixed owner-operated 1.6T endpoint.

Opt-in reuse never changes the selected model. Reused output is clearly labeled;
it is not another inference, independent quality sample or provider-speed gain.
Only public research inputs are allowed by this initial client.
"""
from __future__ import annotations
import argparse
from collections import OrderedDict
from concurrent.futures import Future, TimeoutError as FutureTimeout
import copy
import hashlib
import http.client
import json
import math
from pathlib import Path
import re
import threading
import time
from trillion_runtime import MODEL, REVISION, SERVED_MODEL, canonical, loads

MAX_BYTES=512_000


class InferenceError(RuntimeError):pass


def validate_body(body):
    allowed={'model','messages','metadata','temperature','top_p','max_tokens','stream'}
    if (not isinstance(body,dict) or set(body)-allowed or body.get('model')!=SERVED_MODEL
        or body.get('metadata')!={'data_class':'public'} or body.get('stream',False) is not False):
        raise InferenceError('UNAUTHORIZED_MODEL_OR_NONPUBLIC_REQUEST')
    messages=body.get('messages')
    if (not isinstance(messages,list) or not 1<=len(messages)<=64
        or any(not isinstance(m,dict) or set(m)!={'role','content'}
               or m['role'] not in ('system','user','assistant')
               or not isinstance(m['content'],str) or not m['content'].strip() for m in messages)
        or messages[-1]['role']!='user'):
        raise InferenceError('INVALID_TEXT_MESSAGES')
    limit=body.get('max_tokens',4096)
    if type(limit) is not int or not 1<=limit<=16384:raise InferenceError('INVALID_OUTPUT_BUDGET')
    for name,minimum,maximum in (('temperature',0,2),('top_p',.001,1)):
        value=body.get(name,1)
        if type(value) not in (float,int) or not math.isfinite(value) or not minimum<=value<=maximum:
            raise InferenceError('INVALID_SAMPLING')
    if len(canonical(messages))>65536:raise InferenceError('INPUT_BYTE_LIMIT')
    return copy.deepcopy(body)


def validate_response(response):
    if not isinstance(response,dict) or response.get('model')!=SERVED_MODEL or 'error' in response:
        raise InferenceError('WRONG_MODEL_OR_PROVIDER_ERROR')
    choices=response.get('choices')
    if not isinstance(choices,list) or len(choices)!=1 or not isinstance(choices[0],dict):
        raise InferenceError('INVALID_CHOICES')
    row=choices[0]
    if row.get('finish_reason')!='stop':raise InferenceError('TRUNCATED_OR_NONFINAL_RESPONSE')
    message=row.get('message')
    if (not isinstance(message,dict) or not isinstance(message.get('content'),str)
        or not message['content'].strip() or message.get('tool_calls')):
        raise InferenceError('TEXT_COMPLETION_REQUIRED')
    usage=response.get('usage')
    if usage is not None:
        if not isinstance(usage,dict) or any(type(usage.get(k)) is not int or usage[k]<0
                for k in ('prompt_tokens','completion_tokens','total_tokens')):
            raise InferenceError('INVALID_USAGE')
        if usage['total_tokens']!=usage['prompt_tokens']+usage['completion_tokens']:
            raise InferenceError('INVALID_USAGE_SUM')
    try:wire=canonical(response)
    except (ValueError,TypeError):raise InferenceError('INVALID_JSON_RESPONSE') from None
    if len(wire)>MAX_BYTES:raise InferenceError('RESPONSE_SIZE_LIMIT')
    return copy.deepcopy(response)


class SingleFlight:
    """Bounded, process-local reuse, scoped by authenticated identity and model revision.

    Does not promise exactly-once execution across restarts or multiple processes.
    Caller transports must enforce their own deadlines. Follower timeouts do not
    falsely claim to cancel an upstream generation that has already started.
    """
    def __init__(self,provider,*,ttl_seconds=30,max_entries=8,max_bytes=2_000_000,
                 max_inflight=4,wait_seconds=120,clock=time.monotonic):
        if (type(ttl_seconds) not in (int,float) or not 0<=ttl_seconds<=300
            or type(max_entries) is not int or not 1<=max_entries<=32
            or type(max_bytes) is not int or not MAX_BYTES<=max_bytes<=8_000_000
            or type(max_inflight) is not int or not 1<=max_inflight<=8
            or type(wait_seconds) not in (int,float) or not 1<=wait_seconds<=600):
            raise InferenceError('OPTIMIZER_LIMITS_INVALID')
        self.provider=provider;self.ttl=ttl_seconds;self.max_entries=max_entries
        self.max_bytes=max_bytes;self.max_inflight=max_inflight;self.wait_seconds=wait_seconds
        self.clock=clock;self.lock=threading.Lock();self.cache=OrderedDict();self.inflight={}
        self.cache_bytes=0
        self.provider_slots=threading.BoundedSemaphore(max_inflight)

    def _invoke(self,request):
        if not self.provider_slots.acquire(blocking=False):
            raise InferenceError('INFERENCE_CAPACITY_BUSY')
        try:return validate_response(self.provider(request))
        finally:self.provider_slots.release()

    def _annotate(self,response,origin):
        result=copy.deepcopy(response)
        # Preserve original token accounting as source_usage; cache hits are not fresh generations.
        usage=result.pop('usage',None)
        if usage is not None:result['source_usage']=usage
        result['mj_optimization']={
            'origin':origin,'inference_performed_by_this_request':origin=='generated',
            'source_model':MODEL,'source_revision':REVISION,
            'model_quality_uplift_claimed':False,'smaller_model_fallback':False,
            'reuse_scope':'same_process_same_identity_exact_request',
        }
        if origin=='generated' and usage is not None:result['usage']=usage
        return result

    def complete(self,body,*,namespace,reuse=False):
        request=validate_body(body)
        if not isinstance(namespace,str) or not re.fullmatch('[A-Za-z0-9_-]{1,128}',namespace):
            raise InferenceError('AUTHENTICATED_NAMESPACE_REQUIRED')
        if type(reuse) is not bool:raise InferenceError('EXPLICIT_BOOLEAN_REUSE_REQUIRED')
        if reuse and request.get('temperature',1)!=0:
            raise InferenceError('STOCHASTIC_REQUEST_REUSE_REFUSED')
        if not reuse:
            try:return self._annotate(self._invoke(request),'generated')
            except Exception:raise InferenceError('GENERATION_FAILED') from None
        key=hashlib.sha256(canonical({'namespace':namespace,'revision':REVISION,'request':request})).hexdigest()
        leader=False
        with self.lock:
            now=self.clock()
            for old in [k for k,(expires,_,_) in self.cache.items() if expires<=now]:
                _,_,size=self.cache.pop(old);self.cache_bytes-=size
            if key in self.cache:
                expires,response,size=self.cache.pop(key)
                self.cache[key]=(expires,response,size)
                return self._annotate(response,'cache')
            if key in self.inflight:
                future,waiters=self.inflight[key]
                if waiters>=16:raise InferenceError('TOO_MANY_IDENTICAL_WAITERS')
                self.inflight[key]=(future,waiters+1)
            else:
                if len(self.inflight)>=self.max_inflight:raise InferenceError('INFERENCE_CAPACITY_BUSY')
                future=Future();self.inflight[key]=(future,0);leader=True
        if not leader:
            try:return self._annotate(future.result(timeout=self.wait_seconds),'coalesced')
            except FutureTimeout:raise InferenceError('FOLLOWER_WAIT_TIMEOUT') from None
            except Exception:raise InferenceError('GENERATION_FAILED') from None
            finally:
                with self.lock:
                    if key in self.inflight:
                        f,n=self.inflight[key];self.inflight[key]=(f,max(0,n-1))
        try:
            response=self._invoke(request);size=len(canonical(response))
            with self.lock:
                if self.ttl>0 and size<=self.max_bytes:
                    while self.cache and (len(self.cache)>=self.max_entries or self.cache_bytes+size>self.max_bytes):
                        _,(_,_,old_size)=self.cache.popitem(last=False);self.cache_bytes-=old_size
                    self.cache[key]=(self.clock()+self.ttl,response,size);self.cache_bytes+=size
                future.set_result(response)
            return self._annotate(response,'generated')
        except Exception:
            future.set_exception(InferenceError('GENERATION_FAILED'))
            raise InferenceError('GENERATION_FAILED') from None
        finally:
            with self.lock:self.inflight.pop(key,None)


class LocalTransport:
    """One fixed loopback origin, one pinned model. No redirects/proxy env/cookies."""
    def __init__(self,api_key,timeout=120):
        if (not isinstance(api_key,str) or not 32<=len(api_key)<=512
            or any(ord(c)<33 or ord(c)>126 for c in api_key)
            or type(timeout) not in (int,float) or not 1<=timeout<=600):
            raise InferenceError('INVALID_LOCAL_AUTH_OR_TIMEOUT')
        self.key=api_key;self.timeout=timeout

    def __call__(self,body):
        request=validate_body(body)
        # Classification is consumed locally; vLLM receives only its supported standard fields.
        request.pop('metadata')
        conn=http.client.HTTPConnection('127.0.0.1',18189,timeout=self.timeout)
        try:
            conn.request('POST','/v1/chat/completions',body=canonical(request),
                         headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'})
            response=conn.getresponse();raw=response.read(MAX_BYTES+1)
            if response.status!=200:raise InferenceError('LOCAL_HTTP_'+str(response.status))
            if len(raw)>MAX_BYTES:raise InferenceError('RESPONSE_LIMIT')
            return validate_response(loads(raw))
        except (OSError,ValueError,http.client.HTTPException):
            raise InferenceError('LOCAL_INFERENCE_UNAVAILABLE') from None
        finally:conn.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api-key-file',type=Path)
    parser.add_argument('--public',action='store_true')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--prompt-file',type=Path)
    parser.add_argument('--max-tokens',type=int,default=4096)
    args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({'mode':'plan','model':SERVED_MODEL,'source':MODEL,
                          'origin':'http://127.0.0.1:18189','model_calls':0}))
        return 0
    try:
        if not args.public or args.api_key_file is None or args.prompt_file is None:
            raise InferenceError('EXPLICIT_PUBLIC_INPUT_AND_OWNER_AUTH_REQUIRED')
        if args.prompt_file.stat().st_size>65536:raise InferenceError('PROMPT_LIMIT')
        key=args.api_key_file.read_text('utf-8-sig').strip()
        payload={'model':SERVED_MODEL,'messages':[{'role':'user','content':args.prompt_file.read_text('utf-8-sig')}],
                 'metadata':{'data_class':'public'},'temperature':1,'max_tokens':args.max_tokens,'stream':False}
        result=SingleFlight(LocalTransport(key)).complete(payload,namespace=hashlib.sha256(key.encode()).hexdigest())
        print(json.dumps(result,ensure_ascii=False))
        return 0
    except (OSError,InferenceError):
        print(json.dumps({'status':'FAILED','model':SERVED_MODEL,'smaller_model_fallback':False}))
        return 2


if __name__=='__main__':raise SystemExit(main())
