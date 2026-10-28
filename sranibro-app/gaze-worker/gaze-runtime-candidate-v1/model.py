"""Explicit isolated research runtime. Existing app and FrozenModel stay unchanged."""
from pathlib import Path
import hashlib,importlib.util,json,threading,time,sys
ROOT=Path(__file__).resolve().parent;GOAL=ROOT.parent

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

class CandidateModel:
    def __init__(self,backend='auto',profile_prefix=None):
        import onnxruntime as ort
        if backend not in ('auto','cpu','directml'):raise ValueError('unknown backend')
        self.owner=threading.get_ident();self.closed=False;self.fault=None;self.fallback_reason=None;self.profile=None
        manifest=json.loads((ROOT/'manifest-v3.json').read_text(encoding='utf-8'))
        for relative,expected in manifest['files'].items():
            path=(GOAL/relative).resolve()
            if not path.is_relative_to(GOAL):raise ValueError('candidate artifact outside research root')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('candidate artifact hash mismatch: '+relative)
        graph=GOAL/manifest['graph']
        self.model_id='sha256:'+hashlib.sha256(json.dumps(manifest['files'],sort_keys=True).encode()).hexdigest()
        self.preprocessing_id='fixed-order-canonical-u8-right-mirror-once-v1'
        shared=sys.modules.get('postprocess')
        self._shared_postprocess=shared;self._shared_witness=shared.witness if shared is not None else None
        self.lazy=load('candidate_private_decoder',ROOT/'private_decoder.py')
        self._private_witness=self.lazy.b.pp.witness
        self.pre=load('candidate_guarded_avx_prepare',GOAL/'preprocess-avx-fixed-v1/preprocess.py')
        use_gpu=backend!='cpu' and 'DmlExecutionProvider' in ort.get_available_providers()
        if backend=='directml' and not use_gpu:raise RuntimeError('DirectML unavailable')
        if backend=='auto' and not use_gpu:self.fallback_reason='DirectML unavailable at startup'
        opts=self.lazy.options()
        if profile_prefix is not None:opts.enable_profiling=True;opts.profile_file_prefix=str(profile_prefix)
        providers=['DmlExecutionProvider','CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        try:self.session=ort.InferenceSession(str(graph),sess_options=opts,providers=providers)
        except Exception as e:
            if backend!='auto' or not use_gpu:raise
            self.fallback_reason='DirectML initialization failed: '+str(e)
            self.session=ort.InferenceSession(str(graph),sess_options=opts,providers=['CPUExecutionProvider'])
        self.session.disable_fallback();self.providers=self.session.get_providers()
        if backend=='directml' and 'DmlExecutionProvider' not in self.providers:raise RuntimeError('explicit DirectML fell back')
        if backend=='auto' and use_gpu and 'DmlExecutionProvider' not in self.providers:self.fallback_reason='Runtime selected CPU at startup'
        self.profiling=profile_prefix is not None

    def predict(self,images):
        if threading.get_ident()!=self.owner:raise RuntimeError('wrong inference owner')
        if self.closed:raise RuntimeError('model closed')
        if self.fault is not None:raise RuntimeError('model fault is sticky: '+self.fault)
        import numpy as np
        if images.shape!=(2,200,200) or images.dtype!=np.uint8:raise ValueError('uint8 canonical stereo200x200 required')
        if self.lazy.b.pp.witness is not self._private_witness:raise RuntimeError('private decoder unexpectedly changed')
        try:
            t0=time.perf_counter();feed=self.pre.prepare(images.astype(np.float32)[:,None]/255);t1=time.perf_counter()
            values=self.session.run(None,{k:feed[k] for k in ('encoder_input','raw_input')});t2=time.perf_counter()
            result=self.lazy.decode(images,values)['corroborated'];t3=time.perf_counter()
            self.last_stages_ms=dict(prepare=(t1-t0)*1000,infer=(t2-t1)*1000,decode=(t3-t2)*1000,total=(t3-t0)*1000)
            return result
        except Exception as e:
            self.fault=type(e).__name__+': '+str(e)
            raise

    def close(self):
        if threading.get_ident()!=self.owner:raise RuntimeError('wrong close owner')
        if self.closed:return
        try:
            if self.profiling:self.profile=self.session.end_profiling()
        finally:self.session=None;self.closed=True
