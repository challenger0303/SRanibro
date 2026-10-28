"""Opt-in research variant only; no current candidate/app default change."""
from pathlib import Path
import ctypes,hashlib,importlib.util,json
import numpy as np
ROOT=Path(__file__).resolve().parent;GOAL=ROOT.parent
spec=importlib.util.spec_from_file_location('seeded_base_candidate',GOAL/'gaze-runtime-candidate-v1/model.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

class SeededModel(base.CandidateModel):
    def __init__(self,backend='auto',profile_prefix=None):
        manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
        for name,expected in manifest['files'].items():
            path=(GOAL/name).resolve()
            if not path.is_relative_to(GOAL):raise ValueError('seeded artifact outside research root')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('seeded artifact hash mismatch: '+name)
        self._area_dll=ctypes.CDLL(str(ROOT/'component.dll'));self._area_fn=self._area_dll.component_areas
        self._area_fn.argtypes=[np.ctypeslib.ndpointer(np.uint8,flags='C_CONTIGUOUS'),ctypes.c_int,ctypes.c_int,
            np.ctypeslib.ndpointer(np.int32,flags='C_CONTIGUOUS'),np.ctypeslib.ndpointer(np.uint8,flags='C_CONTIGUOUS')]
        self._area_fn.restype=ctypes.c_int
        super().__init__(backend,profile_prefix)
        try:
            self.lazy.native.area_fn=self._area_fn
            self.lazy.primary.native.area_fn=self._area_fn
            identity=dict(base=self.model_id,seeded=manifest['files'])
            self.model_id='sha256:'+hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        except Exception:
            self.close();raise

    def predict(self,images):
        if self.lazy.native.area_fn is not self._area_fn or self.lazy.primary.native.area_fn is not self._area_fn:
            raise RuntimeError('private seeded area function unexpectedly changed')
        return super().predict(images)
