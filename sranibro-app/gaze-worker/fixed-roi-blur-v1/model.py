"""Separate fixed-ROI-blur research model; original candidates remain unchanged."""
from pathlib import Path
import hashlib,importlib.util,json
ROOT=Path(__file__).resolve().parent;GOAL=ROOT.parent
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
Base=load('blur_model_seeded_base',GOAL/'seeded-component-flood-v1/model.py').SeededModel

class BlurModel(Base):
    def __init__(self,backend='auto',profile_prefix=None):
        manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
        for name,expected in manifest['files'].items():
            path=(GOAL/name).resolve()
            if not path.is_relative_to(GOAL):raise ValueError('blur artifact outside research root')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('blur artifact hash mismatch: '+name)
        self._blur_helper=load('blur_model_private_proxy',ROOT/'decoder.py')
        self._blur_proxy=self._blur_helper.proxy
        super().__init__(backend,profile_prefix)
        try:
            self.lazy.primary.cv2=self._blur_proxy
            self.lazy.native.stable.cv2=self._blur_proxy
            identity=dict(base=self.model_id,blur=manifest['files'])
            self.model_id='sha256:'+hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        except Exception:
            self.close();raise

    def predict(self,images):
        if self.lazy.primary.cv2 is not self._blur_proxy or self.lazy.native.stable.cv2 is not self._blur_proxy:
            raise RuntimeError('private blur proxy unexpectedly changed')
        return super().predict(images)
