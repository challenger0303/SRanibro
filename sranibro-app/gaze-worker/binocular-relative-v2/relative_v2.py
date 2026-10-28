"""Anchored learned image-coordinate displacement, research only."""
import importlib.util
from pathlib import Path
import numpy as np
_path=Path(__file__).resolve().parent.parent/'binocular-relative-v1/relative.py'
_spec=importlib.util.spec_from_file_location('frozen_relative_v1',_path)
_v1=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_v1)

class RelativeGazeV2(_v1.RelativeGaze):
    @staticmethod
    def fit(x,y):
        if len(x)<24:return None
        mean=x.mean(axis=0);centered=x-mean
        values,vectors=np.linalg.eigh(centered.T@centered/len(x))
        basis=vectors[:,values>=1.]
        if basis.shape[1]==0:return None
        design=np.column_stack((centered@basis,np.ones(len(x))))
        weights=np.ones(len(x))
        for _ in range(3):
            root=np.sqrt(weights)
            coef=np.linalg.lstsq(design*root[:,None],y*root[:,None],rcond=None)[0]
            residual=np.linalg.norm(design@coef-y,axis=1)
            weights=np.minimum(1.,2./np.maximum(residual,1e-12))
        singular=np.linalg.svd(coef[:-1],compute_uv=False)
        rms=float(np.sqrt(np.mean(residual**2)))
        if not np.isfinite(coef).all() or rms>2 or singular.min()<.25 or singular.max()>4:return None
        return dict(slope=basis@coef[:-1],basis=basis,samples=x.copy(),low=x.min(axis=0),high=x.max(axis=0),rms=rms,rank=basis.shape[1])

    def estimate(self,eye,partner,time):
        if self.anchor is None:return None,None
        t,pair=self.anchor
        if not 0<=time-t<=.25:return None,None
        delta=partner-pair[1-eye];mapping=self.maps[eye]
        if mapping is not None:
            unsupported=delta-delta@mapping['basis']@mapping['basis'].T
            if (partner>=mapping['low']).all() and (partner<=mapping['high']).all() and np.linalg.norm(mapping['samples']-partner,axis=1).min()<=4 and np.linalg.norm(unsupported)<=1:
                guess=pair[eye]+delta@mapping['slope']
                if np.isfinite(guess).all() and ((guess>=0)&(guess<=199)).all():return guess,'anchored_rank'+str(mapping['rank'])
        if np.linalg.norm(delta)<=2:return pair[eye].copy(),'stationary'
        return None,None
