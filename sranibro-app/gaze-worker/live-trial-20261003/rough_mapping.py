"""Explicit partial-session visual mapping, NOT a passed ray calibration."""
import hashlib,json,math
import numpy as np

def fit_visual(report):
    if not report['integrity']['all_image_hashes_verified']:raise ValueError('unverified images')
    if report['source_provenance']['source_kind']!='live_xr5':raise ValueError('live XR5 required')
    # Fixed policy: first repetition only, all 18 train poses; no checkpoint or
    # threshold selection from the later repeated poses or missing holdout.
    entries=[e for e in report['selected_attempts'] if 1<=e['step']['id']<=18]
    if {e['step']['id'] for e in entries}!=set(range(1,19)) or not all(e['pass_quality'] for e in entries):
        raise ValueError('first complete 18-pose block required')
    eyes={}
    for eye in ('left','right'):
        pixels=np.asarray([e['eyes'][eye]['pupil_median_px'] for e in entries],float)
        x=np.column_stack([np.ones(len(pixels)),(pixels-99.5)/100.])
        target=np.asarray([e['target_hmd_m'] for e in entries],float)
        origin=np.asarray(entries[0]['eye_proxies'][eye]['origin_m'],float)
        delta=target-origin
        if not np.isfinite(x).all() or np.linalg.matrix_rank(x)!=3 or np.any(delta[:,2]>=0):
            raise ValueError('invalid visual fit inputs')
        slopes=delta[:,:2]/(-delta[:,2,None]);coeff=np.linalg.lstsq(x,slopes,rcond=None)[0]
        predicted=np.column_stack([x@coeff,-np.ones(len(x))]);truth=delta.copy()
        predicted/=np.linalg.norm(predicted,axis=1,keepdims=True);truth/=np.linalg.norm(truth,axis=1,keepdims=True)
        err=np.degrees(np.arccos(np.clip((predicted*truth).sum(axis=1),-1,1)))
        eyes[eye]=dict(coeff=coeff.tolist(),origin_m=origin.tolist(),train_rms_deg=float(np.sqrt(np.mean(err**2))))
    result=dict(schema='partial-session-visual-gaze-v1',eyes=eyes,source_identity=report['session_identity'],
                source_events_sha256=report['events_sha256'],fit_step_ids=list(range(1,19)),
                diagnostic_gates_pass=False,adoption_allowed=False,measured_corneal_centre=False,
                holdout_evaluated=False,origin_kind='device_eye_pose_proxy',
                warning='Prior wearing position; unvalidated partial mapping. Not anatomical origin or accuracy certification.')
    result['visual_mapping_id']='sha256:'+hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()
    return result

class VisualMapper:
    def __init__(self,mapping,scope):
        if mapping.get('schema')!='partial-session-visual-gaze-v1' or mapping['adoption_allowed']:
            raise ValueError('explicit non-adopted visual mapping required')
        for key in ('device_id','model_id','preprocessing_id'):
            if mapping['source_identity'][key]!=scope[key]:raise ValueError('visual mapping source differs: '+key)
        self.mapping=mapping;self.scope=dict(scope);self.last={}
    def emit(self,*,eye,pupil_xy,timestamp_s,scope,state=None):
        if scope!=self.scope or eye not in ('left','right'):raise ValueError('visual scope changed')
        if not math.isfinite(timestamp_s) or timestamp_s<=self.last.get(eye,-1):raise ValueError('visual timestamp regressed')
        state=state or ('invalid' if pupil_xy is None else 'observed')
        if state not in ('observed','partner_estimated','invalid') or (state=='invalid')!=(pupil_xy is None):
            raise ValueError('invalid visual observation')
        d=None
        if pupil_xy is not None:
            p=np.asarray(pupil_xy,float)
            if p.shape!=(2,) or not np.isfinite(p).all() or np.any(p<0) or np.any(p>199):raise ValueError('invalid pupil')
            xy=np.r_[1.,(p-99.5)/100.]@np.asarray(self.mapping['eyes'][eye]['coeff'])
            d=np.r_[xy,-1.];d=(d/np.linalg.norm(d)).tolist()
        self.last[eye]=timestamp_s
        return dict(eye=eye,timestamp_s=timestamp_s,direction=d,origin_m=self.mapping['eyes'][eye]['origin_m'],
                    origin_kind='device_eye_pose_proxy',ray_available=d is not None,state=state,
                    diagnostic_only=True,adoption_allowed=False,accuracy_validated=False,
                    measured_corneal_centre=False,visual_mapping_id=self.mapping['visual_mapping_id'],
                    transferred_from_previous_session=True)
