"""Research-only causal2D gaze transfer; no eyelid changes or neural models."""
from collections import deque
import math
import numpy as np


class RelativeGaze:
    def __init__(self, recover=True, median=True):
        self.recover = recover
        self.median = median
        self.reset()

    def reset(self):
        self.recording = None
        self.time = None
        self.history = [deque(maxlen=3), deque(maxlen=3)]
        self.paired_recent = deque(maxlen=3)
        self.pairs = deque(maxlen=240)
        self.anchor = None
        self.maps = [None, None]
        self.pair_count = 0
        self.last_pair_time = None

    @staticmethod
    def point(point, confidence):
        if point is None: return None
        try:
            a = np.asarray(point, dtype=np.float64)
            score = float(confidence)
        except (ValueError, TypeError): return None
        if a.shape != (2,) or not np.isfinite(a).all() or np.any(a < 0) or np.any(a > 199): return None
        if not math.isfinite(score) or not 0 <= score <= 1: return None
        return a.copy()

    @staticmethod
    def fit(x, y):
        if len(x) < 24: return None
        centered = x - x.mean(axis=0)
        if np.linalg.eigvalsh(centered.T @ centered / len(x))[0] < 1: return None
        design = np.column_stack((x, np.ones(len(x))))
        weights = np.ones(len(x))
        coef = None
        for _ in range(3):
            root = np.sqrt(weights)
            coef = np.linalg.lstsq(design * root[:, None], y * root[:, None], rcond=None)[0]
            residual = np.linalg.norm(design @ coef - y, axis=1)
            weights = np.minimum(1., 2. / np.maximum(residual, 1e-12))
        singular = np.linalg.svd(coef[:2], compute_uv=False)
        rms = float(np.sqrt(np.mean(residual ** 2)))
        if not np.isfinite(coef).all() or rms > 2 or singular.min() < .25 or singular.max() > 4: return None
        return dict(coef=coef, samples=x.copy(), low=x.min(axis=0), high=x.max(axis=0), rms=rms)

    def update_mapping(self, time, raw, scores):
        while self.pairs and time - self.pairs[0][0] > 2: self.pairs.popleft()
        if any(p is None for p in raw) or min(scores) < .9:
            self.paired_recent.clear()
            return
        pair = np.stack(raw)
        self.paired_recent.append(pair)
        self.pairs.append((time, pair))
        self.pair_count += 1
        self.last_pair_time = time
        if len(self.paired_recent) == 3:
            recent = np.stack(self.paired_recent)
            anchor = np.median(recent, axis=0)
            if np.linalg.norm(recent - anchor, axis=2).max() <= 2:
                self.anchor = (time, anchor)
        if len(self.pairs) >= 24 and self.pair_count % 8 == 0:
            values = np.stack([p for _, p in self.pairs])
            self.maps = [self.fit(values[:, 1-eye], values[:, eye]) for eye in (0, 1)]

    def estimate(self, eye, partner, time):
        mapping = self.maps[eye]
        if mapping is not None and self.last_pair_time is not None and time-self.last_pair_time <= 2:
            if (partner >= mapping['low']).all() and (partner <= mapping['high']).all() and np.linalg.norm(mapping['samples']-partner, axis=1).min() <= 4:
                guess = np.r_[partner, 1.] @ mapping['coef']
                if np.isfinite(guess).all() and ((guess >= 0) & (guess <= 199)).all(): return guess, 'affine'
        if self.anchor is not None:
            t, pair = self.anchor
            if time-t <= .25 and np.linalg.norm(partner-pair[1-eye]) <= 2:
                guess = pair[eye] + partner-pair[1-eye]
                if ((guess >= 0) & (guess <= 199)).all(): return guess, 'local'
        return None, None

    def step(self, recording, time, centers, confidence):
        try: time = float(time)
        except (ValueError, TypeError): time = float('nan')
        if not math.isfinite(time) or len(centers) != 2 or len(confidence) != 2:
            self.reset()
            return dict(centers=[None,None], observed=[False,False], estimated=[False,False], kind=[None,None])
        if self.recording != recording or self.time is not None and not 0 < time-self.time <= .05: self.reset()
        self.recording, self.time = recording, time
        raw = [self.point(p,s) for p,s in zip(centers,confidence)]
        out = []; kinds = [None,None]
        for eye, point in enumerate(raw):
            if point is None:
                self.history[eye].clear();out.append(None)
            else:
                self.history[eye].append(point)
                out.append(np.median(np.stack(self.history[eye]),axis=0) if self.median and len(self.history[eye])==3 else point.copy())
        self.update_mapping(time, raw, [float(s) if p is not None else 0. for p,s in zip(raw,confidence)])
        # Use observed partner only. Never recursively recover both eyes.
        if self.recover:
            for eye in (0,1):
                if raw[eye] is None and raw[1-eye] is not None:
                    out[eye], kinds[eye] = self.estimate(eye,out[1-eye],time)
        return dict(centers=[p.tolist() if p is not None else None for p in out], observed=[p is not None for p in raw], estimated=[k is not None for k in kinds], kind=kinds)
