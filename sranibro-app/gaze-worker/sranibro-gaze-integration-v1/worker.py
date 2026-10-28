"""Local-only SRanibro gaze child. No camera/network/settings access."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys

HEADER = struct.Struct('<QQQdd')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_exact(stream, size, *, eof=False):
    result = bytearray()
    while len(result) < size:
        block = stream.read(size-len(result))
        if not block:
            if eof and not result:
                return None
            raise EOFError('truncated gaze input')
        result.extend(block)
    return bytes(result)


def canonical(raw):
    import numpy as np
    images = np.frombuffer(raw, dtype=np.uint8).reshape(2, 200, 200).copy()
    images[1] = images[1, :, ::-1]
    return images


class Processor:
    def __init__(self, model, mapping, relative_type, mapper_type, horizontal_sign=1):
        if type(horizontal_sign) not in (int, float) or horizontal_sign not in (-1, 1):
            raise ValueError('horizontal_sign must be -1 or 1')
        self.horizontal_sign = horizontal_sign
        self.model = model
        self.scope = dict(mapping['source_identity'])
        if model.model_id != self.scope['model_id'] or model.preprocessing_id != self.scope['preprocessing_id']:
            raise ValueError('gaze model identity mismatch')
        self.mapper = mapper_type(mapping, self.scope)
        self.relative = relative_type()
        self.last = None

    def step(self, header, raw):
        seq, left, right, lt, rt = header
        gens, times = (left, right), (lt, rt)
        if seq < 1 or min(gens) < 1 or not all(math.isfinite(t) and t >= 0 for t in times) or abs(lt-rt) > .020:
            raise ValueError('invalid stereo header')
        if self.last:
            prev, old_gens, old_times = self.last
            if seq != prev+1 or any(a <= b for a,b in zip(gens,old_gens)) or any(a <= b for a,b in zip(times,old_times)):
                raise ValueError('gaze input order regressed')
            if any(a != b+1 for a,b in zip(gens,old_gens)) or any(a-b > .05 for a,b in zip(times,old_times)):
                self.relative.reset()
        predictions = self.model.predict(canonical(raw))
        raw_centers = [p['center'] for p in predictions]
        result = self.relative.step(self.scope['session_id'], max(times), raw_centers,
                                    [p['visibility'] for p in predictions])
        backend = int('DmlExecutionProvider' in self.model.providers)
        packet = bytearray(struct.pack('<4sQB', b'XGZ1', seq, backend))
        for i, eye in enumerate(('left','right')):
            state = 2 if result['estimated'][i] else (1 if result['observed'][i] else 0)
            ray = self.mapper.emit(eye=eye, pupil_xy=result['centers'][i], timestamp_s=times[i], scope=self.scope,
                                   state=('invalid','observed','partner_estimated')[state])
            x, y, z = ray['direction'] or (0.,0.,0.)
            # Personal mapping orientation, before centering and every output sink.
            # Eye identity, vertical axis, pupil inference and validity stay intact.
            packet.extend(struct.pack('<Bfff', state, self.horizontal_sign*x if state else 0., y, z))
        self.last = seq, gens, times
        return packet


def main(path, backend, profile_prefix=None):
    package = json.loads(Path(path).read_text(encoding='utf-8'))
    if package['schema'] != 'sranibro-xr5-gaze-local-v1' or backend not in ('auto','cpu'):
        raise ValueError('unsupported package/backend')
    # Explicit local package. Not a sandbox or a format for untrusted models.
    for filename, expected in package['verified_files'].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise ValueError('gaze dependency hash mismatch: '+filename)
    sys.path[:0] = package['dependency_paths']
    goal = Path(package['research_root'])
    mapping = json.loads(Path(package['mapping']).read_text(encoding='utf-8'))
    expected_device = 'pimax_xr5:'+package['device_serial']+':preflight'
    if mapping['source_identity']['device_id'] != expected_device:
        raise ValueError('mapping device differs')
    # Keep stdout exclusively binary, including during model imports/inference.
    output = sys.stdout.buffer
    sys.stdout = sys.stderr
    model_type = load('integrated_blur', goal/'fixed-roi-blur-v1/model.py').BlurModel
    relative = load('integrated_relative', goal/'binocular-relative-v2/relative_v2.py').RelativeGazeV2
    mapper = load('integrated_mapper', goal/'live-trial-20261003/rough_mapping.py').VisualMapper
    model = model_type(backend=backend, profile_prefix=profile_prefix)
    try:
        processor = Processor(model, mapping, relative, mapper, package.get('horizontal_sign', 1))
        output.write(b'XRGZ0001'); output.flush()
        while (head := read_exact(sys.stdin.buffer, HEADER.size, eof=True)) is not None:
            raw = read_exact(sys.stdin.buffer, 80000)
            packet = processor.step(HEADER.unpack(head), raw)
            output.write(packet); output.flush()
    finally:
        model.close()


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv)>3 else None)
