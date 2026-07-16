#!/usr/bin/env python3
"""Bake a TinyBrowNet (eyebrow) model into SRanibro's flat `BROWNET1` weights file.

Folds BatchNorm into the preceding conv/linear so the Rust runtime (src/ml/brow_net.rs)
runs plain conv+bias+relu / linear — no BN, no ONNX runtime. Also writes a golden
input/output fixture so the Rust `pytorch_parity` test can assert numerical agreement.

Usage:
    python bake_brow_weights.py --pth model.pth --out <dir>     # bake a trained model
    python bake_brow_weights.py --random --out <dir>            # random model (parity test)

Output in <dir>: brow.bin (BROWNET1), brow_in.f32, brow_out.f32 (1x1x64x64 input + output).
Then in sranibro-rs:  BROW_FIXTURE_DIR=<dir> cargo test --lib pytorch_parity

Self-contained: TinyBrowNet is defined inline (matches the vr_eyebrow model.py), so a .pth
state_dict of the same architecture loads without the original project on PYTHONPATH.
"""
import argparse
import struct
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyBrowNet(nn.Module):
    """Mirror of vr_eyebrow/model.py TinyBrowNet (1-channel default)."""

    def __init__(self, out_dim=3, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1); self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1);          self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1);          self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1);          self.bn4 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128); self.bn_fc = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, out_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.bn_fc(self.fc1(x))))
        return self.fc2(x)


def fold_conv_bn(conv, bn, eps=1e-5):
    """Return folded (weight[oc,ic,3,3], bias[oc]) so relu(bn(conv(x)))==relu(W'x+b')."""
    w = conv.weight.detach().double()
    b = (conv.bias.detach().double() if conv.bias is not None
         else torch.zeros(w.shape[0], dtype=torch.float64))
    g = bn.weight.detach().double()
    beta = bn.bias.detach().double()
    rm = bn.running_mean.detach().double()
    rv = bn.running_var.detach().double()
    s = g / torch.sqrt(rv + eps)
    w2 = w * s.reshape(-1, 1, 1, 1)
    b2 = (b - rm) * s + beta
    return w2.float().numpy(), b2.float().numpy()


def fold_fc_bn(fc, bn, eps=1e-5):
    w = fc.weight.detach().double()
    b = fc.bias.detach().double()
    g = bn.weight.detach().double()
    beta = bn.bias.detach().double()
    rm = bn.running_mean.detach().double()
    rv = bn.running_var.detach().double()
    s = g / torch.sqrt(rv + eps)
    w2 = w * s.reshape(-1, 1)
    b2 = (b - rm) * s + beta
    return w2.float().numpy(), b2.float().numpy()


def detect_out_dim(sd):
    for k in ("fc2.weight", "fc2.0.weight"):
        if k in sd:
            return int(sd[k].shape[0])
    return 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", help="trained TinyBrowNet checkpoint (.pth state_dict)")
    ap.add_argument("--random", action="store_true", help="random model (for parity testing)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--out-dim", type=int, default=3)
    args = ap.parse_args()

    import os
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    if args.pth:
        raw = torch.load(args.pth, map_location="cpu")
        sd = raw.get("state_dict", raw.get("model", raw)) if isinstance(raw, dict) else raw
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        out_dim = detect_out_dim(sd)
        model = TinyBrowNet(out_dim=out_dim)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[warn] missing keys: {missing}", file=sys.stderr)
        if unexpected:
            print(f"[warn] unexpected keys: {unexpected}", file=sys.stderr)
    else:
        out_dim = args.out_dim
        model = TinyBrowNet(out_dim=out_dim)
        # Non-trivial BN running stats so the fold is actually exercised (defaults make
        # BN an identity). Parity-only; not a usable model.
        for bn in (model.bn1, model.bn2, model.bn3, model.bn4, model.bn_fc):
            bn.running_mean.normal_(0.0, 0.5)
            bn.running_var.uniform_(0.5, 1.5)
            bn.weight.data.uniform_(0.5, 1.5)
            bn.bias.data.normal_(0.0, 0.3)

    model.eval()

    # Fold BN -> conv/fc.
    convs = [(model.conv1, model.bn1), (model.conv2, model.bn2),
             (model.conv3, model.bn3), (model.conv4, model.bn4)]
    folded = [fold_conv_bn(c, b) for c, b in convs]
    fc1_w, fc1_b = fold_fc_bn(model.fc1, model.bn_fc)
    fc2_w = model.fc2.weight.detach().numpy().astype(np.float32)
    fc2_b = model.fc2.bias.detach().numpy().astype(np.float32)

    # Write BROWNET1: magic, u32 out_dim, then f32 arrays in module order.
    blob = bytearray(b"BROWNET1")
    blob += struct.pack("<I", out_dim)

    def put(arr):
        blob.extend(np.ascontiguousarray(arr, dtype="<f4").tobytes())

    for (cw, cb) in folded:
        put(cw); put(cb)
    put(fc1_w); put(fc1_b)
    put(fc2_w); put(fc2_b)

    with open(os.path.join(args.out, "brow.bin"), "wb") as f:
        f.write(blob)

    # Golden fixture: a random z-scored 64x64 input + the PyTorch output.
    rng = np.random.default_rng(1)
    inp = rng.standard_normal((1, 1, 64, 64)).astype(np.float32)
    with torch.no_grad():
        out = model(torch.from_numpy(inp)).numpy().reshape(-1).astype(np.float32)
    inp.reshape(-1).astype("<f4").tofile(os.path.join(args.out, "brow_in.f32"))
    out.astype("<f4").tofile(os.path.join(args.out, "brow_out.f32"))

    print(f"baked out_dim={out_dim}  brow.bin={len(blob)}B  out={out.tolist()}  -> {args.out}")


if __name__ == "__main__":
    main()
