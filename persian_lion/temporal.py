"""Every-frame 25-Hz inference from five half-local relative-feature streams."""
from numbers import Integral

import numpy as np
import torch

from .data import blocks
from .model import GLOBAL_DIM, TEAM_DIM


def _validated_inputs(data, period):
    if not {"g", "t"}.issubset(data):
        raise ValueError("Missing relative feature input")
    period = np.asarray(period)
    if period.ndim != 1 or not len(period) or not np.isin(period, (0, 1, 2)).all():
        raise ValueError("Expected aligned nonempty first/second-half periods")
    g, t = np.asarray(data["g"], np.float32), np.asarray(data["t"], np.float32)
    n = len(period)
    if g.shape != (n, GLOBAL_DIM) or t.shape != (n, 2, TEAM_DIM):
        raise ValueError("Expected raw 25-Hz g[N,59], t[N,2,88]")
    if np.isinf(g).any() or np.isinf(t).any():
        raise ValueError("Missing features must use NaNs, not infinities")
    if "native_frame" in data:
        native = np.asarray(data["native_frame"])
        if native.shape != (n,):
            raise ValueError("Misaligned native_frame")
        for a, b in blocks(period):
            if period[a] and not np.all(np.diff(native[a:b]) == 1):
                raise ValueError("Inference requires every original 25-Hz frame")
    return dict(g=g, t=t), period


def probabilities(model, data, period, device="cpu", chunk_steps=2048):
    """Return raw sigmoid live[N], home[N], boundary[N,3] at native 25 Hz.

    Data contains raw physical features. The model's frozen buffers supply
    training-only normalization; no statistics are computed on this match.
    Period 0 placeholders are live=0, home=.5, boundaries=0; emit X there.
    """
    if not isinstance(chunk_steps, Integral) or isinstance(chunk_steps, bool) or chunk_steps < 1:
        raise ValueError("chunk_steps must be a positive integer")
    radius = getattr(model, "radius", None)
    if not isinstance(radius, Integral) or isinstance(radius, bool) or radius < 0:
        raise ValueError("Model must declare its temporal radius")
    data, period = _validated_inputs(data, period)
    output = np.zeros((len(period), 5), np.float32)
    output[:, 1] = 0.5
    requested_device = torch.device(device)
    tensors = list(model.parameters()) + list(model.buffers())
    original_device = tensors[0].device if tensors else requested_device
    if any(t.device != original_device for t in tensors):
        raise ValueError("Model must be on one device")
    original_training = model.training
    try:
        model.to(requested_device)
        model.eval()
        with torch.inference_mode():
            for a, b in blocks(period):
                if period[a] == 0:
                    continue
                for offset in range(5):
                    stream = np.arange(a+offset, b, 5, dtype=np.int64)
                    for start in range(0, len(stream), chunk_steps):
                        stop = min(start+chunk_steps, len(stream))
                        left, right = max(0, start-radius), min(len(stream), stop+radius)
                        chosen = stream[left:right]
                        inputs = [torch.as_tensor(np.ascontiguousarray(data[name][chosen]), device=requested_device).unsqueeze(0)
                                  for name in ("g", "t")]
                        live, home, boundary = model(*inputs)
                        if (live.shape != (1, len(chosen)) or home.shape != (1, len(chosen))
                                or boundary.shape != (1, len(chosen), 3)):
                            raise ValueError("Model outputs lost temporal alignment")
                        logits = torch.cat((live.unsqueeze(-1), home.unsqueeze(-1), boundary), dim=-1)
                        logits = logits[0, start-left:stop-left]
                        if not bool(torch.isfinite(logits).all()):
                            raise ValueError("Nonfinite model logits")
                        output[stream[start:stop]] = torch.sigmoid(logits).float().cpu().numpy()
    finally:
        if original_device != requested_device:
            model.to(original_device)
        model.train(original_training)
    return output[:, 0].copy(), output[:, 1].copy(), output[:, 2:].copy()
