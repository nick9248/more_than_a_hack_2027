"""Symmetric temporal model using frozen relative geometry and physical motion.

No raw player point, player slot, or absolute XY embedding enters this network.
Normalization is fitted once on named external training cases and stored in the
model. Team statistics are shared, so exchanging teams reverses owner logits.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

class TemporalBlock(nn.Module):
    def __init__(self,hidden,dilation,dropout):
        super().__init__()
        self.conv=nn.Conv1d(hidden,hidden,3,padding=dilation,dilation=dilation)
        self.norm=nn.LayerNorm(hidden)
        self.point=nn.Linear(hidden,hidden)
        self.drop=nn.Dropout(dropout)

    def forward(self,x):
        y=self.conv(x.transpose(1,2)).transpose(1,2)
        y=self.point(self.drop(F.gelu(self.norm(y))))
        return x+y


GLOBAL_DIM = 59
TEAM_DIM = 88
CLIP = 8.0
MIN_SCALE = 0.001


class PossessionNet(nn.Module):
    def __init__(self, hidden=64, dilations=(1, 2, 4, 8, 16, 32), dropout=0.2,
                 kernel_size=3, normalization=None):
        super().__init__()
        if kernel_size != 3 or any(not isinstance(d, int) or d <= 0 for d in dilations):
            raise ValueError("Expected positive dilations and the declared kernel size 3")
        self.radius = sum(dilations)
        for name, width in (("g", GLOBAL_DIM), ("t", TEAM_DIM)):
            center = torch.zeros(width) if normalization is None else torch.tensor(normalization[name+"_center"], dtype=torch.float32)
            scale = torch.ones(width) if normalization is None else torch.tensor(normalization[name+"_scale"], dtype=torch.float32)
            if (center.shape != (width,) or scale.shape != (width,) or not torch.isfinite(center).all()
                    or not torch.isfinite(scale).all() or (scale < MIN_SCALE).any()):
                raise ValueError("Invalid fitted feature normalization")
            self.register_buffer(name+"_center", center)
            self.register_buffer(name+"_scale", scale)
        self.project = nn.Linear(2*GLOBAL_DIM + 4*TEAM_DIM, hidden)
        self.temporal = nn.Sequential(*(TemporalBlock(hidden, d, dropout) for d in dilations))
        self.owner = nn.Linear(hidden, 1)
        self.invariant = nn.Sequential(nn.Linear(2*hidden, hidden), nn.GELU(), nn.Linear(hidden, 4))

    def encode_inputs(self, g, t):
        if g.ndim != 3 or g.shape[-1] != GLOBAL_DIM or t.shape != (*g.shape[:2], 2, TEAM_DIM):
            raise ValueError("Expected raw g[B,T,59] and t[B,T,2,88]")
        encoded = []
        for name, value in (("g", g), ("t", t)):
            finite = torch.isfinite(value)
            scaled = (value - getattr(self, name+"_center")) / getattr(self, name+"_scale")
            scaled = torch.where(finite, scaled, torch.zeros_like(scaled)).clamp(-CLIP, CLIP)
            encoded.append(torch.cat((scaled, finite.to(value.dtype)), dim=-1))
        return encoded

    def forward(self, g, t):
        g, t = self.encode_inputs(g, t)
        candidate = [torch.cat((g, t[:, :, side], t[:, :, 1-side]), dim=-1) for side in (0, 1)]
        n = len(g)
        encoded = self.temporal(F.gelu(self.project(torch.cat(candidate, dim=0))))
        h, a = encoded[:n], encoded[n:]
        home_logit = (self.owner(h) - self.owner(a)).squeeze(-1)
        invariant = self.invariant(torch.cat(((h+a)/2, torch.abs(h-a)), dim=-1))
        return invariant[..., 0], home_logit, invariant[..., 1:]
