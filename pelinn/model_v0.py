# pelinn/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LiquidCell(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.in_dim, self.hid_dim = in_dim, hid_dim
        self.Wx = nn.Linear(in_dim, hid_dim, bias=True)
        self.Wh = nn.Linear(hid_dim, hid_dim, bias=False)
        self.Wtx = nn.Linear(in_dim, hid_dim, bias=False)  # for tau(x,h)
        self.Wth = nn.Linear(hid_dim, hid_dim, bias=False)
        self.bt  = nn.Parameter(torch.zeros(hid_dim))
        self.eps = 1e-3

    def forward(self, x, h, dt=0.1):
        tau = F.softplus(self.Wtx(x) + self.Wth(h) + self.bt) + self.eps
        dh = -h / tau + torch.tanh(self.Wx(x) + self.Wh(h))
        return h + dt * dh

class PELiNNQEM(nn.Module):
    def __init__(self, in_dim, hid_dim=64, steps=5):
        super().__init__()
        self.cell = LiquidCell(in_dim, hid_dim)
        self.h0   = nn.Parameter(torch.zeros(hid_dim))
        self.head = nn.Linear(hid_dim, 1)

        self.steps = steps

    def forward(self, x):
        B = x.shape[0]
        h = self.h0.unsqueeze(0).expand(B, -1)
        for _ in range(self.steps):
            h = self.cell(x, h)
        y = torch.tanh(self.head(h))  # range constraint
        return y.squeeze(-1)

def physics_loss(pred, target, groups=None, alpha_inv=0.1):
    """MSE to ideal + invariance across noise scales for same circuit.
       groups: list of index lists; each group are samples of the same circuit under different noise.
    """
    mse = F.mse_loss(pred, target)
    if not groups:
        return mse
    inv = 0.0
    cnt = 0
    for idxs in groups:
        if len(idxs) < 2: continue
        p = pred[idxs]
        inv += (p.unsqueeze(0) - p.unsqueeze(1)).abs().mean()
        cnt += 1
    inv = inv / max(cnt, 1)
    return mse + alpha_inv * inv
