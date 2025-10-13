from dataclasses import dataclass
import torch
Tensor = torch.Tensor

@dataclass
class State:
    # shapes: [B, A] unless noted
    w: Tensor
    y: Tensor
    z: Tensor  # [B, 1]

    def to(self, device, dtype=None):
        cast = (lambda t: t.to(device=device, dtype=dtype) if dtype else t.to(device))
        return State(cast(self.w), cast(self.y), cast(self.z))

    def select(self, idx_or_mask):
        return State(self.w[idx_or_mask], self.y[idx_or_mask], self.z[idx_or_mask])