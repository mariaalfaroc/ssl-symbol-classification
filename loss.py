from typing import Tuple

import torch
import torch.nn.functional as F

def invariance_loss(za: torch.Tensor, zb: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(za, zb)

def variance_loss(za: torch.Tensor, zb: torch.Tensor) -> torch.Tensor:
    eps = 1e-4
    std_za = torch.sqrt(za.var(dim=0) + eps)
    std_zb = torch.sqrt(zb.var(dim=0) + eps)
    std_loss = torch.mean(F.relu(1 - std_za)) + torch.mean(F.relu(1 - std_zb))
    return std_loss

def off_diagonal(z: torch.Tensor) -> torch.Tensor:
    # Return a flattened view of the off-diagonal elements of a square matrix
    n, m = z.shape
    assert n == m
    diag = torch.eye(n, device=z.device)
    return z[~diag.bool()]

def covariance_loss(za: torch.Tensor, zb: torch.Tensor) -> torch.Tensor:
    N, D = za.shape
    za = za - za.mean(dim=0)
    zb = zb - zb.mean(dim=0)
    cov_za = (za.T @ za) / (N - 1)
    cov_zb = (zb.T @ zb) / (N - 1)
    cov_loss = off_diagonal(cov_za).pow_(2).sum() / D + off_diagonal(cov_zb).pow_(2).sum() / D
    return cov_loss

def vicreg_loss(
    za: torch.Tensor,
    zb: torch.Tensor,
    sim_loss_weight: float = 25.0,
    var_loss_weight: float = 25.0,
    cov_loss_weight: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sim_loss = sim_loss_weight * invariance_loss(za, zb)
    var_loss = var_loss_weight * variance_loss(za, zb)
    cov_loss = cov_loss_weight * covariance_loss(za, zb)
    loss = sim_loss + var_loss + cov_loss
    return loss, sim_loss, var_loss, cov_loss
