
from typing import Optional
import math
import warnings

import torch
import torch.nn as nn

from utils.kernel_points import get_kernel_points



#------------  helpers----------------


def _gather_with_neighbors(x: torch.Tensor, n_idx: torch.Tensor):
    """
    gather of neighbor features/coords.
    """
    assert x.dim() == 2 and n_idx.dim() == 2, "expected [N,C] and [N,K]"
    N, C = x.shape
    K = n_idx.shape[1]

    inds = n_idx if n_idx.dtype == torch.long else n_idx.long()

    # Append a zero row at index N for safe padding gather
    pad = torch.zeros(1, C, device=x.device, dtype=x.dtype)
    ext = torch.cat([x, pad], dim=0)  


    safe = torch.where(inds >= 0, inds, torch.full_like(inds, N))
    if (safe > N).any():
        warnings.warn(f"[gather] clamped {(safe > N).sum().item()} neighbor indices > N to N.")
        safe = torch.minimum(safe, torch.full_like(safe, N))

    xg = ext.index_select(0, safe.reshape(-1)).view(-1, K, C)  
    mask = (inds >= 0).unsqueeze(-1).to(xg.dtype)             
    return xg, mask


def closest_pool(features_fine: torch.Tensor, pool_inds: torch.Tensor) -> torch.Tensor:
    """
    features_fine : [Nf, C]
    pool_inds     : [Nc, P] long, -1 padded  (each row = children indices for one coarse point)
    returns       : [Nc, C] pooled features (masked mean over valid children)
    """
    assert features_fine.dim() == 2 and pool_inds.dim() == 2, 
    Nf, C = features_fine.shape
    Nc, P = pool_inds.shape

    # pad row
    zero_row = torch.zeros((1, C), device=features_fine.device, dtype=features_fine.dtype)
    ext = torch.cat([features_fine, zero_row], dim=0)   

    # safe indices
    inds = pool_inds if pool_inds.dtype == torch.long else pool_inds.long()
    safe = torch.where(inds >= 0, inds, torch.full_like(inds, Nf))
    if (safe > Nf).any():
        warnings.warn(f"[closest_pool] clamped {(safe > Nf).sum().item()} indices > Nf to Nf.")
        safe = torch.minimum(safe, torch.full_like(safe, Nf))

    gathered = ext.index_select(0, safe.reshape(-1)).view(Nc, P, C) 
    valid_mask = (inds >= 0).float().unsqueeze(-1)                   
    num = (gathered * valid_mask).sum(dim=1)                         
    den = valid_mask.sum(dim=1).clamp_min(1.0)                      
    return num / den


@torch.no_grad()
def interpolated_upsample(
    features_coarse: torch.Tensor,     
    coarse_pts: torch.Tensor,          
    fine_pts: torch.Tensor,            
    parent_inds: Optional[torch.Tensor] = None,   
    cand_inds: Optional[torch.Tensor]   = None,   
    k: int = 3,
    mode: str = "idw",                 # nearest or idw or gaussian
    power: float = 2.0,                # for idw: w ~ 1 / d^power
    gaussian_sigma: Optional[float] = None,  # if None then, = mean of distances
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Upsample coarse features to fine points
    """
    Nc, C = features_coarse.shape
    Nf = fine_pts.shape[0]
    device = features_coarse.device
    dtypeF = features_coarse.dtype

    # Append zero "pad row" at index Nc for safe gathers
    padF = torch.zeros((1, C), device=device, dtype=dtypeF)
    padX = torch.zeros((1, 3), device=coarse_pts.device, dtype=coarse_pts.dtype)
    extF = torch.cat([features_coarse, padF], dim=0)  
    extX = torch.cat([coarse_pts,      padX], dim=0)  

    # 1) Fast path: parent indices
    if parent_inds is not None:
        idx = parent_inds.view(-1).to(torch.long)
        idx = torch.where(idx >= 0, idx, torch.full_like(idx, Nc))
        if (idx > Nc).any():
            warnings.warn(f"[upsample] clamped {(idx > Nc).sum().item()} indices > Nc to Nc.")
            idx = torch.minimum(idx, torch.full_like(idx, Nc))
        return extF.index_select(0, idx)  # [Nf, C]

    # 2) Candidate path
    if cand_inds is not None:
        idx = cand_inds.to(torch.long)
        idx = torch.where(idx >= 0, idx, torch.full_like(idx, Nc))
        if (idx > Nc).any():
            warnings.warn(f"[upsample] clamped {(idx > Nc).sum().item()} indices > Nc to Nc.")
            idx = torch.minimum(idx, torch.full_like(idx, Nc))
        P = idx.shape[1]

        candX = extX.index_select(0, idx.reshape(-1)).view(Nf, P, 3)
        d = torch.linalg.norm(candX - fine_pts[:, None, :], dim=2)  

        if mode == "nearest":
            j = d.argmin(dim=1)
            idx_nn = idx[torch.arange(Nf, device=idx.device), j]
            return extF.index_select(0, idx_nn)

        if mode == "idw":
            w = 1.0 / (d.clamp_min(eps) ** power)
        elif mode == "gaussian":
            sigma = gaussian_sigma if gaussian_sigma is not None else (d.mean().item() + 1e-12)
            inv_two_sigma2 = 0.5 / (sigma * sigma + 1e-12)
            w = torch.exp(- (d * d) * inv_two_sigma2)
        else:
            raise ValueError("mode must be 'nearest', 'idw', or 'gaussian'.")

        mask = (idx != Nc)
        w = w * mask
        s = w.sum(dim=1, keepdim=True).clamp_min(eps)
        w = w / s

        Fg = extF.index_select(0, idx.reshape(-1)).view(Nf, P, C)
        return (Fg * w.unsqueeze(-1)).sum(dim=1)

    # 3) Fallback: kNN via cdist
    d_full = torch.cdist(fine_pts.to(device), coarse_pts.to(device))     
    k_eff = min(k, Nc)
    d_k, idx_k = torch.topk(d_full, k_eff, dim=1, largest=False)         
    f_k = features_coarse.index_select(0, idx_k.reshape(-1)).view(Nf, k_eff, C)

    zero_mask = d_k[:, 0] < eps
    if zero_mask.any():
        out = torch.zeros((Nf, C), device=device, dtype=dtypeF)
        out[zero_mask] = f_k[zero_mask, 0, :]
        keep = ~zero_mask
        if not keep.any():
            return out
        d_k = d_k[keep]
        f_k = f_k[keep]
        row_idx = keep.nonzero(as_tuple=False).reshape(-1)
    else:
        out = None
        row_idx = None

    if mode == "nearest":
        j = d_k.argmin(dim=1)
        up_part = f_k[torch.arange(f_k.shape[0], device=device), j, :]
    elif mode == "idw":
        w = 1.0 / (d_k.clamp_min(eps) ** power)
        w = w / (w.sum(dim=1, keepdim=True) + 1e-12)
        up_part = torch.einsum('nk,nkc->nc', w, f_k)
    elif mode == "gaussian":
        sigma = gaussian_sigma if gaussian_sigma is not None else (d_k.mean().item() + 1e-12)
        inv_two_sigma2 = 0.5 / (sigma * sigma + 1e-12)
        w = torch.exp(- (d_k * d_k) * inv_two_sigma2)
        w = w / (w.sum(dim=1, keepdim=True) + 1e-12)
        up_part = torch.einsum('nk,nkc->nc', w, f_k)
    else:
        raise ValueError("mode must be 'nearest', 'idw', or 'gaussian'.")

    if out is None:
        return up_part
    else:
        out[row_idx] = up_part
        return out


def _influence_weights(
    dist_nkm: torch.Tensor,                    
    extent: float,
    mask_nk1: Optional[torch.Tensor] = None,    
    mode: str = "linear",
    gaussian_sigma: Optional[torch.Tensor] = None,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Compute influence weights from distances to kernel points.
      - 'linear'  : w = max(0, 1 - d/extent)
      - 'gaussian': w = exp(- d^2 / (2*sigma^2))
    """
    if mode == "linear":
        w = (1.0 - dist_nkm / float(extent)).clamp_min(0.0)
    elif mode == "gaussian":
        sigma = float(gaussian_sigma) if gaussian_sigma is not None else (float(extent) / 1.5)
        inv_2s2 = 0.5 / (sigma * sigma + eps)
        w = torch.exp(- (dist_nkm ** 2) * inv_2s2)
    else:
        raise ValueError("influence mode must be 'linear' or 'gaussian'.")
    if mask_nk1 is not None:
        w = w * mask_nk1
    return w



#------------------non-deformable KPConv Layer with chunking----------------


class KPConvNonDeform(nn.Module):
    """
    Non-deformable Kernel Point Convolution (one-ring kernel points)
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        radius: float,
        num_kernel_points: int = 15,
        add_center: bool = True,
        inward_scale: float = 0.85,
        bias: bool = True,
        influence_mode: str = "linear",          # linear or gaussian
        gaussian_sigma: Optional[float] = None,  # if None, auto extent/1.5
        chunk_K: int = 0,                        # 0 = off; else process neighbors in chunks
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.radius = float(radius)
        self.num_kernel_points = int(num_kernel_points)
        self.influence_mode = influence_mode
        self.gaussian_sigma = gaussian_sigma
        self.chunk_K = int(chunk_K)

        # Fixed kernel points (no grad)
        KP_np = get_kernel_points(
            num_kpoints=self.num_kernel_points,
            KP_extent=self.radius,
            add_center=add_center,
            inward_scale=inward_scale,
            optimize=True, iters=200, step=0.08, momentum=0.85, seed=0
        )
        self.register_buffer('KP', torch.from_numpy(KP_np))  

        # Learnable weights
        self.weights = nn.Parameter(torch.randn(self.num_kernel_points, in_dim, out_dim) / math.sqrt(in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None

    def forward(
        self,
        q_pts: torch.Tensor,           
        s_pts: torch.Tensor,            
        s_feats: torch.Tensor,          
        neighbors: torch.Tensor,        
    ) -> torch.Tensor:
        Nq, K = neighbors.shape
        Cin = self.in_dim
        Cout = self.out_dim
        M = self.num_kernel_points
        KP = self.KP.to(s_feats.dtype)                             

        # No chunking → original (easy) path
        if self.chunk_K <= 0 or self.chunk_K >= K:
            s_xyz, nmask = _gather_with_neighbors(s_pts,   neighbors)  
            s_fea, _     = _gather_with_neighbors(s_feats, neighbors)  
            rel = s_xyz - q_pts[:, None, :]                            
            dist = torch.linalg.norm(rel[:, :, None, :] - KP[None, None, :, :], dim=-1)  
            w = _influence_weights(dist, extent=self.radius, mask_nk1=nmask,
                                   mode=self.influence_mode, gaussian_sigma=self.gaussian_sigma)  
            agg = torch.einsum('nkm,nkc->nmc', w, s_fea)               
            out = torch.einsum('nmc,mco->no', agg, self.weights)        
            if self.bias is not None:
                out = out + self.bias
            return out

        # ---- Chunked path- accumulate without allocating  ---------
        out = s_feats.new_zeros((Nq, Cout))
        Kc = int(self.chunk_K)

        for m in range(M):
            agg_m = s_feats.new_zeros((Nq, Cin))                        
            for start in range(0, K, Kc):
                end = min(K, start + Kc)
                n_chunk = neighbors[:, start:end]                      

                s_xyz_c, nmask_c = _gather_with_neighbors(s_pts,   n_chunk) 
                s_fea_c, _       = _gather_with_neighbors(s_feats, n_chunk)  

                rel_c = s_xyz_c - q_pts[:, None, :]                     
                # distances to kernel point m
                dist_c = torch.linalg.norm(rel_c - KP[m][None, None, :], dim=-1) 
                w_c = _influence_weights(dist_c.unsqueeze(-1), extent=self.radius,
                                         mask_nk1=nmask_c, mode=self.influence_mode,
                                         gaussian_sigma=self.gaussian_sigma).squeeze(-1)  

                # weighted sum over chunk neighbors
                agg_m += torch.einsum('nk,nkc->nc', w_c, s_fea_c)

            # apply kernel weight m - add to output
            out += torch.matmul(agg_m, self.weights[m])              

        if self.bias is not None:
            out = out + self.bias
        return out



# ------------------Blocks-----------------


class UnaryBlock(nn.Module):
    """1x1 'MLP' on point features"""
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class SimpleKPBlock(nn.Module):
    """KPConv -> BN -> ReLU"""
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        radius: float,
        kpoints: int = 15,
        influence_mode: str = "linear",
        gaussian_sigma: Optional[float] = None,
        chunk_K: int = 0,
    ):
        super().__init__()
        self.kp = KPConvNonDeform(in_dim, out_dim, radius,
                                  num_kernel_points=kpoints,
                                  influence_mode=influence_mode,
                                  gaussian_sigma=gaussian_sigma,
                                  chunk_K=chunk_K)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, q_pts, s_pts, s_feats, neighbors):
        x = self.kp(q_pts, s_pts, s_feats, neighbors)   
        x = self.bn(x)
        x = self.act(x)
        return x


class ResKPBlock(nn.Module):
    """
    Residual KPConv block: KPConv(Cin->Cout) -> BN/ReLU -> KPConv(Cout->Cout) + skip -> BN/ReLU
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        radius: float,
        kpoints: int = 15,
        influence_mode: str = "linear",
        gaussian_sigma: Optional[float] = None,
        chunk_K: int = 0,
    ):
        super().__init__()
        self.conv1 = KPConvNonDeform(in_dim, out_dim, radius,
                                     num_kernel_points=kpoints,
                                     influence_mode=influence_mode,
                                     gaussian_sigma=gaussian_sigma,
                                     chunk_K=chunk_K)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.conv2 = KPConvNonDeform(out_dim, out_dim, radius,
                                     num_kernel_points=kpoints,
                                     influence_mode=influence_mode,
                                     gaussian_sigma=gaussian_sigma,
                                     chunk_K=chunk_K)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.act = nn.ReLU(inplace=True)
        self.shortcut = None if in_dim == out_dim else UnaryBlock(in_dim, out_dim)

    def forward(self, q_pts, s_pts, s_feats, neighbors):
        identity = s_feats if self.shortcut is None else self.shortcut(s_feats)
        x = self.conv1(q_pts, s_pts, s_feats, neighbors)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(q_pts, s_pts, x, neighbors)
        x = self.bn2(x)
        x = self.act(x + identity)
        return x
