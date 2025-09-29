from typing import List, Optional
import torch
import torch.nn as nn

from models.blocks import (
    SimpleKPBlock,
    ResKPBlock,
    closest_pool,
    interpolated_upsample,
)


class KPFCNN_Seg(nn.Module):
    """
    Non-deformable KPConv U-Net for point-wise classification
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        radii: List[float],
        widths: Optional[List[int]] = None,
        kpoints: int = 15,
        use_residual: bool = True,
        # KPConv influence
        influence_mode: str = "linear",            # linear or gaussian
        gaussian_sigma: Optional[float] = None,    # if None, KPConv uses extent/1.5
        # Upsampling
        upsample_mode: str = "nearest",            # nearest or interp
        interp_k: int = 3,                         
        interp_mode: str = "idw",                  # nearest or idw or gaussian
        interp_sigma: Optional[float] = None,      # if None and mode="gaussian"; auto from kNN dists
        # Chunking (memory)
        chunkK: int = 0,                           # 0 disables; else chunk neighbors by Kc
    ):
        super().__init__()

        self.L = len(radii)
        self.radii = list(radii)
        self.kpoints = int(kpoints)
        self.influence_mode = influence_mode
        self.gaussian_sigma = gaussian_sigma
        self.upsample_mode = upsample_mode
        self.interp_k = int(interp_k)
        self.interp_mode = interp_mode
        self.interp_sigma = interp_sigma
        self.chunkK = int(chunkK)

        if widths is None:
            base = 64
            widths = [base * (2 ** min(i, 3)) for i in range(self.L)]  
        assert len(widths) == self.L, "widths must have one entry per level"
        self.widths = widths

        Block = ResKPBlock if use_residual else SimpleKPBlock

        # ------------Encoder ----------------------
        enc_blocks = []
        for l in range(self.L):
            inc = in_dim if l == 0 else widths[l - 1]
            outc = widths[l]
            enc_blocks.append(
                Block(
                    inc,
                    outc,
                    radius=self.radii[l],
                    kpoints=self.kpoints,
                    influence_mode=self.influence_mode,
                    gaussian_sigma=self.gaussian_sigma,
                    chunk_K=self.chunkK,                
                )
            )
        self.enc_blocks = nn.ModuleList(enc_blocks)

        # -----------------Decoder----------------------
        dec_blocks = []
        for l in range(self.L - 2, -1, -1):  
            inc = widths[l] + widths[l + 1]  
            outc = widths[l]
            dec_blocks.append(
                SimpleKPBlock(
                    inc,
                    outc,
                    radius=self.radii[l],
                    kpoints=self.kpoints,
                    influence_mode=self.influence_mode,
                    gaussian_sigma=self.gaussian_sigma,
                    chunk_K=self.chunkK,                
                )
            )
        self.dec_blocks = nn.ModuleList(dec_blocks)

        # ----------------Head----------------------
        self.head = nn.Sequential(
            nn.Linear(widths[0], widths[0]),
            nn.ReLU(inplace=True),
            nn.Linear(widths[0], num_classes),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        pts:  List[torch.Tensor] = batch['points']
        nbh:  List[torch.Tensor] = batch['neighbors']
        pools:List[torch.Tensor] = batch['pools']
        ups:  List[torch.Tensor] = batch.get('upsamples', [])
        feats0: torch.Tensor     = batch['features'][0]        

        # ---------Encoder-------
        enc_feats = []
        x = feats0
        for l in range(self.L):
            x = self.enc_blocks[l](pts[l], pts[l], x, nbh[l])     
            enc_feats.append(x)
            if l < self.L - 1:
                x = closest_pool(x, pools[l])                 

        # ---Decoder-------
        for i, l in enumerate(range(self.L - 2, -1, -1)):
            if self.upsample_mode == "nearest":
                parent = ups[l].view(-1).long()                                    
                up = interpolated_upsample(
                    features_coarse = x,                                           
                    coarse_pts      = pts[l + 1],                                  
                    fine_pts        = pts[l],                                      
                    parent_inds     = parent,                                      
                    mode            = "nearest",
                )
            elif self.upsample_mode == "interp":
                up = interpolated_upsample(
                    features_coarse=x,
                    coarse_pts=pts[l + 1],
                    fine_pts=pts[l],
                    k=self.interp_k,
                    mode=self.interp_mode,
                    gaussian_sigma=self.interp_sigma,
                )
            else:
                raise ValueError("upsample_mode must be 'nearest' or 'interp'.")

            cat = torch.cat([enc_feats[l], up], dim=1)                             
            x = self.dec_blocks[i](pts[l], pts[l], cat, nbh[l])                    

        logits = self.head(x)                                                      
        return logits


def build_from_cfg(cfg, in_dim: int, num_classes: int | None = None):
    """
    One-liner builder for KPFCNN_Seg from Config
    """
    assert getattr(cfg, "layer_radii", None) is not None, \

    return KPFCNN_Seg(
        in_dim=in_dim,
        num_classes=(cfg.num_classes if num_classes is None else num_classes),
        radii=list(cfg.layer_radii),
        widths=list(cfg.widths),
        kpoints=cfg.num_kernel_points,
        # KPConv influence
        influence_mode=getattr(cfg, "influence_mode", getattr(cfg, "kp_influence_mode", "linear")),
        gaussian_sigma=getattr(cfg, "kp_gaussian_sigma", None),
        # Decoder upsampling
        upsample_mode=cfg.upsample_mode,
        interp_k=cfg.interp_k,
        interp_mode=cfg.interp_mode,
        interp_sigma=cfg.interp_sigma,
        # Chunking
        chunkK=int(getattr(cfg, "kp_chunkK", 0)),
        use_residual=bool(getattr(cfg, "use_residual", True)),
    )
