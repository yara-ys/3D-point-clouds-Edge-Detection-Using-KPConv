

import math
import numpy as np
from functools import lru_cache

# ----------one ring generator ----------
def _spherical_fibonacci(n: int) -> np.ndarray:
    """Quasi-uniform points on the unit sphere (no center)"""
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    theta = 2.0 * math.pi * i / phi
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y, z], axis=1).astype(np.float32)

def _pairwise_forces(P: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Coulomb-like repulsion: sum_j (pi - pj) / ||pi - pj||^3 (diagonal = 0)"""
    diff  = P[:, None, :] - P[None, :, :]            
    dist2 = np.sum(diff * diff, axis=2) + np.eye(P.shape[0])
    inv_d3 = 1.0 / (np.sqrt(dist2)**3 + eps)
    np.fill_diagonal(inv_d3, 0.0)
    return (diff * inv_d3[:, :, None]).sum(axis=1)    

def _project_to_ball(P: np.ndarray, radius: float) -> np.ndarray:
    norms = np.linalg.norm(P, axis=1, keepdims=True)  
    mask = (norms[:, 0] > radius)
    if np.any(mask):
        P[mask] *= (radius / norms[mask])
    return P

@lru_cache(maxsize=256)
def get_kernel_points(num_kpoints: int,
                      KP_extent: float,
                      add_center: bool = True,
                      inward_scale: float = 0.85,
                      optimize: bool = True,
                      iters: int = 200,
                      step: float = 0.08,
                      momentum: float = 0.85,
                      seed: int = 0) -> np.ndarray:
    """
    One ring kernel points inside a ball of radius KP_extent
      - num_kpoints: total K (includes center if add_center=True)
      - inward_scale: ring radius as a fraction of KP_extent
      - optimize: Coulomb repulsion on the ring (for uniformity)
    """
    if num_kpoints <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    K_center = 1 if add_center else 0
    K_shell  = max(0, num_kpoints - K_center)

    #points on unit sphere at a single radius 
    shell = _spherical_fibonacci(K_shell).astype(np.float64)
    shell *= float(inward_scale)  # pull slightly inward for more interior coverage

    #repulsion optimization to keeps them evenly spread
    if optimize and K_shell > 1:
        P = shell.copy()
        V = np.zeros_like(P)
        for _ in range(iters):
            F = _pairwise_forces(P)
            V = momentum * V + (step * F)
            P = _project_to_ball(P + V, radius=1.0)
        shell = P

    # assemble (+center) and scale to KP_extent
    if add_center:
        pts = np.vstack([np.zeros((1, 3), dtype=np.float64), shell])
    else:
        pts = shell
    return (pts * float(KP_extent)).astype(np.float32)

# ----------visualization (didnt need to use) ----------
def visualize_kernel_points(KP: np.ndarray, KP_extent: float, out_path: str, title: str = ""):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

   
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 15)
    xs = KP_extent * np.outer(np.cos(u), np.sin(v))
    ys = KP_extent * np.outer(np.sin(u), np.sin(v))
    zs = KP_extent * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, linewidth=0.3, alpha=0.4)

    # kernel points
    ax.scatter(KP[:, 0], KP[:, 1], KP[:, 2], s=50, depthshade=True)
    if (np.linalg.norm(KP, axis=1) < 1e-6).any():
        cidx = np.where(np.linalg.norm(KP, axis=1) < 1e-6)[0][0]
        ax.scatter([KP[cidx, 0]], [KP[cidx, 1]], [KP[cidx, 2]], s=120, marker='o')

    ax.set_title(title or f"One-ring KPConv kernel (K={KP.shape[0]})")
    lim = KP_extent * 1.1
    ax.set_xlim([-lim, lim]); ax.set_ylim([-lim, lim]); ax.set_zlim([-lim, lim])
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)