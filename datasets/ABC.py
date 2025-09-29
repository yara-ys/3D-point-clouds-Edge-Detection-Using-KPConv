import os
import re
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset
from plyfile import PlyData
from sklearn.neighbors import NearestNeighbors
from typing import Optional

# -------C++ wrappers (compiled .so) ------------------
# grid subsampling
try:
    from cpp_wrappers.cpp_subsampling.grid_subsampling import subsample as grid_subsample
    from cpp_wrappers.cpp_subsampling.grid_subsampling import subsample_batch as grid_subsample_batch
except (ImportError, AttributeError):
    from cpp_wrappers.cpp_subsampling.grid_subsampling import grid_subsampling as grid_subsample
    def grid_subsample_batch(points, features=None, labels=None, sampleDl=0.1, verbose=0):
        return grid_subsample(points, features, labels, sampleDl, verbose)

# neighbors (batched radius)
from cpp_wrappers.cpp_neighbors.radius_neighbors import batch_query as _batch_query

# ------------------helpers ----------
def _np32(a): return np.ascontiguousarray(a, dtype=np.float32)
def _ni32(a): return np.ascontiguousarray(a, dtype=np.int32)

# -----------------fixed-width padding for neighbors ---------------
def _fix_neighbors_width(idx: np.ndarray, K: int) -> np.ndarray:
    """Ensure neighbors have exactly K columns (-1 padded or clipped)"""
    n, k = idx.shape
    if k == K:
        return idx
    if k > K:
        return idx[:, :K]
    pad = -np.ones((n, K - k), dtype=idx.dtype)
    return np.concatenate([idx, pad], axis=1)


#------------Robust radius neighbors with one-time fallback warning--------------------

_WARNED_NB_FALLBACK = False  #

def radius_neighbors_batch(Q, S, Q_sizes, S_sizes, radius):
    return _batch_query(_np32(Q), _np32(S),
                        _ni32(Q_sizes), _ni32(S_sizes),
                        radius=np.float32(radius))

def _radius_neighbors_safe(q_xyz: np.ndarray,
                           s_xyz: np.ndarray,
                           radius: float,
                           K: int) -> np.ndarray:
    """
    Try C++ batched radius neighbors; if it fails or returns nonsense,fall back to sklearn KD-tree
    """
    global _WARNED_NB_FALLBACK

    # Attempt C++ wrapper
    q = _np32(q_xyz); s = _np32(s_xyz)
    qs = np.array([len(q)], dtype=np.int32)
    ss = np.array([len(s)], dtype=np.int32)

    idx = None
    try:
        raw = radius_neighbors_batch(q, s, qs, ss, float(radius))  d
        idx = np.array(raw, dtype=np.int32, copy=True, order='C')
       
        Ns = s.shape[0]
        bad = (idx >= Ns) | (idx < -1)
        if bad.any():
            idx[bad] = -1
        idx = _fix_neighbors_width(idx, int(K)).astype(np.int32)
    except Exception:
        idx = None

    # Fallback to sklearn 
    if idx is None or idx.size == 0:
        if not _WARNED_NB_FALLBACK:
            print("[ABC] radius neighbors: C++ wrapper unavailable; "
                  "falling back to sklearn KD-tree (slower).")
            _WARNED_NB_FALLBACK = True
        nn = NearestNeighbors(radius=float(radius), algorithm='kd_tree').fit(s)
        lists = nn.radius_neighbors(q, return_distance=False)
        out = np.full((len(q), int(K)), -1, dtype=np.int32)
        for i, li in enumerate(lists):
            m = min(int(K), len(li))
            if m > 0:
                out[i, :m] = li[:m]
        return out

    return idx


#-------------------------IO helpers-------------------------------------


_float_re = re.compile(
    r'(?<![A-Za-z])'                  # avoid picking the letter-part of tokens like 'v1.1'
    r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)'   
    r'(?:[eE][+-]?\d+)?'              # scientific exponent
)

def read_ply_xyz_normals(path):
    with open(path, 'rb') as f:
        ply = PlyData.read(f)
    x = np.stack([ply['vertex'][k] for k in ['x','y','z']], axis=1).astype(np.float32)
    if all(k in ply['vertex'].data.dtype.names for k in ['nx','ny','nz']):
        n = np.stack([ply['vertex'][k] for k in ['nx','ny','nz']], axis=1).astype(np.float32)
    else:
        n = None
    return x, n

def read_lb(path: str, n: Optional[int] = None) -> np.ndarray:
    """ignores blanks/#, uses first token per line."""
    vals = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            line = line.replace(",", " ").replace(";", " ")
            tok = line.split()
            if tok:
                try:
                    vals.append(int(tok[0]))
                except ValueError:
                    continue
    lbl = np.asarray(vals, dtype=np.int64)
    if n is not None and lbl.size != n:
        raise ValueError(f"Label count mismatch in {path}: got {lbl.size} expected {n}")
    return lbl

def read_ssm(path: str, n_points: Optional[int] = None) -> np.ndarray:
    """
    Extracts only numeric tokens,---If n_points is given, uses first n_points*320 floats ,- Else requires total tokens % 320 == 0
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    toks = _float_re.findall(txt)
    if not toks:
        raise ValueError(f"No numeric tokens found in {path}")
    flat = np.array([float(t) for t in toks], dtype=np.float32)

    if n_points is not None:
        need = int(n_points) * 320
        if flat.size < need:
            raise ValueError(f"SSM has only {flat.size} floats but needs {need} in {path}")
        return flat[:need].reshape((int(n_points), 320))

    if flat.size % 320 != 0:
        raise ValueError(f"SSM token count {flat.size} not divisible by 320 in {path}")
    return flat.reshape((-1, 320))


#---------------------Dataset on-disk caching


class ABCDataset(Dataset):
    """
      Builds multi-scale points, neighbors, pools, upsamples,-Adds on-disk caching to avoid recomputing the graph every epoch
    """
    def __init__(
        self,
        cfg,
        split='Train',
        use_ssm=True,
        use_normals=True,
        voxel_size=None,                # if None, use cfg.first_subsampling_dl
        layer_radii_mult=None,          # if None, use cfg.layer_multipliers
        max_neighbors=None,             # int or list per level
        pool_cap: int = 32,             # children kept per coarse (-1 padded)
        normalize_ssm=True
    ):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.use_ssm = use_ssm
        self.use_normals = use_normals
        self.voxel_size = voxel_size if voxel_size is not None else getattr(cfg, 'first_subsampling_dl', 0.01)
        self.layer_multipliers  = layer_radii_mult or getattr(cfg, 'layer_multipliers', [1, 2, 4, 8, 16])
        self.max_neighbors = max_neighbors if max_neighbors is not None else getattr(cfg, 'max_neighbors', 64)
        self.pool_cap = int(pool_cap)
        self.normalize_ssm = normalize_ssm
        self.conv_radius = float(getattr(cfg, 'conv_radius', 2.5))

        # Resolve dirs from cfg
        if split.lower().startswith('train'):
            base_dir = getattr(cfg, 'train_path', None) or os.path.join(cfg.root_path, 'Train')
        else:
            base_dir = getattr(cfg, 'val_path', None) or os.path.join(cfg.root_path, 'Validation')

        ply_dir = os.path.join(base_dir, 'ply')
        lb_dir  = os.path.join(base_dir, 'lb')
        ssm_dir = os.path.join(base_dir, 'SSM_Challenge-ABC')

        self.ids = sorted([os.path.splitext(f)[0] for f in os.listdir(ply_dir) if f.endswith('.ply')])
        self.ply_paths = [os.path.join(ply_dir, f'{i}.ply') for i in self.ids]
        self.lb_paths  = [os.path.join(lb_dir,  f'{i}.lb')  for i in self.ids]
        self.ssm_paths = [os.path.join(ssm_dir, f'{i}.ssm') for i in self.ids]
        self.names = list(self.ids)

        # -----cache dir----
    
        root = getattr(cfg, 'root_path', os.path.dirname(os.path.dirname(ply_dir)))
        self.cache_dir = os.path.join(root, 'cache_abc')
        os.makedirs(self.cache_dir, exist_ok=True)

        # Dataset level ssm normalization
        self.ssm_mu, self.ssm_sigma = None, None
        if self.use_ssm and self.normalize_ssm:
            stats_dir = root
            m_path = os.path.join(stats_dir, 'ssm_mean.npy')
            s_path = os.path.join(stats_dir, 'ssm_std.npy')

            if os.path.isfile(m_path) and os.path.isfile(s_path):
                self.ssm_mu = np.load(m_path)[None, :].astype(np.float32)
                self.ssm_sigma = np.load(s_path)[None, :].astype(np.float32) + 1e-6
            else:
                if self.split.lower().startswith('train'):
                    print("[ABC] Fitting SSM mean/std on Train split (once) ...")
                    samp = []
                    n_take = min(20, len(self.ssm_paths))
                    for j in range(n_take):
                        p_ssm = self.ssm_paths[j]
                        p_ply = self.ply_paths[j]
                        with open(p_ply, "rb") as f:
                            Nj = len(PlyData.read(f)["vertex"].data)
                        s = read_ssm(p_ssm, n_points=Nj)
                        if s.shape[0] > 50000:
                            s = s[np.random.choice(s.shape[0], 50000, replace=False)]
                        samp.append(s)
                    if samp:
                        cat = np.concatenate(samp, 0)
                        mu = cat.mean(0).astype(np.float32)
                        sg = cat.std(0).astype(np.float32)
                        np.save(m_path, mu)
                        np.save(s_path, sg)
                        self.ssm_mu = mu[None, :]
                        self.ssm_sigma = sg[None, :] + 1e-6
                    else:
                        self.ssm_mu = np.zeros((1, 320), np.float32)
                        self.ssm_sigma = np.ones((1, 320), np.float32)
                else:
                    print("[ABC][WARN] SSM stats not found. "
                          "Run Train once to create ssm_mean.npy / ssm_std.npy.")
                    #split-local fallback
                    samp = []
                    n_take = min(10, len(self.ssm_paths))
                    for j in range(n_take):
                        p_ssm = self.ssm_paths[j]
                        p_ply = self.ply_paths[j]
                        with open(p_ply, "rb") as f:
                            Nj = len(PlyData.read(f)["vertex"].data)
                        s = read_ssm(p_ssm, n_points=Nj)
                        if s.shape[0] > 50000:
                            s = s[np.random.choice(s.shape[0], 50000, replace=False)]
                        samp.append(s)
                    if samp:
                        cat = np.concatenate(samp, 0)
                        self.ssm_mu = cat.mean(0, keepdims=True).astype(np.float32)
                        self.ssm_sigma = cat.std(0, keepdims=True).astype(np.float32) + 1e-6
                    else:
                        self.ssm_mu = np.zeros((1, 320), np.float32)
                        self.ssm_sigma = np.ones((1, 320), np.float32)

        #absolute radii provided in cfg
        self.abs_layer_radii = getattr(cfg, 'layer_radii', None)

        #epoch-time downsampling for dev
        self.max_items = getattr(cfg, 'max_items', None)

    def __len__(self):
        n = len(self.ids)
        m = self.max_items
        return min(n, m) if (m is not None and m > 0) else n

    # -------deterministic radii-------------
    def _radius_schedule(self):
        if self.abs_layer_radii is not None:
            return list(self.abs_layer_radii)
        base = float(self.voxel_size)
        return [self.conv_radius * base * float(m) for m in self.layer_multipliers]

    # ---------cache key----------
    def _cache_key(self, idx: int) -> str:
        key = (
            f"id={self.ids[idx]}|vox={self.voxel_size}|mult={self.layer_multipliers}|"
            f"K={self.max_neighbors}|cap={self.pool_cap}|radii={self.abs_layer_radii}|"
            f"use_ssm={self.use_ssm}|use_normals={self.use_normals}"
        )
        h = hashlib.sha1(key.encode()).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"{self.ids[idx]}_{h}.npz")

    def __getitem__(self, idx):
        # -------------------load cache if available-------------------
        cpath = self._cache_key(idx)
        if os.path.isfile(cpath):
            data = np.load(cpath, allow_pickle=False)
            L = int(data['L'])
            return {
                'points':   [torch.from_numpy(data[f'pts{l}'])   for l in range(L)],
                'features': [torch.from_numpy(data['fea0'])],
                'neighbors':[torch.from_numpy(data[f'nei{l}']).long()  for l in range(L)],
                'pools':    [torch.from_numpy(data[f'pool{l}']).long() for l in range(L-1)],
                'upsamples':[torch.from_numpy(data[f'up{l}']).long()   for l in range(L-1)],
                'labels':   torch.from_numpy(data['y0']).long(),
                'radii':    torch.from_numpy(data['radii']).float(),
                'id':       self.ids[idx],
            }

        # -------------build once, then cache----------------
        xyz, nrm = read_ply_xyz_normals(self.ply_paths[idx])
        y = read_lb(self.lb_paths[idx]).astype(np.int64)
        assert xyz.shape[0] == y.shape[0], f"Point/label mismatch in {self.ids[idx]}"

        feats_list = []
        if getattr(self.cfg, "include_xyz", False):
            feats_list.append(xyz.astype(np.float32, copy=False))

        if self.use_normals and (nrm is not None):
            nn = np.linalg.norm(nrm, axis=1, keepdims=True)
            nrm = nrm / (nn + 1e-8)
            feats_list.append(nrm.astype(np.float32, copy=False))

        if self.use_ssm:
            N = xyz.shape[0]
            ssm = read_ssm(self.ssm_paths[idx], n_points=N)
            if self.normalize_ssm and (self.ssm_mu is not None) and (self.ssm_sigma is not None):
                mu = np.asarray(self.ssm_mu, dtype=np.float32).reshape(1, -1)
                sg = np.asarray(self.ssm_sigma, dtype=np.float32).reshape(1, -1)
                ssm = (ssm - mu) / (sg + 1e-6)
            feats_list.append(ssm.astype(np.float32, copy=False))

        feats = (np.concatenate(feats_list, axis=1).astype(np.float32, copy=False)
                 if feats_list else np.zeros((xyz.shape[0], 0), np.float32))

        # L0
        xyz0 = xyz.astype(np.float32, copy=False)
        f0   = feats.astype(np.float32, copy=False)
        y0   = y.astype(np.int64,   copy=False)

        # batched subsampling for deeper levels
        points = [xyz0]
        sizes  = [np.array([len(xyz0)], dtype=np.int32)]
        featsP = [f0]
        L = len(self.layer_multipliers)
        for l in range(1, L):
            voxel_dl = np.float32(self.voxel_size * float(self.layer_multipliers[l]))
            P_prev, B_prev = points[-1], sizes[-1]
            P_next, B_next = grid_subsample_batch(
                _np32(P_prev), _ni32(B_prev),
                sampleDl=voxel_dl, method="barycenters",
                max_p=np.int32(0), verbose=0,
            )
            points.append(P_next.astype(np.float32, copy=False))
            sizes.append(B_next.astype(np.int32,  copy=False))

        # radii and neighbors
        radii = self._radius_schedule()
        neighbors = []
        for l, r in enumerate(radii):
            K = self.max_neighbors[l] if isinstance(self.max_neighbors, (list, tuple)) else int(self.max_neighbors)
            neigh = _radius_neighbors_safe(points[l], points[l], r, K)
            neighbors.append(neigh)

        # pools/upsamples
        pools, upsamples = [], []
        for l in range(L - 1):
            fine = points[l]; coarse = points[l + 1]
            knn = NearestNeighbors(n_neighbors=1).fit(coarse)
            parent = knn.kneighbors(fine, return_distance=False).squeeze(1) 
            up_idx = parent.astype(np.int64)[:, None]                         
            upsamples.append(torch.from_numpy(up_idx))

            Nf = fine.shape[0]; Nc = coarse.shape[0]
            cap = int(self.pool_cap)
            buckets = [[] for _ in range(Nc)]
            for fi, ci in enumerate(parent):
                if len(buckets[ci]) < cap:
                    buckets[ci].append(fi)
            pool_mat = -np.ones((Nc, cap), dtype=np.int64)  # -1 padded
            for ci, lst in enumerate(buckets):
                if lst:
                    Lc = min(len(lst), cap)
                    pool_mat[ci, :Lc] = np.asarray(lst[:Lc], dtype=np.int64)
            pools.append(torch.from_numpy(pool_mat))

        # ---------- save cache-----------------
        try:
            save = {'L': np.int32(L),
                    'fea0': featsP[0].astype(np.float32, copy=False),
                    'y0':   y0.astype(np.int64, copy=False),
                    'radii': np.asarray(radii, dtype=np.float32)}
            for l in range(L):
                save[f'pts{l}'] = points[l].astype(np.float32, copy=False)
                save[f'nei{l}'] = neighbors[l].astype(np.int32,  copy=False)
            for l in range(L - 1):
                save[f'pool{l}'] = pools[l].numpy().astype(np.int64, copy=False)
                save[f'up{l}']   = upsamples[l].numpy().astype(np.int64, copy=False)
            np.savez_compressed(self._cache_key(idx), **save)
        except Exception:
            pass  

        return {
            'points':   [torch.from_numpy(p) for p in points],
            'features': [torch.from_numpy(featsP[0])],
            'neighbors':[torch.from_numpy(n).long() for n in neighbors],
            'pools':    pools,
            'upsamples':upsamples,
            'labels':   torch.from_numpy(y0).long(),
            'radii':    torch.tensor(radii, dtype=torch.float32),
            'id':       self.ids[idx],
        }


#--------------------------------Collate-----------------------------------------------


def abc_collate(batch_list):

    L = len(batch_list[0]['points'])
    batched = {
        'points': [], 'features': [], 'neighbors': [], 'pools': [], 'upsamples': [],
        'batch_ids': [],
        'labels': torch.cat([b['labels'] for b in batch_list], 0),
        'ids': [b['id'] for b in batch_list],
        'radii': batch_list[0]['radii'],
    }

    sizes_per_level, offsets_per_level = [], []
    for l in range(L):
        sizes = [b['points'][l].shape[0] for b in batch_list]
        sizes_per_level.append(sizes)
        offsets_per_level.append(np.cumsum([0] + sizes[:-1]).astype(np.int64))

    for l in range(L):
        pts_l = [b['points'][l] for b in batch_list]
        pts_cat = torch.cat(pts_l, 0)
        batched['points'].append(pts_cat)

        if l == 0:
            fea0 = [b['features'][0] for b in batch_list]
            fea0_cat = torch.cat(fea0, 0) if fea0[0].numel() else torch.zeros((pts_cat.shape[0], 0), dtype=torch.float32)
            batched['features'] = [fea0_cat]

        neighs = []
        for off, item in zip(offsets_per_level[l], batch_list):
            n = item['neighbors'][l].clone()
            mask = (n >= 0)
            n[mask] += int(off)
            neighs.append(n.long())
        batched['neighbors'].append(torch.cat(neighs, 0))

        sizes = sizes_per_level[l]
        batched['batch_ids'].append(torch.from_numpy(
            np.concatenate([np.full(s, i, dtype=np.int32) for i, s in enumerate(sizes)])
        ))

    for l in range(L - 1):
        # pools: offset valid fine indices; keep -1 padding as -1
        rows = []
        for off_f, item in zip(offsets_per_level[l], batch_list):
            pool = item['pools'][l].clone()
            mask = (pool >= 0)
            pool[mask] += int(off_f)
            rows.append(pool.long())
        batched['pools'].append(torch.cat(rows, 0))

        # upsamples: add coarse offset
        ups = []
        for off_c, item in zip(offsets_per_level[l + 1], batch_list):
            up = item['upsamples'][l].clone()
            up += int(off_c)
            ups.append(up.long())
        batched['upsamples'].append(torch.cat(ups, 0))

    return batched
