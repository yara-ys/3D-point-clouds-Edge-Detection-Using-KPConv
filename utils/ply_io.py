import numpy as np
from plyfile import PlyData, PlyElement

# --------- writer ----------
def write_ply_xyz_rgb(path: str, xyz: np.ndarray, rgb: np.ndarray):

    assert xyz.shape[0] == rgb.shape[0], "XYZ and RGB must match in length"
    n = xyz.shape[0]
    verts = np.empty(n, dtype=[('x','f4'),('y','f4'),('z','f4'),
                               ('red','u1'),('green','u1'),('blue','u1')])
    verts['x'], verts['y'], verts['z'] = xyz[:,0], xyz[:,1], xyz[:,2]
    verts['red'], verts['green'], verts['blue'] = rgb[:,0], rgb[:,1], rgb[:,2]
    PlyData([PlyElement.describe(verts, 'vertex')], text=False).write(path)


EDGE_COLOR      = np.array([255,   0,   0], np.uint8)  # red
NONEDGE_COLOR   = np.array([160, 160, 160], np.uint8)  # gray

# Error map colors:
TP_COLOR        = np.array([ 50, 205,  50], np.uint8)  # green (true edge)
FP_COLOR        = np.array([255, 0,   0], np.uint8)  # red (pred edge, gt non-edge)
FN_COLOR        = np.array([ 30, 144, 255], np.uint8)  # blue (missed edge)
TN_COLOR        = np.array([180, 180, 180], np.uint8)  # light gray (true non-edge)


def colorize_predictions(pred: np.ndarray,
                         edge_color: np.ndarray = EDGE_COLOR,
                         nonedge_color: np.ndarray = NONEDGE_COLOR) -> np.ndarray:

    pred = np.asarray(pred).astype(np.int32).ravel()
    rgb = np.empty((pred.shape[0], 3), dtype=np.uint8)
    rgb[pred == 1] = edge_color
    rgb[pred == 0] = nonedge_color
    return rgb

def colorize_errors(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
      TP=green, FP=red, FN=blue, TN=light gray
    """
    pred = np.asarray(pred).astype(np.int32).ravel()
    gt   = np.asarray(gt).astype(np.int32).ravel()
    assert pred.shape == gt.shape, "pred and gt must have same shape"
    rgb = np.empty((pred.shape[0], 3), dtype=np.uint8)

    tp = (pred == 1) & (gt == 1)
    fp = (pred == 1) & (gt == 0)
    fn = (pred == 0) & (gt == 1)
    tn = (pred == 0) & (gt == 0)

    rgb[tp] = TP_COLOR
    rgb[fp] = FP_COLOR
    rgb[fn] = FN_COLOR
    rgb[tn] = TN_COLOR
    return rgb


def save_prediction_ply(path: str, xyz: np.ndarray, pred: np.ndarray):
    """Write PLY colored by predicted labels (edge=red, non-edge=gray)"""
    rgb = colorize_predictions(pred)
    write_ply_xyz_rgb(path, xyz, rgb)

def save_errormap_ply(path: str, xyz: np.ndarray, pred: np.ndarray, gt: np.ndarray):
    """Write PLY colored by TP/FP/FN/TN """
    rgb = colorize_errors(pred, gt)
    write_ply_xyz_rgb(path, xyz, rgb)