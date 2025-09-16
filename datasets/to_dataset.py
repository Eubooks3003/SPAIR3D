# datasets/ply_dataset.py
import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

try:
    from plyfile import PlyData  # pip install plyfile
except Exception:
    PlyData = None

def _read_ply_points(path, instance_field=None):
    if PlyData is None:
        raise ImportError("Please `pip install plyfile` (or switch to open3d).")
    pd = PlyData.read(path)
    v  = pd['vertex'].data
    xyz = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float32)
    # optional colors
    if all(c in v.dtype.names for c in ('red','green','blue')):
        rgb = np.stack([v['red'], v['green'], v['blue']], axis=1).astype(np.float32) / 255.0
    else:
        rgb = None
    # optional instance ids for metrics
    inst = None
    if instance_field and instance_field in v.dtype.names:
        inst = np.asarray(v[instance_field]).astype(np.int64)
    elif 'instance' in v.dtype.names:
        inst = np.asarray(v['instance']).astype(np.int64)
    elif 'label' in v.dtype.names:
        inst = np.asarray(v['label']).astype(np.int64)
    return xyz, rgb, inst

def _normalize_xyz(xyz, mode='unit_sphere'):
    if mode in (None, 'none'):
        return xyz
    if mode == 'zshift_scale_like_uor':
        xyz = xyz.copy()
        xyz[:, 2] -= 7.0/16.0
        xyz *= (16.0/8.0)
        return xyz
    # default: center + scale to ~[-1,1]
    center = np.median(xyz, axis=0, keepdims=True)
    xyz0 = xyz - center
    rad = np.percentile(np.linalg.norm(xyz0, axis=1), 95)
    scale = (rad / 0.9) if rad > 0 else 1.0
    return xyz0 / scale

class PLYDataset(Dataset):
    """
    Looks for <root>/<split>/*.ply (split in {'train','val','test'}), else falls back to <root>/*.ply.
    Returns a dict; use spair3d_collate to get (pos, rgb, batch, Id, layer).
    """
    def __init__(self, root, split='train', glob_pat='*.ply', normalize='unit_sphere',
                 max_points=None, instance_field=None):
        self.split = split
        split_dir = os.path.join(root, split)
        base = split_dir if os.path.isdir(split_dir) else root
        self.files = sorted(glob.glob(os.path.join(base, glob_pat)))
        if not self.files:
            raise FileNotFoundError(f"No PLYs found under {base} with pattern {glob_pat}")
        self.normalize = normalize
        self.max_points = max_points
        self.instance_field = instance_field

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        xyz, rgb, inst = _read_ply_points(path, self.instance_field)
        xyz = _normalize_xyz(xyz, self.normalize)

        # optional downsample (train: random, val/test: deterministic)
        if self.max_points is not None and xyz.shape[0] > self.max_points:
            N = xyz.shape[0]
            if self.split == 'train':
                sel = np.random.choice(N, self.max_points, replace=False)
            else:
                sel = np.linspace(0, N - 1, self.max_points, dtype=np.int64)
            xyz = xyz[sel]
            if rgb is not None: rgb = rgb[sel]
            if inst is not None: inst = inst[sel]

        layer = np.zeros(xyz.shape[0], dtype=np.int64)
        Id    = inst if inst is not None else np.zeros(xyz.shape[0], dtype=np.int64)

        return {
            'pos': torch.from_numpy(xyz),                          # [N,3] float32
            'rgb': torch.from_numpy(rgb) if rgb is not None else None,  # [N,3] float32 or None
            'layer': torch.from_numpy(layer),                      # [N] int64
            'Id': torch.from_numpy(Id),                            # [N] int64
            'file_idx': idx,
            'path': path,
        }

def spair3d_collate(batch_list):
    B = len(batch_list)
    Ns = [b['pos'].shape[0] for b in batch_list]
    total = sum(Ns)

    pos = torch.cat([b['pos'] for b in batch_list], dim=0)
    if batch_list[0]['rgb'] is None:
        rgb = torch.zeros(total, 3, dtype=pos.dtype)
    else:
        rgb = torch.cat([b['rgb'] for b in batch_list], dim=0)

    batch = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(Ns)], dim=0)
    Id    = torch.cat([b['Id'] for b in batch_list], dim=0)
    layer = torch.cat([b['layer'] for b in batch_list], dim=0)
    return (pos, rgb, batch, Id, layer)


def make_ply_loader(root, split, batch_size, num_workers=4, pin_memory=True,
                    normalize='unit_sphere', max_points=None, instance_field=None,
                    shuffle_train=True):
    ds = PLYDataset(root, split=split, normalize=normalize,
                    max_points=max_points, instance_field=instance_field)
    shuffle = (split == 'train') and shuffle_train
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        collate_fn=spair3d_collate,
        drop_last=(split == 'train')  # typical for training
    )

def make_ply_dataset(root, split, normalize="unit_sphere", max_points=300_000, instance_field=None):
    return PLYDataset(root, split, normalize=normalize, max_points=max_points, instance_field=instance_field)
