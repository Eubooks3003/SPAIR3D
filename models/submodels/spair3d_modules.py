import os
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.relaxed_bernoulli import LogitRelaxedBernoulli
from torch.distributions.bernoulli import Bernoulli

from torch_geometric.nn import radius_graph, LayerNorm, GraphNorm
from torch_scatter import scatter_mean, scatter_sum, scatter_log_softmax

from torch.utils.checkpoint import checkpoint as _ckpt

from models.layers.AutoRegistration import AutoRegistrationLayer
from models.layers.PointConv import PointConv, CenterShift
from models.utils import to_sigma, find_voxel_center, voxel_mean_pool

# -------------------- Lightweight profiling helpers --------------------
from contextlib import contextmanager

_ENABLE_MEM    = os.getenv("SPAIR_MEM", "0") == "1"
_ENABLE_SHAPES = os.getenv("SPAIR_SHAPES", "0") == "1"
_ENABLE_NVTX   = os.getenv("SPAIR_NVTX", "0") == "1"

def _mb(x): return x / (1024.0 ** 2)

def _mem(tag: str):
    if _ENABLE_MEM and torch.cuda.is_available():
        torch.cuda.synchronize()
        a = _mb(torch.cuda.memory_allocated())
        r = _mb(torch.cuda.memory_reserved())
        p = _mb(torch.cuda.max_memory_allocated())
        print(f"[MEM] {tag}: alloc={a:.1f}MB reserved={r:.1f}MB peak={p:.1f}MB")

def _reset_peak():
    if _ENABLE_MEM and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def _shape_str(t):
    try:
        return tuple(t.shape)
    except Exception:
        return "?"

def _shapes(tag: str, **tensors):
    if _ENABLE_SHAPES:
        parts = []
        for k, v in tensors.items():
            if isinstance(v, torch.Tensor):
                parts.append(f"{k}={tuple(v.shape)} {str(v.dtype).replace('torch.', '')}")
            elif isinstance(v, (list, tuple)) and len(v) and isinstance(v[0], torch.Tensor):
                parts.append(f"{k}[0]={tuple(v[0].shape)} ... (len={len(v)})")
            else:
                parts.append(f"{k}={type(v).__name__}")
        print(f"[SHAPE] {tag}: " + ", ".join(parts))

def _assert_finite(tag, *tensors):
    for t in tensors:
        if t is None: 
            continue
        if not torch.isfinite(t).all():
            bad = (~torch.isfinite(t)).nonzero(as_tuple=False)[:5]
            print(f"[NaN/Inf] {tag}: found {bad.shape[0]} bad values; first idx: {bad.flatten().tolist()[:10]}")
            return False
    return True

@contextmanager
def _nvtx(name: str):
    if _ENABLE_NVTX and torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield
# ----------------------------------------------------------------------

def ckpt(fn, *args):
    # Wrap so checkpoint sees only Tensor args and a single Tensor/tuple return
    return _ckpt(lambda *xs: fn(*xs), *args)

class SPAIRPointFeatureNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.radius = 1/16
        self.max_num_neighbors = 128
        self.conv1 = PointConv(c_in = 3,  c_mid = 8,  c_out = 8)
        self.conv2 = PointConv(c_in = 8,  c_mid = 16, c_out = 16)
        self.conv3 = PointConv(c_in = 16, c_mid = 32, c_out = 32)

    def forward(self, pos, rgb, batch):
        _shapes("PointFeat.in", pos=pos, batch=batch)

        _assert_finite("PointFeat.pos ", pos)
        _assert_finite("PointFeat.batch ", batch)
        _mem("PointFeat:start")
        with _nvtx("PointFeat.radius_graph"):
            out_index, in_index = radius_graph(pos, self.radius, batch, loop=True,
                                               max_num_neighbors=64, flow='target_to_source')
        _mem("PointFeat:after radius_graph")

        with _nvtx("PointFeat.conv1"):
            out = F.celu(self.conv1(pos, pos, batch, in_index=in_index, out_index=out_index))
        _mem("PointFeat:after conv1")

        self._assert_finite("PointFeat: After Conv1 ", out)

        with _nvtx("PointFeat.conv2"):
            out = F.celu(self.conv2(out, pos, batch, in_index=in_index, out_index=out_index))
        _mem("PointFeat:after conv2")

        self._assert_finite("PointFeat: After Conv2 ", out)


        with _nvtx("PointFeat.conv3"):
            out = F.celu(self.conv3(out, pos, batch, in_index=in_index, out_index=out_index))
        _mem("PointFeat:end")

        self._assert_finite("PointFeat: End ", out)


        return pos, out, batch

    def _assert_finite(self, tag, *tensors):
        for t in tensors:
            if t is None: 
                continue
            if not torch.isfinite(t).all():
                bad = (~torch.isfinite(t)).nonzero(as_tuple=False)[:5]
                print(f"[NaN/Inf] {tag}: found {bad.shape[0]} bad values; first idx: {bad.flatten().tolist()[:10]}")
                return False
        return True

class SPAIRGridFeatureNetwork(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer_norm = cfg.grid_encoder_ln
        self.ar = cfg.grid_encoder_ar
        self.glimpse_type = cfg.glimpse_type
        self.generate_z_pres = cfg.generate_z_pres

        if self.ar:
            self.ar1 = AutoRegistrationLayer(x_dim = 64,  f_hidden = 64,  f_out = 64,  g_hidden = 64,  g_out = 64)
            self.ar2 = AutoRegistrationLayer(x_dim = 128, f_hidden = 128, f_out = 128, g_hidden = 128, g_out = 128)
            self.ar3 = AutoRegistrationLayer(x_dim = 256, f_hidden = 256, f_out = 256, g_hidden = 256, g_out = 256, end_relu=False)
            self.ar4 = AutoRegistrationLayer(x_dim = 256, f_hidden = 256, f_out = 256, g_hidden = 256, g_out = 256, end_relu=False)
            self.ar5 = AutoRegistrationLayer(x_dim = 256, f_hidden = 256, f_out = 256, g_hidden = 256, g_out = 256, end_relu=False)

        self.conv1 = PointConv(16/1, max_num_neighbors = 128, c_in = 32,  c_mid = 32,  c_out = 64)
        self.conv2 = PointConv(2/16, max_num_neighbors = 128, c_in = 64,  c_mid = 64,  c_out = 128)
        self.conv3 = PointConv(2/16, max_num_neighbors = 128, c_in = 128, c_mid = 128, c_out = 256)
        self.conv4 = CenterShift(c_in = 256, c_mid = 256, c_out = 256)

        if self.layer_norm:
            self.norm1 = GraphNorm(64,  eps=1e-5)
            self.norm2 = GraphNorm(128, eps=1e-5)
            self.norm3 = GraphNorm(256, eps=1e-5)


        if self.glimpse_type == "ball":
            self.linear = torch.nn.Linear(in_features = 256, out_features = 9)
        elif self.glimpse_type == "box":
            self.linear = torch.nn.Linear(in_features = 256, out_features = 13)

        
        self.rg_max_k            = getattr(cfg, "rg_max_neighbors_glimpse", 24)
        self.grid_max_k            = getattr(cfg, "grid_max_neighbors_glimpse", 24)

    def forward(self, pos, feature, batch, temperature):
        _reset_peak()
        _shapes("GridFeat.in", pos=pos, feature=feature, batch=batch)
        _mem("GridFeat:start")

        self._assert_finite("Grid Feat Starting", feature)
        max_pos, _ = torch.max(pos, dim=0)
        min_pos, _ = torch.min(pos, dim=0)
        noise = torch.rand_like(min_pos) * (1/8)
        min_pos -= noise

        with _nvtx("GridFeat.voxel_pool_0"):
            (out_index, in_index), pos, batch, pos_sample, batch_sample, voxel_cluster, voxel_cluster_sample, inv = \
                voxel_mean_pool(pos=pos, batch=batch, start=min_pos, end=max_pos, size=0.5 / 16)
            num_pts = scatter_sum(
                torch.ones_like(out_index, dtype=torch.long),
                out_index, dim=0, dim_size=pos_sample.size(0)
            )
            assert (num_pts > 0).all(), "Empty cluster detected before next PointConv"
            feature = feature[inv]
        _shapes("GridFeat.pool0", pos=pos, pos_sample=pos_sample, feature=feature)
        _mem("GridFeat:after pool0")

        with _nvtx("GridFeat.conv1"):
            feature = F.celu(ckpt(
                lambda xin, pin, bin, pout, bout, iidx, oidx:
                    self.conv1(x_in=xin, pos_in=pin, batch_in=bin,
                            pos_out=pout, batch_out=bout,
                            in_index=iidx, out_index=oidx),
                feature, pos, voxel_cluster, pos_sample, voxel_cluster_sample, in_index, out_index
            ))
        _mem("GridFeat:after conv1")

        self._assert_finite("GridFeat.after conv1", feature)
        pos = pos_sample
        batch = batch_sample

        with _nvtx("GridFeat.radius_graph_0"):
            edge_index = radius_graph(pos, 0.5 / 16, batch, loop=True, max_num_neighbors=self.grid_max_k)
        if self.ar:
            with _nvtx("GridFeat.ar1"):
                feature, _, _ = self.ar1(feature, pos, edge_index)
        if self.layer_norm:
            feature = self.norm1(feature, batch)

        with _nvtx("GridFeat.voxel_pool_1"):
            (out_index, in_index), pos, batch, pos_sample, batch_sample, voxel_cluster, voxel_cluster_sample, inv = \
                voxel_mean_pool(pos=pos, batch=batch, start=min_pos, end=max_pos, size=1 / 16)
            num_pts = scatter_sum(
                torch.ones_like(out_index, dtype=torch.long),
                out_index, dim=0, dim_size=pos_sample.size(0)
            )
            assert (num_pts > 0).all(), "Empty cluster detected before next PointConv"
            
            feature = feature[inv]
        _mem("GridFeat:after pool1")

        with _nvtx("GridFeat.conv2"):
            feature = F.celu(ckpt(
                lambda xin, pin, bin, pout, bout, iidx, oidx:
                    self.conv2(x_in=xin, pos_in=pin, batch_in=bin,
                            pos_out=pout, batch_out=bout,
                            in_index=iidx, out_index=oidx),
                feature, pos, voxel_cluster, pos_sample, voxel_cluster_sample, in_index, out_index
            ))

        _mem("GridFeat:after conv2")

        self._assert_finite("GridFeat.after conv2", feature)
        pos = pos_sample
        batch = batch_sample

        with _nvtx("GridFeat.radius_graph_1"):
            edge_index = radius_graph(pos, 2 / 16, batch, loop=True, max_num_neighbors=self.grid_max_k)
        if self.ar:
            with _nvtx("GridFeat.ar2"):
                feature, _, _ = self.ar2(feature, pos, edge_index)
        if self.layer_norm:
            feature = self.norm2(feature, batch)

        with _nvtx("GridFeat.voxel_pool_2"):
            (out_index, in_index), pos, batch, pos_sample, batch_sample, voxel_cluster, voxel_cluster_sample, inv = \
                voxel_mean_pool(pos=pos, batch=batch, start=min_pos, end=max_pos, size=2 / 16)
            feature = feature[inv]

            num_pts = scatter_sum(
                torch.ones_like(out_index, dtype=torch.long),
                out_index, dim=0, dim_size=pos_sample.size(0)
            )
            assert (num_pts > 0).all(), "Empty cluster detected before next PointConv"
        _mem("GridFeat:after pool2")

        with _nvtx("GridFeat.conv3"):
            feature = F.celu(ckpt(
                lambda xin, pin, bin, pout, bout, iidx, oidx:
                    self.conv3(x_in=xin, pos_in=pin, batch_in=bin,
                            pos_out=pout, batch_out=bout,
                            in_index=iidx, out_index=oidx),
                feature, pos, voxel_cluster, pos_sample, voxel_cluster_sample, in_index, out_index
            ))

        _mem("GridFeat:after conv3")

        pos = pos_sample
        batch = batch_sample

        self._assert_finite("GridFeat.after conv3", feature)
        with _nvtx("GridFeat.radius_graph_2"):
            edge_index = radius_graph(pos, 4 / 16, batch, loop=True, max_num_neighbors=self.grid_max_k)

            deg = torch.bincount(edge_index[1], minlength=pos.size(0))
            if (deg == 0).any(): print("WARN: zero-degree nodes in AR graph")
        if self.ar:
            with _nvtx("GridFeat.ar3-5"):
                feature, _, _ = ckpt(lambda f,p,e: self.ar3(f, p, e), feature, pos, edge_index)
                self._assert_finite("GridFeat.after ar3", feature)
                feature, _, _ = ckpt(lambda f,p,e: self.ar4(f, p, e), feature, pos, edge_index)
                self._assert_finite("GridFeat.after ar4", feature)
                feature, _, _ = ckpt(lambda f,p,e: self.ar5(f, p, e), feature, pos, edge_index)
                self._assert_finite("GridFeat.after ar5", feature)


        self._assert_finite("GridFeat. before norm3", feature)
        if self.layer_norm:
            feature = torch.nan_to_num(feature, nan=0.0, posinf=1e6, neginf=-1e6)
            feature = feature.clamp_(-1e4, 1e4)  # tame outliers
            feature = self.norm3(feature, batch)  
            self._assert_finite("GridFeat.after norm3", feature)

        voxel_center = find_voxel_center(pos, start=min_pos, size=2 / 16)

        self._assert_finite("GridFeat.before_center_shift.feature", feature)
        self._assert_finite("GridFeat.before_center_shift.pos", pos)
        self._assert_finite("GridFeat.before_center_shift.voxel_center", voxel_center)

        with _nvtx("GridFeat.center_shift"):
            center_feature = self.conv4(feature, pos, voxel_center)

        with _nvtx("GridFeat.linear"):
            out = self.linear(center_feature)
        # Debug: find origin of non-finites
        if not self._assert_finite("GridFeat.center_feature", center_feature):
            center_feature = torch.nan_to_num(center_feature, nan=0.0, posinf=1e4, neginf=-1e4)
            out = self.linear(center_feature)

        if not self._assert_finite("GridFeat.linear.out", out):
            out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.glimpse_type == "ball":
            mu_pos, sigma_pos, mu_size_ratio, sigma_size_ratio, glimpse__logit_pres = \
                torch.split(out, [3, 3, 1, 1, 1], dim=1)
        else:
            mu_pos, sigma_pos, mu_size_ratio, sigma_size_ratio, glimpse__logit_pres = \
                torch.split(out, [3, 3, 3, 3, 1], dim=1)


        # Final guard before building distributions
        for tag, t in [
            ("mu_pos", mu_pos), ("sigma_pos", sigma_pos),
            ("mu_size_ratio", mu_size_ratio), ("sigma_size_ratio", sigma_size_ratio)
        ]:
            if not self._assert_finite(f"GridFeat.{tag}", t):
                # sanitize and keep going so we can see the rest of the step
                locals()[tag] = torch.nan_to_num(t, nan=0.0, posinf=1e4, neginf=-1e4)

        sigma_pos = to_sigma(sigma_pos).clamp_min(1e-6)
        sigma_size_ratio = to_sigma(sigma_size_ratio).clamp_min(1e-6)

        pos_post       = Normal(mu_pos,        sigma_pos)
        size_ratio_post= Normal(mu_size_ratio, sigma_size_ratio)
        pos_post = Normal(mu_pos, to_sigma(sigma_pos))
        size_ratio_post = Normal(mu_size_ratio, to_sigma(sigma_size_ratio))

        z_pos = pos_post.rsample()
        z_r   = size_ratio_post.rsample()

        pres_post = None
        log_z_pres = None
        glimpse__logit_pres = None

        glimpse__center_offset_ratio = torch.tanh(z_pos)
        glimpse__ball_radius_ratio   = torch.sigmoid(z_r)

        _mem("GridFeat:end")
        return (glimpse__center_offset_ratio, glimpse__ball_radius_ratio, log_z_pres, glimpse__logit_pres), \
               (pos_post, size_ratio_post, pres_post), voxel_center, feature, pos, batch_sample
    
    def _assert_finite(self,tag, *tensors):
        for t in tensors:
            if t is None: 
                continue
            if not torch.isfinite(t).all():
                bad = (~torch.isfinite(t)).nonzero(as_tuple=False)[:5]
                print(f"[NaN/Inf] {tag}: found {bad.shape[0]} bad values; first idx: {bad.flatten().tolist()[:10]}")
                return False
        return True

class SPAIRGlimpseEncoder(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer_norm = cfg.glimpse_encoder_ln
        self.ar = cfg.glimpse_encoder_ar

        # --- memory knobs (configurable) ---
        self.max_pts_per_glimpse = getattr(cfg, "max_pts_per_glimpse", 6000)
        self.rg_max_k            = getattr(cfg, "rg_max_neighbors_glimpse", 16)
        self.pc_max_k            = getattr(cfg, "pc_max_neighbors_glimpse", 16)

        if self.ar:
            self.ar1 = AutoRegistrationLayer(x_dim=3,    f_hidden=8,   f_out=8,   g_hidden=8,   g_out=8)
            self.ar2 = AutoRegistrationLayer(x_dim=32,   f_hidden=32,  f_out=32,  g_hidden=32,  g_out=32)
            self.ar3 = AutoRegistrationLayer(x_dim=128,  f_hidden=128, f_out=128, g_hidden=128, g_out=128)

        # use tighter neighbor caps for PointConv
        self.conv1 = PointConv(0.25, max_num_neighbors=self.pc_max_k, c_in=8 if self.ar else 1, c_mid=16,  c_out=32)
        self.conv2 = PointConv(0.5,  max_num_neighbors=self.pc_max_k, c_in=32,                         c_mid=64,  c_out=128)
        self.conv3 = PointConv(1.0,  max_num_neighbors=self.pc_max_k, c_in=128,                        c_mid=128, c_out=256)

        if self.layer_norm:
            self.norm1 = LayerNorm(16)
            self.norm2 = LayerNorm(64)
            self.norm3 = LayerNorm(128)
            self.norm4 = LayerNorm(256)

        self.linear = torch.nn.Linear(in_features=256, out_features=256)

    def forward(self, rgb, pos, glimpse_member__glimpse_index, glimpse__center, glimpse__batch):
        _reset_peak()
        _shapes("Enc.in", pos=pos, idx=glimpse_member__glimpse_index)
        _mem("Enc:start")

        # --- per-glimpse point cap BEFORE any radius_graph ---
        with _nvtx("Enc.downsample_glimpse"):
            gidx = glimpse_member__glimpse_index
            if self.max_pts_per_glimpse and gidx.numel():
                uniq, inv, counts = torch.unique(gidx, return_inverse=True, return_counts=True)

                # Build keep mask only for groups over the cap
                keep = torch.ones_like(inv, dtype=torch.bool)
                too_big = (counts > self.max_pts_per_glimpse).nonzero(as_tuple=False).flatten()

                # Early exit if nothing is oversized
                if too_big.numel() > 0:
                    for new_gid in too_big.tolist():
                        sel = (inv == new_gid).nonzero(as_tuple=False).flatten()
                        # random subset (deterministic: set a Generator with a seed if you want)
                        perm = torch.randperm(sel.numel(), device=pos.device)
                        drop = sel[perm[self.max_pts_per_glimpse:]]
                        keep[drop] = False

                    # filter tensors
                    pos = pos[keep]
                    gidx = gidx[keep]
                    if rgb is not None:
                        rgb = rgb[keep]
                    glimpse_member__glimpse_index = gidx

        # Now compute ranges for voxel pooling on the filtered set
        min_pos, _ = torch.min(pos, dim=0)
        max_pos, _ = torch.max(pos, dim=0)
        noise = torch.rand_like(min_pos)
        min_pos -= noise

        pos_list = [pos]
        glimpse_index_list = [glimpse_member__glimpse_index]
        in_out_index_list = []

        with _nvtx("Enc.rg_0"):
            edge_index = radius_graph(
                pos, 0.25, glimpse_member__glimpse_index, loop=True,
                max_num_neighbors=self.rg_max_k
            )

        if self.ar:
            with _nvtx("Enc.ar1"):
                feature, _, _ = ckpt(lambda x,p,e: self.ar1(x, p, e), pos, pos, edge_index)
            if self.layer_norm:
                feature = self.norm1(feature, glimpse_member__glimpse_index)

        else:
            feature = rgb

        with _nvtx("Enc.pool0"):
            (out_index, in_index), pos, glimpse_member__glimpse_index, pos_sample, glimpse_member_sample__glimpse_index, \
            voxel_cluster, voxel_cluster_sample, inv = voxel_mean_pool(
                pos=pos, batch=glimpse_member__glimpse_index, start=min_pos, end=max_pos, size=0.25)
        with _nvtx("Enc.conv1"):
            feature = F.celu(ckpt(
                lambda xin, pin, bin, pout, bout, iidx, oidx:
                    self.conv1(x_in=xin, pos_in=pin, batch_in=bin,
                            pos_out=pout, batch_out=bout,
                            in_index=iidx, out_index=oidx),
                feature, pos, voxel_cluster, pos_sample, voxel_cluster_sample, in_index, out_index
            ))

        _mem("Enc:after conv1")

        pos = pos_sample
        glimpse_member__glimpse_index = glimpse_member_sample__glimpse_index
        pos_list.append(pos)
        glimpse_index_list.append(glimpse_member__glimpse_index)
        in_out_index_list.append((in_index, out_index))

        with _nvtx("Enc.rg_1"):
            edge_index = radius_graph(
                pos, 0.5, glimpse_member__glimpse_index, loop=True,
                max_num_neighbors=self.rg_max_k
            )
        if self.ar:
            with _nvtx("Enc.ar2"):
                feature, _, _ = ckpt(lambda x,p,e: self.ar2(x, p, e), feature, pos, edge_index)
        if self.layer_norm:
            feature = self.norm2(feature, glimpse_member__glimpse_index)


        with _nvtx("Enc.pool1"):
            (out_index, in_index), pos, glimpse_member__glimpse_index, pos_sample, glimpse_member_sample__glimpse_index, \
            voxel_cluster, voxel_cluster_sample, inv = voxel_mean_pool(
                pos=pos, batch=glimpse_member__glimpse_index, start=min_pos, end=max_pos, size=0.5)
            feature = feature[inv]
        with _nvtx("Enc.conv2"):
            feature = F.celu(ckpt(
                lambda xin, pin, bin, pout, bout, iidx, oidx:
                    self.conv2(x_in=xin, pos_in=pin, batch_in=bin,
                            pos_out=pout, batch_out=bout,
                            in_index=iidx, out_index=oidx),
                feature, pos, voxel_cluster, pos_sample, voxel_cluster_sample, in_index, out_index
            ))

        _mem("Enc:after conv2")

        pos = pos_sample
        glimpse_member__glimpse_index = glimpse_member_sample__glimpse_index
        pos_list.append(pos)
        glimpse_index_list.append(glimpse_member__glimpse_index)
        in_out_index_list.append((in_index, out_index))

        with _nvtx("Enc.rg_2"):
            edge_index = radius_graph(
                pos, 1.0, glimpse_member__glimpse_index, loop=True,
                max_num_neighbors=self.rg_max_k
            )
        if self.ar:
            with _nvtx("Enc.ar3"):
                feature, _, _ = ckpt(lambda x,p,e: self.ar3(x, p, e), feature, pos, edge_index)
        if self.layer_norm:
            feature = self.norm3(feature, glimpse_member__glimpse_index)


        pos_sample = torch.zeros_like(glimpse__center)
        glimpse_member_sample__glimpse_index = torch.arange(glimpse__center.size(0), dtype=torch.long, device=pos.device)
        in_index = torch.arange(pos.size(0), dtype=torch.long, device=pos.device)
        out_index = glimpse_member__glimpse_index

        with _nvtx("Enc.conv3"):
            feature = F.celu(ckpt(
                lambda xin, pin, bin, pout, bout, iidx, oidx:
                    self.conv3(x_in=xin, pos_in=pin, batch_in=bin,
                            pos_out=pout, batch_out=bout,
                            in_index=iidx, out_index=out_index),
                feature, pos, glimpse_member__glimpse_index,
                pos_sample, glimpse_member_sample__glimpse_index,
                in_index, out_index
            ))

        _mem("Enc:after conv3")

        pos_list.append(pos_sample)
        glimpse_index_list.append(glimpse_member_sample__glimpse_index)
        in_out_index_list.append((in_index, out_index))

        with _nvtx("Enc.linear"):
            out = self.linear(feature)
        mu, sigma = torch.chunk(out, 2, dim=1)

        what_mask_post = Normal(mu, to_sigma(sigma))
        z_what_mask = what_mask_post.rsample()
        z_what, z_mask = torch.chunk(z_what_mask, 2, dim=1)

        _mem("Enc:end")
        return z_what, z_mask, what_mask_post, pos_list, glimpse_index_list, in_out_index_list, feature


class SPAIRGlimpseZPresGenerator(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.radius_max = cfg.max_radius
        self.layer_norm = cfg.glimpse_encoder_ln

        self.z_pres_linear = torch.nn.Linear(in_features = 8, out_features = 1)
        self.ar1 = AutoRegistrationLayer(x_dim = 256, f_hidden = 128, f_out = 64, g_hidden = 64, g_out = 64)
        self.ar2 = AutoRegistrationLayer(x_dim = 64,  f_hidden = 32,  f_out = 32, g_hidden = 32, g_out = 32)
        self.ar3 = AutoRegistrationLayer(x_dim = 32,  f_hidden = 16,  f_out = 16, g_hidden = 16, g_out = 8)

    def forward(self, glimpse__feature, glimpse__center, glimpse__batch,
                glimpse_member__local_pos, glimpse_member__log_mask,
                glimpse_member__glimpse_index, temperature):
        _mem("ZPres:start")

        glimpse_member__normalized_mask = torch.exp(
            scatter_log_softmax(glimpse_member__log_mask, index=glimpse_member__glimpse_index, dim=0)
        )
        glimpse_member__weighted_pos = glimpse_member__local_pos * glimpse_member__normalized_mask
        glimpse__member_center = scatter_sum(glimpse_member__weighted_pos, glimpse_member__glimpse_index, dim=0)

        glimpse__center_local_scale = glimpse__center / self.radius_max

        with _nvtx("ZPres.rg"):
            edge_index = radius_graph(glimpse__center_local_scale, 1, glimpse__batch, loop=True,  max_num_neighbors=getattr(self, "grid_max_k", 32))
        with _nvtx("ZPres.ar1-3"):
            z_pres_feature, _, _ = ckpt(lambda f,p,e: self.ar1(f, p, e),
                                        glimpse__feature, glimpse__center_local_scale, edge_index)
            z_pres_feature, _, _ = ckpt(lambda f,p,e: self.ar2(f, p, e),
                                        z_pres_feature, glimpse__center_local_scale, edge_index)
            z_pres_feature, _, _ = ckpt(lambda f,p,e: self.ar3(f, p, e),
                                        z_pres_feature, glimpse__center_local_scale, edge_index)


        glimpse__logit_pres = self.z_pres_linear(z_pres_feature).squeeze(1)
        glimpse__logit_pres = 8.8 * torch.tanh(glimpse__logit_pres)
        pres_post = Bernoulli(logits=glimpse__logit_pres)
        log_z_pres = F.logsigmoid(
            LogitRelaxedBernoulli(logits=glimpse__logit_pres, temperature=temperature).rsample()
        )
        _mem("ZPres:end")
        return log_z_pres, glimpse__logit_pres, pres_post, glimpse__member_center


class SPAIRGlimpseZPresMLP(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.z_pres_linear = torch.nn.Linear(in_features = 256, out_features = 1)

    def forward(self, glimpse__feature, glimpse_member__local_pos, glimpse_member__log_mask, glimpse_member__glimpse_index, temperature):
        _mem("ZPresMLP:start")
        glimpse_member__normalized_mask = torch.exp(
            scatter_log_softmax(glimpse_member__log_mask, index=glimpse_member__glimpse_index, dim=0)
        )
        glimpse_member__weighted_pos = glimpse_member__local_pos * glimpse_member__normalized_mask
        glimpse__member_center = scatter_sum(glimpse_member__weighted_pos, glimpse_member__glimpse_index, dim=0)

        glimpse__logit_pres = self.z_pres_linear(glimpse__feature).squeeze(1)
        glimpse__logit_pres = 8.8 * torch.tanh(glimpse__logit_pres)
        pres_post = Bernoulli(logits=glimpse__logit_pres)
        log_z_pres = F.logsigmoid(
            LogitRelaxedBernoulli(logits=glimpse__logit_pres, temperature=temperature).rsample()
        )
        _mem("ZPresMLP:end")
        return log_z_pres, glimpse__logit_pres, pres_post, glimpse__member_center


class SPAIRGlimpseMaskDecoder(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.conv1 = PointConv(1,   max_num_neighbors = 64, c_in = 64, c_mid = 64, c_out = 32)
        self.conv2 = PointConv(0.5, max_num_neighbors = 64, c_in = 32, c_mid = 16, c_out = 16)
        self.conv3 = PointConv(0.25,max_num_neighbors = 64, c_in = 16, c_mid = 8,  c_out = 8)
        self.linear = torch.nn.Linear(in_features = 8, out_features = 1)

    def forward(self, z_mask, pos_list, glimpse_index_list, in_out_index_list):
        _mem("MaskDec:start")
        (in_index, out_index) = in_out_index_list[-1]
        with _nvtx("MaskDec.conv1"):
            out = F.celu(ckpt(
                lambda xin, pin, bin, pout, bout, iidx, oidx:
                    self.conv1(x_in=xin, pos_in=pin, batch_in=bin,
                            pos_out=pout, batch_out=bout,
                            in_index=iidx, out_index=oidx),
                z_mask, pos_list[-1], glimpse_index_list[-1],
                pos_list[-2], glimpse_index_list[-2],
                out_index, in_index
            ))

        _mem("MaskDec:after conv1")

        (in_index, out_index) = in_out_index_list[-2]
        with _nvtx("MaskDec.conv2"):
            out = F.celu(ckpt(
                lambda xin, pin, bin, pout, bout, iidx, oidx:
                    self.conv2(x_in=xin, pos_in=pin, batch_in=bin,
                            pos_out=pout, batch_out=bout,
                            in_index=iidx, out_index=oidx),
                out, pos_list[-2], glimpse_index_list[-2],
                pos_list[-3], glimpse_index_list[-3],
                out_index, in_index
            ))

        _mem("MaskDec:after conv2")

        (in_index, out_index) = in_out_index_list[-3]
        with _nvtx("MaskDec.conv3"):
            out = F.celu(ckpt(
                lambda xin, pin, bin, pout, bout, iidx, oidx:
                    self.conv3(x_in=xin, pos_in=pin, batch_in=bin,
                            pos_out=pout, batch_out=bout,
                            in_index=iidx, out_index=oidx),
                out, pos_list[-3], glimpse_index_list[-3],
                pos_list[-4], glimpse_index_list[-4],
                out_index, in_index
            ))

        with _nvtx("MaskDec.linear"):
            out = self.linear(out)
        out = F.logsigmoid(out)
        _mem("MaskDec:end")
        return out


class SPAIRGlimpseRGBDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = PointConv(1,   max_num_neighbors = 64, c_in = 128, c_mid = 128, c_out = 64)
        self.conv2 = PointConv(0.5, max_num_neighbors = 64, c_in = 64,  c_mid = 32,  c_out = 32)
        self.conv3 = PointConv(0.25,max_num_neighbors = 64, c_in = 32,  c_mid = 16,  c_out = 16)
        self.linear = torch.nn.Linear(in_features = 16, out_features = 3)

    def forward(self, z_what, pos_list, glimpse_index_list):
        _mem("RGBDec:start")
        with _nvtx("RGBDec.conv1"):
            out = F.celu(self.conv1(z_what, pos_list[-1], glimpse_index_list[-2]))
        with _nvtx("RGBDec.conv2"):
            out = F.celu(self.conv2(out, pos_list[-2], glimpse_index_list[-3]))
        with _nvtx("RGBDec.conv3"):
            out = F.celu(self.conv3(out, pos_list[-3], glimpse_index_list[-4]))
        with _nvtx("RGBDec.linear"):
            out = self.linear(out)
        _mem("RGBDec:end")
        return out


class SPAIRPointPosDecoder(torch.nn.Module):
    def __init__(self, latent_size = 128, num_points = 1024):
        super().__init__()
        self.num_points = num_points
        self.fc1 = torch.nn.Linear(in_features = latent_size, out_features = 256)
        self.fc2 = torch.nn.Linear(in_features = 256, out_features = 512)
        self.fc3 = torch.nn.Linear(in_features = 512, out_features = 1024)
        self.fc4 = torch.nn.Linear(in_features = 1024, out_features = num_points * 3)

    def forward(self, z, glimpse_index, center_flag = True):
        _mem("PosDec:start")
        x = F.celu(self.fc1(z))
        x = F.celu(self.fc2(x))
        x = F.celu(self.fc3(x))
        x = self.fc4(x)
        x = x.view(x.shape[0], -1, 3)

        if center_flag:
            x_center = torch.mean(x, dim=1, keepdim=True)
            x = x - x_center

        glimpse_index = glimpse_index.unsqueeze(1).repeat(1, self.num_points)
        x = torch.cat(list(x), dim=0)
        pos_predict_glimpse_index = torch.cat(list(glimpse_index), dim=0)
        _mem("PosDec:end")
        return x, pos_predict_glimpse_index


class SPAIRPointPosFlow(torch.nn.Module):
    def __init__(self, latent_size = 128, layer_norm = False):
        super().__init__()
        self.ar1 = AutoRegistrationLayer(x_dim = 3 + latent_size, f_hidden = 128, f_out = 128, g_hidden = 128, g_out = 64 + 3, end_relu=False)
        self.ar2 = AutoRegistrationLayer(x_dim = 64,             f_hidden = 64,  f_out = 64,  g_hidden = 64,  g_out = 32 + 3, end_relu=False)
        self.ar3 = AutoRegistrationLayer(x_dim = 32,             f_hidden = 16,  f_out = 16,  g_hidden = 16,  g_out = 3,      end_relu=False)

        self.layer_norm = layer_norm
        if self.layer_norm:
            self.norm1 = LayerNorm(64)
            self.norm2 = LayerNorm(32)
        self.noise = None

    def forward(self, z, batch, center_flag = True, extra_predict_ratio = 0.25):
        _mem("PosFlow:start")
        if self.noise is None:
            self.noise = Normal(torch.tensor(0.0, device=z.device), torch.tensor(0.3, device=z.device))

        if extra_predict_ratio > 0:
            prob = torch.ones(batch.size(0), device=batch.device)
            sample = torch.multinomial(prob, int(batch.size(0) * extra_predict_ratio))
            batch = torch.cat((batch, batch[sample]), dim=0)

        z = z[batch]
        population = self.noise.sample((batch.size(0), 3))

        with _nvtx("PosFlow.rg1"):
            edge_index = radius_graph(population, 0.2, batch, loop = True, max_num_neighbors=64)

        feature = torch.cat((z, population), dim=1)
        with _nvtx("PosFlow.ar1"):
            feat_pop, _, _ = ckpt(lambda x,p,e: self.ar1(x, p, e), feature, population, edge_index)
        feature, population = torch.split(feat_pop, (64, 3), dim=1)
        if self.layer_norm: feature = self.norm1(feature)
        feature = F.celu(feature)

        with _nvtx("PosFlow.rg2"):
            edge_index = radius_graph(population, 0.1, batch, loop=True, max_num_neighbors=64)
        with _nvtx("PosFlow.ar2"):
            feat_pop, _, _ = ckpt(lambda x,p,e: self.ar2(x, p, e), feature, population, edge_index)
        feature, population = torch.split(feat_pop, (32, 3), dim=1)
        if self.layer_norm: feature = self.norm2(feature)
        feature = F.celu(feature)

        with _nvtx("PosFlow.rg3"):
            edge_index = radius_graph(population, 0.05, batch, loop=True, max_num_neighbors=64)
        with _nvtx("PosFlow.ar3"):
            population, _, _ = ckpt(lambda x,p,e: self.ar3(x, p, e), feature, population, edge_index)


        if center_flag:
            population_center = scatter_mean(population, batch, dim=0)
            population = population - population_center[batch]

        _mem("PosFlow:end")
        return population, batch


class SPAIRGlimpseVAE(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.no_ZPres_generator = False
        self.generate_z_pres = cfg.generate_z_pres

        self.encoder = SPAIRGlimpseEncoder(cfg)
        if self.no_ZPres_generator:
            self.z_pres_mlp = SPAIRGlimpseZPresMLP(cfg)
        else:
            self.z_pres_generator = SPAIRGlimpseZPresGenerator(cfg)
        self.mask_decoder = SPAIRGlimpseMaskDecoder(cfg)

        self.pos_decoder = SPAIRPointPosFlow(latent_size=64)

        self.extra_predict_ratio = cfg.extra_predict_ratio
        self.no_ZPres_generator = cfg.no_ZPres_generator
        self.flow_points_per_glimpse = getattr(cfg, "flow_points_per_glimpse", 64)
        self.flow_budget = getattr(cfg, "flow_budget", 8192) 
        self.flow_chunk_glimpses = getattr(cfg, "flow_chunk_glimpses", 64)

    def forward(self, rgb, glimpse_member__local_pos, glimpse_member__glimpse_index, glimpse__center, glimpse__batch, temperature):
        _reset_peak()
        _shapes("GlimpseVAE.in", local_pos=glimpse_member__local_pos, gidx=glimpse_member__glimpse_index, gcenter=glimpse__center)
        _mem("GlimpseVAE:start")

        with _nvtx("VAE.encoder"):
            (glimpse__z_what,
             glimpse__z_mask,
             glimpse__what_mask_post,
             pos_list,
             glimpse_index_list,
             in_out_index_list,
             glimpse__feature) = self.encoder(rgb, glimpse_member__local_pos, glimpse_member__glimpse_index, glimpse__center, glimpse__batch)
        _mem("VAE:after encoder")

        with _nvtx("VAE.mask_decoder"):
            glimpse_member__log_mask = self.mask_decoder(glimpse__z_mask, pos_list, glimpse_index_list, in_out_index_list)
        _mem("VAE:after mask_decoder")

        if self.no_ZPres_generator:
            with _nvtx("VAE.z_pres_mlp"):
                (glimpse__log_z_pres,
                 glimpse__logit_pres,
                 glimpse__pres_post,
                 glimpse__member_center) = self.z_pres_mlp(glimpse__feature,
                                                           glimpse_member__local_pos,
                                                           glimpse_member__log_mask,
                                                           glimpse_member__glimpse_index,
                                                           temperature)
        else:
            with _nvtx("VAE.z_pres_generator"):
                (glimpse__log_z_pres,
                 glimpse__logit_pres,
                 glimpse__pres_post,
                 glimpse__member_center) = self.z_pres_generator(glimpse__feature,
                                                                 glimpse__center,
                                                                 glimpse__batch,
                                                                 glimpse_member__local_pos,
                                                                 glimpse_member__log_mask,
                                                                 glimpse_member__glimpse_index,
                                                                 temperature)
        _mem("VAE:after z_pres")

        glimpse__center_diff = torch.norm(glimpse__member_center - glimpse__center, 2, dim=1)

        # Compact glimpse IDs
        old_ids = glimpse_member__glimpse_index.long()
        if torch.all(old_ids[1:] >= old_ids[:-1]):
            uniq_ids, new_ids = torch.unique_consecutive(old_ids, return_inverse=True)
        else:
            uniq_ids, new_ids = torch.unique(old_ids, return_inverse=True, sorted=True)
        K = uniq_ids.numel()
        ppg = max(4, min(self.flow_points_per_glimpse, self.flow_budget // max(1, K)))
        flow_batch = torch.arange(K, device=glimpse__z_what.device).repeat_interleave(ppg)

        with _nvtx("VAE.pos_decoder"):
            glimpse_predict__pos, glimpse_predict__glimpse_index = self.pos_decoder(
                glimpse__z_what, flow_batch, True, extra_predict_ratio=0.0
            )
        _mem("VAE:end")

        return (glimpse__z_what,
                glimpse__z_mask,
                glimpse__log_z_pres,
                glimpse__logit_pres,
                glimpse_member__log_mask,
                None,
                glimpse_predict__pos,
                glimpse_predict__glimpse_index,
                glimpse__what_mask_post,
                glimpse__pres_post,
                glimpse__center_diff)


class CoarseEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = PointConv(10, max_num_neighbors = 64, c_in = 256, c_mid = 256, c_out = 512)

    def forward(self, pos, feature, batch):
        _mem("CoarseEnc:start")
        pos_center = scatter_mean(pos, batch, dim=0)
        pos_center_batch = torch.arange(pos_center.size(0), dtype=torch.long, device=pos.device)
        _index = torch.arange(pos.size(0), dtype=torch.long, device=pos.device)
        assert torch.max(_index) == (feature.size(0) - 1), "CoarseEncoder assertion triggered"

        with _nvtx("CoarseEnc.conv1"):
            out = self.conv1(x_in=feature, pos_in=pos, batch_in=batch,
                             pos_out=pos_center, batch_out=pos_center_batch,
                             in_index=_index, out_index=batch)
        mu, sigma = torch.chunk(out, 2, dim=1)
        what_coarse_post = Normal(mu, to_sigma(sigma))
        z_what_coarse = what_coarse_post.rsample()
        _mem("CoarseEnc:end")
        return z_what_coarse, what_coarse_post, pos_center_batch


class CoarseVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CoarseEncoder()
        self.decoder = SPAIRPointPosFlow(latent_size=256)
        # Tunables (keep small and predictable)
        self.flow_points_per_center = 32   # try 16/32/64
        self.flow_chunk_centers     = 64   # chunk coarse centers to cap peak mem

    def forward(self, voxel__pos, voxel__feature, voxel__batch, _ignored_batch):
        _reset_peak()
        _mem("CoarseVAE:start")

        with _nvtx("CoarseVAE.encoder"):
            z_what_coarse, what_coarse_post, pos_center_batch = \
                self.encoder(voxel__pos, voxel__feature, voxel__batch)
            # z_what_coarse: [K, 256], one latent per coarse center

        # Build a compact flow plan: K centers × ppc samples per center
        K   = z_what_coarse.size(0)
        ppc = self.flow_points_per_center

        preds_pos, preds_idx = [], []

        with _nvtx("CoarseVAE.decoder"):
            for s in range(0, K, self.flow_chunk_centers):
                e = min(K, s + self.flow_chunk_centers)
                z_chunk = z_what_coarse[s:e]                # [(e-s), 256]
                # compact group ids [0..(e-s-1)], each repeated ppc times
                flow_batch = torch.arange(e - s, device=z_chunk.device).repeat_interleave(ppc)

                pos_c, idx_c = self.decoder(
                    z_chunk, flow_batch, center_flag=False, extra_predict_ratio=0.0
                )
                # Re-offset indices back to global center ids
                preds_pos.append(pos_c)
                preds_idx.append(idx_c + s)

        coarse_point_predict = torch.cat(preds_pos, dim=0)
        pos_predict_batch    = torch.cat(preds_idx, dim=0)

        _mem("CoarseVAE:end")
        return coarse_point_predict, pos_predict_batch, what_coarse_post

