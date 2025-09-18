# run_spair3d.py
import os, os.path as osp, json, warnings
warnings.filterwarnings("ignore", category=Warning)

import pretty_errors
import forge
from forge import flags
import forge.experiment_tools as fet
from forge.experiment_tools import fprint

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from models.utils import compute_performance, batch_statistic
from datasets.to_dataset import make_ply_dataset  # your PLY-based dataset factory

torch.set_printoptions(threshold=3000, linewidth=200)

# ---------- JSON helpers ----------
def _expand(v):
    return os.path.expanduser(os.path.expandvars(v)) if isinstance(v, str) else v

def load_json_config(path):
    with open(_expand(path), "r") as f:
        raw = json.load(f)
    def rec(x):
        if isinstance(x, dict):
            return {k: rec(v) for k, v in x.items()}
        if isinstance(x, list):
            return [rec(v) for v in x]
        return _expand(x)
    return rec(raw)  # dict

def apply_overrides(cfg, dct):
    for k, v in dct.items():
        setattr(cfg, k, v)

def merge_schedule_overrides(cfg):
    sched = getattr(cfg, "schedule", None)
    if isinstance(sched, dict):
        for k in ["tfb_update_every", "report_loss_every", "ckpt_freq",
                  "dash_plot_every", "run_validation_every"]:
            if k in sched: setattr(cfg, k, sched[k])

def merge_train_overrides(cfg):
    tr = getattr(cfg, "train", None)
    if isinstance(tr, dict):
        if "lr" in tr: cfg.lr = tr["lr"]
        if "iter" in tr: cfg.train_iter = tr["iter"]

# ---------- Flags (single-GPU) ----------
def main_flags():
    flags.DEFINE_string('config', None, 'Path to JSON config (required).')
    flags.DEFINE_boolean('resume', False, 'Resume if True.')
    flags.DEFINE_integer('report_loss_every', 100, 'Iters between loss prints.')
    flags.DEFINE_integer('run_validation_every', 5000, 'Iters between validation.')
    flags.DEFINE_integer('dash_plot_every', 100, 'Iters between dash viz.')
    flags.DEFINE_integer('tfb_update_every', 50, 'Iters between TB updates.')
    flags.DEFINE_integer('ckpt_freq', 5000, 'Iters between checkpoints.')
    flags.DEFINE_integer('num_test', 128, 'Num test iters.')
    flags.DEFINE_integer('train_iter', 100000, 'Total train iterations.')

    # safe defaults (overridable by JSON)
    flags.DEFINE_string('results_dir', osp.join(os.path.expanduser("~"), "checkpoints", "SPAIR3D"), 'Results root.')
    flags.DEFINE_string('run_name', 'run', 'Run folder name.')
    flags.DEFINE_string('data_config', 'datasets/unity_object_room.py', 'Data cfg file.')
    flags.DEFINE_string('model_config', 'models/SS3D.py', 'Model cfg file.')
    flags.DEFINE_integer('batch_size', 1, 'Batch size.')
    flags.DEFINE_float('grad_max_norm', 1.0, 'Grad clip norm.')

# ---- Helpers for batch canonicalization ----
def _pick(d: dict, names):
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return None

def _ensure_batch_from_lengths(lengths, device):
    pieces = [torch.full((int(L),), i, dtype=torch.long, device=device) for i, L in enumerate(lengths)]
    return torch.cat(pieces, dim=0) if pieces else torch.empty(0, dtype=torch.long, device=device)

def _ensure_batch_from_pos(pos, device):
    N = pos.size(0)
    return torch.zeros(N, dtype=torch.long, device=device)

def _canonicalize_pos_batch(pos, bidx, device, log_prefix=""):
    """
    Ensure pos is [N,3] float32 contiguous and batch is int64 [N] on same device.
    Accepts pos as [B,N,3], [N,3], [3,N], or flat [3N].
    Accepts batch as None, [N], or [B,N].
    """
    if not isinstance(pos, torch.Tensor):
        pos = torch.as_tensor(pos)
    pos = pos.to(device=device, dtype=torch.float32, non_blocking=True)

    if pos.dim() == 3 and pos.size(-1) == 3:
        B, N, _ = pos.shape
        pos = pos.reshape(B * N, 3).contiguous()
        if bidx is None:
            bidx = _ensure_batch_from_lengths([N] * B, device)
    elif pos.dim() == 2:
        if pos.size(1) == 3:
            pos = pos.contiguous()
        elif pos.size(0) == 3:
            pos = pos.t().contiguous()
        else:
            raise ValueError(f"{log_prefix}pos has unexpected 2D shape {tuple(pos.shape)}; expected [N,3] or [3,N].")
    elif pos.dim() == 1 and (pos.numel() % 3 == 0):
        pos = pos.view(-1, 3).contiguous()
    else:
        raise ValueError(f"{log_prefix}pos must be [N,3]/[3,N]/[B,N,3]/flat; got shape {tuple(pos.shape)}")

    N = pos.size(0)

    if bidx is None:
        bidx = _ensure_batch_from_pos(pos, device)
    elif isinstance(bidx, torch.Tensor):
        bidx = bidx.to(device=device, dtype=torch.long, non_blocking=True)
        if bidx.dim() == 2:
            bidx = bidx.reshape(-1)
    else:
        bidx = torch.as_tensor(bidx, dtype=torch.long, device=device)

    if bidx.numel() != N:
        raise ValueError(f"{log_prefix}batch length {bidx.numel()} != num points {N}")

    return pos, bidx

def param_groups_no_decay(model, weight_decay=1e-4):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or any(k in n.lower() for k in ['bias','norm','layernorm','bn','graphnorm']):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {'params': decay,    'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]

# ---- Batch extractor (dict, tuple/list, or PyG Batch-like) ----
def extract_batch(batch_data):
    """Return (pos, rgb, batch_idx, Id, layer). May create batch_idx if missing."""
    if isinstance(batch_data, dict):
        pos   = _pick(batch_data, ['pos', 'xyz', 'points'])
        rgb   = _pick(batch_data, ['rgb', 'color', 'colors'])
        bidx  = _pick(batch_data, ['batch', 'batch_idx', 'b'])
        Id    = _pick(batch_data, ['Id', 'instance', 'labels', 'instances'])
        layer = _pick(batch_data, ['layer', 'Layer'])
        if pos is None:
            raise KeyError(f"Batch dict missing 'pos' (available keys: {list(batch_data.keys())})")
        return pos, rgb, bidx, Id, layer

    if isinstance(batch_data, (list, tuple)):
        if len(batch_data) < 1:
            raise KeyError(f"Tuple/list batch too short; got len={len(batch_data)}")
        pos   = batch_data[0]
        rgb   = batch_data[1] if len(batch_data) > 1 else None
        bidx  = batch_data[2] if len(batch_data) > 2 else None
        Id    = batch_data[3] if len(batch_data) > 3 else None
        layer = batch_data[4] if len(batch_data) > 4 else None
        return pos, rgb, bidx, Id, layer

    if hasattr(batch_data, "pos"):
        pos   = getattr(batch_data, "pos")
        rgb   = getattr(batch_data, "rgb", None)
        bidx  = getattr(batch_data, "batch", None)
        Id    = getattr(batch_data, "y", None) or getattr(batch_data, "Id", None)
        layer = getattr(batch_data, "layer", None)
        return pos, rgb, bidx, Id, layer

    raise TypeError(f"Unsupported batch type: {type(batch_data)}")

def main():
    main_flags()
    config = forge.config()
    if not config.config:
        raise SystemExit("Please provide --config <path/to.json>")

    # Load JSON and apply; copy train/schedule into top-level for convenience
    user_cfg = load_json_config(config.config)  # dict
    apply_overrides(config, user_cfg)
    merge_schedule_overrides(config)
    merge_train_overrides(config)
    fprint(f"Loaded JSON config from {config.config} and applied overrides.")

    # Device: single GPU (or CPU fallback)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_master = True  # single-process run

    # ==== Checkpoints & logging ====
    run_root = osp.join(config.results_dir, config.run_name)
    logdir, resume_checkpoint = fet.init_checkpoint(
        run_root,
        config.data_config,
        config.model_config,
        ["models/submodels/spair3d_modules.py"],
        config.resume
    )
    merged_out = osp.join(logdir, "config.merged.json")
    try:
        with open(merged_out, "w") as f:
            json.dump(user_cfg, f, indent=2, sort_keys=True)
        fprint(f"Wrote merged config -> {merged_out}")
    except Exception as e:
        fprint(f"[warn] failed to write merged config: {e}")
    os.makedirs(osp.join(logdir, 'dash_data'), exist_ok=True)

    checkpoint_name = osp.join(logdir, 'model.ckpt')
    dash_logdir = osp.join(logdir, 'dash_data')

    # ==== Datasets / Loaders ====
    ds = getattr(config, "dataset", {})
    ds_root           = ds.get("root")
    ds_norm           = ds.get("normalize", "unit_sphere")
    ds_max_points     = ds.get("max_points", 300_000)
    ds_batch_size     = ds.get("batch_size", config.batch_size)
    ds_num_workers    = ds.get("num_workers", 4)
    ds_instance_field = ds.get("instance_field", None)
    if not ds_root:
        raise SystemExit("configs JSON missing dataset.root")

    train_ds = make_ply_dataset(ds_root, "train", ds_norm, ds_max_points, ds_instance_field)
    val_ds   = make_ply_dataset(ds_root, "val",   ds_norm, ds_max_points, ds_instance_field)

    train_loader = DataLoader(
        train_ds, batch_size=ds_batch_size, shuffle=True,
        num_workers=ds_num_workers, pin_memory=True, persistent_workers=(ds_num_workers > 0)
    )
    val_loader = DataLoader(
        val_ds, batch_size=ds_batch_size, shuffle=False,
        num_workers=ds_num_workers, pin_memory=True, persistent_workers=(ds_num_workers > 0)
    )

    # ==== Model & Optimizer ====
    model = fet.load(config.model_config, config).to(device)

    lr = getattr(config, 'lr', 1e-4)  # from train.lr if provided


    optimizer = torch.optim.Adam(param_groups_no_decay(model), lr=lr, amsgrad=True)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)

    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    iter_idx = 0
    writer = SummaryWriter(logdir) if is_master else None

    if resume_checkpoint is not None:
        fprint(f"Restoring checkpoint from {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimiser_state_dict'])
        iter_idx = checkpoint['iter_idx'] + 1

    model.iter = iter_idx
    if is_master:
        fprint(f"Starting training at iter = {iter_idx}")

    model.train()
    train_iters = getattr(config, "train_iter", getattr(config, "train", {}).get("iter", 100000))

    while iter_idx < train_iters:
        print("++++++++++Iteration ", iter_idx, "+++++++++++++++++")
        for batch_data in train_loader:
            if iter_idx >= train_iters:
                break

            # 1) Get raw fields (may be missing batch)
            pos_raw, _, batch_raw, Id, _ = extract_batch(batch_data)

            # 2) Canonicalize on-device: pos->[N,3] float32, batch->[N] int64
            pos, batch = _canonicalize_pos_batch(pos_raw, batch_raw, device, log_prefix="[train] ")

            optimizer.zero_grad()

            (NLL_forward, NLL_backward, DKL,
             glimpse__size,
             glimpse__batch,
             glimpse__center,
             glimpse__voxel_center,
             _,
             _,
             glimpse__log_z_pres,
             glimpse__logit_pres,
             glimpse__center_diff,
             glimpse_member__log_mask,
             glimpse_member__local_pos,
             glimpse_member__normalized_log_alpha,
             glimpse_member__batch,
             glimpse_member__glimpse_index,
             glimpse_member__point_index,
             glimpse_predict__local_pos,
             glimpse_predict__glimpse_index,
             bg_predict__pos,
             bg_predict__batch,
             bg_log_alpha,
             NLL_backward_fg,
             NLL_backward_bg,
             glimpse_chamfer_predict__local_pos,
             bg_chamfer_predict__pos) = model(pos, None, None, batch)

            # Dash / metrics visualization (rank 0 only)
            if is_master and (iter_idx % config.dash_plot_every) == 0:
                with torch.no_grad():
                    seg, ARI, sc, mSC, _, _ = compute_performance(
                        osp.join(logdir, 'dash_data'), iter_idx, getattr(config, "max_radius", 1.0/8.0),
                        Id.detach().to('cpu') if isinstance(Id, torch.Tensor) else None,
                        pos.detach(), batch.detach(),
                        glimpse__batch.detach(),
                        glimpse__center.detach(),
                        glimpse_member__normalized_log_alpha.detach(),
                        glimpse_member__batch.detach(),
                        glimpse_member__glimpse_index.detach(),
                        glimpse_member__point_index.detach(),
                        glimpse_predict__glimpse_index.detach(),
                        bg_log_alpha.detach(),
                        glimpse_chamfer_predict__local_pos.detach(),
                        bg_chamfer_predict__pos.detach(),
                        majority_vote_flag=False,
                        compute_MMD_CD=False,
                    )
                fprint(f"{iter_idx}: ARI={ARI:.4f}, SC={sc:.4f}, mSC={mSC:.4f}")

            NLL = NLL_forward + NLL_backward
            loss = NLL + DKL

            print("STARTING BACKWARD")
            scaler.scale(loss).backward()          # replaces loss.backward()
            # Unscale BEFORE clipping so clipping sees real grads
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                        max_norm=getattr(config, "grad_max_norm", 1.0))

            # Single step with the scaler
            scaler.step(optimizer)
            scaler.update()

            print("FINISHED UPDATE")

            # TensorBoard (rank 0 only)
            if is_master and (iter_idx % config.tfb_update_every) == 0:
                z_pres = batch_statistic(torch.exp(glimpse__log_z_pres), glimpse__batch)
                D_center = torch.mean(glimpse__center_diff)
                bg_alpha = torch.mean(torch.exp(bg_log_alpha))

                writer.add_scalar('loss', loss.item(), iter_idx)
                writer.add_scalar('NLL', NLL.item(), iter_idx)
                writer.add_scalar('NLL_forward', NLL_forward.item(), iter_idx)
                writer.add_scalar('NLL_backward', NLL_backward.item(), iter_idx)
                writer.add_scalar('DKL', DKL.item(), iter_idx)
                writer.add_scalar('z_pres', z_pres.item(), iter_idx)
                writer.add_scalar('fg_alpha', 1 - bg_alpha.item(), iter_idx)
                writer.add_scalar('NLL_backward_fg', NLL_backward_fg.item(), iter_idx)
                writer.add_scalar('NLL_backward_bg', NLL_backward_bg.item(), iter_idx)
                writer.add_scalar('D_center', D_center, iter_idx)

            # Console (rank 0 only)
            if is_master and (iter_idx % config.report_loss_every) == 0:
                z_pres = batch_statistic(torch.exp(glimpse__log_z_pres), glimpse__batch)
                D_center = torch.mean(glimpse__center_diff)
                bg_alpha = torch.mean(torch.exp(bg_log_alpha))
                fprint(
                    f"{iter_idx}: NLL={NLL.item():.4f},",
                    f" NLL_F={NLL_forward.item():.4f},",
                    f" NLL_B={NLL_backward.item():.4f},",
                    f" NLL_backward_fg={NLL_backward_fg.item():.4f}",
                    f" NLL_backward_bg={NLL_backward_bg.item():.4f}",
                    f" DKL={DKL.item():.4f},",
                    f" Loss={loss.item():.4f},",
                    f" fg_alpha={1 - bg_alpha.item():.4f},",
                    f" z_pres={z_pres:.4f},",
                    f" D_center={D_center.item():.4f}"
                )

            # Checkpoint (rank 0 only)
            if is_master and (iter_idx % config.ckpt_freq) == 0:
                ckpt_file = f'{checkpoint_name}-{iter_idx}'
                fprint(f"Saving model training checkpoint to: {ckpt_file}")
                torch.save(
                    {
                        'iter_idx': iter_idx,
                        'model_state_dict': model.state_dict(),
                        'optimiser_state_dict': optimizer.state_dict(),
                        'elbo': loss.item()
                    },
                    ckpt_file
                )
                writer.flush()

            iter_idx += 1

    if is_master and writer is not None:
        writer.close()

if __name__ == "__main__":
    main()
