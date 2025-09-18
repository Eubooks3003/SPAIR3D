import torch
import torch.nn.functional as F

from torch_geometric.nn import radius, knn_interpolate
from torch_geometric.utils import to_dense_batch

from torch_scatter import scatter_sum

import torch
import torch.nn.functional as F
from torch_geometric.nn import radius
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_sum

class PointConv(torch.nn.Module):
    def __init__(self, radius=0.25/16, max_num_neighbors=64,
                 c_in=1, c_mid=64, c_out=64, pos_dim=3):
        super().__init__()
        self.radius = radius
        self.max_num_neighbors = max_num_neighbors
        self.c_in  = c_in
        self.c_mid = c_mid

        # No biases keeps zeros -> zeros (nice for padded entries)
        self.mlp1 = torch.nn.Linear(pos_dim, 16, bias=False)
        self.mlp2 = torch.nn.Linear(16, c_mid, bias=False)
        self.mlp3 = torch.nn.Linear(c_mid * c_in, c_out)

    @torch.no_grad()
    def _compact_sort(self, out_index, in_index):
        # group by out_index (needed for to_dense_batch); keep stable order
        perm = torch.argsort(out_index, stable=True)
        out_index = out_index[perm]
        in_index  = in_index[perm]
        # remap out_index -> [0..num_groups-1] (dense, consecutive)
        uniq, new_out = torch.unique_consecutive(out_index, return_inverse=True)
        return new_out, in_index, uniq.numel()

    def forward(self, x_in, pos_in, batch_in, pos_out=None, batch_out=None,
                in_index=None, out_index=None):
        if pos_out is None:
            pos_out = pos_in
            batch_out = batch_in

        # Build neighbors if not provided
        if in_index is None or out_index is None:
            # (row, col) with row indexing pos_out, col indexing pos_in
            out_index, in_index = radius(
                x=pos_in, y=pos_out, r=self.radius,
                batch_x=batch_in, batch_y=batch_out,
                max_num_neighbors=self.max_num_neighbors
            )

        if out_index.numel() == 0:
            # No neighbors at all: return zeros for every pos_out
            return pos_out.new_zeros((pos_out.size(0), self.mlp3.out_features))

        vals, perm = torch.sort(out_index)         # works on old PyTorch too
        out_index = vals
        in_index  = in_index[perm]

        # Now compact the group ids so to_dense_batch is tight
        uniq, out_compact = torch.unique_consecutive(out_index, return_inverse=True)
        num_groups = int(uniq.numel())
        # Safe per-group degree (>=1)
        deg = scatter_sum(
            torch.ones_like(out_compact, dtype=torch.float32),
            out_compact, dim=0, dim_size=uniq.numel()
        ).clamp_min(1.0)

        # Local coordinates
        pos_i = pos_out[out_index]  # still index with original out_index
        pos_j = pos_in[in_index]
        pos_local = pos_j - pos_i
        pos_local = torch.nan_to_num(pos_local)

        # Dense grouping by compact ids
        pos_local_dense, mask = to_dense_batch(pos_local, out_compact, fill_value=0.0)

        if x_in is None:
            x_dense = mask.float().unsqueeze(-1)  # [B,K,1]
        else:
            x_neighbors = x_in[in_index]
            # normalize by neighbor count (per-edge weight = 1/deg[group])
            w = (1.0 / deg[out_compact]).unsqueeze(-1)
            x_neighbors = x_neighbors * w
            x_dense, _ = to_dense_batch(x_neighbors, out_compact, fill_value=0.0)

        # MLPs on geometry; zero-out padded slots explicitly
        M = self.mlp1(pos_local_dense)
        M = F.celu(M)
        M = self.mlp2(M)
        M = F.celu(M)
        M = M * mask.unsqueeze(-1)  # padded rows contribute nothing

        # Aggregate: [B,Cin,K] @ [B,K,Cmid] -> [B,Cin,Cmid]
        product = torch.bmm(x_dense.permute(0, 2, 1), M)
        product = product.reshape(product.size(0), -1)  # [B, Cin*Cmid]

        out = self.mlp3(product)  # [B, Cout]
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

        # We built B = num_groups rows, one per unique out point encountered.
        # If you need exactly pos_out.size(0) rows, scatter back (optional).
        if num_groups != pos_out.size(0):
            # Map compact groups back to original out ids we actually touched:
            uniq_out, _ = torch.unique_consecutive(out_index, return_inverse=False)
            full = pos_out.new_zeros((pos_out.size(0), out.size(1)))
            full[uniq_out] = out
            out = full

        return out


class PointDeconv(torch.nn.Module):

    def __init__(self, radius, max_num_neighbors, c_in = 1, c_mid = 64, c_out = 64, pos_dim = 3, k = 3):
        super().__init__()

        self.conv = PointConv(radius, max_num_neighbors, c_in, c_mid, c_out, pos_dim)
        self.k = k

    def forward(self, x_in, pos_in, batch_in, pos_out, batch_out):

        x_out = knn_interpolate(x_in, pos_in, pos_out, batch_in, batch_out, k = self.k)

        out = self.conv(x_out, pos_out, batch_out, pos_out, batch_out)

        return out

class CenterShift(torch.nn.Module):

    def __init__(self, c_in = 1, c_mid = 64, c_out = 64, pos_dim = 3):
        super().__init__()

        self.c_in = c_in
        self.c_out = c_out

        self.mlp11 = torch.nn.Linear(in_features = pos_dim, out_features = 16)
        self.mlp12 = torch.nn.Linear(in_features = 16, out_features = c_mid)
        self.mlp13 = torch.nn.Linear(in_features = c_mid, out_features = c_out * c_in)

    def forward(self, x, pos_i, pos_j):
        """
        move points from pos_i to pos_j
        x: N x c_in
        """
        
        # ! Debugging:
        assert pos_i.size(0) == pos_j.size(0)
        # ! Debugging:

        pos_local = pos_j - pos_i

        W = F.celu(self.mlp11(pos_local))
        W = F.celu(self.mlp12(W))
        W = self.mlp13(W)
        W = W.view(-1, self.c_in, self.c_out)

        out = torch.bmm(x.unsqueeze(1), W).squeeze() # N, c_out

        return out