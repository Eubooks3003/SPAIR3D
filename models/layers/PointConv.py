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
        # Defaults
        if pos_out is None:
            pos_out  = pos_in
            batch_out = batch_in

        # Neighborhoods (radius() may yield zero neighbors for some out nodes!)
        if in_index is None or out_index is None:
            out_index, in_index = radius(pos_in, pos_out, self.radius,
                                        batch_in, batch_out,
                                        max_num_neighbors=self.max_num_neighbors)

        # Edge features
        pos_local = pos_in[in_index] - pos_out[out_index]       # [E, 3]
        M = F.celu(self.mlp1(pos_local))                        # [E, 16]
        M = F.celu(self.mlp2(M))                                # [E, c_mid]

        # Node/edge signal
        if x_in is None:
            x_edge = torch.ones((in_index.numel(), self.c_in), device=pos_in.device, dtype=pos_in.dtype)
        else:
            x_edge = x_in[in_index]                              # [E, c_in]

        # Degree per out node (size N_out), clamped to avoid 0/0
        N_out = pos_out.size(0)
        deg = scatter_sum(
            torch.ones_like(out_index, dtype=pos_in.dtype),
            out_index, dim=0, dim_size=N_out
        ).clamp_min(1.0)                                        # [N_out]

        # Normalize messages by degree of their destination
        x_edge = x_edge / deg[out_index].unsqueeze(1)

        # Outer product per edge: (c_in x c_mid), then sum over neighbors
        #   E, c_in, c_mid  -> flatten to E, (c_in*c_mid), then scatter-sum to N_out
        outer = torch.einsum('ei,ej->eij', x_edge, M)           # [E, c_in, c_mid]
        agg   = scatter_sum(outer.reshape(-1, self.c_in*self.c_mid),
                            out_index, dim=0, dim_size=N_out)   # [N_out, c_in*c_mid]

        out = self.mlp3(agg)                                    # [N_out, c_out]

        # Final safety belt (shouldn’t trigger after the fixes above)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
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