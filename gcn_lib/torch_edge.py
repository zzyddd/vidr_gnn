# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import math
import torch
from torch import nn
import torch.nn.functional as F

def pairwise_distance(x):
    """
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    return torch.cdist(x, x, p=2).pow(2)

def pairwise_distance_FR(x, y):
    """
    Args:
        x: tensor (batch_size, num_points, num_dims)
        y: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: tensor (batch_size, num_points, num_points)
    """
    return torch.cdist(x, y, p=2).pow(2)

def part_pairwise_distance(x, start_idx=0, end_idx=1):
    """
    Args:
        x: tensor (batch_size, num_points, num_dims)
        start_idx/end_idx:
    Returns:
        distance: (batch_size, part_points, num_points)
    """
    x_part = x[:, start_idx:end_idx]
    return torch.cdist(x_part, x, p=2).pow(2)

def xy_pairwise_distance(x, y):
    """
    Args:
        x: tensor (batch_size, num_points, num_dims)
        y: tensor (batch_size, num_points, num_dims)
    Returns:
        distance: (batch_size, num_points, num_points)
    """
    return torch.cdist(x, y, p=2).pow(2)

def norm(t):
    return F.normalize(t, dim=1, eps=1e-10)

def helper(dist_x, dist_code, shift):
    min_val = 0.00001
    dist_code_clamped = dist_code.clamp(min=min_val)
    loss = - dist_code_clamped * (dist_x - shift)
    return loss, dist_code

def compute_distances(x, code, k=16):

    batch_size, num_points, _ = x.shape
    device = x.device

    dist_x_self = pairwise_distance_FR(x, x)  # (B, N, N)
    dist_code_self = pairwise_distance_FR(code, code)  # (B, N, N)

    mask_x = torch.eye(num_points, device=device).bool().unsqueeze(0)  # (1, N, N)
    mask_x = mask_x.expand(batch_size, -1, -1)  # (B, N, N)

    dist_min_x = dist_x_self.masked_fill(mask_x, float('inf'))
    closest_indices_x = dist_min_x.argmin(dim=-1)  # (B, N)

    dist_max_x = dist_x_self.masked_fill(mask_x, -float('inf'))
    farthest_indices_x = dist_max_x.argmax(dim=-1)  # (B, N)

    closest_indices_expanded = closest_indices_x.unsqueeze(1).expand(-1, num_points, -1)  # (B, N, N)
    farthest_indices_expanded = farthest_indices_x.unsqueeze(1).expand(-1, num_points, -1)

    dist_x_knn = torch.gather(dist_x_self, dim=2, index=closest_indices_expanded)  # (B, N, N)
    dist_x_rand = torch.gather(dist_x_self, dim=2, index=farthest_indices_expanded)

    dist_code_knn = torch.gather(dist_code_self, dim=2, index=closest_indices_expanded)
    dist_code_rand = torch.gather(dist_code_self, dim=2, index=farthest_indices_expanded)

    return dist_x_self, dist_code_self, dist_x_knn, dist_code_knn, dist_x_rand, dist_code_rand

def Discriminative_Feature_Reorganization(dist_x_self, dist_code_self, dist_x_knn, dist_code_knn, dist_x_rand,
                                          dist_code_rand):
    """(Discriminative Feature Reorganization Loss)"""
    # pos_intra_loss
    pos_intra_loss, pos_intra_cd = helper(dist_x_self, dist_code_self, 0.12)
    pos_intra_loss = 0.10 * pos_intra_loss
    pos_intra_loss = pos_intra_loss.mean()

    # pos_inter_loss
    pos_inter_loss, pos_inter_cd = helper(dist_x_knn, dist_code_knn, 0.20)
    pos_inter_loss = 1.00 * pos_inter_loss
    pos_inter_loss = pos_inter_loss.mean()

    # neg_inter_loss
    neg_inter_loss, neg_inter_cd = helper(dist_x_rand, dist_code_rand, 1.00)
    neg_inter_loss = 0.15 * neg_inter_loss
    neg_inter_loss = neg_inter_loss.mean()


    Discriminative_Reorganization_Loss = pos_intra_loss + pos_inter_loss + neg_inter_loss

    return Discriminative_Reorganization_Loss

###################
def adjacency_joint_optimization(F, R):
    """
    Args:
        F: [batch_size, num_points, d_f]
        R: [batch_size, num_points, d_r]

    Returns:
        fused: [batch_size, num_points, d_f + d_r]
    """
    assert F.dim() == R.dim() == 3, " [batch, nodes, features]"
    fused = torch.cat([F, R], dim=-1)  # [B, N, d_f+d_r]

    return fused
###################

def Graph_Structure_Refinement(x, y, code, k=16, relative_pos=None):
    """
    Refines the graph structure by first fusing the x and code features,
    and then computing the top-k most similar nodes based on the fused features.

    Args:
        x: tensor of shape (batch_size, num_points, num_dims_x)
        code: tensor of shape (batch_size, num_points, num_dims_code)
        k: int, number of top similar nodes to return

    Returns:
        nn_idx: tensor of shape (batch_size, num_points, k), indices of the top-k similar nodes
    """
    fused_features = adjacency_joint_optimization(x, code)
    if y == None:
        dist = pairwise_distance(fused_features.detach())
    else:
        num_dims_fused = int(x.size(-1) + code.size(-1))
        num_dims_x = int(x.size(-1))
        proj = nn.Linear(num_dims_fused, num_dims_x).to(x.device)
        fused_features_proj = proj(fused_features)
        dist = xy_pairwise_distance(fused_features_proj.detach(), y.detach())
    if relative_pos is not None:
        dist += relative_pos
    _, nn_idx = torch.topk(-dist, k=k, dim=-1)  # topk returns indices of top-k smallest values
    return nn_idx

def dense_knn_matrix(x, code, k=16, relative_pos=None):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points, k) (batch_size, num_points, k)
    """
    x = x.transpose(2, 1).squeeze(-1)
    y = None
    code = code.transpose(2, 1).squeeze(-1)
    batch_size, n_points, n_dims = x.shape
    n_part = 10000
    if n_points > n_part:
        nn_idx_list = []
        groups = math.ceil(n_points / n_part)
        for i in range(groups):
            start_idx = n_part * i
            end_idx = min(n_points, n_part * (i + 1))
            dist = part_pairwise_distance(x.detach(), start_idx, end_idx)
            if relative_pos is not None:
                dist += relative_pos[:, start_idx:end_idx]
            _, nn_idx_part = torch.topk(-dist, k=k)
            nn_idx_list += [nn_idx_part]
        nn_idx = torch.cat(nn_idx_list, dim=1)
    else:
        if batch_size != 1:
            dist_x_self, dist_code_self, dist_x_knn, dist_code_knn, dist_x_rand, dist_code_rand = compute_distances(x,
                                                                                                                    code,
                                                                                                                    k)

            Discriminative_Reorganization_Loss = Discriminative_Feature_Reorganization(
                dist_x_self, dist_code_self, dist_x_knn, dist_code_knn, dist_x_rand, dist_code_rand
            )

        else:
            dist = pairwise_distance(x.detach())
            Discriminative_Reorganization_Loss = torch.tensor([0]).cuda

        nn_idx = Graph_Structure_Refinement(x, y, code, k, relative_pos)
    center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0), Discriminative_Reorganization_Loss

def xy_dense_knn_matrix(x, y, code, k=16, relative_pos=None):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points, k) (batch_size, num_points, k)
    """
    Discriminative_Reorganization_Loss = 0
    x = x.transpose(2, 1).squeeze(-1)
    y = y.transpose(2, 1).squeeze(-1)
    code = code.transpose(2, 1).squeeze(-1)
    batch_size, n_points, n_dims = x.shape
    if batch_size != 1:
        dist_x_self, dist_code_self, dist_x_knn, dist_code_knn, dist_x_rand, dist_code_rand = compute_distances(x, code,
                                                                                                                k)
        Discriminative_Reorganization_Loss = Discriminative_Feature_Reorganization(
            dist_x_self, dist_code_self, dist_x_knn, dist_code_knn, dist_x_rand, dist_code_rand
        )

    with torch.no_grad():
        nn_idx = Graph_Structure_Refinement(x, y, code, k, relative_pos)
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0), Discriminative_Reorganization_Loss


class DenseDilated(nn.Module):
    """
    Find dilated neighbor from neighbor list
    edge_index: (2, batch_size, num_points, k)
    """

    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilated, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index):
        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index[:, :, :, randnum]
            else:
                edge_index = edge_index[:, :, :, ::self.dilation]
        else:
            edge_index = edge_index[:, :, :, ::self.dilation]
        return edge_index


class DenseDilatedKnnGraph(nn.Module):
    """
    Find the neighbors' indices based on dilated knn
    """

    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0, C=192):
        super(DenseDilatedKnnGraph, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
        self._dilated = DenseDilated(k, dilation, stochastic, epsilon)

    def forward(self, x, code, y=None, relative_pos=None):
        B, C, H, W = x.shape
        if y is not None:
            #### normalize
            x = F.normalize(x, p=2.0, dim=1)
            y = F.normalize(y, p=2.0, dim=1)
            ####
            edge_index, Discriminative_Reorganization_Loss = xy_dense_knn_matrix(x, y, code, self.k * self.dilation,
                                                                                 relative_pos)
        else:

            #### normalize
            code = F.normalize(code, p=2.0, dim=1)
            x = F.normalize(x, p=2.0, dim=1)
            ####
            edge_index, Discriminative_Reorganization_Loss = dense_knn_matrix(x, code, self.k * self.dilation,
                                                                              relative_pos)

        return self._dilated(edge_index), Discriminative_Reorganization_Loss
