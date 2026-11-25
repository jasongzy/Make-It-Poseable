import json
import math
import os
from argparse import Namespace
from dataclasses import dataclass
from functools import cached_property

import torch
import torch.amp
import torch.nn.functional as F
import trimesh
from pytorch3d.ops import knn_points
from pytorch3d.transforms import Rotate, random_rotations
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import util.lr_sched as lr_sched
import util.misc as misc
from model import MIP, Output
from util.dataset_mixamo_mip import PoseDataGen
from util.utils import (
    apply_transform,
    apply_transform_normals,
    compose_transform_matrix,
    fps,
    get_normalize_transl_scaling,
    indices_flatten,
    indices_unflatten,
    kl_loss,
    knn,
    lexsort_3d_torch,
    robust_dataloader,
)


@dataclass(frozen=False)
class GT:
    data: PoseDataGen
    global_transform_source: torch.Tensor
    global_transform_target: torch.Tensor
    device: torch.device
    model: MIP = None
    args: Namespace = None

    @cached_property
    def global_transform_source_matrix(self):
        """(B, Ps, 4, 4)"""
        return self.global_transform_source.to(self.device, non_blocking=True)

    @cached_property
    def global_transform_target_matrix(self):
        """(B, Pt, 4, 4)"""
        return self.global_transform_target.to(self.device, non_blocking=True)

    @cached_property
    def target_scaling(self):
        """(B, Pt, 3)"""
        rotation_scaling_part = self.global_transform_target[..., :3, :3]
        scaling = torch.norm(rotation_scaling_part, p=2, dim=-2)
        return scaling

    @cached_property
    def source_joints(self):
        """(B, Ps, K, 3)"""
        return apply_transform(self.data.source_joints, self.global_transform_source).to(self.device, non_blocking=True)

    @cached_property
    def source_pts(self):
        """(B, Ps, Ns, 3)"""
        return apply_transform(self.data.source_pts, self.global_transform_source).to(self.device, non_blocking=True)

    @cached_property
    def source_normals(self):
        """(B, Ps, Ns, 3)"""
        return apply_transform_normals(self.data.source_normals, self.global_transform_source).to(
            self.device, non_blocking=True
        )

    @cached_property
    def source_pts_normals(self):
        """(B, Ps, Ns, 6)"""
        return torch.cat((self.source_pts, self.source_normals), dim=-1)

    @cached_property
    def source_skeleton(self):
        """(B, Ps, Ns, 3)"""
        return apply_transform(self.data.source_skeleton, self.global_transform_source).to(
            self.device, non_blocking=True
        )

    @cached_property
    def source_bw(self):
        """(B, Ps, Ns, K)"""
        return self.data.source_bw.to(self.device, non_blocking=True)

    @cached_property
    def source_inputs(self):
        return self.source_pts_normals if self.args.ae_type in ("triposg", "hy3d2.1") else self.source_pts

    @cached_property
    def target_joints(self):
        """(B, Ps, K, 3)"""
        return apply_transform(self.data.target_joints, self.global_transform_target).to(self.device, non_blocking=True)

    @cached_property
    def target_queries(self):
        """(B, Ps, Nt, 3)"""
        target_queries = apply_transform(self.data.target_queries, self.global_transform_target).to(
            self.device, non_blocking=True
        )
        target_queries.requires_grad = True  # for SDF gradients
        return target_queries

    @cached_property
    def target_surface_mask(self):
        """(B, Ps, Nt) bool"""
        return self.data.target_surface_mask.to(self.device, non_blocking=True)

    @cached_property
    def target_surface_pts(self):
        """(B, Ps, Nt', 3)"""
        return self.target_queries[self.target_surface_mask].reshape(*self.data.target_surface_normals.shape)

    @cached_property
    def target_surface_normals(self):
        """(B, Ps, Nt', 3)"""
        return apply_transform_normals(self.data.target_surface_normals, self.global_transform_target).to(
            self.device, non_blocking=True
        )

    @cached_property
    def target_surface_pts_normals(self):
        """(B, Ps, Ns, 6)"""
        return torch.cat((self.target_surface_pts, self.target_surface_normals), dim=-1)

    @cached_property
    def target_pts(self):
        """(B, Ps, Ns, 3)"""
        return apply_transform(self.data.target_pts, self.global_transform_target).to(self.device, non_blocking=True)

    @cached_property
    def target_normals(self):
        """(B, Ps, Ns, 3)"""
        return apply_transform_normals(self.data.target_normals, self.global_transform_target).to(
            self.device, non_blocking=True
        )

    @cached_property
    def target_pts_normals(self):
        """(B, Ps, Ns, 6)"""
        return torch.cat((self.target_pts, self.target_normals), dim=-1)

    @cached_property
    def target_inputs(self):
        return self.target_pts_normals if self.args.ae_type in ("triposg", "hy3d2.1") else self.target_pts

    @cached_property
    def target_pts_orig(self):
        """(B, Ps, Ns, 3)"""
        return apply_transform(self.data.target_pts_orig, self.global_transform_target).to(
            self.device, non_blocking=True
        )

    @cached_property
    def target_normals_orig(self):
        """(B, Ps, Ns, 3)"""
        return apply_transform_normals(self.data.target_normals_orig, self.global_transform_target).to(
            self.device, non_blocking=True
        )

    @cached_property
    def target_pts_normals_orig(self):
        """(B, Ps, Ns, 6)"""
        return torch.cat((self.target_pts_orig, self.target_normals_orig), dim=-1)

    @cached_property
    def target_inputs_orig(self):
        return self.target_pts_normals_orig if self.args.ae_type in ("triposg", "hy3d2.1") else self.target_pts_orig

    @cached_property
    def target_skeleton(self):
        """(B, Ps, Ns, 3)"""
        return apply_transform(self.data.target_skeleton, self.global_transform_target).to(
            self.device, non_blocking=True
        )

    def get_target_added_points(self, fps_idx: torch.Tensor):
        """
        Returns:
            fps_idx_add: (B*Ps, num_latents_add) based on `target_inputs_orig`
            target_xyz_add: (B*Ps, num_latents_add, 3)
        """
        target_inputs = self.target_inputs_orig.contiguous()
        B, P, N, C = target_inputs.shape
        # target_inputs = target_inputs.clone()
        # target_inputs = target_inputs[:, :, torch.randperm(target_inputs.shape[-2], device=target_inputs.device)]
        # target_inputs.view(-1, C)[fps_idx] = self.target_inputs.reshape(-1, C)[fps_idx]

        target_xyz = target_inputs[..., :3].reshape(B * P, N, 3)
        fps_inputs = self.target_inputs.reshape(-1, C)[fps_idx]
        fps_xyz = fps_inputs[:, :3].reshape(B * P, -1, 3)

        num_latents = fps_xyz.shape[1]
        num_latents_add = self.args.num_latents_add
        assert 0 < num_latents_add < num_latents

        # Select `num_latents_add` points from `fps_xyz_orig` that are farthest from `fps_xyz`
        fps_xyz_orig, fps_idx_orig = fps(target_xyz, num_latents)
        fps_idx_orig = indices_unflatten(fps_idx_orig, N=N, B=B * P)  # (B*P, num_latents)
        dists = knn_points(fps_xyz_orig, fps_xyz, K=1).dists
        farthest_idx = torch.topk(dists.squeeze(-1), num_latents_add, dim=-1).indices  # (B*P, num_latents_add)
        fps_idx_add = torch.gather(fps_idx_orig, 1, farthest_idx)  # (B*P, num_latents_add)

        # Sort by x-y-z
        target_xyz_add = torch.gather(target_xyz, 1, fps_idx_add.unsqueeze(-1).repeat(1, 1, 3))
        add_sorted_idx = lexsort_3d_torch(target_xyz_add)

        # # Sort by sortkey
        # target_sortkey = self.data.target_sortkey_orig.reshape(-1, N).to(target_inputs.device)
        # # target_xyz_sorted = torch.gather(target_xyz, 1, torch.argsort(target_sortkey, -1).unsqueeze(-1).repeat(1, 1, 3))  # fmt:skip
        # # import matplotlib.pyplot as plt  # fmt:skip
        # # colors = plt.get_cmap("plasma")(torch.linspace(0, 1, target_xyz_sorted.shape[1]).cpu().numpy())[:, :3]
        # # trimesh.PointCloud(target_xyz_sorted[0].cpu().numpy(), colors=colors).export("test.glb")
        # target_sortkey_add = torch.gather(target_sortkey, 1, fps_idx_add)
        # add_sorted_idx = torch.argsort(target_sortkey_add, -1)

        fps_idx_add = torch.gather(fps_idx_add, 1, add_sorted_idx)
        # target_xyz_add = target_xyz.reshape(-1, 3)[indices_flatten(fps_idx_add, N=N)].reshape(B * P, -1, 3)
        target_xyz_add = torch.gather(target_xyz, 1, fps_idx_add.unsqueeze(-1).repeat(1, 1, 3))
        # import matplotlib.pyplot as plt  # fmt:skip
        # colors = plt.get_cmap("plasma")(torch.linspace(0, 1, target_xyz_add.shape[1]).cpu().numpy())[:, :3]
        # trimesh.PointCloud(target_xyz_add[0].cpu().numpy(), colors=colors).export("test.glb")

        return fps_idx_add, target_xyz_add

    def target_latents(self, fps_idx: torch.Tensor):
        """(B, Ps, num_latents, D)"""
        assert self.model is not None
        assert fps_idx is not None

        target_xyz_add = None

        if self.args.remesh_type != "none" and self.args.use_unremeshed_target:
            target_inputs = self.target_inputs_orig.contiguous()
            B, P, N, C = target_inputs.shape
            # target_inputs = target_inputs.clone()
            # target_inputs = target_inputs[:, :, torch.randperm(target_inputs.shape[-2], device=target_inputs.device)]
            # target_inputs.view(-1, C)[fps_idx] = self.target_inputs.reshape(-1, C)[fps_idx]
            fps_inputs = self.target_inputs.reshape(-1, C)[fps_idx]
            target_xyz = target_inputs[..., :3].reshape(B * P, N, 3)
            fps_xyz = fps_inputs[:, :3].reshape(B * P, -1, 3)
            nearest_idx = knn(target_xyz, fps_xyz, K=1).squeeze(-1)
            nearest_idx_ = indices_flatten(nearest_idx, N=N)
            if self.args.remesh_type == "project":
                target_inputs.view(-1, C)[nearest_idx_] = fps_inputs
            if self.args.num_latents_add > 0:
                fps_idx_add, target_xyz_add = self.get_target_added_points(fps_idx)
                target_xyz_add = target_xyz_add.reshape(B, P, self.args.num_latents_add, 3)
                nearest_idx_ = indices_flatten(torch.cat([nearest_idx, fps_idx_add], dim=-1), N=N)
            fps_idx = nearest_idx_
            # fps_xyz = target_xyz.view(-1, 3)[fps_idx].reshape(B * P, -1, 3)
        else:
            target_inputs = self.target_inputs

        with torch.no_grad():
            # Do not call self.model.pretrained_ae() here, because it changes the behavior of the "mean" encoding type.
            target_latents, _ = self.model.eval().encode(target_inputs, fps_idx)
            # if self.args.encoding_type == "mean_var":
            #     mean, logvar = target_latents.split([64, 64], dim=-1)
            #     from models_ae import DiagonalGaussianDistribution
            #     posterior = DiagonalGaussianDistribution(mean, logvar)
            #     target_latents = posterior.sample()
        return target_latents, target_xyz_add

    def get_target_add_region(self, fps_idx: torch.Tensor):
        target_inputs = self.target_inputs_orig.contiguous()
        B, P, N, C = target_inputs.shape

        target_xyz = target_inputs[..., :3].reshape(B * P, N, 3)
        fps_inputs = self.target_inputs.reshape(-1, C)[fps_idx]
        fps_xyz = fps_inputs[:, :3].reshape(B * P, -1, 3)

        num_latents = fps_xyz.shape[1]
        num_latents_add = self.args.num_latents_add
        assert 0 < num_latents_add < num_latents

        # Select `num_latents_add` points from `fps_xyz_orig` that are farthest from `fps_xyz`
        fps_xyz_orig, fps_idx_orig = fps(target_xyz, num_latents)
        fps_idx_orig = indices_unflatten(fps_idx_orig, N=N, B=B * P)  # (B*P, num_latents)
        dists = knn_points(fps_xyz_orig, fps_xyz, K=1).dists
        farthest_idx = torch.topk(dists.squeeze(-1), num_latents_add, dim=-1).indices  # (B*P, num_latents_add)
        fps_idx_add = torch.gather(fps_idx_orig, 1, farthest_idx)  # (B*P, num_latents_add)

        target_xyz_add = torch.gather(target_xyz, 1, fps_idx_add.unsqueeze(-1).repeat(1, 1, 3))

        return target_xyz_add.reshape(B, P, self.args.num_latents_add, 3)

    @cached_property
    def target_occ(self):
        """(B, Pt, Nt', 1)"""
        if self.args.use_encoding_gt:
            assert self.model is not None
            if self.args.remesh_type != "none" and self.args.use_unremeshed_target:
                target_inputs = self.target_inputs_orig
            else:
                target_inputs = self.target_inputs
            with torch.no_grad():
                with self.model.pretrained_ae():
                    occ = self.model.eval().forward(target_inputs, self.target_queries, skip_transformer=True).occ
            occ = F.sigmoid(occ)
        else:
            occ = self.data.target_occ.float()
        return occ.to(self.device, non_blocking=True)

    @cached_property
    def target_sdf(self):
        """(B, Pt, Nt', 1)"""
        if self.args.use_encoding_gt:
            assert self.model is not None
            target_inputs = self.target_inputs if self.args.remesh_type == "none" else self.target_inputs_orig
            with torch.no_grad():
                with self.model.pretrained_ae():
                    sdf = self.model.eval().forward(target_inputs, self.target_queries, skip_transformer=True).sdf
            if self.model.ae_type == "triposg":
                sdf = torch.sigmoid(sdf) * 2 - 1
        else:
            sdf = self.data.target_sdf
            # assert torch.allclose(self.target_scaling[..., 0], self.target_scaling[..., 1])
            # assert torch.allclose(self.target_scaling[..., 1], self.target_scaling[..., 2])
            sdf = sdf * self.target_scaling[..., 0:1].unsqueeze(-1)

        # from util.dataset_mixamo_mip import get_sdf
        # import numpy as np

        # B, Pt, Nt = self.data.target_queries.shape[:3]
        # global_transform_target = self.global_transform_target.numpy()
        # target_queries = apply_transform(self.data.target_queries, self.global_transform_target).numpy()
        # sdf_ = np.empty((B, Pt, Nt, 1), dtype=np.float32)
        # for b in range(B):
        #     for p in range(Pt):
        #         mesh = self.data.meta[b].meshes[p].copy()
        #         mesh.vertices = apply_transform(mesh.vertices, global_transform_target[b, p])
        #         sdf_[b, p] = get_sdf(mesh, target_queries[b, p])[..., None]
        # sdf_ = torch.from_numpy(sdf_)
        return sdf.to(self.device, non_blocking=True)


def get_sdf_gradients(sdf_output: torch.Tensor, points_input: torch.Tensor):
    """
    Computes the gradients of the SDF output with respect to the input points.
    Args:
        sdf_output: Output tensor from the SDF model. Shape (..., N, 1).
        points_input: Input points tensor that requires grad. Shape (..., N, 3).
    Returns:
        Gradients of sdf_output w.r.t. points_input. Shape (..., N, 3).
    """
    assert points_input.requires_grad and sdf_output.requires_grad
    gradients = torch.autograd.grad(
        outputs=sdf_output,
        inputs=[points_input],
        grad_outputs=torch.ones_like(sdf_output),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradients


def get_eikonal_loss(sdf_gradients: torch.Tensor):
    """
    SDF is a solution of the eikonal equation,
        E_x|∇f(x)| = 1, x ~ P(D). Where f(x) is the SDF and P(D) is some distribution in R^3
    Eikonal loss is a loss that enforces the gradient to be a unit vector.
    Args:
        sdf_gradients: Gradients of the SDF w.r.t. 3D points. Shape (..., N, 3).
    """
    assert sdf_gradients.shape[-1] == 3
    norm_gradients = torch.linalg.norm(sdf_gradients, dim=-1)  # (..., N)
    loss = F.mse_loss(norm_gradients, torch.ones_like(norm_gradients))
    return loss


def get_surface_normal_loss(
    all_sdf_gradients: torch.Tensor, surface_mask: torch.Tensor, gt_surface_normals: torch.Tensor
):
    """
    Computes the surface normal loss for surface points: (1 - <predicted_normal, gt_normal>).
    Args:
        all_sdf_gradients: SDF gradients for all input points. Shape (..., N, 3).
        surface_mask: Boolean mask indicating surface points. Shape (..., N).
        gt_surface_normals: Ground truth normals for actual surface points. Shape (..., S, 3), where S is the number of surface points per batch item.
    """
    # Select gradients corresponding to surface points using the mask.
    # This will directly yield a flat tensor of shape (TotalTrueInMask, 3).
    surface_gradients = all_sdf_gradients[surface_mask]
    assert surface_gradients.numel() > 0
    gt_surface_normals_flat = gt_surface_normals.reshape(-1, 3)
    assert surface_gradients.shape[0] == gt_surface_normals_flat.shape[0]

    norm_predicted_sdf_gradients = torch.linalg.norm(surface_gradients, dim=-1, keepdim=True)
    pred_normals = surface_gradients / (norm_predicted_sdf_gradients + 1e-8)
    cos_sim = F.cosine_similarity(pred_normals, gt_surface_normals_flat, dim=-1)
    loss_per_surface_point = 1.0 - cos_sim
    return loss_per_surface_point.mean()


def align_added_latents(
    pred_latents: torch.Tensor,
    gt_latents: torch.Tensor,
    pred_xyz: torch.Tensor,
    gt_xyz: torch.Tensor,
    num_latents_add: int,
):
    """
    Align newly added latents using Hungarian algorithm (bipartite matching) based on spatial distance of 3D coordinates.
    Args:
        pred_latents: (B, P, M_total, D)
        gt_latents: (B, P, M_total, D)
        pred_xyz: (B, P, N_add, 3)
        gt_xyz: (B, P, N_add, 3)
        num_latents_add: int (N_add)
    Returns:
        gt_latents_aligned: (B, P, M_total, D) reordered GT latents
        gt_xyz_aligned: (B, P, N_add, 3) reordered GT coordinates
    """
    import numpy as np
    import scipy.optimize

    B, P, M_total, D = pred_latents.shape
    M_orig = M_total - num_latents_add

    gt_latents_orig = gt_latents[:, :, :M_orig, :]
    gt_latents_add = gt_latents[:, :, M_orig:, :]

    # Compute pairwise distance matrix
    pred_xyz_flat = pred_xyz.view(B * P, num_latents_add, 3)
    gt_xyz_flat = gt_xyz.view(B * P, num_latents_add, 3)
    cost_matrix = torch.cdist(pred_xyz_flat, gt_xyz_flat, p=2.0)  # (B*P, N_add, N_add)
    cost_matrix_np = cost_matrix.detach().cpu().numpy()

    # Solve assignment per batch element (linear_sum_assignment is CPU-only)
    aligned_indices_np = np.empty((B * P, num_latents_add), dtype=np.int64)
    for i in range(B * P):
        _, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix_np[i])
        aligned_indices_np[i] = col_ind

    aligned_indices = torch.from_numpy(aligned_indices_np).to(device=pred_latents.device)
    aligned_indices = aligned_indices.view(B, P, num_latents_add)

    # Reorder gt_latents_add and gt_xyz using gathered indices
    aligned_indices_gt = aligned_indices.unsqueeze(-1).expand(B, P, num_latents_add, D)
    gt_latents_add_aligned = torch.gather(gt_latents_add, 2, aligned_indices_gt)

    aligned_indices_xyz = aligned_indices.unsqueeze(-1).expand(B, P, num_latents_add, 3)
    gt_xyz_aligned = torch.gather(gt_xyz, 2, aligned_indices_xyz)

    gt_latents_aligned = torch.cat([gt_latents_orig, gt_latents_add_aligned], dim=2)

    return gt_latents_aligned, gt_xyz_aligned


def chamfer_loss(pred_xyz: torch.Tensor, target_xyz: torch.Tensor):
    from pytorch3d.loss import chamfer_distance

    B, P = pred_xyz.shape[:2]
    pred_flat = pred_xyz.view(B * P, -1, 3)
    target_flat = target_xyz.view(B * P, -1, 3)
    loss, _ = chamfer_distance(pred_flat, target_flat)
    return loss


def get_sdf_weights_for_added_points(queries_xyz: torch.Tensor, target_queries: torch.Tensor, radius: float = 0.1):
    B, P, Nt, _ = target_queries.shape
    pred_flat = queries_xyz.view(B * P, -1, 3)
    queries_flat = target_queries.view(B * P, Nt, 3)

    # Nearest distance from each target_query to all predicted anchor points `queries_xyz`
    dists = knn_points(queries_flat, pred_flat, K=1).dists.squeeze(-1)

    mask = dists < (radius**2)
    # Gaussian decay: weight approaches 1 when distance is close, and gradually approaches 0 beyond radius
    sigma = radius / 2.0
    weights = torch.exp(-dists / (2 * sigma**2))

    return mask.view(B, P, Nt, 1), weights.view(B, P, Nt, 1)


def get_loss(output: Output, gt: GT, args: Namespace, eval=False):
    occ = output.occ
    sdf = output.sdf

    loss = 0.0
    loss_value_dict: dict[str, float] = {}
    vis_data = {}

    if args.use_latent_loss:
        target_latents, target_xyz_pc = gt.target_latents(output.fps_idx_target)

        if args.num_latents_add > 0:
            assert target_xyz_pc is not None
            assert output.xyz_add is not None
            assert output.xyz_add.shape == target_xyz_pc.shape
            target_latents, target_xyz_add_aligned = align_added_latents(
                output.latents, target_latents, output.xyz_add, target_xyz_pc, args.num_latents_add
            )
            # xyz coordinate penalty to help quickly find spatial anchors for added latents
            loss_xyz = F.mse_loss(output.xyz_add, target_xyz_add_aligned)
            loss += loss_xyz * 1.0
            loss_value_dict["loss/xyz_add"] = loss_xyz.item()

        if args.encoding_type == "mean_var" and False:
            loss_latent = 0.01 * kl_loss(*output.latents.chunk(2, dim=-1), *target_latents.chunk(2, dim=-1))
        else:
            loss_latent = F.smooth_l1_loss(output.latents, target_latents)
        loss += loss_latent
        loss_value_dict["loss/tokens"] = loss_latent.item()

    # if True:
    if (not args.use_latent_loss) or eval:
        if occ is not None and sdf is None:
            if eval:
                loss_occ = F.binary_cross_entropy_with_logits(occ, gt.target_occ)
            else:
                # mean_pos = torch.mean((gt.target_occ == 1).float())
                # pos_weight = torch.tensor([(1.0 - mean_pos) / (mean_pos + 1e-6)], device=occ.device)
                loss_occ = F.binary_cross_entropy_with_logits(occ, gt.target_occ, pos_weight=None)
                # loss_occ = F.smooth_l1_loss(F.sigmoid(occ), gt.target_occ)
            loss += loss_occ
            loss_value_dict["loss/occ"] = loss_occ.item()
            # scn = trimesh.Scene()
            # scn.add_geometry(trimesh.PointCloud(gt.target_surface_pts[-1, 0].cpu().numpy()))
            # # scn.add_geometry(trimesh.PointCloud(gt.target_queries[-1, 0][F.sigmoid(occ[-1,0,:,0]) >= 0.5].cpu().numpy()))
            # # scn.add_geometry(trimesh.PointCloud(gt.target_queries[-1, 0][F.sigmoid(occ[-1,0,:,0]) < 0.5].cpu().numpy()))
            # scn.add_geometry(trimesh.PointCloud(gt.target_queries[-1, 0][gt.target_occ[-1,0,:,0] == 1].cpu().numpy()))
            # scn.add_geometry(trimesh.PointCloud(gt.target_queries[-1, 0][gt.target_occ[-1,0,:,0] == 0].cpu().numpy()))
            # scn.export("test.glb")

        if sdf is not None:
            if args.ae_type == "triposg":
                sdf = torch.sigmoid(sdf) * 2 - 1

            sdf_mask = None
            if args.num_latents_add > 0:
                assert output.xyz_add is not None
                target_xyz_pc = gt.get_target_add_region(output.fps_idx_target)
                loss_xyz_add_chamfer = chamfer_loss(output.xyz_add, target_xyz_pc)
                loss += loss_xyz_add_chamfer * 1.0
                loss_value_dict["loss/xyz_add_chamfer"] = loss_xyz_add_chamfer.item()
                sdf_mask, sdf_weights = get_sdf_weights_for_added_points(output.xyz_add, gt.target_queries, radius=0.1)

            # loss_sdf = F.smooth_l1_loss(sdf, gt.target_sdf)
            if sdf_mask is not None and not eval:
                # if sdf_mask.sum() > 0:
                #     loss_sdf = F.smooth_l1_loss(sdf[sdf_mask], gt.target_sdf[sdf_mask])
                # else:
                #     loss_sdf = (sdf * 0.0).sum()
                loss_sdf = (
                    F.smooth_l1_loss(sdf, gt.target_sdf, reduction="none") * sdf_weights
                ).sum() / sdf_weights.sum().clamp(min=1e-6)
            else:
                loss_sdf = F.smooth_l1_loss(sdf, gt.target_sdf)

            # from matplotlib import pyplot as plt
            # plt.close()
            # plt.plot((sdf[gt.target_sdf.abs() <= 0.01]).flatten().detach().cpu().numpy(), gt.target_sdf[gt.target_sdf.abs() <= 0.01].flatten().cpu().numpy(), "o")
            # plt.savefig("test.jpg")

            loss += loss_sdf
            loss_value_dict["loss/sdf"] = loss_sdf.item()

            if not eval and (args.use_sdf_eikonal_loss or args.use_sdf_normal_loss):
                sdf_grad = get_sdf_gradients(sdf, gt.target_queries)

                if args.use_sdf_eikonal_loss:
                    loss_eikonal = get_eikonal_loss(sdf_grad)
                    loss += loss_eikonal * 0.01
                    loss_value_dict["loss/sdf_eikonal"] = loss_eikonal.item()

                if args.use_sdf_normal_loss:
                    loss_surface_normal = get_surface_normal_loss(
                        sdf_grad, gt.target_surface_mask, gt.target_surface_normals
                    )
                    loss += loss_surface_normal * 0.1
                    loss_value_dict["loss/sdf_surface_normal"] = loss_surface_normal.item()

    assert isinstance(loss, torch.Tensor), "No loss"
    loss_value = loss.item()
    loss_value_dict["loss"] = loss_value

    if not math.isfinite(loss_value):
        raise RuntimeError(f"Loss is NaN: {loss_value_dict}")

    return loss, loss_value_dict, vis_data


def get_global_transforms(data: PoseDataGen, aug_rotation=False):
    if aug_rotation:
        rotate_source = Rotate(R=random_rotations(len(data))).get_matrix().unsqueeze(1).transpose(-1, -2)
    else:
        rotate_source = data.hips_transform_source
    transl_source, scaling_source = get_normalize_transl_scaling(
        apply_transform(data.source_pts, rotate_source), keep_ratio=True, recenter=aug_rotation
    )
    rotate_target = data.hips_transform_target
    transl_target, scaling_target = get_normalize_transl_scaling(
        apply_transform(data.target_queries, rotate_target), keep_ratio=True, recenter=False
    )

    # Align scaling
    scaling_all = torch.cat((scaling_source[..., 0, 0], scaling_target[..., 0, 0]), dim=-1)
    scaling_min = scaling_all.min(dim=-1, keepdim=True).values
    scaling_aligned = torch.eye(4, dtype=scaling_all.dtype, device=scaling_all.device)
    scaling_aligned = scaling_aligned.unsqueeze(0).repeat(*scaling_all.shape[:2], 1, 1)
    scaling_aligned[:, :, 0, 0] = scaling_min
    scaling_aligned[:, :, 1, 1] = scaling_min
    scaling_aligned[:, :, 2, 2] = scaling_min
    scaling_source, scaling_target = scaling_aligned.split([scaling_source.shape[-3], scaling_target.shape[-3]], dim=-3)

    global_transform_source = compose_transform_matrix([rotate_source, transl_source, scaling_source])
    global_transform_target = compose_transform_matrix([rotate_target, transl_target, scaling_target])
    return global_transform_source, global_transform_target


@robust_dataloader
def train_one_epoch(
    model: MIP,
    criterion: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler: misc.NativeScalerWithGradNormCount,
    max_norm: float = 0,
    log_writer: SummaryWriter = None,
    args: Namespace = None,
    report_every: int = None,
    data_loader_val: DataLoader = None,
):
    data_loader.dataset.shuffle_train_anim()

    accum_iter = args.accum_iter
    if log_writer is not None:
        print(f"log_dir: {log_writer.log_dir}")

    optimizer.zero_grad()
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter=" | ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.2e}"))
    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, 50, f"Epoch: [{epoch}]")):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # Ground truth
        data: PoseDataGen
        global_transform_source, global_transform_target = get_global_transforms(data, aug_rotation=args.aug_rotation)
        gt = GT(
            data,
            global_transform_source,
            global_transform_target,
            device,
            model=model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model,
            args=args,
        )
        # with torch.no_grad():
        #     vis_batch, vis_pose = 0, 0
        #     import bpy; bpy.ops.export_scene.fbx(filepath="vis.fbx", check_existing=False, use_selection=False, add_leaf_bones=False, bake_anim=True)  # fmt:skip
        #     data.meta[vis_batch].meshes[-1].apply_transform(global_transform_source[vis_batch, 0].cpu().numpy()).export("vis-source-gt-mesh.glb")  # fmt:skip
        #     data.meta[vis_batch].meshes[:-1][vis_pose].apply_transform(global_transform_target[vis_batch, vis_pose].cpu().numpy()).export("vis-target-gt-mesh.glb")  # fmt:skip
        #     trimesh.Scene([trimesh.PointCloud(gt.source_pts[vis_batch, 0].cpu().numpy()), trimesh.PointCloud(gt.source_skeleton[vis_batch, 0].cpu().numpy())]).export("vis-source.glb")  # fmt:skip
        #     trimesh.Scene([trimesh.PointCloud(gt.target_surface_pts[vis_batch, vis_pose].cpu().numpy()), trimesh.PointCloud(gt.target_queries[vis_batch, vis_pose].cpu().numpy()), trimesh.PointCloud(gt.target_joints[vis_batch, vis_pose].nan_to_num().cpu().numpy())]).export("vis-target-sample.glb")  # fmt:skip
        #     trimesh.Scene([trimesh.PointCloud(gt.target_pts[vis_batch, vis_pose].cpu().numpy()), trimesh.PointCloud(gt.target_skeleton[vis_batch, vis_pose].cpu().numpy())]).export("vis-target-gt.glb")  # fmt:skip
        #     with model.pretrained_ae():
        #         model.eval().to_mesh(gt.source_inputs[vis_batch, 0][None][None], skip_transformer=True)[0][0].export("vis-source-mesh.glb")  # fmt:skip
        #         model.eval().to_mesh(gt.target_inputs[vis_batch, vis_pose][None][None], skip_transformer=True)[0][0].export("vis-target-mesh.glb")  # fmt:skip
        #     model.eval().to_mesh(gt.source_inputs[vis_batch][None], source_skeleton=gt.source_skeleton[vis_batch][None], target_skeleton=gt.target_skeleton[vis_batch, vis_pose][None][None], source_bw=gt.source_bw[vis_batch][None], skip_transformer=False, perturb_hands=args.perturb_hands)[0][0].export("vis-target-output.glb")  # fmt:skip

        # Forward
        with torch.amp.autocast("cuda", enabled=args.ae_type == "hy3d2.1"):
            model.train()
            output: Output = model(
                gt.source_inputs,
                gt.target_queries,
                source_skeleton=gt.source_skeleton,
                target_skeleton=gt.target_skeleton,
                source_bw=gt.source_bw,
                skip_decoding=args.use_latent_loss,
                perturb_hands=args.perturb_hands,
            )
            # latents = gt.target_latents(output.fps_idx_target)[0, 0:1]
            # latents = model.ae.decode(latents)
            # model.ae.enable_flashvdm_decoder(enabled=True, mc_algo="mc")
            # mesh_list = model.ae.latents2mesh(
            #     latents,
            #     output_type="trimesh",
            #     bounds=1.01,
            #     mc_level=0.0,
            #     num_chunks=20000,
            #     octree_resolution=256,
            #     mc_algo="mc",
            #     enable_pbar=True,
            # )
            # from util.Hunyuan3D_21.hy3dshape.hy3dshape.pipelines import export_to_trimesh
            # mesh_list = export_to_trimesh(mesh_list)
            # mesh_list[0].export("test.glb")
            loss, loss_value_dict, _ = get_loss(output, gt, args=args)

        for k, v in loss_value_dict.items():
            loss_value_dict[k] = misc.all_reduce_mean(v)

        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=(data_iter_step + 1) % accum_iter == 0,
            named_parameters=model.named_parameters(),
        )
        # for name, param in model.named_parameters():
        #     if param.requires_grad and not torch.isfinite(param).all():
        #         print(f"Parameter {name} is not finite, fixing it")
        #         param.requires_grad = False
        #         param.nan_to_num_(nan=0.0)
        #         param.requires_grad = True
        # print unused params
        # for name, param in model.named_parameters():
        #     if param.requires_grad and param.grad is None:
        #         print(name)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        metric_logger.update(loss=loss_value_dict["loss"])

        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        if (data_iter_step + 1) % accum_iter == 0:
            # # We use epoch_1000x as the x-axis in tensorboard. This calibrates different curves when batch size changes.
            # step = int((data_iter_step / len(data_loader) + epoch) * 1000)
            step = data_iter_step + epoch * len(data_loader)

            if log_writer is not None and misc.is_main_process():
                for k, v in loss_value_dict.items():
                    log_writer.add_scalar(k, v, step)
                log_writer.add_scalar("lr", max_lr, step)

            if report_every is not None:
                if report_every < 1:
                    report_every = int(len(data_loader) * report_every)
                else:
                    report_every = int(report_every)
                report_every = max(1, report_every)
                if data_iter_step % report_every == 0:
                    # if args.output_dir and data_iter_step != 0:
                    #     misc.save_model(
                    #         args=args,
                    #         model=model,
                    #         model_without_ddp=(
                    #             model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                    #         ),
                    #         optimizer=optimizer,
                    #         loss_scaler=loss_scaler,
                    #         epoch=f"step{step:06}",
                    #     )
                    test_stats, vis_data = evaluate(data_loader_val, model, device, args)
                    log_stats = {
                        **{f"test_{k}": v for k, v in test_stats.items()},
                        "step": step,
                    }
                    if args.log_dir and misc.is_main_process():
                        if log_writer is not None:
                            for k, v in log_stats.items():
                                if k in ("step",):
                                    continue
                                log_writer.add_scalar(f"eval/{k.removeprefix('test_')}", v, step)
                            if vis_data:
                                mesh: list[trimesh.Trimesh] = vis_data["mesh"]
                                for i, m in enumerate(mesh):
                                    if m is None:
                                        continue
                                    log_writer.add_mesh(
                                        f"eval/vis/pose{i}",
                                        vertices=m.vertices[None, ...],
                                        # faces=m.faces[None, ...],
                                        global_step=step,
                                    )
                        with open(os.path.join(args.log_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                            f.write(json.dumps(log_stats) + "\n")

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
@robust_dataloader
def evaluate(data_loader: DataLoader, model: MIP, device: torch.device, args: Namespace):
    criterion = torch.nn.MSELoss()
    metric_logger = misc.MetricLogger(delimiter=" | ")
    model.eval()
    vis_data_list = []
    for data in metric_logger.log_every(data_loader, 50, "Test:"):
        data: PoseDataGen
        global_transform_source, global_transform_target = get_global_transforms(data, aug_rotation=False)
        gt = GT(
            data,
            global_transform_source,
            global_transform_target,
            device,
            model=model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model,
            args=args,
        )
        with torch.amp.autocast("cuda", enabled=args.ae_type == "hy3d2.1"):
            output = model(
                gt.source_inputs,
                gt.target_queries,
                source_skeleton=gt.source_skeleton,
                target_skeleton=gt.target_skeleton,
                source_bw=gt.source_bw,
                perturb_hands=args.perturb_hands,
            )
            _, loss_value_dict, vis_data = get_loss(output, gt, args=args, eval=True)
        metric_logger.update(**loss_value_dict)
        vis_data_list.append(vis_data)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(f"* loss {metric_logger.loss.global_avg:.3f}")
    vis_data = {k: torch.cat([d[k] for d in vis_data_list], dim=0) for k in vis_data_list[0].keys()}
    if misc.is_main_process() and False:
        vis_batch_index = 0
        m = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        vis_data["mesh"] = m.to_mesh(
            gt.source_inputs[vis_batch_index : vis_batch_index + 1],
            source_skeleton=gt.source_skeleton[vis_batch_index : vis_batch_index + 1],
            target_skeleton=gt.target_skeleton[vis_batch_index : vis_batch_index + 1],
            skip_transformer=False,
        )[0]
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, vis_data
