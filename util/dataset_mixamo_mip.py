import os
import sys
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import torch
from tqdm import tqdm

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import warnings

import trimesh
from torch.utils.data import default_collate

from util import blender_utils, dataset_mixamo
from util.blender_utils import bpy
from util.dataset_mixamo import BONES_IDX_DICT, MIXAMO_PREFIX, MetaData, get_hips_transform, seed_worker
from util.utils import (
    HiddenPrints,
    _sample_mesh,
    apply_transform,
    apply_transform_normals,
    dqs,
    get_nearest_on_mesh,
    get_rotation_about_point,
    get_sdf,
    remesh,
    sample_mesh_visible,
)

warnings.filterwarnings("ignore", category=RuntimeWarning, module="trimesh.triangles")


def sample_on_surface(mesh: trimesh.Trimesh, num_points: int) -> tuple[np.ndarray, ...]:
    """
    Samples points on the surface of the mesh and their normals.

    Args:
        mesh (trimesh.Trimesh): The input trimesh object.
        num_points (int): The number of points to sample.

    Returns:
        points (np.ndarray): Sampled points, shape (N, 3).
        normals (np.ndarray): Normals of the sampled points, shape (N, 3).
        sdf (np.ndarray): SDF values (all zeros), shape (N, 1).
        occupancy (np.ndarray): Occupancy values (all ones), shape (N, 1).
    """
    if num_points <= 0:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 1)), np.empty((0, 1), dtype=int)

    points_on_surface, face_indices = trimesh.sample.sample_surface(mesh, count=num_points)
    points_on_surface = points_on_surface.astype(np.float32)
    normals_on_surface = mesh.face_normals[face_indices].astype(np.float32)

    sdf_on_surface = np.zeros((points_on_surface.shape[0], 1), dtype=np.float32)
    occupancy_on_surface = np.ones((points_on_surface.shape[0], 1), dtype=int)
    return points_on_surface, normals_on_surface, sdf_on_surface, occupancy_on_surface


def sample_near_surface(mesh: trimesh.Trimesh, num_points: int, point_only=False) -> tuple[np.ndarray, ...]:
    """
    Samples points near the surface of the mesh.

    Args:
        mesh (trimesh.Trimesh): The input trimesh object.
        num_points (int): The number of points to sample.

    Returns:
        points (np.ndarray): Sampled points, shape (N, 3).
        sdf (np.ndarray): SDF values, shape (N, 1).
        occupancy (np.ndarray): Occupancy values, shape (N, 1).
    """
    if num_points <= 0:
        return np.empty((0, 3)), np.empty((0, 1)), np.empty((0, 1), dtype=int)

    # Perturbation scale, e.g., 2% of the mesh's characteristic scale.
    perturb_sigma = 0.02 * mesh.scale
    if perturb_sigma <= 1e-7:  # Prevent excessively small or zero perturbation
        perturb_sigma = 0.01  # A small absolute fallback

    base_points_near, face_indices = trimesh.sample.sample_surface(mesh, count=num_points)
    base_points_near = base_points_near.astype(np.float32)

    # Get normals from faces corresponding to the sampled points
    normals_near = mesh.face_normals[face_indices].astype(np.float32)

    # Perturb points along normals with random magnitude and direction (in/out)
    # Ensure perturbation has some magnitude (0.1*sigma to sigma)
    perturb_magnitudes = np.random.uniform(low=0.1, high=1.0, size=(base_points_near.shape[0], 1)) * perturb_sigma
    perturb_signs = np.random.choice([-1, 1], size=(base_points_near.shape[0], 1))
    actual_perturb_dist = (perturb_signs * perturb_magnitudes).astype(np.float32)

    points_near_surface = base_points_near + normals_near * actual_perturb_dist

    if point_only:
        sdf_near_surface = np.full((points_near_surface.shape[0], 1), np.nan, dtype=np.float32)
        occupancy_near_surface = np.empty((points_near_surface.shape[0], 1), dtype=int)
    else:
        sdf_near_surface = get_sdf(mesh, points_near_surface).astype(np.float32)[..., None]
        occupancy_near_surface = (sdf_near_surface <= 0).astype(int)
    return points_near_surface, sdf_near_surface, occupancy_near_surface


def sample_in_volume(
    mesh: trimesh.Trimesh,
    num_points: int,
    point_only=False,
    bbox_extend=1.1,
    bbox_transform: np.ndarray = None,
) -> tuple[np.ndarray, ...]:
    """
    Samples points within the bounding box of the mesh.

    Args:
        mesh (trimesh.Trimesh): The input trimesh object.
        num_points (int): The number of points to sample.
    Returns:
        points (np.ndarray): Sampled points, shape (N, 3).
        sdf (np.ndarray): SDF values, shape (N, 1).
        occupancy (np.ndarray): Occupancy values, shape (N, 1).
    """
    if num_points <= 0:
        return np.empty((0, 3)), np.empty((0, 1)), np.empty((0, 1), dtype=int)

    bbox: trimesh.primitives.Box = mesh.bounding_box
    if bbox_extend > 1.0:
        # Get the center and extents of the original bounding box
        center = bbox.centroid
        extents = bbox.extents
        # Determine the side length of the new cubic box
        max_extent = np.max(extents)
        cube_side_length = max_extent * bbox_extend
        # Define the bounds of the cubic box
        half_side = cube_side_length / 2.0
        min_bound = center - half_side
        max_bound = center + half_side
        # Sample points uniformly within the cubic box
        points_volume = np.random.uniform(low=min_bound, high=max_bound, size=(num_points, 3)).astype(np.float32)
        if bbox_transform is not None:
            assert bbox_transform.shape == (4, 4)
            points_volume = apply_transform(points_volume, bbox_transform)

        if point_only:
            sdf_volume = np.full((num_points, 1), np.nan, dtype=np.float32)
            occupancy_volume = np.empty((num_points, 1), dtype=int)
        else:
            # clamp_distance_factor = 0.1
            clamp_distance_factor = None

            if clamp_distance_factor is not None:  # Clamping for efficiency
                sdf_volume = np.zeros(num_points)
                clamp_dist = cube_side_length * clamp_distance_factor
                dist_to_bbox_signed = -bbox.nearest.signed_distance(points_volume)

                # 1. Points far outside the bbox surface (clamp SDF)
                #    If dist_to_bbox_signed is negative (inside), this condition is false.
                far_mask = dist_to_bbox_signed > clamp_dist

                # 2. Points inside or close to the bbox surface (calculate exact SDF)
                exact_sdf_mask = ~far_mask

                # Calculate exact SDF for points inside or close to bbox
                if np.any(exact_sdf_mask):
                    sdf_volume[exact_sdf_mask] = get_sdf(mesh, points_volume[exact_sdf_mask])

                # Clamp SDF for points far outside bbox surface
                if np.any(far_mask):
                    sdf_volume[far_mask] = dist_to_bbox_signed[far_mask]

            else:  # No clamping, calculate exact SDF for all points
                sdf_volume = get_sdf(mesh, points_volume)
    else:
        # Sample points randomly within the mesh's bounding box.
        points_volume = bbox.sample_volume(count=num_points).astype(np.float32)
        if bbox_transform is not None:
            assert bbox_transform.shape == (4, 4)
            points_volume = apply_transform(points_volume, bbox_transform)
        if point_only:
            sdf_volume = np.full((num_points, 1), np.nan, dtype=np.float32)
            occupancy_volume = np.empty((num_points, 1), dtype=int)
        else:
            sdf_volume = get_sdf(mesh, points_volume)

    if not point_only:
        sdf_volume = sdf_volume.astype(np.float32)[..., None]
        occupancy_volume = (sdf_volume <= 0).astype(int)
    return points_volume, sdf_volume, occupancy_volume


def sample_mesh_sdf(
    mesh: trimesh.Trimesh,
    num_total_points: int,
    point_only=False,
    bbox_extend=1.1,
    bbox_transform: np.ndarray = None,
) -> tuple[np.ndarray, ...]:
    """
    Generates a specified number of spatial sample points for a given mesh
    and returns their occupancy, SDF values, and specifically the on-surface points and their normals.
    Points are divided into on-surface, near-surface, and volume points.

    Args:
        mesh: trimesh.Trimesh: The input mesh to sample from.
        num_total_points (int): Total number of points to sample.

    Returns:
        all_points (np.ndarray): All sampled points (on-surface, near-surface, volume), shape (num_total_points, 3). \
        sdf_values (np.ndarray): SDF value for each point in all_points, shape (num_total_points, 1). \
        occupancy_values (np.ndarray): Occupancy value (0 or 1) for each point in all_points, shape (num_total_points, 1). \
        surface_points_mask (np.ndarray): Bool mask (True if point is on the surface), shape (num_total_points,). \
        surface_normals (np.ndarray): Normals for each point in surface_points, shape (num_on_surface, 3).
    """
    if not isinstance(num_total_points, int) or num_total_points <= 0:
        raise ValueError("num_total_points must be a positive integer")

    # Distribute points among categories.
    n_points_per_category = num_total_points // 3
    num_on_surface = n_points_per_category
    num_near_surface = n_points_per_category
    num_volume = num_total_points - num_on_surface - num_near_surface

    all_points_list = []
    all_sdf_list = []
    all_occupancy_list = []

    # 1. On-surface points
    if num_on_surface > 0:
        pts_s, normals_s, sdf_s, occ_s = sample_on_surface(mesh, num_on_surface)
        assert pts_s.shape[0] == num_on_surface, f"Got {pts_s.shape[0]}/{num_on_surface} surface points"
        all_points_list.append(pts_s)
        all_sdf_list.append(sdf_s)
        all_occupancy_list.append(occ_s)

    # 2. Near-surface points
    if num_near_surface > 0:
        pts_n, sdf_n, occ_n = sample_near_surface(mesh, num_near_surface, point_only=point_only)
        assert pts_n.shape[0] == num_near_surface, f"Got {pts_n.shape[0]}/{num_near_surface} near-surface points"
        all_points_list.append(pts_n)
        all_sdf_list.append(sdf_n)
        all_occupancy_list.append(occ_n)

    # 3. Volume points
    if num_volume > 0:
        pts_v, sdf_v, occ_v = sample_in_volume(
            mesh, num_volume, point_only=point_only, bbox_extend=bbox_extend, bbox_transform=bbox_transform
        )
        assert pts_v.shape[0] == num_volume, f"Got {pts_v.shape[0]}/{num_volume} volume points"
        all_points_list.append(pts_v)
        all_sdf_list.append(sdf_v)
        all_occupancy_list.append(occ_v)

    # Concatenate results.
    final_points = np.vstack(all_points_list)
    surface_points_mask = np.zeros(final_points.shape[0], dtype=bool)
    surface_points_mask[:num_on_surface] = True
    # assert (final_points[surface_points_mask] == pts_s).all()
    final_sdf = np.concatenate(all_sdf_list)
    final_occupancy = np.concatenate(all_occupancy_list)
    assert final_points.shape[0] == num_total_points, f"Expected {num_total_points} points, got {final_points.shape[0]}"

    return final_points, final_sdf, final_occupancy, surface_points_mask, normals_s


@dataclass(frozen=True)
class PoseDataGen:
    source_joints: torch.Tensor = None
    """(B, Ps, K, 3)"""
    source_pts: torch.Tensor = None
    """(B, Ps, Ns, 3)"""
    source_normals: torch.Tensor = None
    """(B, Ps, Ns, 3)"""
    source_skeleton: torch.Tensor = None
    """(B, Ps, Ns, 3)"""
    source_bw: torch.Tensor = None
    """(B, Ps, Ns, K)"""

    target_queries: torch.Tensor = None
    """(B, Pt, Nt, 3)"""
    target_occ: torch.Tensor = None
    """(B, Pt, Nt, 1)"""
    target_sdf: torch.Tensor = None
    """(B, Pt, Nt, 1)"""
    target_surface_mask: torch.Tensor = None
    """(B, Pt, Nt)"""
    target_surface_normals: torch.Tensor = None
    """(B, Pt, Nt', 3)"""

    target_joints: torch.Tensor = None
    """(B, Pt, K, 3)"""
    target_pts: torch.Tensor = None
    """(B, Pt, Ns, 3)"""
    target_normals: torch.Tensor = None
    """(B, Pt, Ns, 3)"""
    target_skeleton: torch.Tensor = None
    """(B, Pt, Ns, 3)"""
    target_pts_orig: torch.Tensor = None
    """(B, Pt, Ns, 3)"""
    target_normals_orig: torch.Tensor = None
    """(B, Pt, Ns, 3)"""
    target_sortkey_orig: torch.Tensor = None
    """(B, Pt, Ns)"""

    meta: list[MetaData] = None

    def __len__(self):
        assert isinstance(
            self.meta, list
        ), f"Non-batchified {self.__class__.__name__} can not be counted or subscripted"
        return len(self.meta)

    def __getitem__(self, index: int):
        len(self)  # guard
        return self.__class__(**{k: v[index] for k, v in self.__dict__.items()})

    @cached_property
    def hips_transform_source(self):
        """(B, P, 4, 4)
        transformation that moves Hips to the origin,
        and rotates body to face forward in the positive direction of z-axis
        and right-to-left oriented in the positive direction of x-axis.
        """
        bones_idx_dict = self.meta[0].bones_idx_dict
        assert len(bones_idx_dict) == self.source_joints.shape[-2]
        if any(
            x not in bones_idx_dict
            for x in [f"{MIXAMO_PREFIX}Hips", f"{MIXAMO_PREFIX}RightUpLeg", f"{MIXAMO_PREFIX}LeftUpLeg"]
        ):
            warnings.warn("Cannot find hips plane bones, return identity transform")
            hips_transform = torch.eye(4).to(self.source_joints).unsqueeze(0).repeat(self.source_joints.shape[0], 1, 1)
        else:
            hips = self.source_joints[..., bones_idx_dict[f"{MIXAMO_PREFIX}Hips"], :]
            rightupleg = self.source_joints[..., bones_idx_dict[f"{MIXAMO_PREFIX}RightUpLeg"], :]
            leftupleg = self.source_joints[..., bones_idx_dict[f"{MIXAMO_PREFIX}LeftUpLeg"], :]
            if torch.isnan(torch.stack([hips, rightupleg, leftupleg])).any():
                raise RuntimeError("Cannot find hips plane")
            hips_transform = get_hips_transform(hips, rightupleg, leftupleg)
        return hips_transform

    @cached_property
    def hips_transform_target(self):
        """(B, P, 4, 4)
        transformation that moves Hips to the origin,
        and rotates body to face forward in the positive direction of z-axis
        and right-to-left oriented in the positive direction of x-axis.
        """
        bones_idx_dict = self.meta[0].bones_idx_dict
        assert len(bones_idx_dict) == self.target_joints.shape[-2]
        if any(
            x not in bones_idx_dict
            for x in [f"{MIXAMO_PREFIX}Hips", f"{MIXAMO_PREFIX}RightUpLeg", f"{MIXAMO_PREFIX}LeftUpLeg"]
        ):
            warnings.warn("Cannot find hips plane bones, return identity transform")
            hips_transform = torch.eye(4).to(self.target_joints).unsqueeze(0).repeat(self.target_joints.shape[0], 1, 1)
        else:
            hips = self.target_joints[..., bones_idx_dict[f"{MIXAMO_PREFIX}Hips"], :]
            rightupleg = self.target_joints[..., bones_idx_dict[f"{MIXAMO_PREFIX}RightUpLeg"], :]
            leftupleg = self.target_joints[..., bones_idx_dict[f"{MIXAMO_PREFIX}LeftUpLeg"], :]
            if torch.isnan(torch.stack([hips, rightupleg, leftupleg])).any():
                raise RuntimeError("Cannot find hips plane")
            hips_transform = get_hips_transform(hips, rightupleg, leftupleg)
        return hips_transform


def project_points_to_bones(
    pts: np.ndarray,
    pts_bone_ids: np.ndarray,
    joints_head: np.ndarray,
    joints_tail: np.ndarray,
    return_local_coords=False,
) -> np.ndarray:
    """
    Projects points onto specified bone lines.

    For each point in `pts`, it's projected onto the infinite line
    defined by `joints_head` and `joints_tail` at the index
    specified by `pts_bone_ids`.

    Args:
        pts (np.ndarray): Points to project. Shape (B, N, 3).
        pts_bone_ids (np.ndarray): Integer array of bone indices (0 to K-1) for each point. Shape (B, N).
        joints_head (np.ndarray): Starting points of K lines for each batch. Shape (B, K, 3).
        joints_tail (np.ndarray): Ending points of K lines for each batch. Shape (B, K, 3).

    Returns:
        np.ndarray: Projected points. Shape (B, N, 3).
    """
    # Create batch indices for advanced indexing
    # batch_indices will be (B, 1) to correctly broadcast with pts_bone_ids (B, N)
    batch_indices = np.arange(pts.shape[0])[:, np.newaxis]

    # Gather the corresponding head (A) and tail (B) points for each point in pts
    # A_selected and B_selected will have shape (B, N, 3)
    A_selected = joints_head[batch_indices, pts_bone_ids]
    B_selected = joints_tail[batch_indices, pts_bone_ids]

    # Calculate line direction vectors (v = B - A)
    # line_dir_vecs will have shape (B, N, 3)
    line_dir_vecs = B_selected - A_selected

    # Calculate vectors from line start (A) to the points (P) (w = P - A)
    # vec_A_to_pts will have shape (B, N, 3)
    vec_A_to_pts = pts - A_selected

    # Calculate the parameter t for projection: t = dot(w, v) / dot(v, v)
    # w_dot_v and v_dot_v will have shape (B, N)
    w_dot_v = np.sum(vec_A_to_pts * line_dir_vecs, axis=2)
    v_dot_v = np.sum(line_dir_vecs * line_dir_vecs, axis=2)

    # Initialize t with zeros.
    # This handles cases where v_dot_v is zero (i.e., joint_head == joint_tail).
    # In such cases, the line is a point, and the projection is A_selected.
    # t=0 leads to projected_point = A_selected + 0 * v = A_selected.
    t = np.zeros_like(v_dot_v)

    # A small epsilon to avoid division by zero due to floating point inaccuracies
    # for non-zero length line segments.
    epsilon = 1e-8
    valid_v_mask = v_dot_v > epsilon

    # Calculate t only for valid line segments (where v_dot_v > epsilon)
    t[valid_v_mask] = w_dot_v[valid_v_mask] / v_dot_v[valid_v_mask]

    # Reshape t from (B, N) to (B, N, 1) for broadcasting
    t_reshaped = t[:, :, np.newaxis]

    # Calculate projected points: P_proj = A + t * v
    projected_pts = A_selected + t_reshaped * line_dir_vecs

    return (projected_pts, t) if return_local_coords else projected_pts


def project_points_to_cylinders(
    pts: np.ndarray,
    pts_bone_ids: np.ndarray,
    joints_head: np.ndarray,
    joints_tail: np.ndarray,
    diameter: float = None,
) -> np.ndarray:
    """
    Projects points onto the surfaces of specified cylinders.

    For each point in `pts`, it is first projected onto the infinite axis line
    defined by `joints_head` and `joints_tail` at the index specified by
    `pts_bone_ids`. Then, it is moved radially from the axis to the
    cylinder's surface.

    Args:
        pts (np.ndarray): Points to project. Shape (B, N, 3).
        pts_bone_ids (np.ndarray): Integer array of bone indices (0 to K-1) for each point. Shape (B, N).
        joints_head (np.ndarray): Starting points of K cylinder axes for each batch. Shape (B, K, 3).
        joints_tail (np.ndarray): Ending points of K cylinder axes for each batch. Shape (B, K, 3).
        diameter (Optional[Union[float, np.ndarray]]): The diameter of the cylinders.
            - If a float, all cylinders share the same diameter.
            - If None, the diameter is adaptive to the bone length
              (diameter = bone_length * ADAPTIVE_DIAMETER_RATIO).
            - (Advanced) Can be a (K,) array to specify per-cylinder diameters.

    Returns:
        np.ndarray: Projected points. Shape (B, N, 3).
    """
    # Create batch indices for advanced indexing
    batch_indices = np.arange(pts.shape[0])[:, np.newaxis]

    # --- Step 1: Determine Radius for Each Point ---
    # Calculate all bone vectors and their lengths to be used in adaptive mode
    bone_vecs = joints_tail - joints_head  # Shape (B, K, 3)
    bone_lengths = np.linalg.norm(bone_vecs, axis=2)  # Shape (B, K)

    if diameter is None:
        # Adaptive mode: diameter is proportional to bone length
        diameters = bone_lengths * 0.25
        if joints_head.shape[1] == len(BONES_IDX_DICT):
            diameters[:, BONES_IDX_DICT[f"{MIXAMO_PREFIX}Head"]] *= 2
            diameters[:, BONES_IDX_DICT[f"{MIXAMO_PREFIX}Hips"]] *= 4
            diameters[:, BONES_IDX_DICT[f"{MIXAMO_PREFIX}Spine"]] *= 4
            diameters[:, BONES_IDX_DICT[f"{MIXAMO_PREFIX}Spine1"]] *= 4
            diameters[:, BONES_IDX_DICT[f"{MIXAMO_PREFIX}Spine2"]] *= 4
    elif isinstance(diameter, (float, int)):
        # Fixed mode: all cylinders have the same diameter
        diameters = np.full_like(bone_lengths, float(diameter))
    else:
        # Advanced mode: a specific diameter is provided for each cylinder
        assert diameter.shape == (K,)
        diameters = np.broadcast_to(diameter, bone_lengths.shape)

    # Select the correct radius for each point.
    # radii_selected will have shape (B, N).
    radii = diameters / 2.0
    radii_selected = radii[batch_indices, pts_bone_ids]

    # --- Step 2: Project Points onto the Cylinder Axis ---
    # Gather the corresponding head (A) and tail (B) points for each point
    A_selected = joints_head[batch_indices, pts_bone_ids]
    B_selected = joints_tail[batch_indices, pts_bone_ids]

    # Calculate axis direction vectors (v = B - A)
    line_dir_vecs = B_selected - A_selected

    # Calculate vectors from axis start (A) to the points (P) (w = P - A)
    vec_A_to_pts = pts - A_selected

    # Calculate the parameter t for projection: t = dot(w, v) / dot(v, v)
    w_dot_v = np.sum(vec_A_to_pts * line_dir_vecs, axis=2)
    v_dot_v = np.sum(line_dir_vecs * line_dir_vecs, axis=2)

    # Initialize t and handle zero-length axes to avoid division by zero
    t = np.zeros_like(v_dot_v)
    epsilon = 1e-8
    valid_v_mask = v_dot_v > epsilon
    t[valid_v_mask] = w_dot_v[valid_v_mask] / v_dot_v[valid_v_mask]

    # Reshape t for broadcasting from (B, N) to (B, N, 1)
    t_reshaped = t[:, :, np.newaxis]

    # Calculate the projected points on the central axis: P_axis = A + t * v
    pts_axis_proj = A_selected + t_reshaped * line_dir_vecs

    # --- Step 3: Move Points from Axis to Cylinder Surface ---
    # Calculate the radial vectors from the axis to the original points
    radial_vec = pts - pts_axis_proj  # Shape (B, N, 3)

    # Calculate the distance from the axis for each point
    dist_from_axis = np.linalg.norm(radial_vec, axis=2)  # Shape (B, N)

    # Safely normalize the radial vectors to get unit direction vectors.
    # For points on the axis (dist_from_axis ~ 0), the vector becomes (0,0,0).
    dist_reshaped = dist_from_axis[:, :, np.newaxis]
    radial_unit_vec = np.divide(radial_vec, dist_reshaped, out=np.zeros_like(radial_vec), where=dist_reshaped > epsilon)

    # Calculate the final offset vector from the axis to the surface
    radii_reshaped = radii_selected[:, :, np.newaxis]
    offset_from_axis = radii_reshaped * radial_unit_vec

    # Final projected point is the axis projection plus the radial offset
    projected_pts = pts_axis_proj + offset_from_axis

    return projected_pts


def sharp_sample_pointcloud(mesh: trimesh.Trimesh, num: int):
    """
    Sample points and normals preferentially from sharp edges of the mesh.

    Sharp edges are detected based on the angle between vertex normals and face normals.
    Points are sampled along these edges proportionally to edge length.

    Args:
        mesh (trimesh.Trimesh): Input mesh to sample from.
        num (int, optional): Number of points to sample from sharp edges. Default is 16384.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - samples: Sampled points along sharp edges, shape (num, 3).
            - normals: Corresponding interpolated normals, shape (num, 3).
    """
    V = mesh.vertices
    N = mesh.face_normals
    VN = mesh.vertex_normals
    F = mesh.faces
    VN2 = np.ones(V.shape[0])
    for i in range(3):
        dot = np.stack((VN2[F[:, i]], np.sum(VN[F[:, i]] * N, axis=-1)), axis=-1)
        VN2[F[:, i]] = np.min(dot, axis=-1)

    sharp_mask = VN2 < 0.985
    # collect edge
    edge_a = np.concatenate((F[:, 0], F[:, 1], F[:, 2]))
    edge_b = np.concatenate((F[:, 1], F[:, 2], F[:, 0]))
    num_faces = F.shape[0]
    face_indices_for_edges = np.tile(np.arange(num_faces), 3)
    sharp_edge = sharp_mask[edge_a] * sharp_mask[edge_b]
    edge_a = edge_a[sharp_edge > 0]
    edge_b = edge_b[sharp_edge > 0]
    sharp_face_indices = face_indices_for_edges[sharp_edge > 0]

    sharp_verts_a = V[edge_a]
    sharp_verts_b = V[edge_b]
    sharp_verts_an = VN[edge_a]
    sharp_verts_bn = VN[edge_b]

    weights = np.linalg.norm(sharp_verts_b - sharp_verts_a, axis=-1)
    weights_sum = np.sum(weights)
    if weights.sum() == 0:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty(0, dtype=int)
    weights /= weights_sum

    random_number = np.random.rand(num)
    w = np.random.rand(num, 1)
    index = np.searchsorted(weights.cumsum(), random_number)
    samples: np.ndarray = w * sharp_verts_a[index] + (1 - w) * sharp_verts_b[index]
    normals: np.ndarray = w * sharp_verts_an[index] + (1 - w) * sharp_verts_bn[index]
    face_indices: np.ndarray = sharp_face_indices[index]
    return samples, normals, face_indices


def get_weights(
    pts: np.ndarray, verts: np.ndarray, weights_per_vert: np.ndarray, vert_idx: np.ndarray
) -> tuple[np.ndarray, ...]:
    """
    Args:
        pts: (N, 3) points on the surface of the mesh
        verts: (V, 3) vertices of the mesh
        weights_per_vert: (V, K) weights per vertex
        vert_idx: (N, 3) vertex indices of each point
    Returns:
        (N, K) weights of each point
        (N,) max weight index of each point
        (N, K) weights of each point, but only the max weight is kept as 1.0, others are 0.0
    """
    # weights = weights_per_vert[vert_idx].mean(axis=1)  # mean of the three vertices of a face
    barycentric_coords = trimesh.triangles.points_to_barycentric(triangles=verts[vert_idx], points=pts)
    barycentric_coords = barycentric_coords.astype(np.float32)
    barycentric_coords = np.nan_to_num(barycentric_coords, nan=1 / 3, posinf=1 / 3, neginf=1 / 3)
    barycentric_coords = np.clip(barycentric_coords, 0, 1)
    weights = np.einsum("ij,ijk->ik", barycentric_coords, weights_per_vert[vert_idx])
    weights_max_idx = np.argmax(weights, axis=-1)
    # Keep the max weight as 1.0, and others as 0.0
    weights_maxonly = np.zeros_like(weights)
    weights_maxonly[np.arange(weights.shape[0]), weights_max_idx] = 1.0
    return weights, weights_max_idx, weights_maxonly


class MixamoDatasetGen(dataset_mixamo.MixamoDataset):

    def __init__(
        self,
        *args,
        num_sample_points=32768,
        sharp_edge_ratio=0.0,
        num_poses_source=1,
        num_poses_target=1,
        num_target_points=2048 * 3,
        sample_frames=None,
        rest_as_target=False,
        get_sdf=False,
        align_sample_points=True,
        sample_bbox_extend=1.1,
        skeleton_type="cylinders",
        remesh_type="none",
        remesh_method="winding_number",
        ipose_rate=0.0,
        **xargs,
    ):
        assert num_poses_source > 0 and num_poses_target >= 0
        self.num_poses_source = num_poses_source
        self.num_poses_target = num_poses_target
        if sample_frames is None:
            sample_frames = num_poses_source + num_poses_target
            if rest_as_target:
                sample_frames -= 1  # include_rest=True
        else:
            assert len(sample_frames) == (
                num_poses_source + num_poses_target - 1 if rest_as_target else num_poses_source + num_poses_target
            )
        super().__init__(
            *args,
            sample_points=1 if (align_sample_points and remesh_type == "none") else num_sample_points,
            sample_vertices=-1,
            include_rest=rest_as_target,
            get_normals=True,
            get_face_index=True,
            sample_frames=sample_frames,
            **xargs,
        )
        self.num_sample_points = num_sample_points
        if sharp_edge_ratio > 0:
            assert align_sample_points, "Sharp edge sampling requires align_sample_points=True"
        self.sharp_edge_ratio = sharp_edge_ratio
        self.num_target_points = num_target_points
        self.get_sdf = get_sdf
        self.align_sample_points = align_sample_points
        self.sample_bbox_extend = sample_bbox_extend
        self.skeleton_type = skeleton_type
        self.remesh_type = remesh_type
        if self.remesh_type != "none":
            assert self.align_sample_points, f"{self.remesh_type=} requires align_sample_points=True"
            assert self.sharp_edge_ratio == 0.0, f"{self.remesh_type=} is not compatible with sharp edge sampling"
        self.remesh_method = remesh_method
        self.ipose_rate = ipose_rate

    def __getitem__(self, index: int):
        data_raw = super().__getitem__(index)

        # target first (including rest), then source
        target_joints = np.nan_to_num(data_raw.joints[: self.num_poses_target])
        target_joints_tail = np.nan_to_num(data_raw.joints_tail[: self.num_poses_target])
        target_pts = np.array(data_raw.pts[: self.num_poses_target])
        target_pts_orig = target_pts.copy()
        target_normals = np.array(data_raw.pts_normal[: self.num_poses_target])
        target_normals_orig = target_normals.copy()
        target_verts = np.array(data_raw.verts[: self.num_poses_target])
        source_joints = np.nan_to_num(data_raw.joints[self.num_poses_target :])
        source_joints_tail = np.nan_to_num(data_raw.joints_tail[self.num_poses_target :])
        source_pts = data_raw.pts[self.num_poses_target :]
        source_normals = data_raw.pts_normal[self.num_poses_target :]
        source_verts = np.array(data_raw.verts[self.num_poses_target :])
        source_pts_faces = data_raw.pts_faces[self.num_poses_target :]
        faces = data_raw.meta.faces

        joints_transform = np.array(data_raw.joints_transform)
        nan_mask = np.any(np.isnan(joints_transform), axis=(-2, -1))
        joints_transform[nan_mask] = np.identity(4, dtype=joints_transform.dtype)
        source_vertices_weights = np.nan_to_num(data_raw.weights[self.num_poses_target :])
        target_pts_faces = np.array(data_raw.pts_faces[: self.num_poses_target])
        target_vertices_weights = np.nan_to_num(data_raw.weights[: self.num_poses_target])
        target_hips_transform = data_raw.hips_transform[: self.num_poses_target]

        if self.align_sample_points:  # override source sampling points

            if (
                self.remesh_type != "none"
                and self.ipose_rate > 0
                and (self.split != "train" or np.random.rand() < self.ipose_rate)
            ):  # override source pose
                with HiddenPrints(enable=True):
                    for action in bpy.data.actions:
                        bpy.data.actions.remove(action)
                    char_armature = blender_utils.get_armature_obj()
                    i_armature = blender_utils.load_file(os.path.join(self.data_dir, "animation_extra/i-pose.fbx"))[0]
                    blender_utils.set_action(char_armature, i_armature.animation_data.action)
                    bpy.data.objects.remove(i_armature, do_unlink=True)
                    bpy.context.scene.frame_set(2)
                    blender_utils.update()
                source_rest_to_pose = joints_transform[self.num_poses_target :]
                source_pose_to_rest = np.linalg.inv(source_rest_to_pose)
                source_rest_to_i = blender_utils.get_pose_bones(char_armature)[-1]
                source_rest_to_i = dataset_mixamo.reorganize_bone_data(
                    source_rest_to_i.T,
                    blender_utils.get_rest_bones(char_armature)[-1],
                    template_dict=self.bones_idx_dict,
                ).T.astype(np.float32)
                nan_mask = np.any(np.isnan(source_rest_to_i), axis=(-2, -1))
                source_rest_to_i[nan_mask] = np.identity(4, dtype=source_rest_to_i.dtype)
                source_pose_to_i = source_rest_to_i @ source_pose_to_rest
                source_joints = apply_transform(source_joints, source_pose_to_i)
                source_joints_tail = apply_transform(source_joints_tail, source_pose_to_i)
                source_skinning_transform = np.einsum("bkij,bnk->bnij", source_pose_to_i, source_vertices_weights)
                source_verts = apply_transform(source_verts, source_skinning_transform)
                for i, mesh in enumerate(data_raw.meta.meshes[self.num_poses_target :]):
                    mesh.vertices = source_verts[i]
                joints_transform[self.num_poses_target :] = source_rest_to_i

            for batch_idx, v in enumerate(source_verts):
                mesh = trimesh.Trimesh(v, faces, process=False, maintain_order=True)
                if self.sharp_edge_ratio > 0:
                    num_sharp_edge = int(self.sharp_edge_ratio * self.num_sample_points)
                    assert num_sharp_edge > 0
                    num_uniform = self.num_sample_points - num_sharp_edge
                else:
                    num_uniform = self.num_sample_points
                    num_sharp_edge = 0

                # import matplotlib.pyplot as plt # fmt:skip
                # pts, normals, face_index, visible_mask = sample_mesh_visible(mesh, num_uniform)
                # target_mesh = trimesh.Trimesh(data_raw.verts[0], faces, process=False, maintain_order=True)
                # target_mesh.visual.face_colors = plt.get_cmap("plasma")(1 - visible_mask)
                # target_mesh.export("test.glb")

                if self.remesh_type == "none":
                    pts, face_index = _sample_mesh(mesh, num_uniform, return_normal=True, return_face_index=True)
                elif self.remesh_type == "project":
                    if self.remesh_method == "alpha_wrap":
                        pts = _sample_mesh(
                            remesh(mesh, alpha_fraction=0.02), num_uniform, return_normal=False, return_face_index=False
                        )
                        pts, normals, face_index = get_nearest_on_mesh(mesh, pts, use_kdtree=True)
                    elif self.remesh_method == "winding_number":
                        pts, normals, face_index, visible_mask = sample_mesh_visible(mesh, num_uniform)
                        # print(f"Visible ratio: {visible_mask.sum() / visible_mask.shape[0]}")
                    else:
                        raise NotImplementedError(f"{self.remesh_method=}")
                    pts = np.concatenate([pts, normals], axis=-1)
                elif self.remesh_type == "replace":
                    pts = _sample_mesh(
                        remesh(mesh, alpha_fraction=0.006), num_uniform, return_normal=True, return_face_index=False
                    )
                    _, _, face_index = get_nearest_on_mesh(mesh, pts[..., :3], use_kdtree=False)
                else:
                    raise NotImplementedError(f"{self.remesh_type=}")

                pts, normals = np.split(pts, 2, axis=-1)
                if num_sharp_edge > 0:
                    sharp_pts, sharp_normals, sharp_pts_faces = sharp_sample_pointcloud(mesh, num_sharp_edge)
                    if sharp_pts.shape[0] < num_sharp_edge:
                        # If not enough sharp points, fill the rest with uniform points
                        sharp_pts_, sharp_pts_faces_ = _sample_mesh(
                            mesh, num_sharp_edge - sharp_pts.shape[0], return_normal=True, return_face_index=True
                        )
                        sharp_pts_, sharp_normals_ = np.split(sharp_pts_, 2, axis=-1)
                        sharp_pts = np.concatenate((sharp_pts, sharp_pts_), axis=0)
                        sharp_normals = np.concatenate((sharp_normals, sharp_normals_), axis=0)
                        sharp_pts_faces = np.concatenate((sharp_pts_faces, sharp_pts_faces_), axis=0)
                    pts = np.concatenate((pts, sharp_pts), axis=0)
                    normals = np.concatenate((normals, sharp_normals), axis=0)
                    face_index = np.concatenate((face_index, sharp_pts_faces), axis=0)
                    assert pts.shape[0] == normals.shape[0] == face_index.shape[0] == self.num_sample_points
                source_pts[batch_idx] = pts.astype(np.float32)
                source_normals[batch_idx] = normals.astype(np.float32)
                source_pts_faces[batch_idx] = face_index

        source_pts_weights = []
        source_pts_bones = []
        source_pts_weights_max = []
        for batch_idx, w in enumerate(source_vertices_weights):
            v = faces[source_pts_faces[batch_idx]]
            weights, weights_max_idx, weights_max = get_weights(source_pts[batch_idx], source_verts[batch_idx], w, v)
            source_pts_weights.append(weights)
            source_pts_bones.append(weights_max_idx)
            source_pts_weights_max.append(weights_max)
        if self.skeleton_type == "lines":
            project_fn = project_points_to_bones
        elif self.skeleton_type == "cylinders":
            project_fn = project_points_to_cylinders
        else:
            raise NotImplementedError(f"{self.skeleton_type=}")
        source_skeleton = project_fn(
            np.array(source_pts), np.array(source_pts_bones), source_joints, source_joints_tail
        )  # (Ps, Ns, 3)

        ref_idx = 0  # the first source pose is the base to get all the target poses
        source_to_rest = np.linalg.inv(joints_transform[self.num_poses_target :][ref_idx])
        rest_to_target = joints_transform[: self.num_poses_target]
        source_to_target = np.einsum("bnij,njk->bnik", rest_to_target, source_to_rest)  # (Pt, K, 4, 4)
        skeleton_source_to_target = np.einsum(
            "bkij,nk->bnij", source_to_target, source_pts_weights_max[ref_idx]
        )  # (Pt, Ns, 4, 4)
        target_skeleton = apply_transform(
            source_skeleton[ref_idx][np.newaxis, ...].repeat(skeleton_source_to_target.shape[0], axis=0),
            skeleton_source_to_target,
        )  # (Pt, Ns, 3)

        if self.remesh_type != "none" and False:
            target_pts_bones = []
            for batch_idx, w in enumerate(target_vertices_weights):
                v = faces[target_pts_faces[batch_idx]]
                _, weights_max_idx, _ = get_weights(target_pts[batch_idx], target_verts[batch_idx], w, v)
                target_pts_bones.append(weights_max_idx)
            target_pts_bones = np.array(target_pts_bones)
            _, target_pts_local = project_points_to_bones(
                target_pts,
                target_pts_bones,
                target_joints,
                target_joints_tail,
                return_local_coords=True,
            )
            target_sortkey = target_pts_bones * np.abs(target_pts_local).max() * 3 + target_pts_local
            # import matplotlib.pyplot as plt
            # norm_func = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)
            # colors = plt.get_cmap("plasma")(norm_func(target_sortkey[0]))[:, :3]
            # trimesh.PointCloud(target_pts[0], colors=colors).export("test.glb")
        else:
            target_sortkey = np.full(target_pts.shape[:2], np.nan)

        if self.align_sample_points:
            # Override target sampling points to align with source sampling points by manually applying skinning
            if self.dual_quat_skinning:
                pts_source_to_target = dqs(source_to_target, source_pts_weights[ref_idx]).astype(np.float32)
                assert np.isfinite(pts_source_to_target).all()
            else:
                pts_source_to_target = np.einsum(
                    "bkij,nk->bnij", source_to_target, source_pts_weights[ref_idx]
                )  # (Pt, Ns, 4, 4)
            target_pts = apply_transform(
                source_pts[ref_idx][np.newaxis, ...].repeat(pts_source_to_target.shape[0], axis=0),
                pts_source_to_target,
            )  # (Pt, Ns, 3)
            target_normals = apply_transform_normals(
                source_normals[ref_idx][np.newaxis, ...].repeat(pts_source_to_target.shape[0], axis=0),
                pts_source_to_target,
            )  # (Pt, Ns, 3)

        target_queries = []
        target_occ = []
        target_sdf = []
        target_surface_mask = []
        target_surface_normals = []
        for batch_idx, v in enumerate(target_verts):
            mesh = trimesh.Trimesh(vertices=v, faces=faces, process=False)
            bbox_transform = target_hips_transform[batch_idx, :3, :3].T
            # bbox_transform = get_rotation_about_point(bbox_transform, target_joints[batch_idx, 0])
            bbox_transform = get_rotation_about_point(bbox_transform, mesh.bounding_box.centroid)
            final_points, final_sdf, final_occupancy, surface_mask, surface_normals = sample_mesh_sdf(
                mesh,
                self.num_target_points,
                point_only=not self.get_sdf,
                bbox_extend=self.sample_bbox_extend,
                bbox_transform=bbox_transform,
            )
            target_queries.append(final_points)
            target_occ.append(final_occupancy)
            target_sdf.append(final_sdf)
            target_surface_mask.append(surface_mask)
            target_surface_normals.append(surface_normals)

        # After batchified by `collate`, values in data will follow their type hinting (torch.Tensor), but not yet now
        data = PoseDataGen(
            source_joints=np.array(source_joints),
            source_pts=np.array(source_pts),
            source_normals=np.array(source_normals),
            source_skeleton=np.array(source_skeleton),
            source_bw=np.array(source_pts_weights),
            target_queries=np.array(target_queries),
            target_occ=np.array(target_occ),
            target_sdf=np.array(target_sdf),
            target_surface_mask=np.array(target_surface_mask),
            target_surface_normals=np.array(target_surface_normals),
            target_joints=np.array(target_joints),
            target_pts=np.array(target_pts),
            target_normals=np.array(target_normals),
            target_skeleton=np.array(target_skeleton),
            target_pts_orig=np.array(target_pts_orig),
            target_normals_orig=np.array(target_normals_orig),
            target_sortkey_orig=np.array(target_sortkey),
            meta=data_raw.meta,
        )
        return data


def collate(batch: list[PoseDataGen]):
    meta_all = []
    for data in batch:
        meta_all.append(data.meta)
    data_all = [
        {k: v for k, v in filter(lambda x: not isinstance(x[1], MetaData), data.__dict__.items())} for data in batch
    ]
    data_all: dict[str, torch.Tensor] = default_collate(data_all)
    posedata = PoseDataGen(**data_all, meta=meta_all)
    return posedata


if __name__ == "__main__":
    from time import perf_counter

    from torch.utils.data import DataLoader

    from util.utils import fix_random

    fix_random()
    dataset = MixamoDatasetGen(
        "../data/Mixamo",
        # extra_character_dir="../data/Mixamo/character_humanrig_refined",
        num_poses_source=1,
        num_poses_target=2,
        rest_as_target=True,
        num_sample_points=32768,
        sharp_edge_ratio=0.0,
        get_sdf=False,
        align_sample_points=True,
        dual_quat_skinning=False,
        random_fingers=True,
        skeleton_type="cylinders",
        remesh_type="project",
        remesh_method="alpha_wrap",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=collate,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )
    print(len(dataset))

    tic = perf_counter()
    for i, data in enumerate(tqdm(dataloader, dynamic_ncols=True)):
        print(perf_counter() - tic)
        data: PoseDataGen
        meta = data.meta
        trimesh.Scene([trimesh.PointCloud(data.target_pts[0, 0].numpy()), trimesh.PointCloud(data.target_skeleton[0, 0].numpy())]).export("test.glb")  # fmt:skip
        pass
