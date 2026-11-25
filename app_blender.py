import argparse

import numpy as np


def create_pc_object(points: np.ndarray, name="PointCloud"):
    import bpy

    assert points.ndim == 2 and points.shape[1] == 3, f"Expected points to be of shape (N, 3), got {points.shape}"
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(points.tolist(), [], [])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def preprocess(args: argparse.Namespace, output_path: str = None):
    import bpy
    import torch
    import torch.nn.functional as F

    import util.blender_utils as blender_utils
    from engine import GT, get_global_transforms
    from util.dataset_mixamo_mip import MixamoDatasetGen, PoseDataGen, collate
    from util.utils import HiddenPrints, apply_transform, fix_random

    fix_random()

    assert isinstance(args, argparse.Namespace)

    with HiddenPrints(suppress_err=False):
        blender_utils.reset()

        dataset = MixamoDatasetGen(
            data_dir=None,
            character_subdir=[args.input_path],
            animation_subdir=None,
            num_sample_points=args.num_sample_points,
            sharp_edge_ratio=False,
            num_poses_source=1,
            num_poses_target=1,
            num_target_points=2048 * 3,
            sample_frames=[0, -1],
            rest_as_target=False,
            split="val",
            get_sdf=False,
            align_sample_points=True,
            skeleton_type="cylinders",
            dual_quat_skinning=False,
            sample_bbox_extend=1.3,
            keep_raw_skeleton=args.keep_raw_skeleton,
        )
        data: PoseDataGen = collate([dataset[0]])

        global_transform_source, global_transform_target = get_global_transforms(data, aug_rotation=False)
        # global_transform_target = global_transform_target[:1, :1]
        global_transform_inv = torch.inverse(global_transform_target.squeeze(0))
        P = global_transform_inv.shape[0]
        matrix_world = torch.from_numpy(np.array(blender_utils.get_armature_obj().matrix_world)).unsqueeze(0)
        matrix_world = matrix_world.repeat(P, 1, 1)
        global_transform_inv = torch.bmm(matrix_world.to(global_transform_inv), global_transform_inv)
        rot_x_neg90 = torch.tensor([[[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]])
        rot_x_neg90 = rot_x_neg90.repeat(P, 1, 1)
        global_transform_inv = torch.bmm(rot_x_neg90.to(global_transform_source), global_transform_inv)

        gt = GT(data, global_transform_source, global_transform_target, device="cpu", model=None, args=args)
        source_inputs = gt.source_inputs.squeeze(0).squeeze(0).numpy()
        source_skeleton = gt.source_skeleton.squeeze(0).squeeze(0).numpy()
        source_bw = gt.source_bw.squeeze(0).squeeze(0).numpy()
        source_bw_onehot = F.one_hot(torch.argmax(gt.source_bw, dim=-1), num_classes=gt.source_bw.shape[-1])
        source_bw_onehot = source_bw_onehot.squeeze(0).squeeze(0).numpy()
        target_skeleton = gt.target_skeleton.squeeze(0).squeeze(0).numpy()
        global_transform_inv = global_transform_inv.squeeze(0).numpy()
        output_data = dict(
            source_inputs=source_inputs,
            source_skeleton=source_skeleton,
            source_bw=source_bw,
            target_skeleton=target_skeleton,
            global_transform_inv=global_transform_inv,
        )

        if output_path is not None:
            np.save(output_path, output_data)

        if args.input_vis_path is not None:
            mesh = data.meta[0].meshes[-1].copy()
            mesh.vertices = apply_transform(mesh.vertices, global_transform_source.squeeze(0).squeeze(0).numpy())
            mesh.export(args.input_vis_path)

        if args.skeleton_fbx_path is not None:
            for mesh_obj in blender_utils.get_all_mesh_obj():
                bpy.data.objects.remove(mesh_obj, do_unlink=True)
            armature_obj = blender_utils.get_armature_obj()
            armature_obj.animation_data_clear()
            with blender_utils.Mode("POSE", armature_obj):
                bpy.ops.pose.select_all(action="SELECT")
                bpy.ops.pose.transforms_clear()
            armature_obj.matrix_world = global_transform_source.squeeze(0).squeeze(0).numpy().T
            blender_utils.update()
            skeleton_obj = create_pc_object(gt.source_skeleton.squeeze(0).squeeze(0).numpy(), name="skeleton")
            blender_utils.set_armature_parent([skeleton_obj], armature_obj)
            blender_utils.set_weights([skeleton_obj], source_bw_onehot, data.meta[0].bones_idx_dict)
            blender_utils.update()
            bpy.ops.export_scene.fbx(
                filepath=args.skeleton_fbx_path,
                check_existing=False,
                use_selection=False,
                add_leaf_bones=False,
                bake_anim=False,
            )

        return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("func", type=str)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    input_args = np.load(args.input_path, allow_pickle=True).item()
    globals()[args.func](args=input_args, output_path=args.output_path)
