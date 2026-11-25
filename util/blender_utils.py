"""https://docs.blender.org/api/current/info_advanced_blender_as_bpy.html"""

import math
import os

import bpy
import numpy as np
from bpy.types import Action, Armature, Context, Mesh, Object

# isort: split
import bmesh
import mathutils

USE_WORLD_COORDINATES = False


class Mode:
    def __init__(self, mode_name="EDIT", active_obj: Object = None):
        self.mode = mode_name
        self.active = active_obj
        self.pre_active = None
        self.pre_mode = "OBJECT"

    def __enter__(self):
        self.pre_active = bpy.context.view_layer.objects.active
        if self.pre_active is not None:
            self.pre_mode = bpy.context.object.mode
        bpy.context.view_layer.objects.active = self.active
        bpy.ops.object.mode_set(mode=self.mode)
        return self.active

    def __exit__(self, exc_type, exc_val, exc_tb):
        bpy.ops.object.mode_set(mode=self.pre_mode)
        bpy.context.view_layer.objects.active = self.pre_active


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def update(context: Context = None):
    if context is None:
        context = bpy.context
    context.view_layer.update()
    context.scene.update_tag()
    for obj in context.scene.objects:
        # obj.hide_render = obj.hide_render
        obj.update_tag()


def remove_all(delete_actions=True):
    for obj in bpy.data.objects.values():
        bpy.data.objects.remove(obj, do_unlink=True)
    for coll in bpy.data.collections:
        bpy.data.collections.remove(coll)
    if delete_actions:
        for action in bpy.data.actions:
            bpy.data.actions.remove(action, do_unlink=True)
    bpy.ops.outliner.orphans_purge(do_recursive=True)


def remove_empty():
    childless_empties = [e for e in bpy.data.objects if e.type.startswith("EMPTY") and not e.children]
    bpy.data.batch_remove(childless_empties)
    for coll in bpy.data.collections:
        if len(coll.all_objects) == 0:
            bpy.data.collections.remove(coll)


def remove_collection(coll_name: str):
    if coll_name not in bpy.data.collections:
        return
    coll = bpy.data.collections[coll_name]
    for c in coll.children:
        remove_collection(c)
    bpy.data.collections.remove(coll, do_unlink=True)


def remove_unused_actions(ignore_protection=True):
    for action in bpy.data.actions:
        if action.users == 0 or (ignore_protection and action.users == 1 and action.use_fake_user):
            bpy.data.actions.remove(action)


def load_file(filepath: str, *args, **kwargs) -> "list[Object]":
    old_objs = set(bpy.context.scene.objects)
    if filepath.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=filepath, *args, **kwargs)
    elif filepath.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=filepath, *args, **kwargs)
    elif filepath.endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=filepath, *args, **kwargs)
    elif filepath.endswith(".ply"):
        bpy.ops.wm.ply_import(filepath=filepath, *args, **kwargs)
    else:
        raise RuntimeError(f"Invalid input file: {filepath}")
    imported_objs = set(bpy.context.scene.objects) - old_objs
    imported_objs = sorted(imported_objs, key=lambda x: x.name)
    print("Imported:", imported_objs)
    return imported_objs


def select_all():
    bpy.ops.object.select_all(action="SELECT")


def deselect():
    bpy.ops.object.select_all(action="DESELECT")


def select_objs(obj_list: "list[Object]" = None, deselect_first=False):
    if not obj_list:
        obj_list = bpy.context.scene.objects
    if deselect_first:
        deselect()
    for obj in obj_list:
        obj.select_set(True)


def select_mesh(obj_list: "list[Object]" = None, all=True, deselect_first=False):
    if not obj_list:
        obj_list = bpy.context.scene.objects
    if deselect_first:
        deselect()
    for obj in obj_list:
        if obj.type == "MESH":
            if all:
                obj.select_set(True)
            else:
                break


class Select:
    """
    Deselecting before and after selecting the specified objects.
    """

    def __init__(self, objs: "Object | list[Object]" = None):
        self.objs = (objs,) if isinstance(objs, Object) else objs
        self.objs: "tuple[Object]" = tuple(self.objs)

    def __enter__(self):
        select_objs(self.objs, deselect_first=True)
        return self.objs

    def __exit__(self, exc_type, exc_val, exc_tb):
        deselect()


def get_type_objs(obj_list: "list[Object]" = None, type="MESH", sort=True) -> "list[Object]":
    if not obj_list:
        obj_list = bpy.context.scene.objects
    type_obj_list = [obj for obj in obj_list if obj.type == type]
    if sort:
        type_obj_list = sorted(type_obj_list, key=lambda x: x.name)
    return type_obj_list


def get_all_mesh_obj(obj_list: "list[Object]" = None):
    return get_type_objs(obj_list, "MESH")


def get_all_armature_obj(obj_list: "list[Object]" = None):
    return get_type_objs(obj_list, "ARMATURE")


def get_armature_obj(obj_list: "list[Object]" = None) -> Object:
    if not obj_list:
        obj_list = bpy.context.scene.objects
    for obj in obj_list:
        if obj.type == "ARMATURE":
            return obj


def set_armature_parent(mesh_obj_list: "list[Object]", armature_obj: Object, type="ARMATURE", no_inv=False):
    with Select(mesh_obj_list):
        # the active object will be the parent of all selected objects
        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.parent_set(type=type)
        if no_inv:
            bpy.ops.object.parent_no_inverse_set(keep_transform=False)
    return armature_obj


def get_keyframes(obj_list: "list[Object]" = None, mute_global_anim=False) -> "list[int]":
    if not obj_list:
        obj_list = bpy.context.scene.objects
    keyframes = []
    for obj in obj_list:
        anim = obj.animation_data
        if anim is not None and anim.action is not None:
            if mute_global_anim and len(anim.action.groups) > 0:
                anim.action.groups[0].mute = True
            for fcu in anim.action.fcurves:
                for keyframe in fcu.keyframe_points:
                    x = keyframe.co.x
                    x = math.ceil(x)
                    if x not in keyframes:
                        keyframes.append(x)
        shape_keys = obj.data.shape_keys if hasattr(obj.data, "shape_keys") else None
        if shape_keys:
            action = shape_keys.animation_data.action if shape_keys.animation_data else None
            if action:
                for fcurve in action.fcurves:
                    if fcurve.data_path.startswith("key_blocks"):
                        for keyframe in fcurve.keyframe_points:
                            x = keyframe.co.x
                            x = math.ceil(x)
                            if x not in keyframes:
                                keyframes.append(x)
    return keyframes


def get_bones_idx_dict(armature_obj: Object):
    if armature_obj is None:
        return None
    bones_idx_dict: "dict[str, int]" = {bone.name: i for i, bone in enumerate(armature_obj.data.bones)}
    return bones_idx_dict


def get_rest_bones(armature_obj: Object):
    if armature_obj is None:
        return None, None, None
    rest_bones = []
    rest_bones_tail = []
    bones_idx_dict: "dict[str, int]" = {}
    armature_data: Armature = armature_obj.data
    for i, bone in enumerate(armature_data.bones):
        pos = bone.head_local
        pos_tail = bone.tail_local
        if USE_WORLD_COORDINATES:
            pos = armature_obj.matrix_world @ pos
            pos_tail = armature_obj.matrix_world @ pos_tail
        rest_bones.append(pos)
        rest_bones_tail.append(pos_tail)
        bones_idx_dict[bone.name] = i
    rest_bones = np.stack(rest_bones, axis=0)
    rest_bones_tail = np.stack(rest_bones_tail, axis=0)
    return rest_bones, rest_bones_tail, bones_idx_dict


def set_rest_bones(
    armature_obj: Object,
    head: np.ndarray = None,
    tail: np.ndarray = None,
    bones_idx_dict: "dict[str, int]" = None,
    remove_absent_bones=True,
    reset_as_rest=False,
):
    assert armature_obj is not None, "Armature object is None"
    armature_data: Armature = armature_obj.data

    if reset_as_rest:
        # select_mesh(armature_obj.children, all=True, deselect_first=True)
        mesh_list = []
        for obj in armature_obj.children:
            if obj.type != "MESH":
                continue
            mesh_list.append(obj)
            bpy.context.view_layer.objects.active = obj
            if obj.modifiers:
                bpy.ops.object.modifier_apply(modifier=obj.modifiers[0].name)
            with Select(obj):
                bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
        with Mode("POSE", armature_obj):
            bpy.ops.pose.armature_apply(selected=False)
        set_armature_parent(mesh_list, armature_obj)

    if head is not None:
        assert bones_idx_dict is not None
        with Mode("EDIT", armature_obj):
            bones_absent = []
            for bone in armature_data.edit_bones:
                bone.use_connect = False
                if bone.name not in bones_idx_dict:
                    bones_absent.append(bone)
            if remove_absent_bones:
                for bone in bones_absent:
                    armature_data.edit_bones.remove(bone)
            for bone in armature_data.edit_bones:
                if bone.name in bones_idx_dict:
                    bone_roll = bone.matrix.to_3x3().copy() @ mathutils.Vector((0.0, 0.0, 1.0))
                    bone.head = head[bones_idx_dict[bone.name]]
                    if tail is not None:
                        bone.tail = tail[bones_idx_dict[bone.name]]
                    bone.align_roll(bone_roll)
            # for bone in armature_data.edit_bones:
            #     if bone.parent is not None and len(bone.parent.children) == 1:
            #         bone.parent.tail = bone.head

    if reset_as_rest:
        with Mode("POSE", armature_obj):
            bpy.ops.pose.select_all(action="SELECT")
            bpy.ops.pose.transforms_clear()

    return armature_obj


def get_pose_bones(armature_obj: Object):
    if armature_obj is None:
        return None, None, None
    bones = []
    bones_tail = []
    bones_rotation_relative_to_posed = []
    bones_rotation_relative_to_rest = []
    bones_transform_global = []
    for bone in armature_obj.pose.bones:
        # pos = bone.matrix @ bone.location
        pos = bone.head
        pos_tail = bone.tail
        if USE_WORLD_COORDINATES:
            pos = armature_obj.matrix_world @ pos
            pos_tail = armature_obj.matrix_world @ pos_tail
        bones.append(np.array(pos))
        bones_tail.append(np.array(pos_tail))

        # rot_rel = bone.rotation_quaternion  # same as bone.matrix_basis.to_quaternion(), relative to posed parent in rest local coordinates
        # To armature coordinates:
        rot_rel = bone.bone.matrix_local @ bone.matrix_basis @ bone.bone.matrix_local.inverted()
        rot_rel_rest = rot_rel.copy()  # relative to rest parent
        if bone.parent is not None:
            parent_r2p = bone.parent.matrix @ bone.parent.bone.matrix_local.inverted()
            rot_rel = parent_r2p @ rot_rel @ parent_r2p.inverted()
        if USE_WORLD_COORDINATES:
            rot_rel = armature_obj.matrix_world @ rot_rel @ armature_obj.matrix_world.inverted()
            rot_rel_rest = armature_obj.matrix_world @ rot_rel_rest @ armature_obj.matrix_world.inverted()
        bones_rotation_relative_to_posed.append(np.array(rot_rel.to_quaternion()))
        bones_rotation_relative_to_rest.append(np.array(rot_rel_rest.to_quaternion()))

        # https://blender.stackexchange.com/questions/44637/how-can-i-manually-calculate-bpy-types-posebone-matrix-using-blenders-python-ap
        # PoseBone.head == PoseBone.matrix @ PoseBone.bone.matrix_local.inverted() @ PoseBone.bone.head_local
        # PoseBone.bone.matrix_local: initial (zero) pose to rest pose (in armature coordinates)
        # PoseBone.matrix: initial (zero) pose to current pose (in armature coordinates)
        matrix = bone.matrix @ bone.bone.matrix_local.inverted()
        # matrix: rest pose to current pose (in armature coordinates)
        if USE_WORLD_COORDINATES:
            # posed = matrix @ rest --> world @ posed = (world @ matrix @ world^(-1)) @ (world @ rest)
            matrix = armature_obj.matrix_world @ matrix @ armature_obj.matrix_world.inverted()
        bones_transform_global.append(matrix)

    bones = np.stack(bones, axis=0)
    bones_tail = np.stack(bones_tail, axis=0)
    bones_rotation_relative_to_posed = np.stack(bones_rotation_relative_to_posed, axis=0)
    bones_rotation_relative_to_rest = np.stack(bones_rotation_relative_to_rest, axis=0)
    bones_transform_global = np.stack(bones_transform_global, axis=0)
    return bones, bones_tail, bones_rotation_relative_to_posed, bones_rotation_relative_to_rest, bones_transform_global


def set_bone_pose(armature_obj: Object, pose: np.ndarray, bones_idx_dict: "dict[str, int]", local=False):
    assert armature_obj is not None, "Armature object is None"
    for bone in armature_obj.pose.bones:
        if bone.name in bones_idx_dict:
            bone_pose = pose[bones_idx_dict[bone.name]]
            print(f"{bone.name}: {bone_pose}")
            if local:
                bone.matrix_basis = mathutils.Quaternion(bone_pose).normalized().to_matrix().to_4x4()
            else:
                bone.matrix = mathutils.Matrix(bone_pose) @ bone.bone.matrix_local
            bpy.context.view_layer.update()
    return armature_obj


def get_evaluated_mesh(mesh_obj: Object):
    mesh: Mesh = mesh_obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    return mesh


def get_evaluated_vertices(mesh_obj: Object):
    # mesh = get_evaluated_mesh(mesh_obj)
    # if USE_WORLD_COORDINATES:
    #     verts = np.array([mesh_obj.matrix_world @ v.co for v in mesh.vertices])
    # else:
    #     verts = np.array([v.co for v in mesh.vertices])

    depsgraph = bpy.context.evaluated_depsgraph_get()
    bm = bmesh.new()
    bm.from_object(mesh_obj, depsgraph)
    bm.verts.ensure_lookup_table()
    if USE_WORLD_COORDINATES:
        verts = np.array([mesh_obj.matrix_world @ v.co for v in bm.verts])
    else:
        verts = np.array([v.co for v in bm.verts])

    # vert_idx = []
    # for f in bm.faces:
    #     vert_idx.append([v.index for v in f.verts])
    # assert all((a==b).all() for a, b in zip(get_faces(mesh_obj), vert_idx))
    # import trimesh; trimesh.Trimesh(verts, vert_idx, process=False, maintain_order=True).export(file_obj="test.ply")  # fmt: skip

    bm.free()
    return verts


def get_faces(mesh_obj: Object):
    bm = bmesh.new()
    bm.from_mesh(mesh_obj.data)
    bm.faces.ensure_lookup_table()
    vert_idx = [[v.index for v in f.verts] for f in bm.faces]
    bm.free()
    assert len(set(map(len, vert_idx))) == 1, "All faces should be triangles"
    return np.array(vert_idx)


def get_rest_vertices(mesh_obj_list: "list[Object]", bones_idx_dict: "dict[str, int]" = None):
    verts_all = []
    bw_all = []
    faces_all = []
    for mesh_obj in mesh_obj_list:
        mesh_data: Mesh = mesh_obj.data
        verts = mesh_data.vertices
        if bones_idx_dict is not None:
            weights = {}
            for v in verts:
                for group in v.groups:
                    weights.setdefault(group.group, {})[v.index] = group.weight
            bw = np.zeros((len(verts), len(bones_idx_dict)))
            for bone_index, vertex_weights in weights.items():
                # print(f"Bone {bone_index} ({mesh_obj.vertex_groups[bone_index].name})")
                for vert_index, vert_weight in sorted(vertex_weights.items(), key=lambda x: x[1], reverse=True):
                    # print(f"Vertex {vert_index} = {vert_weight}")
                    bw[vert_index, bones_idx_dict[mesh_obj.vertex_groups[bone_index].name]] = vert_weight
            bw_all.append(bw)
        if USE_WORLD_COORDINATES:
            verts_pos = np.array([mesh_obj.matrix_world @ v.co for v in mesh_data.vertices])
        else:
            verts_pos = np.array([v.co for v in mesh_data.vertices])
        verts_all.append(verts_pos)
        faces = get_faces(mesh_obj)
        faces_all.append(faces)
    if verts_all:
        verts_nums = np.cumsum(list(map(len, verts_all)))
        verts_all = np.concatenate(verts_all, axis=0)
        if faces_all:
            for i in range(len(faces_all)):
                if i >= 1:
                    faces_all[i] += verts_nums[i - 1]
            faces_all = np.concatenate(faces_all, axis=0)
        else:
            faces_all = None
    else:
        verts_all = None
        faces_all = None
    bw_all = np.concatenate(bw_all, axis=0) if bw_all else None
    return verts_all, faces_all, bw_all


def transfer_weights(source_bone_name: str, target_bone_name: str, mesh_obj_list: "list[Object]"):
    if isinstance(mesh_obj_list, Object):
        mesh_obj_list = [mesh_obj_list]
    for obj in mesh_obj_list:
        source_group = obj.vertex_groups.get(source_bone_name)
        if source_group is None:
            return
        source_i = source_group.index
        target_group = obj.vertex_groups.get(target_bone_name)
        if target_group is None:
            target_group = obj.vertex_groups.new(name=target_bone_name)

        for v in obj.data.vertices:
            for g in v.groups:
                if g.group == source_i:
                    target_group.add((v.index,), g.weight, "ADD")
        obj.vertex_groups.remove(source_group)


def remove_empty_vgroups(mesh_obj_list: "list[Object]"):
    if isinstance(mesh_obj_list, Object):
        mesh_obj_list = [mesh_obj_list]
    for obj in mesh_obj_list:
        vertex_groups = obj.vertex_groups
        groups = {r: None for r in range(len(vertex_groups))}

        for vert in obj.data.vertices:
            for vg in vert.groups:
                i = vg.group
                if i in groups:
                    del groups[i]

        lis = list(groups)
        lis.sort(reverse=True)
        for i in lis:
            vertex_groups.remove(vertex_groups[i])


def set_weights(mesh_obj_list: "list[Object]", weights: np.ndarray, bones_idx_dict: "dict[str, int]"):
    assert len(mesh_obj_list) > 0, "No mesh object"
    vertices_num = [len(mesh_obj.data.vertices) for mesh_obj in mesh_obj_list]
    assert sum(vertices_num) == weights.shape[0], "The number of vertices does not match the number of weights"
    weights_list = np.split(weights, np.cumsum(vertices_num)[:-1])
    # assert list(map(len, weights_list)) == vertices_num
    for mesh_obj, bw in zip(mesh_obj_list, weights_list):
        mesh_data: Mesh = mesh_obj.data
        mesh_obj.vertex_groups.clear()
        for bone_name, bone_index in bones_idx_dict.items():
            group = mesh_obj.vertex_groups.new(name=bone_name)
            for v in mesh_data.vertices:
                v_w = bw[v.index, bone_index]
                if v_w > 1e-3:
                    group.add([v.index], v_w, "REPLACE")
        mesh_data.update()
    return mesh_obj_list


def get_pose_vertices(mesh_obj_list: "list[Object]"):
    verts_deformed_all = []
    for mesh_obj in mesh_obj_list:
        verts_deformed = get_evaluated_vertices(mesh_obj)
        verts_deformed_all.append(verts_deformed)
    verts_deformed_all = np.concatenate(verts_deformed_all, axis=0) if verts_deformed_all else None
    return verts_deformed_all


def get_shape_keys(mesh_obj: Object, ignore_basis=True, ignore_empty=False):
    if mesh_obj is None:
        return None
    assert mesh_obj.type == "MESH"
    mesh: Mesh = mesh_obj.data
    shape_keys: dict[str, np.ndarray] = {}
    if not mesh.shape_keys:
        return shape_keys
    key_blocks = mesh.shape_keys.key_blocks
    num_vertices = len(mesh.vertices)

    basis_kb = key_blocks[0]
    assert all(
        kb.relative_key == basis_kb for kb in key_blocks
    ), "Not all shape keys are relative to the basis shape key"
    basis_co = np.empty(num_vertices * 3, dtype=np.float64)
    basis_kb.points.foreach_get("co", basis_co)
    basis_co = basis_co.reshape(num_vertices, 3)

    for i, kb in enumerate(key_blocks):
        co = np.empty(num_vertices * 3, dtype=np.float64)
        kb.points.foreach_get("co", co)
        co = co.reshape(num_vertices, 3)
        delta = co - basis_co
        if ignore_basis and i == 0:
            continue
        if ignore_empty and i != 0 and np.abs(delta).max() < 1e-6:
            continue
        shape_keys[kb.name] = delta

    return shape_keys


def set_shape_keys(mesh_obj: Object, shape_keys: dict[str, np.ndarray], clear_existing=True):
    assert mesh_obj and mesh_obj.type == "MESH"
    mesh: Mesh = mesh_obj.data
    n_vertices = shape_keys[next(iter(shape_keys))].shape[0]
    assert (
        len(mesh.vertices) == n_vertices
    ), f"Vertex number mismatch: {n_vertices} (from input) v.s. {len(mesh.vertices)} (from mesh)"

    if mesh.shape_keys and clear_existing:
        for kb in mesh.shape_keys.key_blocks[:][::-1]:
            mesh_obj.shape_key_remove(kb)
    if not mesh.shape_keys:
        mesh_obj.shape_key_add(name="Basis", from_mix=False)
    basis_kb = mesh.shape_keys.key_blocks[0]

    basis_co = np.empty(n_vertices * 3, dtype=np.float64)
    basis_kb.points.foreach_get("co", basis_co)
    basis_co = basis_co.reshape(n_vertices, 3)

    for name, delta in shape_keys.items():
        if name in mesh.shape_keys.key_blocks:
            kb = mesh.shape_keys.key_blocks[name]
        else:
            kb = mesh_obj.shape_key_add(name=name)

        kb.relative_key = basis_kb
        new_co = basis_co + delta
        new_co_flat = new_co.ravel()
        kb.points.foreach_set("co", new_co_flat)
        kb.points.update()

    mesh_obj.show_only_shape_key = False
    mesh.update()
    return mesh


def transfer_all_shape_keys(mesh_src: Object, mesh_tgt: Object, clear_existing=True):
    assert mesh_src and mesh_src.type == "MESH"
    assert mesh_tgt and mesh_tgt.type == "MESH"
    if mesh_src.data.shape_keys is None:
        raise ValueError("Source object has no shape keys!")
    if mesh_tgt.data.shape_keys and clear_existing:
        for kb in mesh_tgt.data.shape_keys.key_blocks[:][::-1]:
            mesh_tgt.shape_key_remove(kb)
    bpy.context.view_layer.objects.active = mesh_tgt
    with Select(mesh_src):
        for idx in range(1, len(mesh_src.data.shape_keys.key_blocks)):
            mesh_src.active_shape_key_index = idx
            print(f"Copying Shape Key: {mesh_src.active_shape_key.name}")
            bpy.ops.object.shape_key_transfer()
    mesh_tgt.data.update()
    mesh_tgt.show_only_shape_key = False
    return mesh_tgt


def set_action(armature_obj: Object, action: Action):
    if not armature_obj.animation_data:
        armature_obj.animation_data_create()
    armature_obj.animation_data.action = action
    if hasattr(armature_obj.animation_data, "action_slot"):
        # For Blender 4.4 and later
        armature_obj.animation_data.action_slot = armature_obj.animation_data.action_suitable_slots[0]
    return armature_obj


def mesh_quads2tris(obj_list: "list[Object]" = None):
    if not obj_list:
        obj_list = bpy.context.scene.objects
    for obj in obj_list:
        if obj.type == "MESH":
            with Mode("EDIT", obj):
                bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")


def get_enabled_addons() -> "list[str]":
    return [x.module for x in bpy.context.preferences.addons]


def enable_arp(armature_obj: Object, addon_path=os.path.join(os.path.dirname(__file__), "auto_rig_pro")):
    import sys

    assert os.path.isfile(os.path.join(addon_path, "__init__.py")), "Auto-Rig Pro not found"
    dirname, addon_name = os.path.split(addon_path)
    if addon_name in get_enabled_addons():
        return
    sys.path.insert(0, dirname)
    with Mode("POSE", armature_obj):
        # import addon_utils
        # addon_utils.enable(addon_name)
        bpy.ops.preferences.addon_enable(module=addon_name)


def retarget(source_armature: Object, target_armature: Object, inplace=False):
    enable_arp(target_armature)
    scn = bpy.context.scene
    scn.source_rig = source_armature.name
    if inplace:
        scn.arp_retarget_in_place = True
    scn.target_rig = target_armature.name
    bpy.ops.arp.auto_scale()
    bpy.ops.arp.build_bones_list()
    hips = scn.bones_map_v2["mixamorig:Hips"]
    scn.bones_map_index = list(scn.bones_map_v2).index(hips)
    hips.set_as_root = True
    bpy.ops.arp.retarget()
    return target_armature


def enable_3dgs_render(addon_path=os.path.join(os.path.dirname(__file__), "3dgs-render-blender-addon")):
    import sys

    assert os.path.isfile(os.path.join(addon_path, "__init__.py")), "3DGS Render Blender Addon not found"
    dirname, addon_name = os.path.split(addon_path)
    if addon_name in get_enabled_addons():
        return
    sys.path.insert(0, dirname)
    # import addon_utils
    # addon_utils.enable(addon_name)
    bpy.ops.preferences.addon_enable(module=addon_name)


def load_3dgs(filepath: str) -> "list[Object]":
    enable_3dgs_render()
    old_objs = set(bpy.context.scene.objects)
    bpy.ops.sna.dgs__import_ply_as_splats_8458e(filepath=filepath)
    bpy.ops.sna.dgs__stop_camera_update_9ad85()
    imported_objs = set(bpy.context.scene.objects) - old_objs
    imported_objs = sorted(imported_objs, key=lambda x: x.name)
    print("Imported:", imported_objs)
    return imported_objs


def load_mixamo_anim(char_file: str, anim_file: str, do_retarget=False, inplace=False, to_tris=False):
    char_objs = load_file(char_file) if isinstance(char_file, str) else char_file
    char_armature = get_armature_obj(char_objs)

    if anim_file is not None:
        anim_objs = load_file(anim_file)
        anim_armature = get_armature_obj(anim_objs)
        print(anim_armature)
        print(anim_armature.animation_data)
        assert (
            anim_armature.animation_data is not None and len(bpy.data.actions) > 0
        ), f"Animation not found in {anim_file}"

        set_action(char_armature, anim_armature.animation_data.action)
        if do_retarget:
            retarget(anim_armature, char_armature, inplace=inplace)
            for action in bpy.data.actions:
                if action is not char_armature.animation_data.action:
                    bpy.data.actions.remove(action, do_unlink=True)
        for obj in anim_objs:
            bpy.data.objects.remove(obj, do_unlink=True)

    if to_tris:
        mesh_quads2tris(char_objs)
    return char_objs


if __name__ == "__main__":
    bpy.ops.wm.read_factory_settings(use_empty=False)
    obj_list: "list[Object]" = list(bpy.context.scene.objects)
    print(obj_list)
    obj = obj_list[0]
    print(obj)
