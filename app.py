import gc
import os
import queue
import sys
import tempfile
import threading
import time
import warnings
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from glob import glob

import gradio as gr
import numpy as np
import torch
import trimesh

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from util.utils import TimePrints, Timing, apply_transform, fix_random, remesh

# Task queue for blender tasks to be executed in the main thread
blender_task_queue = queue.Queue()


def is_main_thread():
    import threading

    return threading.current_thread() is threading.main_thread()


# Monkey patching gradio to use let gr.Info & gr.Warning also print on console
def _log_message(
    message: str,
    level="info",
    duration: float | None = 10,
    visible: bool = True,
    *args,
    **xargs,
):
    from gradio.context import LocalContext

    if level in ("info", "success"):
        print(message)
    elif level == "warning":
        warnings.warn(message)

    blocks = LocalContext.blocks.get(None)
    event_id = LocalContext.event_id.get(None)
    if blocks is not None and event_id is not None:
        # Function called outside of Gradio if blocks is None
        # Or from /api/predict if event_id is None
        blocks._queue.log_message(
            event_id=event_id, log=message, level=level, duration=duration, visible=visible, *args, **xargs
        )


import gradio.helpers

gradio.helpers.log_message = _log_message


@dataclass()
class DB:
    output_dir: str = None
    input_path: str = None
    input_vis_path: str = None
    source_pose_path: str = None
    source_pose_fbx_path: str = None
    source_pose_vis_path: str = None
    target_pose_path: str = None
    target_pose_vis_path: str = None
    output_path: str = None

    source_inputs: np.ndarray = None
    source_skeleton: np.ndarray = None
    target_skeleton: np.ndarray = None
    source_bw: np.ndarray = None
    global_transform_inv: np.ndarray = None

    def clear(self):
        for k in self.__dict__:
            self.__dict__[k] = None
        return self


def clear(db: DB = None):
    if db is not None:
        db.clear()
    gc.collect()
    torch.cuda.empty_cache()
    print("Memory cleared")
    return db


def prepare_input(input_path: str, db: DB, keep_raw_skeleton=False, export_temp=False):
    if not (input_path and os.path.isfile(input_path)):
        raise gr.Error(f"Input file not found: '{input_path}', please re-upload the file")
    print(f"{input_path=}")
    db.input_path = input_path

    if export_temp:
        output_dir = tempfile.mkdtemp()
    else:
        output_dir = os.path.join(os.path.dirname(input_path), os.path.splitext(os.path.basename(input_path))[0])
        os.makedirs(output_dir, exist_ok=True)
    db.output_dir = output_dir
    db.input_vis_path = os.path.join(output_dir, "input.glb")
    db.source_pose_path = os.path.join(output_dir, "source_pose.ply")
    db.source_pose_fbx_path = os.path.join(output_dir, "source_pose.fbx")
    db.source_pose_vis_path = os.path.join(output_dir, "source_pose.glb")
    db.target_pose_path = os.path.join(output_dir, "target_pose.ply")
    db.target_pose_vis_path = os.path.join(output_dir, "target_pose.glb")
    db.output_path = os.path.join(output_dir, "output.glb")

    with Timing(msg="Pre-process done in", print_fn=gr.Success):
        input_data = dict(
            input_path=input_path,
            num_sample_points=N,
            skeleton_fbx_path=db.source_pose_fbx_path,
            ae_type=model.ae_type,
            input_vis_path=db.input_vis_path,
            keep_raw_skeleton=keep_raw_skeleton,
        )
        input_data = Namespace(**input_data)
        if is_main_thread():
            from app_blender import preprocess

            output_data = preprocess(args=input_data)
        else:
            # Pass the preprocess task to the main thread through queue
            event = threading.Event()
            result_box = []
            blender_task_queue.put(("preprocess", input_data, event, result_box))
            # Wait until the main thread finishes executing the blender task
            event.wait()
            # If an exception happened during main thread execution, raise it here
            if len(result_box) > 0 and isinstance(result_box[0], Exception):
                raise gr.Error(f"Blender failed: {str(result_box[0])}")
            output_data = result_box[0]

    db.source_inputs = output_data["source_inputs"]
    db.source_skeleton = output_data["source_skeleton"]
    db.target_skeleton = output_data.get("target_skeleton", db.source_skeleton.copy())
    db.source_bw = output_data["source_bw"]
    db.global_transform_inv = output_data.get("global_transform_inv")

    trimesh.PointCloud(db.source_skeleton).export(db.source_pose_path)
    trimesh.PointCloud(db.source_skeleton).export(db.source_pose_vis_path)
    trimesh.PointCloud(db.target_skeleton).export(db.target_pose_path)

    return {
        state: db,
        input_3d_vis: db.input_vis_path,
        input_source_pose: [db.source_pose_path, db.source_pose_fbx_path],
        input_source_pose_vis: db.source_pose_vis_path,
        input_target_pose: db.target_pose_path,
    }


@torch.no_grad()
def model_forward(
    source_inputs: torch.Tensor,
    source_skeleton: torch.Tensor,
    target_skeleton: torch.Tensor,
    source_bw: torch.Tensor,
    octree_resolution=512,
    perturb_hands=False,
    skip_add=False,
    skip_transformer=False,
) -> trimesh.Trimesh:
    target_mesh = model.to_mesh(
        source_inputs.to(device),
        source_skeleton=source_skeleton.to(device),
        target_skeleton=target_skeleton.to(device),
        source_bw=source_bw.to(device),
        skip_transformer=skip_transformer,
        octree_resolution=octree_resolution,
        perturb_hands=perturb_hands,
        skip_add=skip_add,
        post_process=True,
    )[0][0]
    return target_mesh


def postprocess(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = remesh(mesh, alpha_fraction=0.002)
    mesh = trimesh.smoothing.filter_taubin(mesh, iterations=20)
    return mesh


def infer(
    db: DB,
    octree_resolution=512,
    refine_hands=False,
    skip_add=False,
    skip_transformer=False,
    restore_global=True,
    do_postprocess=False,
):
    source_inputs = torch.from_numpy(db.source_inputs).unsqueeze(0).unsqueeze(0)
    source_skeleton = torch.from_numpy(db.source_skeleton).unsqueeze(0).unsqueeze(0)
    target_skeleton = torch.from_numpy(db.target_skeleton).unsqueeze(0).unsqueeze(0)
    source_bw = torch.from_numpy(db.source_bw).unsqueeze(0).unsqueeze(0)

    with Timing(msg="Model inference done in", print_fn=gr.Info):
        target_mesh = model_forward(
            source_inputs,
            source_skeleton,
            target_skeleton,
            source_bw,
            octree_resolution=octree_resolution,
            perturb_hands=refine_hands,
            skip_add=skip_add,
            skip_transformer=skip_transformer,
        )
        if restore_global and db.global_transform_inv is not None:
            target_mesh.vertices = apply_transform(target_mesh.vertices, db.global_transform_inv)

    if do_postprocess:
        target_mesh = postprocess(target_mesh)
    target_mesh.export(db.output_path)
    return {output_3d: db.output_path}


def finish(db: DB = None):
    if db is not None and db.output_dir and os.path.isdir(db.output_dir):
        print(f"Outputs stored in '{db.output_dir}'")
    clear(db)
    return {state: gr.skip() if db is None else db}


@Timing(msg="All done in", print_fn=gr.Success)
def _pipeline(
    db: DB,
    octree_resolution=512,
    refine_hands=False,
    skip_add=False,
    skip_transformer=False,
    restore_global=True,
    do_postprocess=False,
):
    if db is None or db.source_inputs is None:
        raise gr.Error("Please complete `Preprocess` first")
    with TimePrints():
        print("*" * 50)
    # Magic sleep to fix the random pydantic_core._pydantic_core.ValidationError in Gradio: https://github.com/gradio-app/gradio/issues/9366#issuecomment-2412903101
    time.sleep(0.1)
    yield infer(
        db=db,
        octree_resolution=octree_resolution,
        refine_hands=refine_hands,
        skip_add=skip_add,
        skip_transformer=skip_transformer,
        restore_global=restore_global,
        do_postprocess=do_postprocess,
    )
    time.sleep(0.1)
    yield finish(db=None)  # keep the outputs for possible re-animation later


def init_models(model_path: str):
    from model import MIP

    global device, N, model

    if "model" not in globals() or model is None:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        fix_random(1)

        N = 32768

        model = MIP(
            ae_type="hy3d2.1",
            encoding_type="mean_var",
            transformer_type="decoder",
            N=N,
            sharp_edge_ratio=0.0,
            input_bw=False,
            use_bone_embedding=False,
            use_bone_embedding_bw=False,
            use_displ_embedding=False,
            use_point_embedding=False,
            use_latent_embedding=True,
            num_latents_add=512,
            pos_latents_add="post_transformer",
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load(model_path)
    gr.Success("Model loaded successfully!")
    model.to(device).eval()

    clear()


def init_blocks():
    global demo, state, input_3d, input_3d_vis, input_source_pose, input_target_pose, input_source_pose_vis, input_target_pose_vis, output_3d

    title = "Make-It-Poseable"
    description = f"""
    <center>
    <h1> {title} </h1>
    </center>
    """
    camera_position = [90, None, 2.5]

    with gr.Blocks(title=title, delete_cache=(3600, 3660)) as demo:
        gr.Markdown(description)

        state = gr.State(DB())
        with gr.Row():

            # Inputs
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        input_3d = gr.File(
                            label="Input 3D Model", file_types=[".fbx"], file_count="single", interactive=True
                        )
                        with gr.Accordion("Settings", open=False):
                            octree_resolution = gr.Dropdown(
                                label="Octree Resolution", choices=[256, 512, 1024], value=512, interactive=True
                            )
                            refine_hands = gr.Checkbox(label="Refine Hands", value=False, interactive=True)
                            skip_add = gr.Checkbox(label="Skip Additional Latents", value=False, interactive=True)
                            skip_transformer = gr.Checkbox(
                                label="Skip Transformer",
                                info="Sampling & VAE encoding-decoding only",
                                value=False,
                                interactive=True,
                            )
                            do_postprocess = gr.Checkbox(label="Postprocess", value=False, interactive=True)
                            restore_global = gr.Checkbox(
                                label="Restore Global Transform",
                                info="Restore all the output assets to the initial input coordinates (otherwise will be normalized).",
                                value=True,
                                interactive=True,
                            )
                            keep_raw_skeleton = gr.Checkbox(
                                label="Keep Raw Skeleton",
                                info="zero-shot for non-bipedal characters",
                                value=False,
                                interactive=True,
                            )
                    with gr.Column():
                        input_3d_vis = gr.Model3D(
                            label="Input 3D Model",
                            display_mode="solid",
                            camera_position=camera_position,
                            interactive=False,
                        )

                with gr.Row():
                    input_source_pose = gr.File(
                        label="Input Source Pose",
                        file_types=[".ply", ".fbx"],
                        file_count="multiple",
                        interactive=False,
                    )
                    input_target_pose = gr.File(
                        label="Input Target Pose", file_types=[".ply"], file_count="single", interactive=True
                    )
                with gr.Row():
                    input_source_pose_vis = gr.Model3D(
                        label="Input Source Pose",
                        display_mode="point_cloud",
                        camera_position=camera_position,
                        interactive=False,
                    )
                    input_target_pose_vis = gr.Model3D(
                        label="Input Target Pose",
                        display_mode="point_cloud",
                        camera_position=camera_position,
                        interactive=False,
                    )

                with gr.Row():
                    preprocess_btn = gr.Button("Preprocess", variant="secondary")
                    submit_btn = gr.Button("Run", variant="primary")
                with gr.Row():
                    stop_btn = gr.Button("Stop", variant="stop")
                    clear_btn = gr.ClearButton()

                if example_paths := sorted(glob("./data/examples/*.fbx")):
                    gr.Examples(
                        examples=example_paths,
                        inputs=[input_3d],
                        label="Examples",
                        cache_examples=False,
                        examples_per_page=20,
                    )

            inputs = (
                input_3d,
                octree_resolution,
                refine_hands,
                skip_add,
                skip_transformer,
                restore_global,
                keep_raw_skeleton,
                do_postprocess,
                input_3d_vis,
                input_source_pose,
                input_target_pose,
                input_source_pose_vis,
                input_target_pose_vis,
            )

            # Outputs
            with gr.Column():
                with gr.Row():
                    output_3d = gr.Model3D(
                        label="Output 3D Model",
                        display_mode="solid",
                        camera_position=camera_position,
                        interactive=False,
                    )

            outputs = (output_3d,)

            # Events

            def clear_components(inputs: dict):
                return [None] * len(inputs) if len(inputs) > 1 else None

            def prepare_target_skeleton(ply_path: str, db: DB):
                if not ply_path or db is None or db.target_pose_path is None or db.target_pose_vis_path is None:
                    return {input_target_pose_vis: None}
                pc: trimesh.PointCloud = trimesh.load(ply_path, process=False, maintain_order=True)
                db.target_skeleton = pc.vertices.astype(db.source_skeleton.dtype)
                pc.export(db.target_pose_path)
                pc.export(db.target_pose_vis_path)
                return {input_target_pose: db.target_pose_path, input_target_pose_vis: db.target_pose_vis_path}

            def pipeline(inputs: dict, progress=gr.Progress()):
                progress(0, "Starting...")
                if device.type == "cpu":
                    gr.Warning("Running on CPU will take a much longer time", duration=None)

                yield from progress.tqdm(
                    _pipeline(
                        db=inputs[state],
                        octree_resolution=inputs[octree_resolution],
                        refine_hands=inputs[refine_hands],
                        skip_add=inputs[skip_add],
                        skip_transformer=inputs[skip_transformer],
                        restore_global=inputs[restore_global],
                        do_postprocess=inputs[do_postprocess],
                    )
                )
                # gr.Success("Finished successfully!")

            input_target_pose.change(
                fn=prepare_target_skeleton,
                inputs=[input_target_pose, state],
                outputs={input_target_pose, input_target_pose_vis},
            )

            preprocess_event = preprocess_btn.click(
                fn=prepare_input,
                inputs=[input_3d, state, keep_raw_skeleton],
                outputs=set(inputs + (state,)),
            )

            submit_event = submit_btn.click(fn=clear_components, inputs=set(outputs), outputs=outputs).success(
                fn=pipeline, inputs=set(inputs + (state,)), outputs=set(outputs + (state,))
            )

            stop_btn.click(fn=lambda: [], cancels=[preprocess_event, submit_event]).success(
                fn=lambda: gr.Warning("Job cancelled") or []
            )
            clear_btn.click(fn=clear_components, inputs=set(outputs), outputs=outputs)
            # .success(
            #     fn=clear, inputs=state, outputs=state
            # )

            demo.unload(fn=lambda: not clear(state.value) or None)  # just to make sure fn returns None

    return demo


if __name__ == "__main__":
    import bpy

    parser = ArgumentParser()
    parser.add_argument("--checkpoint", default="output/mip/completion", help="Stage 2 checkpoint file or directory")
    args = parser.parse_args()
    init_models(os.path.abspath(args.checkpoint))
    demo = init_blocks()

    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        allowed_paths=[os.path.abspath("data/examples")],
        show_error=True,
        ssr_mode=False,
        share=False,
        prevent_thread_lock=True,
    )

    # Enter the main thread consumer loop to execute Blender rendering tasks
    import app_blender

    print("Main thread entering Blender task event loop...")
    while True:
        try:
            # Block until there's a task.
            task = blender_task_queue.get()
            if task is None:
                continue
            task_type, args, event, result_box = task

            try:
                if task_type == "preprocess":
                    output_data = app_blender.preprocess(args)
                    result_box.append(output_data)
                # Expand for other backend tasks if needed
            except Exception as e:
                import traceback

                traceback.print_exc()
                result_box.append(e)

            # Unblock the Gradio thread
            event.set()
        except KeyboardInterrupt:
            print("Interrupted by user. Exiting task loop...")
            break
