import argparse
import gc
import os
import signal
import sys
import warnings
from contextlib import suppress
from datetime import datetime
from functools import lru_cache, partial, wraps
from time import perf_counter
from typing import Callable, ParamSpec, TypeVar

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from pytorch3d.transforms import Transform3d

Tensor_or_Array = TypeVar("Tensor_or_Array", torch.Tensor, np.ndarray)


class HiddenPrints:
    def __init__(self, enable=True, suppress_err=False, use_fd=False):
        self.enabled = enable
        self.suppress_err = suppress_err
        self.use_fd = use_fd

        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr

        self._null_out = None
        self._null_err = None
        self._saved_fds = []
        self._null_fds = []

    def __enter__(self):
        if not self.enabled:
            return

        if self.use_fd:
            null_fd_out = os.open(os.devnull, os.O_RDWR)
            self._null_fds.append(null_fd_out)
            saved_fd_out = os.dup(1)
            self._saved_fds.append(saved_fd_out)
            os.dup2(null_fd_out, 1)
            if self.suppress_err:
                null_fd_err = os.open(os.devnull, os.O_RDWR)
                self._null_fds.append(null_fd_err)
                saved_fd_err = os.dup(2)
                self._saved_fds.append(saved_fd_err)
                os.dup2(null_fd_err, 2)

        else:
            self._null_out = open(os.devnull, "w")
            sys.stdout = self._null_out
            if self.suppress_err:
                self._null_err = open(os.devnull, "w")
                sys.stderr = self._null_err

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return

        if self.use_fd:
            os.dup2(self._saved_fds[0], 1)
            if self.suppress_err:
                os.dup2(self._saved_fds[1], 2)
            for fd in self._null_fds + self._saved_fds:
                os.close(fd)

        else:
            sys.stdout = self._original_stdout
            if self._null_out:
                self._null_out.close()
            if self.suppress_err:
                sys.stderr = self._original_stderr
                if self._null_err:
                    self._null_err.close()


class TimePrints:
    def __init__(self, enable=True):
        self.enabled = enable
        self._original_stdout = sys.stdout

    def write(self, text):
        if self.enabled:
            timestamp = datetime.now().strftime("%m/%d %H:%M:%S")
            self._original_stdout.write(text.replace("\n", " [{}]\n".format(str(timestamp))))
        else:
            self._original_stdout.write(text)

    def flush(self):
        self._original_stdout.flush()

    def enable(self):
        self.enabled = True
        sys.stdout = self

    def disable(self):
        self.enabled = False
        sys.stdout = self._original_stdout

    def __enter__(self):
        self.enable()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disable()


class Timing:
    def __init__(self, enable=True, msg="Time cost:", repeat_times=1, print_fn=print, print_even_on_error=False):
        self.enable = enable
        self.msg = str(msg)
        self.repeat_times = repeat_times
        if not isinstance(print_fn, (list, tuple)):
            print_fn = [print_fn]
        self.print_fn = tuple(print_fn)
        self.print_even_on_error = print_even_on_error

    def __enter__(self):
        if self.enable:
            self.tic = perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enable and (exc_type is None or self.print_even_on_error):
            self.toc = perf_counter()
            info = f"{self.msg} {(self.toc - self.tic) / self.repeat_times:.2f}s"
            for fn in self.print_fn:
                fn(info)

    T = TypeVar("T")

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        import inspect

        @wraps(func)
        def wrapped_function(*args, **kwargs):
            with self:
                output = func(*args, **kwargs)
                if inspect.isgeneratorfunction(func):
                    yield from output
                else:
                    return output

        return wrapped_function


P = ParamSpec("P")
R = TypeVar("R")


def robust_dataloader(func: Callable[P, R]) -> Callable[P, R]:
    """
    A generic decorator that seamlessly intercepts DataLoader and injects breakpoint resume capability,
    defending against OS-level crashes (e.g., bpy Segmentation fault) during multiprocessing.
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:

        class _CachedSampler:
            """A dummy sampler that yields from a pre-cached sequence."""

            def __init__(self, sequence):
                self.sequence = sequence

            def __iter__(self):
                return iter(self.sequence)

            def __len__(self):
                return len(self.sequence)

        class RobustWrapper:
            def __init__(self, original_dl: torch.utils.data.DataLoader):
                self.original_dl = original_dl
                self.dataset = original_dl.dataset
                self.target_length = len(original_dl)
                self.epoch_sequence = None

            def __len__(self):
                return self.target_length

            def _create_loader(self, skip_batches: int):
                dl_kwargs = {"dataset": self.dataset}

                if self.original_dl.batch_sampler is not None:
                    if self.epoch_sequence is None:
                        self.epoch_sequence = list(self.original_dl.batch_sampler)
                    remaining = self.epoch_sequence[skip_batches:]
                    if not remaining:
                        return None
                    dl_kwargs["batch_sampler"] = _CachedSampler(remaining)
                else:
                    if self.epoch_sequence is None:
                        self.epoch_sequence = list(self.original_dl.sampler)
                    remaining = self.epoch_sequence[skip_batches:]
                    if not remaining:
                        return None
                    dl_kwargs["sampler"] = _CachedSampler(remaining)
                    dl_kwargs["batch_size"] = None

                # Inherit valid properties from the original dataloader
                valid_keys = [
                    "num_workers",
                    "collate_fn",
                    "pin_memory",
                    "timeout",
                    "worker_init_fn",
                    "multiprocessing_context",
                    "generator",
                    "persistent_workers",
                    "pin_memory_device",
                ]
                for key in valid_keys:
                    if hasattr(self.original_dl, key):
                        dl_kwargs[key] = getattr(self.original_dl, key)

                if dl_kwargs.get("num_workers", 0) > 0 and hasattr(self.original_dl, "prefetch_factor"):
                    dl_kwargs["prefetch_factor"] = self.original_dl.prefetch_factor

                return torch.utils.data.DataLoader(**dl_kwargs)

            def __iter__(self):
                yielded_batches = 0
                current_dl = self._create_loader(skip_batches=0)
                if current_dl is None:
                    return
                current_iter = iter(current_dl)

                # Disable PyTorch's asynchronous signal handling to prevent mid-execution interrupts
                if hasattr(signal, "SIGCHLD"):
                    signal.signal(signal.SIGCHLD, signal.SIG_DFL)

                while yielded_batches < self.target_length:
                    try:
                        batch = next(current_iter)
                        yield batch
                        yielded_batches += 1
                    except StopIteration:
                        break
                    except RuntimeError as e:
                        error_msg = str(e).lower()
                        crash_keywords = [
                            "segmentation fault",
                            "killed by signal",
                            "exited unexpectedly",
                        ]

                        if any(kw in error_msg for kw in crash_keywords):
                            warnings.warn(
                                f"Intercepted OS-level worker crash at batch {yielded_batches}/{self.target_length}. "
                                f"Cleaning up memory and seamlessly restarting dataloader...",
                            )

                            # Safely terminate the background pin_memory_thread to prevent memory/FD leaks
                            if hasattr(current_iter, "_pin_memory_thread_done_event"):
                                current_iter._pin_memory_thread_done_event.set()
                            if hasattr(current_iter, "_worker_result_queue"):
                                with suppress(Exception):
                                    current_iter._worker_result_queue.put((None, None))
                                current_iter._worker_result_queue.cancel_join_thread()
                            # Prevent deadlock in pin_memory_thread.join() during cleanup
                            if hasattr(current_iter, "_pin_memory_thread"):
                                del current_iter._pin_memory_thread
                            # if hasattr(current_iter, "_shutdown"):
                            #     current_iter._shutdown = True

                            del current_iter
                            del current_dl
                            gc.collect()

                            current_dl = self._create_loader(skip_batches=yielded_batches)
                            if current_dl is None:
                                break
                            current_iter = iter(current_dl)

                            if hasattr(signal, "SIGCHLD"):
                                signal.signal(signal.SIGCHLD, signal.SIG_DFL)
                        else:
                            raise e

        # Intercept and wrap DataLoader arguments
        new_args = list(args)
        for i, arg in enumerate(new_args):
            if isinstance(arg, torch.utils.data.DataLoader):
                new_args[i] = RobustWrapper(arg)

        for k, v in kwargs.items():
            if isinstance(v, torch.utils.data.DataLoader):
                kwargs[k] = RobustWrapper(v)

        return func(*new_args, **kwargs)

    return wrapper


def fix_random(seed=0):
    import random

    import torch.backends.cudnn
    import torch.cuda

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    # os.environ["PYTHONHASHSEED"] = "0"
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    elif isinstance(v, str):
        v = v.strip().lower()
        if v in ("yes", "true", "t", "y", "1"):
            return True
        elif v in ("no", "false", "f", "n", "0"):
            return False
    raise argparse.ArgumentTypeError(f"Unsupported value encountered: {v}")


def _str2list(v: str, element_type=None) -> list:
    if not isinstance(v, (list, tuple, set)):
        v = v.lstrip("[").rstrip("]")
        v = v.split(",")
        v = list(map(str.strip, v))
        if element_type is not None:
            v = list(map(element_type, v))
    return v


def str2list(type=None):
    return partial(_str2list, element_type=type)


def dir_path(path: str):
    if os.path.isdir(path):
        return path
    else:
        raise NotADirectoryError(os.path.abspath(path))


def file_path(path: str):
    if os.path.isfile(path):
        return path
    else:
        raise FileNotFoundError(os.path.abspath(path))


def find_ckpt(ckpt_dir: str, epoch=-1, prefix="checkpoint-", suffix=".pth"):
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(os.path.abspath(ckpt_dir))
    if os.path.isfile(ckpt_dir):
        return ckpt_dir
    elif not os.path.isdir(ckpt_dir):
        raise NotADirectoryError(os.path.abspath(ckpt_dir))
    if epoch >= 0:
        filepath = os.path.join(ckpt_dir, f"{prefix}{epoch}{suffix}")
        if os.path.isfile(filepath):
            return filepath
        else:
            raise FileNotFoundError(os.path.abspath(filepath))
    file_list = [p for p in os.listdir(ckpt_dir) if p.startswith(prefix) and p.endswith(suffix)]
    if not file_list:
        raise FileNotFoundError(os.path.abspath(ckpt_dir))
    file_list.sort(
        key=lambda x: (
            int(x.split(prefix)[-1].split(suffix)[0]) if x.split(prefix)[-1].split(suffix)[0].isdigit() else -99
        ),
        reverse=True,
    )
    return os.path.join(ckpt_dir, file_list[0])


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def init_dist():
    from datetime import timedelta

    assert torch.cuda.is_available(), "cuda is not available"
    assert (
        "RANK" in os.environ and "WORLD_SIZE" in os.environ
    ), "To use distributed mode, use `python -m torch.distributed.launch` or `torchrun` to launch the program"

    local_rank = int(os.environ["LOCAL_RANK"])
    device_count = torch.cuda.device_count()

    if int(os.environ["LOCAL_WORLD_SIZE"]) <= device_count:
        backend = "nccl"
        device_id = local_rank
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    else:
        if local_rank == 0:
            print("Using gloo as backend, because NCCL does not support using the same device for multiple ranks")
        backend = "gloo"
        device_id = local_rank % device_count

    # torch.cuda.set_device(device_id)
    devices_list = os.environ.get("CUDA_VISIBLE_DEVICES", list(range(device_count)))
    devices_list = [int(d.strip()) for d in devices_list.split(",")] if isinstance(devices_list, str) else devices_list
    os.environ["CUDA_VISIBLE_DEVICES"] = str(devices_list[device_id])

    dist.init_process_group(backend=backend, init_method="env://", timeout=timedelta(hours=10))
    synchronize()
    return local_rank, dist.get_world_size()


def get_local_index(total_num: int) -> tuple[int, int, int]:
    import math

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank, world_size = init_dist()
        num_per_node = math.ceil(total_num / world_size)
        idx_begin = local_rank * num_per_node
        if local_rank != world_size - 1:
            if idx_begin == 0:
                idx_begin = -1
            idx_end = (local_rank + 1) * num_per_node
        else:
            idx_end = total_num
    else:
        local_rank, world_size = 0, 1
        idx_begin, idx_end = -1, total_num
    if idx_begin == -1:
        idx_begin = 0
    return local_rank, idx_begin, idx_end


class DummySummaryWriter:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, *args, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def get_normalize_transl_scaling(points: torch.Tensor, keep_ratio=True, recenter=True):
    """
    points: (..., N, 3)
    """
    min_vals = torch.min(points, dim=-2, keepdim=True)[0]
    max_vals = torch.max(points, dim=-2, keepdim=True)[0]
    if not recenter:
        max_vals = torch.maximum(max_vals.abs(), min_vals.abs())
        min_vals = -max_vals
    if keep_ratio:
        scaling = 2.0 / torch.max(max_vals - min_vals, dim=-1, keepdim=True)[0]
        scaling = scaling.tile(3)
    else:
        scaling = 2.0 / (max_vals - min_vals)
    if recenter:
        center = (max_vals + min_vals) / 2.0
        transl = -center
    else:
        transl = torch.zeros_like(min_vals)

    sh = points.shape[:-2]
    transl_matrix = torch.eye(4, dtype=points.dtype, device=points.device)
    if sh:
        transl_matrix = transl_matrix.unsqueeze(0).repeat(*sh, 1, 1)
    transl_matrix[..., :3, 3] = transl.squeeze(-2)
    scaling_matrix = torch.eye(4, dtype=points.dtype, device=points.device)
    if sh:
        scaling_matrix = scaling_matrix.unsqueeze(0).repeat(*sh, 1, 1)
    scaling_matrix[..., 0, 0] = scaling.squeeze(-2)[..., 0]
    scaling_matrix[..., 1, 1] = scaling.squeeze(-2)[..., 1]
    scaling_matrix[..., 2, 2] = scaling.squeeze(-2)[..., 2]
    return transl_matrix, scaling_matrix


def compose_transform_matrix(matrix_list: list[torch.Tensor]) -> torch.Tensor:
    """
    Args:
        matrix_list: list of (..., 4, 4) left-multiplied transformation matrices
    Returns:
        (..., 4, 4) left-multiplied transformation matrix
    """
    result = matrix_list[0]
    for i in range(1, len(matrix_list)):
        result = torch.einsum("...ij,...jk->...ik", matrix_list[i], result)
    return result


def get_normalize_matrix(points: np.ndarray, keep_ratio=True, recenter=True):
    """
    points: (..., N, 3)
    """
    min_vals = np.min(points, axis=-2, keepdims=True)
    max_vals = np.max(points, axis=-2, keepdims=True)
    if not recenter:
        max_vals = np.maximum(np.abs(max_vals), np.abs(min_vals))
        min_vals = -max_vals
    if keep_ratio:
        scaling = 2.0 / np.max(max_vals - min_vals, axis=-1, keepdims=True)
        scaling = np.tile(scaling, 3)
    else:
        scaling = 2.0 / (max_vals - min_vals)
    if recenter:
        center = (max_vals + min_vals) / 2.0
        transl = -center
    else:
        transl = np.zeros_like(min_vals)

    sh = points.shape[:-2]
    transl_matrix = np.eye(4, dtype=points.dtype)
    if sh:
        transl_matrix = np.tile(transl_matrix, (*sh, 1, 1))
    transl_matrix[..., :3, 3] = transl.squeeze(-2)
    scaling_matrix = np.eye(4, dtype=points.dtype)
    if sh:
        scaling_matrix = np.tile(scaling_matrix, (*sh, 1, 1))
    scaling_matrix[..., 0, 0] = scaling.squeeze(-2)[..., 0]
    scaling_matrix[..., 1, 1] = scaling.squeeze(-2)[..., 1]
    scaling_matrix[..., 2, 2] = scaling.squeeze(-2)[..., 2]
    return scaling_matrix @ transl_matrix


def get_normalize_transform(points: torch.Tensor, keep_ratio=True, recenter=True):
    """PyTorch3D version
    points: (B, N, 3)
    """
    from pytorch3d.transforms import Scale, Translate

    min_vals = torch.min(points, dim=1, keepdim=True)[0]
    max_vals = torch.max(points, dim=1, keepdim=True)[0]
    if not recenter:
        max_vals = torch.maximum(max_vals.abs(), min_vals.abs())
        min_vals = -max_vals
    if keep_ratio:
        scaling = 2.0 / torch.max(max_vals - min_vals, dim=2, keepdim=True)[0]
        scaling = scaling.tile(3)
    else:
        scaling = 2.0 / (max_vals - min_vals)
    if recenter:
        center = (max_vals + min_vals) / 2.0
        transl = Translate(-center.squeeze(1))
    scaling = Scale(scaling.squeeze(1))
    transform = transl.compose(scaling) if recenter else scaling
    return transform


def get_homogeneous(xyz: Tensor_or_Array) -> Tensor_or_Array:
    """
    Args:
        xyz: (..., 3)
    Returns:
        (..., 4)
    """
    module = torch if isinstance(xyz, torch.Tensor) else np
    ones = module.ones_like(xyz[..., 0:1])
    xyz_homo = module.concatenate((xyz, ones), -1)
    return xyz_homo


def apply_transform(xyz: Tensor_or_Array, transform: Tensor_or_Array) -> Tensor_or_Array:
    """
    Args:
        xyz: (..., 3)
        transform: (..., 4, 4) left-multiplied transformation matrix
    Returns:
        (..., 3)
    """
    module = torch if isinstance(xyz, torch.Tensor) else np
    xyz_homo = get_homogeneous(xyz)
    if xyz.shape[:-1] != transform.shape[:-2]:
        equation = "...ij,...nj->...ni"
    else:
        equation = "...ij,...j->...i"
    xyz_homo_transformed = module.einsum(equation, transform, xyz_homo)
    xyz_homo_transformed = xyz_homo_transformed[..., :3]
    return xyz_homo_transformed


def apply_transform_normals(normals: Tensor_or_Array, transform: Tensor_or_Array) -> Tensor_or_Array:
    """
    Args:
        normals: (..., 3)
        transform: (..., 4, 4) left-multiplied transformation matrix
    Returns:
        (..., 3)
    """
    module = torch if isinstance(normals, torch.Tensor) else np
    transform = transform[..., :3, :3]
    if normals.shape[:-1] != transform.shape[:-2]:
        equation = "...ij,...nj->...ni"
    else:
        equation = "...ij,...j->...i"
    normals_transformed = module.einsum(equation, transform, normals)
    if module is torch:
        normals_transformed = F.normalize(normals_transformed, p=2, dim=-1)
    else:
        normals_transformed / (np.linalg.norm(normals_transformed, axis=-1, keepdims=True) + 1e-8)
    return normals_transformed


def matrix_to_quat(matrix: Tensor_or_Array) -> Tensor_or_Array:
    """
    Args:
        matrix (..., 3, 3) rotation matrix
    Returns:
        (..., 4) quaternions
    """
    module = torch if isinstance(matrix, torch.Tensor) else np
    if module is torch:
        from pytorch3d.transforms import matrix_to_quaternion

        # ! no need for `transpose` since the PyTorch3D's quaternion also expects a right-multiplied version (opposite vector part)
        quat = matrix_to_quaternion(matrix)
    else:
        from scipy.spatial.transform import Rotation as R

        quat = R.from_matrix(matrix).as_quat(canonical=True, scalar_first=True)
    return quat


def quat_to_matrix(quat: Tensor_or_Array) -> Tensor_or_Array:
    """
    Args:
        quat: (..., 4) quaternions
    Returns:
        (..., 3, 3) rotation matrix
    """
    module = torch if isinstance(quat, torch.Tensor) else np

    # Prevent NaN
    zero_norm_mask = ~module.isclose((quat * quat).sum(-1), module.ones_like(quat[..., 0]))
    zero_norm_mask = zero_norm_mask[..., None]
    if zero_norm_mask.any():
        r_zero = module.zeros_like(quat)
        r_zero[..., 0] = 1.0
        quat = module.where(module.broadcast_to(zero_norm_mask, quat.shape), r_zero, quat)

    if module is torch:
        from pytorch3d.transforms import quaternion_to_matrix

        matrix = quaternion_to_matrix(quat)
    else:
        from scipy.spatial.transform import Rotation as R

        matrix = R.from_quat(quat, scalar_first=True).as_matrix()
    return matrix


def decompose_transform(transform: Tensor_or_Array, return_quat=True, return_concat=True) -> Tensor_or_Array:
    """Decompose homogeneous transformation matrix into translation + rotation + scaling (in concatenation)
    The correct re-compose order: rotation, scaling, translation
    Args:
        transform: (..., 4, 4) left-multiplied transformation matrix
    """
    module = torch if isinstance(transform, torch.Tensor) else np
    norm_fn = (lambda x: torch.norm(x, dim=-2)) if module is torch else (lambda x: np.linalg.norm(x, axis=-2))
    transl = transform[..., :3, 3]
    scaling = norm_fn(transform[..., :3, :3])
    rotation = transform[..., :3, :3] / scaling[..., None, :]
    if return_quat:
        rotation = matrix_to_quat(rotation)
    elif return_concat:
        rotation = rotation.reshape(*rotation.shape[:-2], 3 * 3)
    if return_concat:
        return module.concatenate((transl, rotation, scaling), -1)
    return transl, rotation, scaling


def compose_transform(transform: Tensor_or_Array) -> Tensor_or_Array:
    """
    Args:
        transform: (..., 3+r+3) decomposed translation + rotation + scaling (in concatenation)
    Returns:
        (..., 4, 4) left-multiplied transformation matrix (rotation + scaling + translation)
    """
    if isinstance(transform, (tuple, list)):  # return_concat=False
        if len(transform) == 3:
            transl, rotation, scaling = transform
        elif len(transform) == 2:
            transl, rotation = transform
            scaling = None
        else:
            raise ValueError(f"Invalid transform: {transform}")
    else:
        print(f"Assuming translation and scaling both have ndim=3 in {transform.shape[-1]=}")
        transl, rotation, scaling = transform[..., :3], transform[..., 3:-3], transform[..., -3:]

    module = torch if isinstance(rotation, torch.Tensor) else np
    if rotation.shape[-2:] != (3, 3):  # return_quat=False
        if rotation.shape[-1] == 9:
            rotation = rotation.reshape(*rotation.shape[:-1], 3, 3)
        elif rotation.shape[-1] == 4:
            rotation = quat_to_matrix(rotation)
            assert not module.isnan(rotation).any()
        else:
            raise ValueError(f"Invalid rotation shape: {rotation.shape}")

    matrix = module.zeros((*rotation.shape[:-2], 4, 4), dtype=rotation.dtype)
    if module is torch:
        matrix = matrix.to(rotation)
    matrix[..., 3, 3] = 1.0
    matrix[..., :3, :3] = rotation
    if scaling is not None:
        matrix[..., :3, :3] *= scaling[..., None, :]
    matrix[..., :3, 3] = transl
    return matrix


def quat_transl_to_dualquat(quat: torch.Tensor, transl: torch.Tensor, transl_first=False):
    """
    Args:
        quat: (..., 4)
        transl: (..., 3)
    Returns:
        (..., 8) rotation + translation as quaternions
    """
    from pytorch3d.transforms import quaternion_raw_multiply, standardize_quaternion

    q_r = standardize_quaternion(F.normalize(quat, dim=-1))
    q_d = 0.5 * quaternion_raw_multiply(torch.cat([torch.zeros_like(transl[..., :1]), transl], dim=-1), q_r)
    # translation quaternion should not normalized/standardized
    return torch.cat([q_d, q_r], dim=-1) if transl_first else torch.cat([q_r, q_d], dim=-1)


def dualquat_to_quat_transl(dq: torch.Tensor, concat=False, transl_first=False):
    """
    Args
    ----------
        dq: (..., 8) rotation + translation as quaternions

    Returns
    ----------
        quat:
            (..., 4)
        transl:
            (..., 3)
    """
    from pytorch3d.transforms import quaternion_invert, quaternion_raw_multiply

    q_r, q_d = dq[..., :4], dq[..., 4:]
    if transl_first:
        q_r, q_d = q_d, q_r
    transl = 2.0 * quaternion_raw_multiply(q_d, quaternion_invert(q_r))
    transl = transl[..., 1:]
    quat_transl = (transl, q_r) if transl_first else (q_r, transl)
    if concat:
        quat_transl = torch.cat(quat_transl, dim=-1)
    return quat_transl


def matrix_to_dualquat(matrix: torch.Tensor):
    """
    Args
    ----------
        matrix: (..., 4, 4)

    Returns
    ----------
        dq: (..., 8) rotation + translation as quaternions
    """
    R = matrix[..., :3, :3]
    t = matrix[..., :3, 3]
    q_rot = matrix_to_quat(R)
    dq = quat_transl_to_dualquat(q_rot, t)
    return dq


def dualquat_to_matrix(dq: torch.Tensor):
    """
    Args
    ----------
        dq: (..., 8) rotation + translation as quaternions

    Returns
    ----------
        matrix: (..., 4, 4)
    """
    q_rot, transl = dualquat_to_quat_transl(dq)
    rotation = quat_to_matrix(q_rot)
    matrix = torch.eye(4, device=rotation.device, dtype=rotation.dtype).expand(*rotation.shape[:-2], 4, 4).clone()
    matrix[..., :3, :3] = rotation
    matrix[..., :3, 3] = transl
    return matrix


def matrix_to_ortho6d(matrix: torch.Tensor):
    """
    Args:
        rotation matrix: (..., 3, 3)
    Returns:
        (..., 6) ortho6d
    """
    sh = matrix.shape
    return matrix[..., :-1].transpose(-1, -2).reshape(*sh[:-2], 6)


def ortho6d_to_matrix(ortho6d: torch.Tensor):
    """On the Continuity of Rotation Representations in Neural Networks
    https://github.com/papagina/RotationContinuity/blob/master/sanity_test/code/tools.py
    Args:
        ortho6d: (..., 6)
    Returns:
        (..., 3, 3) rotation matrix
    """
    x_raw = ortho6d[..., 0:3]
    y_raw = ortho6d[..., 3:6]

    x = F.normalize(x_raw, dim=-1)
    z = torch.cross(x, y_raw, dim=-1)
    z = F.normalize(z, dim=-1)
    y = torch.cross(z, x, dim=-1)

    return torch.stack((x, y, z), dim=-1)


def get_rotation_center(matrix: Tensor_or_Array) -> Tensor_or_Array:
    """https://math.stackexchange.com/questions/4397763/3d-rotation-matrix-around-a-point-not-origin/4397766
    Args:
        matrix: (..., 4, 4) homogeneous transformation
    Returns:
        (..., 3)
    """
    module = torch if isinstance(matrix, torch.Tensor) else np

    R = matrix[..., :3, :3]
    t = matrix[..., :3, 3]

    # From the formula: t = v - Rv
    # Therefore: v - Rv = t
    # (I - R)v = t
    # v = (I - R)^(-1) t

    I = module.eye(3)
    I = module.broadcast_to(I, R.shape)
    return module.linalg.solve(I - R, t)


def get_rotation_about_point(rotation: Tensor_or_Array, center: Tensor_or_Array) -> Tensor_or_Array:
    """
    Args:
        rotation: (..., 3, 3)
        center: (..., 3)
    Returns:
        (..., 4, 4) homogeneous transformation matrix
    """
    module = torch if isinstance(rotation, torch.Tensor) else np
    translation = center - module.matmul(rotation, center[..., None]).squeeze(-1)
    batch_shape = rotation.shape[:-2]
    matrix = module.zeros(batch_shape + (4, 4), dtype=rotation.dtype)
    if module is torch:
        matrix = matrix.to(rotation)
    matrix[..., :3, :3] = rotation
    matrix[..., :3, 3] = translation
    matrix[..., 3, 3] = 1.0
    return matrix


def compose_transform_trt(transform: Tensor_or_Array) -> Tensor_or_Array:
    """
    Args:
        transform: (..., 3+r+3) source + rotation + target (in concatenation)
    Returns:
        (..., 4, 4) transformation matrix (translation (from source to origin) + rotation + translation (from origin to target))
    """
    if isinstance(transform, (tuple, list)):
        assert len(transform) == 3, f"Invalid transform: {transform}"
        source, rotation, target = transform
    else:
        source, rotation, target = transform[..., :3], transform[..., 3:-3], transform[..., -3:]

    module = torch if isinstance(rotation, torch.Tensor) else np
    if rotation.shape[-2:] != (3, 3):
        if rotation.shape[-1] == 9:
            rotation = rotation.reshape(*rotation.shape[:-1], 3, 3)
        elif rotation.shape[-1] == 4:
            rotation = quat_to_matrix(rotation)
            assert not module.isnan(rotation).any()
        else:
            raise ValueError(f"Invalid rotation shape: {rotation.shape}")

    matrix = get_rotation_about_point(rotation, source)
    matrix[..., :3, 3] += target - source
    return matrix


def to_pose_local(pose_repr: torch.Tensor, input_mode: str, return_quat=True) -> torch.Tensor:
    """
    Args:
        pose_repr: (..., r)
    Returns:
        (..., 4) if `return_quat` else (..., 3, 3)
    """
    if input_mode == "local_quat":
        return pose_repr if return_quat else quat_to_matrix(pose_repr)
    elif input_mode == "local_ortho6d":
        matrix = ortho6d_to_matrix(pose_repr)
        return matrix_to_quat(matrix) if return_quat else matrix
    else:
        raise NotImplementedError(f"{input_mode=}")


def to_pose_matrix(pose_repr: torch.Tensor, input_mode: str, source: torch.Tensor = None) -> torch.Tensor:
    """
    Args:
        pose_repr: (..., t+r)
    Returns:
        (..., 4, 4)
    """
    if input_mode == "transl_quat":
        return compose_transform(torch.split(pose_repr, [3, 4], dim=-1))
    elif input_mode == "dual_quat":
        transl_quat = dualquat_to_quat_transl(pose_repr, concat=False, transl_first=True)
        return compose_transform(transl_quat)
    elif input_mode == "transl_ortho6d":
        transl, rotation = torch.split(pose_repr, [3, 6], dim=-1)
        rotation = ortho6d_to_matrix(rotation)
        rotation = rotation.reshape(*rotation.shape[:-2], 3 * 3)
        return compose_transform([transl, rotation])
    elif input_mode == "transl_matrix":
        return compose_transform(torch.split(pose_repr, [3, 9], dim=-1))
    elif input_mode == "target_quat":
        target, rotation = torch.split(pose_repr, [3, 4], dim=-1)
        assert source is not None and source.shape == target.shape
        return compose_transform_trt([source, rotation, target])
    elif input_mode == "target_ortho6d":
        target, rotation = torch.split(pose_repr, [3, 6], dim=-1)
        assert source is not None and source.shape == target.shape
        rotation = ortho6d_to_matrix(rotation)
        return compose_transform_trt([source, rotation, target])
    elif input_mode == "target_matrix":
        target, rotation = torch.split(pose_repr, [3, 9], dim=-1)
        assert source is not None and source.shape == target.shape
        return compose_transform_trt([source, rotation, target])
    else:
        raise NotImplementedError(f"{input_mode=}")


def pose_local_to_global(
    pose_local: torch.Tensor,
    joints: torch.Tensor,
    parents: torch.Tensor,
    global_transl: torch.Tensor = None,
    relative_to_source=False,
) -> torch.Tensor:
    """Reference:https://github.com/vchoutas/smplx/blob/main/smplx/lbs.py

    Args:
        pose_local: (..., K, 3, 3)
            rotation axis: global coordinates; rotation origin: source joint with target-posed (or source-posed if `relative_to_source`) parents
        joints: (..., K, 3)
        parents: (K,)
            parent indices from the kinematic tree (root joint has no effect; make sure children appear after parents)

    Returns:
        pose_global: (..., K, 4, 4)
            The rigid transformations for all the joints with respect to the origin
        posed_joints: (..., K, 3)
            The locations of the joints after applying the pose rotations
    """
    if pose_local.shape[-1] == 4:
        pose_local = quat_to_matrix(pose_local)
    elif pose_local.shape[-1] == 6:
        pose_local = ortho6d_to_matrix(pose_local)
    assert pose_local.shape[-2:] == (3, 3)
    assert pose_local.shape[-3] == joints.shape[-2] == parents.shape[0]
    K = pose_local.shape[-3]

    root_matrix = get_rotation_about_point(pose_local[..., 0, :, :], joints[..., 0, :])
    root_posed = joints[..., 0, :]
    if global_transl is not None:
        root_matrix[..., :3, 3] += global_transl
        root_posed = root_posed + global_transl
    pose_global = [root_matrix]
    posed_joints = [root_posed]

    for i in range(1, K):
        parent_matrix = pose_global[parents[i].item()]
        posed = apply_transform(joints[..., i, :], parent_matrix)
        # No need for rotation here, since the local rotation is about itself
        posed_joints.append(posed)
        if relative_to_source:
            matrix = get_rotation_about_point(pose_local[..., i, :, :], joints[..., i, :])
            pose = torch.matmul(parent_matrix, matrix)
        else:
            matrix = get_rotation_about_point(pose_local[..., i, :, :], posed)
            pose = torch.matmul(matrix, parent_matrix)
        pose_global.append(pose)

    pose_global = torch.stack(pose_global, dim=-3)
    posed_joints = torch.stack(posed_joints, dim=-2)

    return pose_global, posed_joints


def pose_rot_to_global(
    pose_rot: torch.Tensor, joints: torch.Tensor, parents: torch.Tensor, global_transl: torch.Tensor = None
) -> torch.Tensor:
    """
    Args:
        pose_rot: (..., K, 3, 3)
            global rotations
        joints: (..., K, 3)
        parents: (K,)
            parent indices from the kinematic tree (root joint has no effect; make sure children appear after parents)

    Returns:
        pose_global: (..., K, 4, 4)
            The rigid transformations for all the joints with respect to the origin
        posed_joints: (..., K, 3)
            The locations of the joints after applying the pose rotations
    """
    if pose_rot.shape[-1] == 4:
        pose_rot = quat_to_matrix(pose_rot)
    elif pose_rot.shape[-1] == 6:
        pose_rot = ortho6d_to_matrix(pose_rot)
    assert pose_rot.shape[-2:] == (3, 3)
    assert pose_rot.shape[-3] == joints.shape[-2] == parents.shape[0]
    K = pose_rot.shape[-3]

    root_matrix = get_rotation_about_point(pose_rot[..., 0, :, :], joints[..., 0, :])
    root_posed = joints[..., 0, :]
    if global_transl is not None:
        root_matrix[..., :3, 3] += global_transl
        root_posed = root_posed + global_transl
    pose_global = [root_matrix]
    posed_joints = [root_posed]

    for i in range(1, K):
        parent_matrix = pose_global[parents[i].item()]
        posed = apply_transform(joints[..., i, :], parent_matrix)
        posed_joints.append(posed)
        matrix = get_rotation_about_point(pose_rot[..., i, :, :], joints[..., i, :])
        matrix[..., :3, 3] += posed - joints[..., i, :]
        pose_global.append(matrix)

    pose_global = torch.stack(pose_global, dim=-3)
    posed_joints = torch.stack(posed_joints, dim=-2)

    return pose_global, posed_joints


def filter_nan(x: torch.Tensor, target: torch.Tensor):
    x = x.reshape(-1, x.shape[-1])
    target = target.reshape(-1, target.shape[-1])
    mask_row = (~torch.isnan(target)).all(dim=-1)
    x, target = x[mask_row, :], target[mask_row, :]
    mask_col = (~torch.isnan(target)).all(dim=0)
    x, target = x[:, mask_col], target[:, mask_col]
    return x, target


def _sample_mesh(mesh: trimesh.Trimesh, num_pts: int, return_normal=False, return_face_index=False):
    if isinstance(mesh, trimesh.PointCloud):
        assert not return_normal, "Point clouds do not have normals"
        verts = mesh.vertices
        samples = verts[np.random.choice(verts.shape[0], size=num_pts, replace=num_pts > verts.shape[0])]
    else:
        samples, face_index = trimesh.sample.sample_surface(mesh, num_pts)
        if return_normal:
            normals = mesh.face_normals[face_index]
            samples = np.concatenate([samples, normals], axis=-1)
        if return_face_index:
            return np.array(samples), np.array(face_index)
    return np.array(samples)


def get_nearest_on_mesh(mesh: trimesh.Trimesh, query_points: np.ndarray, use_kdtree=False):
    if use_kdtree:
        from scipy.spatial import cKDTree

        kdtree_sample_factor = 4
        n_samples = int(kdtree_sample_factor * max(len(query_points), len(mesh.vertices)))
        samples, face_indices_samples = trimesh.sample.sample_surface(mesh, n_samples)
        normals_samples = mesh.face_normals[face_indices_samples]
        kdtree = cKDTree(samples)
        _, indices = kdtree.query(query_points, k=1)
        surface_pts = samples[indices]
        normals = normals_samples[indices]
        face_indices = face_indices_samples[indices]
    else:
        surface_pts, _, face_indices = mesh.nearest.on_surface(query_points)
        normals = mesh.face_normals[face_indices]

    return surface_pts, normals, face_indices


def sample_mesh_visible(mesh: trimesh.Trimesh, num_pts: int, method="winding_number") -> tuple[np.ndarray, ...]:
    if method == "ray":
        face_centers = mesh.triangles.mean(axis=1)
        face_normals = mesh.face_normals
        # Small epsilon based on mesh scale to avoid self-intersection
        epsilon = 1e-6 * mesh.scale
        # Ray origins: slightly offset outward along normal
        ray_origins = face_centers + epsilon * face_normals
        # Ray directions: outward along normal
        ray_directions = face_normals
        # Create accelerated ray intersector (uses embree if available for efficiency)
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
        # Check if the rays intersect any other faces
        hits = intersector.intersects_any(ray_origins, ray_directions)
        # Visible faces are those where the outward ray does not hit anything
        visible_mask = ~hits
        visible_mask = visible_mask.astype(float)

    elif method == "winding_number":
        import gpytoolbox

        query = np.array(mesh.vertices)
        query += mesh.vertex_normals * (1e-6 * mesh.scale)
        W = gpytoolbox.fast_winding_number(query, mesh.vertices, mesh.faces)
        visible_mask = (W[mesh.faces].mean(1) <= 0.1).astype(float)

    else:
        raise NotImplementedError(f"{method=}")

    face_weights = mesh.area_faces * visible_mask

    if np.sum(face_weights) == 0:
        raise ValueError("No visible faces detected in the mesh.")

    samples, face_indices = trimesh.sample.sample_surface(mesh, num_pts, face_weight=face_weights)
    normals = mesh.face_normals[face_indices]

    return samples, normals, face_indices, visible_mask


def sample_near_positions(
    mesh: trimesh.Trimesh, positions: np.ndarray, num_pts: np.ndarray, radius: float, get_normals=False
):
    samples = []
    for i, center in enumerate(positions):
        if num_pts[i] <= 0:
            continue
        box = trimesh.creation.box(extents=(radius, radius, radius))
        box.vertices += center
        # trimesh.util.concatenate([mesh, box, trimesh.creation.axis()]).export("sample.ply")
        attn_mesh = (
            trimesh.PointCloud(mesh.vertices[mesh.kdtree.query_ball_point(center, radius)])
            if isinstance(mesh, trimesh.PointCloud)
            else mesh.slice_plane(box.facets_origin, -box.facets_normal)
        )
        if attn_mesh.vertices.shape[0] == 0:
            continue
        samples.append(_sample_mesh(attn_mesh, num_pts[i], return_normal=get_normals))
    return np.concatenate(samples, axis=0) if samples else np.empty((0, 6 if get_normals else 3))


def get_geometry_guided_sampling_centers(mesh: trimesh.Trimesh, num_pts: int, radius: float, num_fps=10, num_centers=5):
    import potpourri3d as pp3d
    from scipy.spatial import distance_matrix
    from torch_cluster import fps

    if isinstance(mesh, trimesh.PointCloud):
        raise NotImplementedError("Point cloud is not supported")

    # Preprocess mesh to prevent error in geodesic distance calculation
    mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
    verts = np.array(mesh.vertices)
    idx = fps(torch.from_numpy(verts), ratio=num_fps / verts.shape[0]).numpy()

    # mesh_ = mesh.copy()
    # # trimesh.smoothing.filter_humphrey(mesh_, iterations=10, beta=1.0)
    # # diff = np.linalg.norm(mesh_.vertices - mesh.vertices, axis=-1)
    # diff = mesh_.simplify_quadric_decimation(face_count=mesh_.faces.shape[0] / 10).nearest.vertex(mesh_.vertices)[0]
    # import matplotlib; cmap = matplotlib.colormaps.get_cmap("plasma")  # fmt: skip
    # mesh_ = trimesh.Trimesh(vertices=mesh_.vertices, faces=mesh_.faces, vertex_colors=cmap(diff / diff.max())[:, :3])
    # mesh_.export("sample.obj")

    # area_ratio = []
    # for i_v in idx:
    #     box = trimesh.creation.box(extents=(radius, radius, radius))
    #     box.vertices += verts[i_v]
    #     mesh_sliced = mesh.slice_plane(box.facets_origin, -box.facets_normal)
    #     trimesh.repair.fill_holes(mesh_sliced)
    #     mesh_area = mesh_sliced.facets_area.sum()
    #     # oriented_box = mesh_sliced.bounding_cylinder
    #     oriented_box = mesh_sliced.bounding_box_oriented
    #     box_area = oriented_box.facets_area.sum()
    #     area_ratio.append(mesh_area / box_area)
    #     if area_ratio[-1] > 0.1:
    #         mesh_ = trimesh.util.concatenate([mesh_, oriented_box])
    # mesh_.export("sample.obj")

    dist = distance_matrix(verts[idx], verts)
    solver = pp3d.MeshHeatMethodDistanceSolver(verts, mesh.faces)
    dist_geo = np.empty((idx.shape[0], verts.shape[0]))
    for i, i_v in enumerate(idx):
        dist_geo[i] = solver.compute_distance(i_v)
    dist_ratio = dist / (dist_geo + 1e-8)
    dist_ratio = (dist_ratio < 0.25).sum(-1) / verts.shape[0]
    idx_top = idx[np.argsort(dist_ratio)[::-1][:num_centers]]
    geo_centers = verts[idx_top]

    # Refine these centers within local neighborhood
    for i, geo_center in enumerate(geo_centers):
        neighbors = verts[mesh.kdtree.query_ball_point(geo_center, radius)]
        geo_centers[i] = np.mean(neighbors, axis=0)

    geo_num_pts = [num_pts // num_centers] * (num_centers - 1)
    geo_num_pts.append(num_pts - sum(geo_num_pts))
    return geo_centers, geo_num_pts


def sample_mesh(
    mesh: trimesh.Trimesh,
    num_points: int,
    get_normals=False,
    get_face_index=False,
    attn_ratio=0.0,
    attn_centers: np.ndarray = None,
    attn_rel_radius=0.15,
    attn_geo_ratio=0.0,
):
    if attn_ratio <= 0 and attn_geo_ratio <= 0:
        pts = _sample_mesh(mesh, num_points, return_normal=get_normals, return_face_index=get_face_index)

    else:
        num_pts_attn = int(num_points * attn_ratio)
        num_pts_attn_geo = int(num_points * attn_geo_ratio)
        num_points_main = num_points - num_pts_attn - num_pts_attn_geo
        pts = _sample_mesh(mesh, num_points_main, return_normal=get_normals)

        if attn_centers is None or num_pts_attn == 0:
            attn_centers = np.empty((0, 3))
            num_pts_attn_list = []
        else:
            num_pts_attn_list = [num_pts_attn // len(attn_centers)] * (len(attn_centers) - 1)
            num_pts_attn_list.append(num_pts_attn - sum(num_pts_attn_list))
        attn_radius = attn_rel_radius * (np.max(mesh.vertices) - np.min(mesh.vertices))

        if num_pts_attn_geo > 0:
            geo_centers, geo_num_pts = get_geometry_guided_sampling_centers(
                mesh, num_pts=num_pts_attn_geo, radius=attn_radius
            )
            attn_centers = np.concatenate([attn_centers, geo_centers], axis=0)
            num_pts_attn_list.extend(geo_num_pts)

        pts_attn = sample_near_positions(mesh, attn_centers, num_pts_attn_list, attn_radius, get_normals=get_normals)
        pts = np.concatenate([pts, pts_attn], axis=0)
        if pts.shape[0] < num_points:  # some of the specified attention regions are empty, resample over the whole mesh
            pts_extra = _sample_mesh(mesh, num_points - pts.shape[0], return_normal=get_normals)
            pts = np.concatenate([pts, pts_extra], axis=0)

    # assert pts.shape[0] == num_points
    # colors = np.ones_like(pts)
    # colors[:num_points_main, :] = 0
    # trimesh.Trimesh(pts, vertex_colors=colors).export("sample.ply")
    return pts


def load_gs(path: str, compatible=True):
    """https://github.com/3DTopia/LGM/blob/main/core/gs.py"""
    from plyfile import PlyData

    plydata = PlyData.read(path)

    xyz = np.stack(
        (
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ),
        axis=1,
    )
    # print("Number of points at loading : ", xyz.shape[0])

    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    shs = np.zeros((xyz.shape[0], 3))
    shs[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    shs[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
    shs[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    gaussians = np.concatenate([xyz, opacities, scales, rots, shs], axis=1)
    gaussians = torch.from_numpy(gaussians).float()  # cpu

    if compatible:
        gaussians[..., 3:4] = torch.sigmoid(gaussians[..., 3:4])
        gaussians[..., 4:7] = torch.exp(gaussians[..., 4:7])
        gaussians[..., 11:] = 0.28209479177387814 * gaussians[..., 11:] + 0.5

    return gaussians


def save_gs(gaussians: torch.Tensor, path: str, compatible=True, prune=False):
    """https://github.com/3DTopia/LGM/blob/main/core/gs.py"""
    # gaussians: [N, 14]

    from plyfile import PlyData, PlyElement

    means3D = gaussians[..., 0:3].contiguous().float()
    opacity = gaussians[..., 3:4].contiguous().float()
    scales = gaussians[..., 4:7].contiguous().float()
    rotations = gaussians[..., 7:11].contiguous().float()
    shs = gaussians[..., 11:].unsqueeze(1).contiguous().float()  # [N, 1, 3]

    if prune:
        # prune by opacity
        mask = opacity.squeeze(-1) >= 0.005
        means3D = means3D[mask]
        opacity = opacity[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        shs = shs[mask]

    # invert activation to make it compatible with the original ply format
    if compatible:
        # opacity = kiui.op.inverse_sigmoid(opacity)
        inverse_sigmoid = lambda x: torch.log(x) - torch.log(1 - x)
        opacity = inverse_sigmoid(opacity)
        scales = torch.log(scales + 1e-8)
        shs = (shs - 0.5) / 0.28209479177387814

    xyzs = means3D.detach().cpu().numpy()
    f_dc = shs.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = opacity.detach().cpu().numpy()
    scales = scales.detach().cpu().numpy()
    rotations = rotations.detach().cpu().numpy()

    l = ["x", "y", "z"]
    # All channels except the 3 DC
    for i in range(f_dc.shape[1]):
        l.append(f"f_dc_{i}")
    l.append("opacity")
    for i in range(scales.shape[1]):
        l.append(f"scale_{i}")
    for i in range(rotations.shape[1]):
        l.append(f"rot_{i}")

    dtype_full = [(attribute, "f4") for attribute in l]

    elements = np.empty(xyzs.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyzs, f_dc, opacities, scales, rotations), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")

    PlyData([el]).write(path)


def transform_gs(gs: torch.Tensor, transform: Transform3d):
    from pytorch3d.transforms import quaternion_raw_multiply

    gs = gs.clone()
    assert isinstance(gs, torch.Tensor) and gs.ndim == 2 and gs.shape[-1] == 14
    if not isinstance(transform, Transform3d):
        transform = Transform3d(matrix=torch.tensor(transform).transpose(-1, -2))
    transform = transform.to(gs.device)
    transl, rotation, scaling = decompose_transform(
        transform.get_matrix().transpose(-1, -2), return_quat=True, return_concat=False
    )

    xyz = gs[..., :3]
    if len(transform) > 1:
        xyz = xyz.unsqueeze(-2)
    xyz = transform.transform_points(xyz)
    if len(transform) > 1:
        xyz = xyz.squeeze(-2)
    gs[..., :3] = xyz

    gs[..., 4:7] *= scaling
    gs[..., 7:11] = quaternion_raw_multiply(rotation, gs[..., 7:11])

    return gs


def make_archive(input_path: str, output_path: str):
    import shutil
    import zipfile

    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if os.path.isfile(input_path):
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(input_path, os.path.basename(input_path))
    elif os.path.isdir(input_path):
        shutil.make_archive(output_path, "zip", input_path)
    else:
        raise NotImplementedError


def convert_3d_format(input_path: str, output_path: str):
    import aspose.threed as a3d

    scene = a3d.Scene.from_file(input_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    scene.save(output_path)


def dqs(transform_matrix: np.ndarray, bw: np.ndarray) -> np.ndarray:
    """
    Dual Quaternion Skinning (DQS)

    Args:
        transform_matrix: (B, K, 4, 4)
        bw: (N, K)

    Returns:
        (B, N, 4, 4)
    """
    # --- Step 1: Convert all bone matrices to dual quaternions ---
    dqs = matrix_to_dualquat(torch.from_numpy(transform_matrix)).numpy()

    # --- Step 2: Execute DQS blending logic ---
    # Separate the real and dual parts
    dqs_real, dqs_dual = dqs[..., :4], dqs[..., 4:]

    # Ensure shortest path for interpolation (Antipodality)
    ref_quat_real = dqs_real[:, 0:1, :]
    dots = np.einsum("bkx,bix->bk", dqs_real, ref_quat_real)
    signs = np.where(dots < 0, -1.0, 1.0)[..., np.newaxis]

    dqs_real_flipped, dqs_dual_flipped = dqs_real * signs, dqs_dual * signs

    # Linearly blend dual quaternions using skinning weights
    blended_dqs_real = np.einsum("nk,bkx->bnx", bw, dqs_real_flipped)
    blended_dqs_dual = np.einsum("nk,bkx->bnx", bw, dqs_dual_flipped)

    # Normalize the blended dual quaternions
    norm = np.linalg.norm(blended_dqs_real, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-8)

    blended_dqs_real_normalized = blended_dqs_real / norm
    blended_dqs_dual_normalized = blended_dqs_dual / norm

    # Re-concatenate the normalized real and dual parts
    blended_dq = np.concatenate([blended_dqs_real_normalized, blended_dqs_dual_normalized], axis=-1)

    # --- Step 3: Convert the final blended dual quaternions back to matrices ---
    final_matrices = dualquat_to_matrix(torch.from_numpy(blended_dq)).numpy()
    return final_matrices


def indices_flatten(idx: torch.Tensor, N: int) -> torch.Tensor:
    """
    Prepares indices for indexing a flattened tensor.

    This function converts a batch of indices `idx` (shape B, K) into a 1D
    tensor `idx_flatten`, suitable for selecting elements from a flattened
    tensor like `pts.reshape(B*N, 3)[idx_flatten]`.

    Args:
        idx: (B, K)
        N: The size of the `pts` tensor in its 1st dimension, i.e., the number of points per batch.

    Returns:
        idx_flatten: (B*K,)
    """
    B, K = idx.shape
    # For the b-th batch, all indices need an offset of b * N.
    # torch.arange(B) -> [0, 1, 2, ..., B-1]
    # offset -> [0, N, 2*N, ..., (B-1)*N]
    offset = torch.arange(B, device=idx.device) * N
    idx_with_offset = idx + offset.unsqueeze(1)
    idx_flatten = idx_with_offset.reshape(-1)
    return idx_flatten


def indices_unflatten(idx_flatten: torch.Tensor, N: int, B: int) -> torch.Tensor:
    """
    Converts flattened indices back to a 2D tensor of indices.

    This function takes a 1D tensor of indices `idx_flatten` and converts it
    back to a 2D tensor of indices suitable for indexing a tensor with shape
    (B, N, 3) using `torch.gather`.

    Args:
        idx_flatten: (B*K,)
        N: The size of the `pts` tensor in its 1st dimension, i.e., the number of points per batch.
        B: The original batch size.

    Returns:
        idx: (B, K)
    """
    idx = idx_flatten % N
    idx = idx.reshape(B, -1)
    return idx


def kl_loss(mean: torch.Tensor, logvar: torch.Tensor, mean_gt: torch.Tensor = None, logvar_gt: torch.Tensor = None):
    if mean_gt is None or logvar_gt is None:
        return 0.5 * torch.mean(torch.pow(mean, 2) + torch.exp(logvar) - 1.0 - logvar)
    else:
        return 0.5 * torch.mean(
            torch.pow(mean - mean_gt, 2) * torch.exp(-logvar_gt)
            + torch.exp(logvar - logvar_gt)
            - 1.0
            - logvar
            + logvar_gt
        )


@lru_cache(maxsize=1)
def get_grid(grid_density: int):
    x = np.linspace(-1, 1, grid_density + 1)
    y = np.linspace(-1, 1, grid_density + 1)
    z = np.linspace(-1, 1, grid_density + 1)
    xv, yv, zv = np.meshgrid(x, y, z)
    return np.stack([xv, yv, zv]).astype(np.float32).reshape(3, -1).T


def get_sdf(mesh: trimesh.Trimesh, points: np.ndarray, engine="pysdf") -> np.ndarray:
    """
    Points OUTSIDE the mesh will have POSITIVE distance
    Points INSIDE the mesh will have NEGATIVE distance
    """
    if engine == "trimesh":
        proximity_query = trimesh.proximity.ProximityQuery(mesh)
        sdf = -proximity_query.signed_distance(points)
    elif engine == "pysdf":
        from pysdf import SDF

        sdf_fn = SDF(mesh.vertices, mesh.faces)
        sdf = -sdf_fn(points)
    else:
        raise NotImplementedError(f"{engine=}")
    return sdf


@lru_cache(maxsize=1)
def get_vae_forward():
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from models_ae import create_autoencoder

    vae_model = create_autoencoder(dim=512, M=512, N=32768, latent_dim=8, deterministic=True).eval()
    vae_model.load_state_dict(
        torch.load(
            os.path.join(os.path.dirname(__file__), "../output/ae/ae_d512_m512/checkpoint-199.pth"), map_location="cpu"
        )["model"],
        strict=True,
    )

    @torch.no_grad()
    def vae_forward(pc, queries):
        return vae_model.forward(pc.float(), queries.float())["logits"].unsqueeze(-1)

    return vae_forward


def remesh(mesh: trimesh.Trimesh, method="pymeshlab", **kwargs) -> trimesh.Trimesh:
    if method == "mc":
        import mcubes

        normalize_matrix = get_normalize_matrix(mesh.vertices, keep_ratio=True, recenter=True)
        normalize_matrix_inv = np.linalg.inv(normalize_matrix)
        grid_density = 127
        grid = get_grid(grid_density=grid_density)
        grid = apply_transform(grid, normalize_matrix_inv)
        sdf = -get_sdf(mesh, grid)
        sdf = sdf.reshape(grid_density + 1, grid_density + 1, grid_density + 1).transpose(1, 0, 2)
        verts, faces = mcubes.marching_cubes(sdf, 0)
        verts *= 2.0 / grid_density
        verts -= 1.0
        verts = apply_transform(verts, normalize_matrix_inv)
        mesh_output = trimesh.Trimesh(verts, faces, process=True)

        # Keep the largest connected_component
        components = mesh_output.split(only_watertight=False)
        if len(components) > 1:
            largest_component = max(components, key=lambda m: m.vertices.shape[0])
            mesh_output = largest_component

    elif method == "vae":
        import mcubes

        normalize_matrix = get_normalize_matrix(mesh.vertices, keep_ratio=True, recenter=True)
        normalize_matrix_inv = np.linalg.inv(normalize_matrix)
        grid_density = 127
        grid = get_grid(grid_density=grid_density)

        pc = _sample_mesh(mesh, 32768)
        pc = apply_transform(pc, normalize_matrix)
        logits = get_vae_forward()(torch.from_numpy(pc)[None], torch.from_numpy(grid)[None])
        logits = logits.numpy().reshape(grid_density + 1, grid_density + 1, grid_density + 1).transpose(1, 0, 2)

        verts, faces = mcubes.marching_cubes(logits, 0)
        verts *= 2.0 / grid_density
        verts -= 1.0
        verts = apply_transform(verts, normalize_matrix_inv)
        mesh_output = trimesh.Trimesh(verts, faces, process=True)

    elif method == "pymeshlab":
        import pymeshlab

        ms = pymeshlab.MeshSet()
        pyml_mesh = pymeshlab.Mesh(mesh.vertices, mesh.faces)
        ms.add_mesh(pyml_mesh)
        # ms.apply_coord_laplacian_smoothing()
        # ms.apply_filter("meshing_isotropic_explicit_remeshing", targetlen=pymeshlab.PercentageValue(2), adaptive=False)
        # with HiddenPrints(suppress_err=True, use_fd=True):
        #     ms.apply_filter("generate_surface_reconstruction_vcg", voxsize=pymeshlab.PercentageValue(0.5))
        # ms.apply_filter("generate_resampled_uniform_mesh", cellsize=pymeshlab.PercentageValue(0.5), mergeclosevert=True)
        ms.apply_filter("generate_alpha_wrap", alpha_fraction=kwargs.get("alpha_fraction", 0.02))
        current_mesh = ms.current_mesh()
        mesh_output = trimesh.Trimesh(vertices=current_mesh.vertex_matrix(), faces=current_mesh.face_matrix())

    elif method == "pymeshfix":
        import pymeshfix

        components = mesh.split(only_watertight=False)
        mesh_output = []
        for component in components:
            vclean, fclean = pymeshfix.clean_from_arrays(component.vertices, component.faces)
            mesh_output.append(trimesh.Trimesh(vclean, fclean, process=True))
        mesh_output = trimesh.util.concatenate(mesh_output)

    else:
        raise NotImplementedError(f"{method=}")

    return mesh_output


def fps(pc: torch.Tensor, num_samples: int, idx: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        pc: [B, N, D], first 3 channels are xyz
    Returns:
        [B, `num_samples`, D]
    """
    from torch_cluster import fps as fps_fn

    B, N, D = pc.shape

    if idx is None:
        flattened = pc[..., :3].reshape(B * N, 3)
        batch = torch.arange(B).to(pc.device)
        batch = torch.repeat_interleave(batch, N)
        pos = flattened
        ratio = 1.0 * num_samples / N
        idx = fps_fn(pos, batch, ratio=ratio)
    else:
        assert idx.shape == (B * num_samples,)

    sampled_pc = pc.reshape(B * N, D)[idx]
    sampled_pc = sampled_pc.reshape(B, -1, D)
    # trimesh.Trimesh(sampled_pc[0, :, :3].cpu()).export("sample.ply")

    return sampled_pc, idx


def knn(xyz1: torch.Tensor, xyz2: torch.Tensor, K: int, backend="pytorch3d") -> torch.Tensor:
    """For each point in `xyz2`, find the indices of the `K` nearest points in `xyz1`
    Args:
        xyz1: (B, N, 3)
        xyz2: (B, M, 3)
        K: int <= M
    Returns:
        (B, M, K)
    """
    assert 0 < K < xyz2.shape[-2] > K, f"Invalid {K=}"

    if backend in ("pytorch3d"):
        from pytorch3d.ops import knn_points

        top_k_nearest_idx = knn_points(xyz2, xyz1, K=K, return_nn=False).idx

    elif backend == "mmcv":
        from mmcv.ops.knn import knn

        top_k_nearest_idx = knn(K, xyz1.contiguous(), xyz2.contiguous()).transpose(-1, -2)

    else:
        raise NotImplementedError(f"{backend=}")

    return top_k_nearest_idx.long()


def lexsort_3d_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: (B, N, 3)
    Returns:
        (B, N)
    """
    weights = torch.tensor([1e6, 1e3, 1e0], device=x.device, dtype=x.dtype)
    sort_values = torch.sum(x * weights, dim=-1)
    return torch.argsort(sort_values, dim=-1, stable=True)


if __name__ == "__main__":
    fix_random()
