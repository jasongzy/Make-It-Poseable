import logging
import os
import sys
import warnings
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import torch.amp
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from einops import repeat

from models_ae import Attention, DiagonalGaussianDistribution, create_autoencoder
from util.transformer import QK_Norm_TransformerBlock
from util.utils import find_ckpt, fps, indices_flatten, indices_unflatten

warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")
from util.Hunyuan3D_21.hy3dshape.hy3dshape.models.autoencoders import ShapeVAE
from util.Hunyuan3D_21.hy3dshape.hy3dshape.pipelines import export_to_trimesh
from util.Hunyuan3D_21.hy3dshape.hy3dshape.postprocessors import DegenerateFaceRemover, FloaterRemover

logging.getLogger("hy3dgen.shapgen").setLevel(logging.WARNING)


@dataclass
class Output:
    fps_idx: torch.Tensor = None
    fps_idx_target: torch.Tensor = None
    latents: torch.Tensor = None
    latents_sample: torch.Tensor = None
    occ: torch.Tensor = None
    sdf: torch.Tensor = None
    xyz_add: torch.Tensor = None


class Embedder3D(nn.Module):
    def __init__(self, dim=48, concat_input=True):
        super().__init__()

        assert dim % 6 == 0
        self.embedding_dim = dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack(
            [
                torch.cat([e, torch.zeros(self.embedding_dim // 6), torch.zeros(self.embedding_dim // 6)]),
                torch.cat([torch.zeros(self.embedding_dim // 6), e, torch.zeros(self.embedding_dim // 6)]),
                torch.cat([torch.zeros(self.embedding_dim // 6), torch.zeros(self.embedding_dim // 6), e]),
            ]
        )
        self.register_buffer("basis", e, persistent=False)  # 3 x 16
        self.basis: torch.Tensor
        self.concat_input = concat_input

    def embed(self, xyz: torch.Tensor):
        """
        Args:
            xyz: [B, N, 3]
        Returns:
            [B, N, `dim]
        """
        projections = torch.einsum("bnd,de->bne", xyz, self.basis)
        return torch.cat([projections.sin(), projections.cos()], dim=2)

    def forward(self, xyz: torch.Tensor):
        """
        Args:
            xyz: [B, N, 3]
        Returns:
            [B, N, `dim`(+3)]
        """
        embeddings = self.embed(xyz)
        if self.concat_input:
            embeddings = torch.cat([embeddings, xyz], dim=-1)
        return embeddings


class JointsEmbedder(nn.Module):
    def __init__(self, include_tail=False, embed_dim=48, out_dim=512, concat_input=True, out_mlp=True):
        super().__init__()
        self.embed = Embedder3D(embed_dim, concat_input=concat_input)
        self.point_num = 2 if include_tail else 1
        self.embedding_dim = self.point_num * (embed_dim + (3 if concat_input else 0))
        self.out_mlp = out_mlp
        if self.out_mlp:
            self.mlp = nn.Linear(self.embedding_dim, out_dim)

    def forward(self, joints: torch.Tensor):
        """
        Args:
            joints: [B, N, D]
        Returns:
            [B, N, `out_dim` if `out_mlp` else `embedding_dim`]
        """
        B, N, D = joints.shape
        assert D == self.point_num * 3
        out = self.embed(joints.view(B, N * self.point_num, 3)).view(B, N, self.embedding_dim)
        return self.mlp(out) if self.out_mlp else out


class SkeletonEncoder(nn.Module):
    def __init__(
        self,
        num_pts=32768,
        num_latents=512,
        embed_dim=512,
        bone_num=52,
        use_bone_embedding=True,
        use_bone_embedding_bw=True,
        use_displ_embedding=False,
        use_point_embedding=False,
        use_latent_embedding=False,
    ):
        super().__init__()

        from models_ae import FeedForward, PointEmbed, PreNorm
        from util.Hunyuan3D_21.hy3dshape.hy3dshape.models.autoencoders.attention_blocks import (
            ResidualCrossAttentionBlock,
        )

        self.num_latents = num_latents
        self.point_embed = PointEmbed(dim=embed_dim)
        # self.cross_attend_blocks = nn.ModuleList(
        #     [
        #         PreNorm(
        #             embed_dim,
        #             Attention(embed_dim, embed_dim, heads=1, dim_head=embed_dim, drop_path_rate=0.1),
        #             context_dim=embed_dim,
        #         ),
        #         PreNorm(embed_dim, FeedForward(embed_dim, drop_path_rate=0.1)),
        #     ]
        # )
        # if embed_dim == 512:
        #     ckpt_path = "output/ae/ae_d512_m512/checkpoint-199.pth"
        #     self.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model"], strict=False)
        #     print(f"Loaded base encoder from {ckpt_path}")
        self.cross_attn = ResidualCrossAttentionBlock(width=embed_dim, heads=1, qkv_bias=False, qk_norm=True)

        self.use_bone_embedding = use_bone_embedding
        self.use_bone_embedding_bw = use_bone_embedding_bw
        if self.use_bone_embedding:
            self.bone_embedding = nn.Embedding(bone_num, embed_dim)
        self.use_displ_embedding = use_displ_embedding
        self.use_point_embedding = use_point_embedding
        if self.use_point_embedding:
            self.point_embedding = nn.Embedding(num_pts, embed_dim)
        self.use_latent_embedding = use_latent_embedding
        if self.use_latent_embedding:
            self.latent_embedding = nn.Embedding(self.num_latents, embed_dim)

    def forward(
        self,
        pc: torch.Tensor,
        fps_idx: torch.Tensor = None,
        bw: torch.Tensor = None,
        displ: torch.Tensor = None,
        point_idx: torch.Tensor = None,
        latent_idx: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            pc: [B, N, D]
        Returns:
            [B, N, `embed_dim`]
        """
        B, N, D = pc.shape

        sampled_pc, fps_idx = fps(pc, self.num_latents, fps_idx)

        pc_embeddings = self.point_embed(pc)
        sampled_pc_embeddings = self.point_embed(sampled_pc)
        embed_dim = pc_embeddings.shape[-1]

        if self.use_bone_embedding:
            assert bw is not None and bw.shape == (B, N, self.bone_embedding.num_embeddings)
            if not self.use_bone_embedding_bw:
                bw = F.one_hot(torch.argmax(bw, dim=-1), num_classes=bw.shape[-1]).to(bw)
            bw_embeddings = torch.einsum("bnk,kd->bnd", bw, self.bone_embedding.weight)
            pc_embeddings = pc_embeddings + bw_embeddings
            sampled_pc_embeddings = sampled_pc_embeddings + bw_embeddings.reshape(B * N, embed_dim)[fps_idx].reshape(
                B, -1, embed_dim
            )

        if self.use_displ_embedding:
            assert displ is not None and displ.shape == (B, N, 3)
            displ_embeddings = self.point_embed(displ)
            pc_embeddings = pc_embeddings + displ_embeddings
            sampled_pc_embeddings = sampled_pc_embeddings + displ_embeddings.reshape(B * N, embed_dim)[fps_idx].reshape(
                B, -1, embed_dim
            )

        if self.use_point_embedding:
            assert point_idx is not None and point_idx.shape == (N,)
            point_embeddings = self.point_embedding(point_idx).expand(B, -1, -1)
            pc_embeddings = pc_embeddings + point_embeddings
            sampled_pc_embeddings = sampled_pc_embeddings + point_embeddings.reshape(B * N, embed_dim)[fps_idx].reshape(
                B, -1, embed_dim
            )

        # cross_attn, cross_ff = self.cross_attend_blocks
        # x = cross_attn(sampled_pc_embeddings, context=pc_embeddings, mask=None) + sampled_pc_embeddings
        # x = cross_ff(x) + x
        x = self.cross_attn(sampled_pc_embeddings, pc_embeddings)
        if self.use_latent_embedding:
            # latent_embeddings = self.latent_embedding.weight
            assert latent_idx is not None and latent_idx.shape == (self.num_latents,)
            latent_embeddings = self.latent_embedding(latent_idx)
            x = x + latent_embeddings
        return x


class InputAttention(nn.Module):
    def __init__(self, feat_dim: int, heads=8, dim_head=64, *args, **xargs):
        super().__init__()
        self.norm = nn.LayerNorm(feat_dim)
        self.norm_context = nn.LayerNorm(feat_dim)
        self.attn = Attention(query_dim=feat_dim, heads=heads, dim_head=dim_head, *args, **xargs)
        nn.init.zeros_(self.attn.to_out.weight)
        nn.init.zeros_(self.attn.to_out.bias)

    def forward(self, feat: torch.Tensor, context: torch.Tensor, return_score=False):
        """
        Args:
            feat: [..., D]
            context: [..., k, D]
        Returns:
            [..., D]
        """
        sh = feat.shape
        feat = feat.view(-1, 1, sh[-1])
        context = context.view(-1, context.shape[-2], context.shape[-1])
        context = torch.cat([feat, context], dim=-2)
        out = self.attn(self.norm(feat), self.norm_context(context), return_score=return_score)
        if return_score:
            out, attn_score = out
        out = out + feat
        out = out.reshape(*sh)
        if return_score:
            attn_score = attn_score.reshape(*sh[:-1], -1, context.shape[-2])
            return out, attn_score
        return out


class TransformerAdd(nn.Module):
    def __init__(self, dim: int, num_add: int, num_layers=16, embed_dim=64):
        super().__init__()

        from models_ae import PointEmbed

        self.num_add = num_add
        self.encoder = nn.Sequential(
            nn.LayerNorm(dim, bias=False),
            *[QK_Norm_TransformerBlock(dim, 64, use_qk_norm=True) for _ in range(num_layers // 2)],
        )
        self.query_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, self.num_add * 3),
            nn.Tanh(),
        )
        self.point_embed = PointEmbed(dim=embed_dim)
        self.query_proj = nn.Sequential(
            nn.Linear(embed_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(dim, bias=False),
            *[QK_Norm_TransformerBlock(dim, 64, use_qk_norm=True) for _ in range(num_layers // 2)],
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: [B, N, D]
        Returns:
            [B, N_add, D]
        """
        context_refined = self.encoder(context)  # [B, N, D]
        pooled_context = torch.max(context_refined, dim=1)[0]  # [B, D]

        queries_xyz = self.query_mlp(pooled_context)  # [B, N_add * 3]
        queries_xyz = queries_xyz.reshape(-1, self.num_add, 3)  # [B, N_add, 3]
        queries = self.query_proj(self.point_embed(queries_xyz.detach()))  # [B, N_add, D]

        decoder_input = torch.cat([context_refined, queries], dim=1)  # [B, N + N_add, D]
        decoder_output = self.decoder(decoder_input)  # [B, N + N_add, D]

        return decoder_output[:, -self.num_add :, :], queries_xyz


class MIP(nn.Module):
    def __init__(
        self,
        ae_type="hy3d2.1",
        encoding_type="mean_var",
        transformer_type="decoder",
        N=32768,
        sharp_edge_ratio=0.0,
        input_bw=False,
        use_bone_embedding=True,
        use_bone_embedding_bw=True,
        use_displ_embedding=False,
        use_point_embedding=False,
        use_latent_embedding=False,
        num_latents_add=0,
        pos_latents_add="post_transformer",
        clone_latents_add=False,
    ):
        super().__init__()

        self.ae_type = ae_type
        self.N = N
        assert 0.0 <= sharp_edge_ratio <= 1.0
        if sharp_edge_ratio and ae_type != "hy3d2.1":
            raise NotImplementedError
        self.sharp_edge_ratio = sharp_edge_ratio
        self.input_bw = input_bw
        self.use_bone_embedding = use_bone_embedding
        self.use_displ_embedding = use_displ_embedding
        self.num_latents_add = num_latents_add
        assert pos_latents_add in ("pre_transformer", "post_transformer", "output")
        self.pos_latents_add = pos_latents_add
        self.clone_latents_add = clone_latents_add
        if self.clone_latents_add:
            assert self.num_latents_add > 0, "Latent cloning only applies when adding latents"
            assert self.pos_latents_add == "post_transformer", "Latent cloning requires `post_transformer` position"

        if self.ae_type == "vecset":
            self.embed_dim = 512
            self.num_latents = 512
            deterministic = True
            self.ae = create_autoencoder(
                dim=self.embed_dim, M=self.num_latents, N=self.N, latent_dim=8, deterministic=deterministic
            )
            self.input_dims = [3]
            self.input_normal = False
            if self.input_normal:
                self.input_dims.append(3)
                self.normal_embed = JointsEmbedder(out_dim=self.embed_dim)
                nn.init.zeros_(self.normal_embed.mlp.weight)
                nn.init.zeros_(self.normal_embed.mlp.bias)
            else:
                self.input_dims.append(0)
            self.input_dim = sum(self.input_dims)
            if self.input_dim == self.input_dims[0]:
                self.input_attention = False
            else:
                self.input_attention = True
            if self.input_attention:
                self.input_attn = InputAttention(self.embed_dim)
            self.grid_density = 128
            self.grid = None
            self.ae_decoder_modules: list[nn.Module] = [self.ae.layers, self.ae.decoder_cross_attn, self.ae.to_outputs]
            # if hasattr(self.ae, "proj"):
            #     self.ae_decoder_list.append(self.ae.proj)
            if self.ae.decoder_ff is not None:
                self.ae_decoder_modules.append(self.ae.decoder_ff)
        elif self.ae_type == "triposg":
            triposg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "util", "TripoSG"))
            sys.path.insert(0, triposg_path)
            from util.TripoSG.scripts.inference_vae import TripoSGVAEModel

            self.ae = TripoSGVAEModel.from_pretrained(
                os.path.join(triposg_path, "pretrained_weights", "TripoSG"), subfolder="vae"
            )
            self.embed_dim = 64
            self.num_latents = 2048
            self.input_dim = 6
            self.input_normal = True
            self.ae_decoder_modules: list[nn.Module] = [self.ae.decoder]
        elif self.ae_type == "hy3d2.1":
            pc_sharpedge_size = int(self.N * self.sharp_edge_ratio)
            pc_size = self.N - pc_sharpedge_size
            self.ae = ShapeVAE.from_pretrained(
                "tencent/Hunyuan3D-2.1",
                device="cpu",
                # dtype=torch.float32,
                use_safetensors=False,
                variant="fp16",
                pc_size=pc_size,
                pc_sharpedge_size=pc_sharpedge_size,
                downsample_ratio=self.N // 4096,
            )
            self.embed_dim = 64
            self.num_latents = 4096
            self.input_dim = 6
            self.input_normal = True
            self.ae_decoder_modules: list[nn.Module] = [self.ae.transformer, self.ae.geo_decoder]
        else:
            raise NotImplementedError(f"{self.ae_type=}")
        self._ae_pretrained = [deepcopy(self.ae).to("cpu").eval()]
        self._is_pretrained_ae = False

        self.encoding_type = encoding_type
        if self.encoding_type == "sample":
            pass
        elif self.encoding_type == "mean":
            pass
        elif self.encoding_type == "mean_var":
            assert not (self.ae_type == "vecset" and deterministic)
            self.embed_dim *= 2
        else:
            raise NotImplementedError(f"{self.encoding_type=}")

        self.transformer_type = transformer_type
        self.feat_dim = 512

        if self.transformer_type == "encoder":
            self.learnable_latents = nn.Parameter(torch.randn(1, self.num_latents, self.embed_dim))
            self.transformer_encoder = nn.ModuleList(
                [QK_Norm_TransformerBlock(self.embed_dim, 64, use_qk_norm=True) for _ in range(12)]
            )
            self.transformer_modules: list[nn.Module] = [self.learnable_latents, self.transformer_encoder]
        elif self.transformer_type == "decoder":
            source_input_dim = self.embed_dim
            info_dim = self.embed_dim // 2 if self.encoding_type == "mean_var" else self.embed_dim
            self.bw_dim = 52
            self.skeleton_encoder = SkeletonEncoder(
                num_pts=self.N,
                num_latents=self.num_latents,
                embed_dim=info_dim,
                bone_num=self.bw_dim,
                use_bone_embedding=self.use_bone_embedding,
                use_bone_embedding_bw=use_bone_embedding_bw,
                use_displ_embedding=self.use_displ_embedding,
                use_point_embedding=use_point_embedding,
                use_latent_embedding=use_latent_embedding,
            )
            source_input_dim += info_dim
            if self.input_bw:
                self.bw_encoder = nn.Sequential(nn.Linear(self.bw_dim, info_dim), nn.SiLU())
                source_input_dim += info_dim
            # self.source_input_mlp = nn.Linear(source_input_dim, self.feat_dim, bias=False)
            self.source_input_mlp = nn.Sequential(
                nn.Linear(source_input_dim, self.feat_dim, bias=False), nn.LayerNorm(self.feat_dim)
            )
            # self.target_input_mlp = nn.Linear(info_dim, self.feat_dim, bias=False)
            self.target_input_mlp = nn.Sequential(
                nn.Linear(info_dim, self.feat_dim, bias=False), nn.LayerNorm(self.feat_dim)
            )
            self.transformer_decoder = nn.Sequential(
                nn.LayerNorm(self.feat_dim, bias=False),
                *[QK_Norm_TransformerBlock(self.feat_dim, 64, use_qk_norm=True) for _ in range(24)],
            )
            # self.output_mlp = nn.Linear(self.feat_dim, self.embed_dim)
            self.output_mlp = nn.Sequential(
                nn.LayerNorm(self.feat_dim, bias=False),
                nn.Linear(self.feat_dim, self.feat_dim),
                nn.SiLU(),
                nn.Linear(self.feat_dim, self.embed_dim),
            )
            self.transformer_modules: list[nn.Module] = [
                self.skeleton_encoder,
                self.source_input_mlp,
                self.target_input_mlp,
                self.transformer_decoder,
                self.output_mlp,
            ]
            if self.input_bw:
                self.transformer_modules.append(self.bw_encoder)
            if self.num_latents_add > 0:
                dim_latents_add = self.embed_dim if self.pos_latents_add == "output" else self.feat_dim
                if self.pos_latents_add in ("pre_transformer", "post_transformer"):
                    # self.context_mlp = nn.Sequential(
                    #     nn.Linear(dim_latents_add * 2, dim_latents_add),
                    #     nn.SiLU(),
                    #     nn.Linear(dim_latents_add, dim_latents_add),
                    # )
                    self.transformer_add = TransformerAdd(
                        dim=dim_latents_add, num_add=self.num_latents_add, num_layers=16, embed_dim=info_dim
                    )
                    if self.clone_latents_add:
                        self.add_latents_mlp = nn.Sequential(
                            nn.LayerNorm(dim_latents_add, bias=False),
                            nn.Linear(dim_latents_add, dim_latents_add),
                            nn.SiLU(),
                            nn.Linear(dim_latents_add, dim_latents_add),
                        )
                        last_block: nn.Linear = self.add_latents_mlp[-1]
                        nn.init.zeros_(last_block.weight)
                        nn.init.zeros_(last_block.bias)
                else:
                    self.add_latents = nn.Parameter(torch.randn(1, self.num_latents_add, dim_latents_add))
                    self.transformer_add = nn.Sequential(
                        nn.LayerNorm(dim_latents_add, bias=False),
                        *[QK_Norm_TransformerBlock(dim_latents_add, 64, use_qk_norm=True) for _ in range(16)],
                    )
            elif self.num_latents_add < 0:
                assert self.pos_latents_add == "post_transformer"
                # self.add_latents = nn.Parameter(torch.randn(1, self.num_latents, self.feat_dim))
                self.transformer_add = nn.Sequential(
                    nn.LayerNorm(self.feat_dim, bias=False),
                    *[QK_Norm_TransformerBlock(self.feat_dim, 64, use_qk_norm=True) for _ in range(16)],
                )
                # self.context_mlp = nn.Sequential(
                #     nn.Linear(self.feat_dim * 2, self.feat_dim), nn.SiLU(), nn.Linear(self.feat_dim, self.feat_dim)
                # )
                self.add_latents_mlp = nn.Sequential(
                    nn.LayerNorm(self.feat_dim, bias=False),
                    nn.Linear(self.feat_dim, self.feat_dim),
                    nn.SiLU(),
                    nn.Linear(self.feat_dim, self.feat_dim),
                )
                last_block: nn.Linear = self.add_latents_mlp[-1]
                nn.init.zeros_(last_block.weight)
                nn.init.zeros_(last_block.bias)
        else:
            raise NotImplementedError(f"{self.transformer_type=}")

    def adapt_ckpt(self, ckpt: dict[str, torch.Tensor]):
        params2rename = [("skeleton_encoder.token_embedding.weight", "skeleton_encoder.latent_embedding.weight")]
        for old, new in params2rename:
            if old in ckpt:
                ckpt[new] = ckpt.pop(old)
        return ckpt

    def load(self, pth_path: str, epoch=-1, strict=True):
        pth_path = find_ckpt(pth_path, epoch=epoch)
        checkpoint = torch.load(pth_path, map_location="cpu", weights_only=True)
        model_state_dict = self.adapt_ckpt(checkpoint["model"])
        self.load_state_dict(model_state_dict, strict=strict)
        print(f"Loaded model from {pth_path}")
        return self

    def load_base(self, pth_path: str):
        if self.ae_type == "vecset":
            self.ae.load_state_dict(torch.load(pth_path, map_location="cpu")["model"], strict=True)
            print(f"Loaded base model from {pth_path}")
        return self

    def freeze_ae(self, include_decoder=True):
        for param in self.ae.parameters():
            param.requires_grad = False
        if not include_decoder:
            for m in self.ae_decoder_modules:
                if self.ae_type == "hy3d2.1":
                    m.to(dtype=torch.float32)
                for param in m.parameters():
                    param.requires_grad = True
        return self

    def freeze_transformer(self):
        for m in self.transformer_modules:
            for param in m.parameters():
                param.requires_grad = False
        return self

    @contextmanager
    def pretrained_ae(self, enabled=True):
        """Temporarily use the unmodified version of pretrained autoencoder."""
        if enabled:
            ae = self.ae
            device = next(ae.parameters()).device
            self._ae_pretrained[0] = self._ae_pretrained[0].to(device)
            self.ae = self._ae_pretrained[0]
            self._is_pretrained_ae = True
        try:
            yield self
        finally:
            if enabled:
                self.ae = ae
                self._is_pretrained_ae = False

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.ae.eval()
        return self

    def embed(self, pc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pc: [B, N, `self.input_dim`]
        Returns:
            [B, `self.num_latents`, `self.embed_dim`]
        """
        extra_embeddings = []
        if self.input_dim > 3:
            pc, normal = pc.split(self.input_dims, dim=-1)
            if normal.shape[-1] > 0:
                extra_embeddings.append(self.normal_embed(normal))

        pc_embeddings = self.ae.point_embed(pc)

        if extra_embeddings:
            if self.input_attention:
                pc_embeddings = self.input_attn(pc_embeddings, torch.stack(extra_embeddings, dim=-2))
                # pc_embeddings, attn_score = self.input_attn(
                #     pc_embeddings, torch.stack(extra_embeddings, dim=-2), return_score=True
                # )
                # if pc.shape[1] > 512:
                #     import matplotlib
                #     cmap = matplotlib.colormaps.get_cmap("plasma")
                #     vis_head = 0
                #     trimesh.PointCloud(pc[0].cpu().numpy(), colors=cmap(attn_score[0, :, vis_head, -1].cpu().numpy())[:, :3], process=False).export("normal_attn.glb")  # fmt: skip
            else:
                pc_embeddings = pc_embeddings + torch.stack(extra_embeddings, dim=0).sum(0)

        return pc_embeddings

    def encode(self, pc: torch.Tensor, fps_idx: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pc: [B, P, N, `self.input_dim`]
        Returns:
            [B, P, `self.num_latents`, `self.embed_dim`]
        """
        B, P, N, C = pc.shape
        assert C == self.input_dim
        M, E = self.num_latents, self.embed_dim

        pc = pc.reshape(-1, N, C)

        if fps_idx is not None and self.num_latents_add == 0:
            assert fps_idx.shape == (B * P * M,)

        posterior = None
        if self.ae_type == "vecset":
            # _, x = self.ae.encode(pc)

            sampled_pc, fps_idx = fps(pc, self.num_latents, fps_idx)
            sampled_pc_embeddings = self.embed(sampled_pc)
            pc_embeddings = self.embed(pc)
            cross_attn, cross_ff = self.ae.cross_attend_blocks
            x = cross_attn(sampled_pc_embeddings, context=pc_embeddings, mask=None) + sampled_pc_embeddings
            x = cross_ff(x) + x

            if hasattr(self.ae, "mean_fc"):
                mean = self.ae.mean_fc(x)
                logvar = self.ae.logvar_fc(x)
                posterior = DiagonalGaussianDistribution(mean, logvar)

        elif self.ae_type == "triposg":
            posterior, fps_idx = self.ae.encode(pc, fps_idx)
            posterior = posterior.latent_dist

        elif self.ae_type == "hy3d2.1":
            pc = torch.cat([pc, torch.zeros_like(pc[..., :1])], dim=-1)
            posterior, fps_idx = self.ae.encode(pc, fps_idx, sample_posterior=False)

        if posterior is not None:
            if self.encoding_type == "sample":
                x = posterior.sample()
            elif self.encoding_type == "mean":
                x = posterior.sample() if self._is_pretrained_ae else posterior.mode()
            elif self.encoding_type == "mean_var":
                x = torch.cat([posterior.mean, posterior.logvar], dim=-1)

        return x.reshape(B, P, -1, E), fps_idx

    def decode(self, x: torch.Tensor, queries: torch.Tensor, learnable_embeddings: torch.Tensor = None):
        """
        Args:
            x: [B, P, `self.num_latents`, `self.embed_dim`]
            queries: [B, P, Nq, 3]
        Returns:
            [B, P, Nq, 1]
        """
        assert x.shape[:2] == queries.shape[:2], "Batch size mismatch"
        B, P, M, E = x.shape
        _, _, Nq, C = queries.shape
        assert C == 3

        x = x.reshape(-1, M, E)
        queries = queries.reshape(-1, Nq, 3)

        if self.ae_type == "vecset":
            # o = self.ae.decode(x, queries)
            if hasattr(self.ae, "proj"):
                x = self.ae.proj(x)
            for self_attn, self_ff in self.ae.layers:
                x = self_attn(x) + x
                x = self_ff(x) + x
            # cross attend from decoder queries to latents
            queries_embeddings = self.embed(queries)
            if learnable_embeddings is not None:
                queries_embeddings = torch.cat((queries_embeddings, learnable_embeddings), dim=1)
            logits = self.ae.decoder_cross_attn(queries_embeddings, context=x)
            # optional decoder feedforward
            if self.ae.decoder_ff is not None:
                logits = logits + self.ae.decoder_ff(logits)
            occ = self.ae.to_outputs(logits).reshape(B, P, Nq, 1)
            out = Output(occ=occ)

        elif self.ae_type == "triposg":
            logits = self.ae.decode(x, sampled_points=queries).sample.reshape(B, P, Nq, 1)
            out = Output(sdf=logits)

        elif self.ae_type == "hy3d2.1":
            x = self.ae.decode(x)
            # self.ae.enable_flashvdm_decoder(enabled=True)
            # self.ae.geo_decoder.set_cross_attention_processor(self.ae.volume_decoder.processor)
            # self.ae.volume_decoder.processor.topk = True
            logits = self.ae.geo_decoder(queries, latents=x).reshape(B, P, Nq, 1)
            out = Output(sdf=logits)

        return out

    def forward(
        self,
        source_pts: torch.Tensor,
        target_queries: torch.Tensor,
        source_skeleton: torch.Tensor = None,
        target_skeleton: torch.Tensor = None,
        source_bw: torch.Tensor = None,
        fps_idx: torch.Tensor = None,
        source_latents: torch.Tensor = None,
        skip_transformer=False,
        skip_decoding=False,
        perturb_hands=False,
        skip_add=False,
    ) -> Output:
        """
        Args:
            source_pts: [B, Ps, Ns, `self.input_dim`]
            target_queries: [B, Pt, Nt, 3]
        Returns:
            [B, Pt, Nt, 1]
        """
        assert source_pts.shape[0] == target_queries.shape[0], "Batch size mismatch"
        B, Ps, Ns, _ = source_pts.shape
        _, Pt, _, _ = target_queries.shape
        M, E, D = self.num_latents, self.embed_dim, self.feat_dim

        if source_latents is None:
            if perturb_hands:
                hands_indices = tuple(range(9, 25)) + tuple(range(28, 44))
                hands_mask = torch.any(source_bw[..., hands_indices] >= 0.1, dim=-1)
                hands_skeleton = source_skeleton[hands_mask]
                source_pts[hands_mask] = torch.cat((hands_skeleton, torch.zeros_like(hands_skeleton)), dim=-1)
            source_latents, fps_idx = self.encode(source_pts, fps_idx)  # [B, Ps, M, E]
        else:
            assert source_latents.shape == (B, Ps, M, E)
            assert fps_idx is not None

        xyz_add = None
        if skip_transformer:
            assert Ps == Pt
            target_latents = source_latents
            fps_idx_target = fps_idx
        else:
            if self.transformer_type == "encoder":
                assert Ps == Pt == 1
                learnable_tokens = self.learnable_latents.expand(B, -1, -1)
                input_tokens = torch.cat((learnable_tokens, source_latents.reshape(B, -1, E)), dim=1)  # [B, 2*M, E]
                intermediate_tokens = input_tokens
                for layer in self.transformer_encoder:
                    intermediate_tokens = layer(intermediate_tokens)
                target_latents, _ = intermediate_tokens.split(
                    [learnable_tokens.shape[1], source_latents.shape[1] * source_latents.shape[2]], dim=1
                )  # [B, M, E]
                target_latents = target_latents.unsqueeze(1)  # [B, 1, M, E]
            elif self.transformer_type == "decoder":
                assert source_skeleton is not None and source_skeleton.shape == (B, Ps, Ns, 3)
                if self.use_bone_embedding:
                    assert source_bw is not None and source_bw.shape == (B, Ps, Ns, self.bw_dim)
                    source_bw = source_bw.reshape(B * Ps, Ns, self.bw_dim)
                if self.use_displ_embedding:
                    source_displ = torch.zeros_like(source_skeleton).reshape(B * Ps, -1, 3)
                else:
                    source_displ = None
                point_idx = torch.randperm(Ns, device=source_pts.device)
                latent_idx = torch.randperm(M, device=source_pts.device)
                source_skeleton_latents = self.skeleton_encoder(
                    source_skeleton.reshape(B * Ps, -1, 3),
                    fps_idx,
                    bw=source_bw,
                    displ=source_displ,
                    point_idx=point_idx,
                    latent_idx=latent_idx,
                )
                source_inputs = [source_latents, source_skeleton_latents.reshape(B, Ps, M, -1)]
                if self.input_bw:
                    assert source_bw is not None and source_bw.shape == (B, Ps, Ns, self.bw_dim)
                    bw_latents = self.bw_encoder(
                        source_bw.reshape(B * Ps * Ns, self.bw_dim)[fps_idx].reshape(B, Ps, -1, self.bw_dim)
                    )
                    source_inputs.append(bw_latents)
                source_input_tokens = self.source_input_mlp(torch.cat(source_inputs, dim=-1))  # [B, Ps, M, D]

                assert target_skeleton is not None and target_skeleton.shape[:2] == (B, Pt)
                ref_index = 0
                fps_idx_target = indices_unflatten(fps_idx, Ns, B * Ps)  # [B*Ps, M]
                fps_idx_target = fps_idx_target.reshape(B, Ps, -1)[:, ref_index : ref_index + 1, :].repeat(
                    1, Pt, 1
                )  # [B, Pt, M]
                fps_idx_target = indices_flatten(fps_idx_target.reshape(B * Pt, -1), Ns)
                if self.use_bone_embedding:
                    target_bw = repeat(
                        source_bw.reshape(B, Ps, Ns, self.bw_dim)[:, ref_index], "b n d -> (b pt) n d", pt=Pt
                    )
                else:
                    target_bw = None
                if self.use_displ_embedding:
                    target_displ = target_skeleton - source_skeleton[:, ref_index : ref_index + 1]
                    target_displ = target_displ.reshape(B * Pt, -1, 3)
                else:
                    target_displ = None
                target_skeleton_latents = self.skeleton_encoder(
                    target_skeleton.reshape(B * Pt, -1, 3),
                    fps_idx_target,
                    bw=target_bw,
                    displ=target_displ,
                    point_idx=point_idx,
                    latent_idx=latent_idx,
                )  # [B*Pt, M, E]
                target_input_tokens = self.target_input_mlp(target_skeleton_latents)  # [B*Pt, M, D]
                # Repeat input tokens for each target pose
                repeated_source_input_tokens = repeat(
                    source_input_tokens.reshape(B, -1, D), "b np d -> (b pt) np d", pt=Pt, np=Ps * M
                )  # [B*Pt, Ps*M, D]
                input_tokens = torch.cat((repeated_source_input_tokens, target_input_tokens), dim=1)

                if not skip_add and self.num_latents_add > 0 and self.pos_latents_add == "pre_transformer":
                    # input_tokens_add = torch.cat((input_tokens, self.add_latents.expand(B * Pt, -1, -1)), dim=1)
                    # input_tokens_add = self.transformer_add(input_tokens_add)[:, -self.num_latents_add :, :]
                    input_tokens_add, xyz_add = self.transformer_add(input_tokens)
                    xyz_add = xyz_add.reshape(B, Pt, self.num_latents_add, 3).to(source_pts)
                    input_tokens = torch.cat((input_tokens, input_tokens_add), dim=1)

                intermediate_tokens = input_tokens
                intermediate_tokens: torch.Tensor = self.transformer_decoder(intermediate_tokens)  # [B*Pt, (Ps+1)*M, D]
                target_output_tokens = intermediate_tokens[:, Ps * M :, :]  # [B*Pt, M, D]

                if not skip_add and self.num_latents_add > 0 and self.pos_latents_add == "post_transformer":
                    # context_latents = input_tokens
                    # context_latents = target_input_tokens
                    # context_latents = self.context_mlp(torch.cat((target_input_tokens, target_output_tokens), dim=-1))
                    # context_latents = torch.cat((repeated_source_input_tokens, context_latents), dim=1)
                    # target_tokens_add = torch.cat((context_latents, self.add_latents.expand(B * Pt, -1, -1)), dim=1)
                    # target_tokens_add = self.transformer_add(target_tokens_add)[:, -self.num_latents_add :, :]
                    # context_latents = target_output_tokens
                    context_latents = torch.cat((input_tokens, target_output_tokens), dim=1)
                    target_tokens_add, xyz_add = self.transformer_add(context_latents)
                    xyz_add = xyz_add.reshape(B, Pt, self.num_latents_add, 3).to(source_pts)
                    if self.clone_latents_add:
                        target_tokens_add = self.add_latents_mlp(target_tokens_add)
                        from pytorch3d.ops import knn_points

                        # Find the nearest latent in the predicted target_output_tokens for each target_tokens_add (based on xyz_add & target_skeleton),
                        # and then use target_tokens_add as the residual
                        target_xyz_flat = target_skeleton.reshape(B * Pt, -1, 3)  # [B*Pt, Ns, 3]
                        fps_idx_target_unflatten = indices_unflatten(fps_idx_target, Ns, B * Pt)
                        fps_idx_target_unflatten = fps_idx_target_unflatten.reshape(B * Pt, -1)  # [B*Pt, M]
                        fps_idx_target_unflatten = fps_idx_target_unflatten.unsqueeze(-1).expand(-1, -1, 3)
                        latent_xyz = torch.gather(target_xyz_flat, 1, fps_idx_target_unflatten)  # [B*Pt, M, 3]
                        xyz_add_flat = xyz_add.reshape(B * Pt, self.num_latents_add, 3)
                        nearest_idx = knn_points(xyz_add_flat, latent_xyz, K=1).idx  # [B*Pt, N_add, 1]
                        # base_tokens_xyz = torch.gather(latent_xyz, 1, nearest_idx.expand(-1, -1, 3))  # [B*Pt, N_add, 3]
                        # trimesh.PointCloud(base_tokens_xyz[0].cpu().numpy()).export("test.glb")
                        nearest_idx_expanded = nearest_idx.expand(-1, -1, self.feat_dim)
                        base_tokens = torch.gather(target_output_tokens, 1, nearest_idx_expanded)  # [B*Pt, N_add, D]
                        target_tokens_add = base_tokens.detach() + target_tokens_add
                    target_output_tokens = torch.cat((target_output_tokens, target_tokens_add), dim=1)
                elif not skip_add and self.num_latents_add < 0:
                    # context_latents = self.context_mlp(torch.cat((target_input_tokens, target_output_tokens), dim=-1))
                    # context_latents = torch.cat((repeated_source_input_tokens, context_latents), dim=1)
                    # target_tokens_add = torch.cat((context_latents, self.add_latents.expand(B * Pt, -1, -1)), dim=1)
                    target_tokens_add = torch.cat((input_tokens, target_output_tokens), dim=1)
                    target_tokens_add = self.transformer_add(target_tokens_add)[:, -self.num_latents :, :]
                    target_tokens_add = self.add_latents_mlp(target_tokens_add)
                    target_output_tokens = target_output_tokens + target_tokens_add

                target_latents: torch.Tensor = self.output_mlp(target_output_tokens)  # [B*Pt, M, E]

                if not skip_add and self.num_latents_add > 0 and self.pos_latents_add == "output":
                    target_latents_add = torch.cat((target_latents, self.add_latents.expand(B * Pt, -1, -1)), dim=1)
                    target_latents_add = self.transformer_add(target_latents_add)[:, -self.num_latents_add :, :]
                    target_latents = torch.cat((target_latents, target_latents_add), dim=1)

                target_latents = target_latents.reshape(B, Pt, -1, E)  # [B, Pt, M, E]
                # assert Ps == 1
                # target_latents = target_latents + source_latents

        if self.encoding_type == "mean_var":
            mean, logvar = target_latents.split([E // 2, E // 2], dim=-1)
            posterior = DiagonalGaussianDistribution(mean, logvar)
            target_latents_sample = posterior.sample()
        else:
            target_latents_sample = target_latents
        if skip_decoding:
            return Output(
                fps_idx=fps_idx,
                fps_idx_target=fps_idx_target,
                latents=target_latents,
                latents_sample=target_latents_sample,
                xyz_add=xyz_add,
            )
        out = self.decode(target_latents_sample, queries=target_queries)  # [B, Pt, Nt, 1]

        if out.occ is None and out.sdf is not None:
            occ = torch.zeros_like(out.sdf)
            occ[out.sdf <= 0] = 1.0
            out.occ = occ

        out.fps_idx = fps_idx
        out.fps_idx_target = fps_idx_target
        out.latents = target_latents
        out.latents_sample = target_latents_sample
        out.xyz_add = xyz_add
        return out

    def get_grid(self):
        if self.grid is None:
            x = np.linspace(-1, 1, self.grid_density + 1)
            y = np.linspace(-1, 1, self.grid_density + 1)
            z = np.linspace(-1, 1, self.grid_density + 1)
            xv, yv, zv = np.meshgrid(x, y, z)
            self.grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32)).view(3, -1).transpose(0, 1)[None]
        return self.grid

    @torch.no_grad()
    def marching_cubes(self, occ: torch.Tensor):
        """
        Args:
            occ: [B, N, 1]
        Returns:
            mesh_list: B * [trimesh.Trimesh]
        """
        import mcubes

        mesh_list: list[trimesh.Trimesh] = []
        for volume in (
            occ.view(-1, self.grid_density + 1, self.grid_density + 1, self.grid_density + 1)
            .permute(0, 2, 1, 3)
            .cpu()
            .numpy()
        ):
            verts, faces = mcubes.marching_cubes(volume, 0)
            verts *= 2.0 / self.grid_density
            verts -= 1.0
            mesh = trimesh.Trimesh(verts, faces)
            mesh_list.append(mesh)
        return mesh_list

    @torch.no_grad()
    def marching_cubes_hier(self, geometric_func):
        """hierarchical_extract_geometry"""
        from util.TripoSG.scripts.inference_vae import hierarchical_extract_geometry

        output = hierarchical_extract_geometry(
            geometric_func,
            self.ae.device,
            bounds=(-1.005, -1.005, -1.005, 1.005, 1.005, 1.005),
            dense_octree_depth=8,
            hierarchical_octree_depth=9,
        )
        meshes = [
            trimesh.Trimesh(None if mesh_v_f[0] is None else mesh_v_f[0].astype(np.float32), mesh_v_f[1])
            for mesh_v_f in output
        ]
        return meshes

    @torch.no_grad()
    def to_mesh(
        self, pc: torch.Tensor, skip_transformer=False, post_process=False, octree_resolution=512, **kwargs
    ) -> list[list[trimesh.Trimesh]]:
        """forward + marching cubes
        Args:
            pc: [B, P, N, `self.input_dim`]
        Returns:
            mesh_list: B * [P * [trimesh.Trimesh]]
        """
        B, P, N, C = pc.shape
        source_skeleton: torch.Tensor = kwargs.get("source_skeleton", None)
        if source_skeleton is not None:
            assert source_skeleton.shape[:2] == (B, P)
        source_bw: torch.Tensor = kwargs.get("source_bw", None)
        if source_bw is not None:
            assert source_bw.shape[:3] == (B, P, N)
        target_skeleton: torch.Tensor = kwargs.get("target_skeleton", None)
        if target_skeleton is not None:
            assert target_skeleton.shape[0] == B
        if skip_transformer:
            P_out = P
        else:
            assert target_skeleton is not None
            P_out = target_skeleton.shape[1]

        if self.ae_type == "vecset":
            output = self.forward(
                pc,
                self.get_grid().to(pc.device).unsqueeze(0).repeat(B, P_out, 1, 1),
                skip_transformer=skip_transformer,
                **kwargs,
            )
            mesh_list = self.marching_cubes(output.occ.view(B * P_out, -1, 1))

        elif self.ae_type == "triposg":
            mesh_list = []
            if skip_transformer:
                for batch in pc.reshape(B * P, N, C).split(1, dim=0):
                    mesh_list.append(
                        self.marching_cubes_hier(
                            geometric_func=lambda x: self.forward(
                                batch.unsqueeze(0), x.unsqueeze(0), skip_transformer=True
                            ).sdf.squeeze(0)
                        )
                    )
            else:
                for b, batch in enumerate(pc.split(1, dim=0)):
                    source_skeleton_ = None if source_skeleton is None else source_skeleton[b : b + 1]
                    source_bw_ = None if source_bw is None else source_bw[b : b + 1]
                    for target_skeleton_ in target_skeleton[b : b + 1].split(1, dim=1):
                        mesh_list.append(
                            self.marching_cubes_hier(
                                geometric_func=lambda x: self.forward(
                                    batch,
                                    x.unsqueeze(0),
                                    source_skeleton=source_skeleton_,
                                    target_skeleton=target_skeleton_,
                                    source_bw=source_bw_,
                                    skip_transformer=False,
                                ).sdf.squeeze(0)
                            )
                        )

        elif self.ae_type == "hy3d2.1":
            with torch.amp.autocast("cuda", enabled=True):
                latents = self.forward(
                    pc,
                    torch.empty_like(pc)[:, :1, ...].repeat(1, P_out, 1, 1),
                    skip_transformer=skip_transformer,
                    skip_decoding=True,
                    **kwargs,
                ).latents_sample
                latents = latents.reshape(B * P_out, -1, latents.shape[-1])
                latents = self.ae.decode(latents)
                self.ae.enable_flashvdm_decoder(enabled=True, mc_algo="mc")
                mesh_list = self.ae.latents2mesh(
                    latents,
                    output_type="trimesh",
                    bounds=1.01,
                    mc_level=0.0,
                    num_chunks=20000,
                    octree_resolution=octree_resolution,
                    mc_algo="mc",
                    enable_pbar=True,
                )
                self.ae.enable_flashvdm_decoder(enabled=False)
            mesh_list = export_to_trimesh(mesh_list)
            if post_process:
                for i, mesh in enumerate(mesh_list):
                    mesh = FloaterRemover()(mesh)
                    mesh = DegenerateFaceRemover()(mesh)
                    mesh_list[i] = mesh

        return np.array(mesh_list).reshape(B, P_out).tolist()
