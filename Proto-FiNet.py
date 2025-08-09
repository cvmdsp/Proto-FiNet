import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import fvcore.nn.weight_init as weight_init
from torch.cuda.amp import autocast
from geoseg.models.vit_modeling import LayerScale
from .d2util import configurable, Conv2d
from geoseg.models.position_encoding import PositionEmbeddingSine
from geoseg.models.sinkhorn import distributed_sinkhorn
from geoseg.models.scalemae_encoder import ScaleMAE_Encoder
from geoseg.models.LoRA_ScaleMAE_Encoder import LoRA_ScaleMAE_Encoder




class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False, is_mhsa_float32=False, use_layer_scale=False):
        super().__init__()
        self.is_mhsa_float32 = is_mhsa_float32
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.ls1 = LayerScale(d_model, init_values=1e-5) if use_layer_scale else nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        if self.is_mhsa_float32:
            with autocast(enabled=False):
                tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                      key_padding_mask=tgt_key_padding_mask)[0]
        else:
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(self.ls1(tgt2))
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        if self.is_mhsa_float32:
            with autocast(enabled=False):
                tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                                      key_padding_mask=tgt_key_padding_mask)[0]
        else:
            tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(self.ls1(tgt2))

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False, is_mhsa_float32=False, use_layer_scale=False):
        super().__init__()
        self.is_mhsa_float32 = is_mhsa_float32
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.ls1 = nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        if self.is_mhsa_float32:
            with autocast(enabled=False):
                tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                           key=self.with_pos_embed(memory, pos),
                                           value=memory, attn_mask=memory_mask,
                                           key_padding_mask=memory_key_padding_mask)[0]
        else:
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                       key=self.with_pos_embed(memory, pos),
                                       value=memory, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(self.ls1(tgt2))
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        if self.is_mhsa_float32:
            with autocast(enabled=False):
                tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                           key=self.with_pos_embed(memory, pos),
                                           value=memory, attn_mask=memory_mask,
                                           key_padding_mask=memory_key_padding_mask)[0]

        else:
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                       key=self.with_pos_embed(memory, pos),
                                       value=memory, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask)[0]

        tgt = tgt + self.dropout(self.ls1(tgt2))

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False, use_layer_scale=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.ls1 = LayerScale(d_model, init_values=1e-5) if use_layer_scale else nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(self.ls1(tgt))
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(self.ls1(tgt2))
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP2(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# class MultiScaleMaskedTransformerDecoder3d(nn.Module):
#
#     def __init__(
#             self,
#             in_channels,
#             mask_classification=True,
#             *,
#             num_classes: int,
#             hidden_dim: int,
#             num_queries: int,
#             nheads: int,
#             dim_feedforward: int,
#             dec_layers: int,
#             pre_norm: bool,
#             mask_dim: int,
#             enforce_input_project: bool,
#             non_object: bool,
#             num_feature_levels: int,  # new
#             is_masking: bool,  # new
#             is_masking_argmax: bool,  # new
#             is_mhsa_float32: bool,  # new
#             no_max_hw_pe: bool,  # new
#             use_layer_scale: bool,  # new
#     ):
#         super().__init__()
#
#         self.no_max_hw_pe = no_max_hw_pe
#         self.is_masking = is_masking
#         self.is_masking_argmax = is_masking_argmax
#         self.num_classes = num_classes
#         self.mask_classification = mask_classification
#
#         # positional encoding
#         N_steps = 128  # hidden_dim // 3
#         self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
#
#         # define Transformer decoder here
#         self.num_heads = nheads
#         self.num_layers = dec_layers
#         self.transformer_self_attention_layers = nn.ModuleList()
#         self.transformer_cross_attention_layers = nn.ModuleList()
#         self.transformer_ffn_layers = nn.ModuleList()
#
#         for _ in range(self.num_layers):
#             self.transformer_self_attention_layers.append(
#                 SelfAttentionLayer(
#                     d_model=hidden_dim,
#                     nhead=nheads,
#                     dropout=0.0,
#                     normalize_before=pre_norm,
#                     is_mhsa_float32=is_mhsa_float32,
#                     use_layer_scale=use_layer_scale,
#                 )
#             )
#
#             self.transformer_cross_attention_layers.append(
#                 CrossAttentionLayer(
#                     d_model=hidden_dim,
#                     nhead=nheads,
#                     dropout=0.0,
#                     normalize_before=pre_norm,
#                     is_mhsa_float32=is_mhsa_float32,
#                     use_layer_scale=use_layer_scale,
#                 )
#             )
#
#             self.transformer_ffn_layers.append(
#                 FFNLayer(
#                     d_model=hidden_dim,
#                     dim_feedforward=dim_feedforward,
#                     dropout=0.0,
#                     normalize_before=pre_norm,
#                     use_layer_scale=use_layer_scale,
#                 )
#             )
#
#         self.decoder_norm = nn.LayerNorm(hidden_dim)
#
#         self.num_queries = num_queries
#         # learnable query features
#         self.query_feat = nn.Embedding(num_queries, hidden_dim)
#         # learnable query p.e.
#         self.query_embed = nn.Embedding(num_queries, hidden_dim)
#
#         self.num_feature_levels = num_feature_levels
#         self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
#         self.input_proj = nn.ModuleList()
#         # print("1111")
#         for _ in range(self.num_feature_levels):
#             # print("111")
#             if in_channels != hidden_dim or enforce_input_project:
#                 self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
#                 weight_init.c2_xavier_fill(self.input_proj[-1])
#                 # print("1")
#             else:
#                 self.input_proj.append(nn.Sequential())
#                 # print("11")
#         # self.level_embed = nn.Embedding(self.num_feature_levels, in_channels)
#         if self.mask_classification:
#             self.class_embed = nn.Linear(hidden_dim, num_classes + int(non_object))
#         self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
#
#     @classmethod
#     def from_config(cls, cfg, in_channels, mask_classification):
#         if isinstance(cfg, dict):
#             ret = {}
#             ret["in_channels"] = in_channels
#             ret["mask_classification"] = mask_classification
#             ret["num_classes"] = cfg["num_classes"]
#             ret["hidden_dim"] = cfg["hidden_dim"]
#             ret["num_queries"] = cfg["num_queries"]
#             ret["nheads"] = cfg["nheads"]
#             ret["dim_feedforward"] = cfg["dim_feedforward"]
#             ret["dec_layers"] = cfg["dec_layers"] - 1
#             ret["pre_norm"] = cfg["pre_norm"]
#             ret["enforce_input_project"] = cfg["enforce_input_project"]
#             ret["mask_dim"] = cfg["mask_dim"]
#
#         else:
#             ret = {}
#             ret["in_channels"] = in_channels
#             ret["mask_classification"] = mask_classification
#             ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
#             ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
#             ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
#             ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
#             ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
#             assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
#             ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
#             ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
#             ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ
#             ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
#
#         return ret
#
#     def forward(self, x, mask_features, mask=None):
#         # if self.num_feature_levels > 1 and not isinstance(x, torch.Tensor):
#         #     assert len(x) == self.num_feature_levels, "x {} num_feature_levels {} ".format(x.shape,
#         #                                                                                    self.num_feature_levels)
#         # else:
#         # x = [x]
#         src = []
#         pos = []
#         size_list = []
#
#         size_list.append(x.shape[-2:])
#
#         pos.append(self.pe_layer(x, None).flatten(2))
#
#         src.append(x.flatten(2) + self.level_embed.weight[0][None, :, None])  # b*c*dhw
#         # src.append(x.flatten(2) + self.level_embed.weight.unsqueeze(1))
#         # first_tensor_shape = src[0].shape
#         # print(src[0].shape,pos[0].shape)
#
#         # flatten NxCxDxHxW to DHWxNxC
#         pos[0] = pos[0].permute(2, 0, 1)
#         src[0] = src[0].permute(2, 0, 1)  # b*c*dhw--> dhw*b*c
#
#         _, bs, _ = src[0].shape  # dhw*b*c
#
#         # QxNxC
#         query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)  # p.e.
#         output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)  # qxbxc
#
#         predictions_class = []
#         predictions_mask = []
#
#         # prediction heads on learnable query features
#         outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features,
#                                                                                attn_mask_target_size=size_list[0],
#                                                                                mask_classification=self.mask_classification)
#         # if self.mask_classification:
#         #     predictions_class.append(outputs_class)
#         # predictions_mask.append(outputs_mask)
#         #
#         # for i in range(self.num_layers):
#         #     level_index = i % self.num_feature_levels
#         test = (attn_mask.sum(-1) == attn_mask.shape[-1])
#         attn_mask[test] = False
#         # attn_mask = None
#         # attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
#
#         # print(src[0].shape,pos[0].shape)
#         output = self.transformer_cross_attention_layers[0](
#             output, src[0],
#             memory_mask=attn_mask,
#             memory_key_padding_mask=None,
#             pos=pos[0] if not self.no_max_hw_pe else None, query_pos=query_embed
#         )
#
#         output = self.transformer_self_attention_layers[0](
#             output, tgt_mask=None,
#             tgt_key_padding_mask=None,
#             query_pos=query_embed
#         )
#
#         # FFN
#         output = self.transformer_ffn_layers[0](
#             output
#         )
#
#         outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features,
#                                                                                attn_mask_target_size=size_list[0],
#                                                                                mask_classification=self.mask_classification)
#
#         #     if self.mask_classification:
#         #         predictions_class.append(outputs_class)
#         #     predictions_mask.append(outputs_mask)
#         #
#         # if self.mask_classification:
#         #     assert len(predictions_class) == self.num_layers + 1
#
#         # out = {
#         #     'pred_logits': predictions_class[-1] if self.mask_classification else None,
#         #     'pred_masks': predictions_mask[-1],
#         #     'aux_outputs': self._set_aux_loss(
#         #         predictions_class if self.mask_classification else None, predictions_mask
#         #     )
#         # }
#
#         return outputs_mask
#
#     def forward_prediction_heads(self, output, mask_features, attn_mask_target_size, mask_classification=True):
#         # print("1",output.shape)
#         decoder_output = self.decoder_norm(output)  # qbc       dxhxw
#         # print("2", decoder_output.shape)
#         decoder_output = decoder_output.transpose(0, 1)
#         # print("3", decoder_output.shape)
#
#         outputs_class = self.class_embed(decoder_output) if mask_classification else None  # (b, num_query, n_class+1)
#
#         mask_embed = self.mask_embed(decoder_output)  # bqc
#         # print("4", mask_embed.shape)
#         # print("mask_features", mask_features.shape)
#         outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
#         # print("mask", outputs_mask.shape)
#
#         attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
#         # print(attn_mask.shape)
#         if self.is_masking_argmax:
#             attn_mask = torch.argmax(attn_mask.flatten(2), dim=1)  # 4 1 1
#             attn_mask = nn.functional.one_hot(attn_mask, num_classes=self.num_classes)
#             attn_mask = attn_mask.permute((0, 2, 1)).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1).bool()
#         else:
#             attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
#                                                                                                              1) < 0.5).bool()
#
#         # print(attn_mask.shape)
#         attn_mask = attn_mask.detach()
#
#         return outputs_class, outputs_mask, attn_mask
#
#     @torch.jit.unused
#     def _set_aux_loss(self, outputs_class, outputs_seg_masks):
#         if self.mask_classification:
#             return [
#                 {"pred_logits": a, "pred_masks": b}
#                 for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
#             ]
#         else:
#             return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
#
class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256):
        super(ProjectionHead, self).__init__()

        self.proj = self.mlp2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_in, proj_dim, 1))

    def forward(self, x):
        return l2_normalize(self.proj(x))

def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)
class MultiScaleMaskedTransformerDecoder3d(nn.Module):

    def __init__(
            self,
            in_channels,
            mask_classification=True,
            *,
            num_classes: int,
            num_prototype: int,
            hidden_dim: int,
            num_queries: int,
            nheads: int,
            dim_feedforward: int,
            dec_layers: int,
            pre_norm: bool,
            mask_dim: int,
            enforce_input_project: bool,
            non_object: bool,
            num_feature_levels: int,  # new
            is_masking: bool,  # new
            is_masking_argmax: bool,  # new
            is_mhsa_float32: bool,  # new
            no_max_hw_pe: bool,  # new
            use_layer_scale: bool,  # new
            scale_dim: int,
    ):
        super().__init__()

        self.no_max_hw_pe = no_max_hw_pe
        self.is_masking = is_masking
        self.is_masking_argmax = is_masking_argmax
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_prototype = num_prototype
        self.mask_classification = mask_classification
        self.scale_dim = scale_dim

        # positional encoding
        N_steps = 128  #hidden_dim // 3
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                    is_mhsa_float32=is_mhsa_float32,
                    use_layer_scale=use_layer_scale,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                    is_mhsa_float32=is_mhsa_float32,
                    use_layer_scale=use_layer_scale,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                    use_layer_scale=use_layer_scale,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Parameter(
    torch.randn(self.num_classes, self.num_prototype, hidden_dim),
    requires_grad=True
)
        # learnable query p.e.
        self.query_embed = nn.Parameter(
    torch.randn(self.num_classes, self.num_prototype, hidden_dim),
    requires_grad=True
)

        self.num_feature_levels = num_feature_levels
        #self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.level_embed = nn.Embedding(4, hidden_dim)
        self.input_proj = nn.ModuleList()
        #print("1111")
        for _ in range(self.num_feature_levels):
            #print("111")
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
                #print("1")
            else:
                self.input_proj.append(nn.Sequential())
                #print("11")
        #self.level_embed = nn.Embedding(self.num_feature_levels, in_channels)
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + int(non_object))
        self.mask_embed = MLP2(hidden_dim, hidden_dim, mask_dim, 3)
        self.mask_norm = nn.LayerNorm(self.num_classes)
        self.proj_head = ProjectionHead(hidden_dim, hidden_dim)
        self.feat_norm = nn.LayerNorm(hidden_dim)
        # self.linear = nn.Linear(128, 256)
        self.conv_layer = nn.Conv2d(in_channels=scale_dim, out_channels=hidden_dim, kernel_size=1)

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        if isinstance(cfg, dict):
            ret = {}
            ret["in_channels"] = in_channels
            ret["mask_classification"] = mask_classification
            ret["num_classes"] = cfg["num_classes"]
            ret["hidden_dim"] = cfg["hidden_dim"]
            ret["num_queries"] = cfg["num_queries"]
            ret["nheads"] = cfg["nheads"]
            ret["dim_feedforward"] = cfg["dim_feedforward"]
            ret["dec_layers"] = cfg["dec_layers"] - 1
            ret["pre_norm"] = cfg["pre_norm"]
            ret["enforce_input_project"] = cfg["enforce_input_project"]
            ret["mask_dim"] = cfg["mask_dim"]

        else:
            ret = {}
            ret["in_channels"] = in_channels
            ret["mask_classification"] = mask_classification
            ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
            ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
            ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
            ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
            ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
            assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
            ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
            ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
            ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ
            ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        return ret

    def forward(self, scale, x1, x2, x3, gt_seg, mask_features, mask=None):
        # if self.num_feature_levels > 1 and not isinstance(x, torch.Tensor):
        #     assert len(x) == self.num_feature_levels, "x {} num_feature_levels {} ".format(x.shape,
        #                                                                                    self.num_feature_levels)
        # else:
        #x = [x]
        src = []
        pos = []
        size_list = []

        # print(scale.shape)
        # scale_dim = scale.shape[1]
        # print(scale_dim)

        # conv_layer = nn.Conv2d(in_channels=scale_dim, out_channels=2*scale_dim, kernel_size=1)

        scale = self.conv_layer(scale)


        size_list.append(x1.shape[-2:])

        size_list.append(x2.shape[-2:])
        size_list.append(x3.shape[-2:])
        size_list.append(scale.shape[-2:])
        # size_list.append(x4.shape[-2:])
        # print("Shape of x1:", size_list[0])
        # print("Shape of x2:", size_list[1])
        # print("Shape of x3:", size_list[2])

        pos.append(self.pe_layer(x1, None).flatten(2))

        src.append(x1.flatten(2) + self.level_embed.weight[0][None, :, None])  # b*c*dhw

        pos.append(self.pe_layer(x2, None).flatten(2))

        src.append(x2.flatten(2) + self.level_embed.weight[1][None, :, None])  # b*c*dhw

        pos.append(self.pe_layer(x3, None).flatten(2))

        src.append(x3.flatten(2) + self.level_embed.weight[1][None, :, None])  # b*c*dhw

        pos.append(self.pe_layer(scale, None).flatten(2))

        src.append(scale.flatten(2) + self.level_embed.weight[1][None, :, None])  # b*c*dhw

        pos[3] = pos[3].permute(2, 0, 1)
        src[3] = src[3].permute(2, 0, 1)

        #src.append(x.flatten(2) + self.level_embed.weight.unsqueeze(1))
        # first_tensor_shape = src[0].shape
        #print(src[0].shape,pos[0].shape)

        # flatten NxCxDxHxW to DHWxNxC
        pos[0] = pos[0].permute(2, 0, 1)
        src[0] = src[0].permute(2, 0, 1)  #b*c*dhw--> dhw*b*c

        _, bs, _ = src[0].shape  #dhw*b*c

        # QxNxC
        query_embed = self.query_embed.unsqueeze(2).repeat(1, 1, bs, 1)  # p.e.
        output = self.query_feat.unsqueeze(2).repeat(1, 1, bs, 1)  #nxkxbxc
        num_queries = self.num_classes * self.num_prototype
        query_embed = query_embed.view(num_queries, bs, self.hidden_dim)  # [num_queries, bs, hidden_dim]
        output = output.view(num_queries, bs, self.hidden_dim)  # [num_queries, bs, hidden_dim]
        output = self.transformer_cross_attention_layers[0](
            output, src[3],

            memory_key_padding_mask=None,
            pos=pos[3] if not self.no_max_hw_pe else None, query_pos=query_embed
        )

        output = self.transformer_self_attention_layers[0](
            output, tgt_mask=None,
            tgt_key_padding_mask=None,
            query_pos=query_embed
        )

        # FFN
        output = self.transformer_ffn_layers[0](
            output
        )



        predictions_class = []
        predictions_mask = []
        contrast_logits = []
        contrast_target = []

        output = output.view(self.num_classes, self.num_prototype, bs, self.hidden_dim)

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask, proto_logits, proto_target  = self.forward_prediction_heads(output, num_queries, bs, mask_features,gt_seg,
                                                                               attn_mask_target_size=size_list[0],
                                                                               mask_classification=self.mask_classification)
        predictions_mask.append(outputs_mask)
        contrast_logits.append(proto_logits)
        # contrast_target.append(proto_target)


        output = output.view(num_queries, bs, self.hidden_dim)
        # if self.mask_classification:
        #     predictions_class.append(outputs_class)
        # predictions_mask.append(outputs_mask)
        #
        # for i in range(self.num_layers):
        #     level_index = i % self.num_feature_levels
        # test = (attn_mask.sum(-1) == attn_mask.shape[-1])
        # attn_mask[test] = False
        #attn_mask = None
        attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

        #print(src[0].shape,pos[0].shape)
        output = self.transformer_cross_attention_layers[1](
            output, src[0],
            memory_mask=attn_mask,
            memory_key_padding_mask=None,
            pos=pos[0] if not self.no_max_hw_pe else None, query_pos=query_embed
        )

        output = self.transformer_self_attention_layers[1](
            output, tgt_mask=None,
            tgt_key_padding_mask=None,
            query_pos=query_embed
        )

        # FFN
        output = self.transformer_ffn_layers[1](
            output
        )

        output = output.view(self.num_classes, self.num_prototype, bs, self.hidden_dim)


        outputs_class, outputs_mask, attn_mask, proto_logits, proto_target = self.forward_prediction_heads(output,num_queries, bs, mask_features,gt_seg,
                                                                               attn_mask_target_size=size_list[1],
                                                                              mask_classification=self.mask_classification)
        # outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, num_queries, bs, mask_features,
        #                                                                        attn_mask_target_size=size_list[1],
        #                                                                        mask_classification=self.mask_classification)
        output = output.view(num_queries, bs, self.hidden_dim)

        predictions_mask.append(outputs_mask)
        contrast_logits.append(proto_logits)
        # contrast_target.append(proto_target)

        # pos.append(self.pe_layer(x2, None).flatten(2))
        #
        # src.append(x2.flatten(2) + self.level_embed.weight[1][None, :, None])  # b*c*dhw
        # src.append(x.flatten(2) + self.level_embed.weight.unsqueeze(1))
        # first_tensor_shape = src[0].shape
        # print(src[0].shape,pos[0].shape)

        # flatten NxCxDxHxW to DHWxNxC
        pos[1] = pos[1].permute(2, 0, 1)
        src[1] = src[1].permute(2, 0, 1)  # b*c*dhw--> dhw*b*c



        # if self.mask_classification:
        #     predictions_class.append(outputs_class)
        # predictions_mask.append(outputs_mask)
        #
        # for i in range(self.num_layers):
        #     level_index = i % self.num_feature_levels
        # test = (attn_mask.sum(-1) == attn_mask.shape[-1])
        # attn_mask[test] = False
        #attn_mask = None
        attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

        # print(src[0].shape,pos[0].shape)
        output = self.transformer_cross_attention_layers[2](
            output, src[1],
            memory_mask=attn_mask,
            memory_key_padding_mask=None,
            pos=pos[1] if not self.no_max_hw_pe else None, query_pos=query_embed
        )

        output = self.transformer_self_attention_layers[2](
            output, tgt_mask=None,
            tgt_key_padding_mask=None,
            query_pos=query_embed
        )

        # FFN
        output = self.transformer_ffn_layers[2](
            output
        )

        output = output.view(self.num_classes, self.num_prototype, bs, self.hidden_dim)
        outputs_class, outputs_mask, attn_mask, proto_logits, proto_target = self.forward_prediction_heads(output,num_queries, bs, mask_features,gt_seg,
                                                                               attn_mask_target_size=size_list[2],
                                                                              mask_classification=self.mask_classification)


        output = output.view(num_queries, bs, self.hidden_dim)

        predictions_mask.append(outputs_mask)
        contrast_logits.append(proto_logits)
        # contrast_target.append(proto_target)

        # pos.append(self.pe_layer(x3, None).flatten(2))
        #
        # src.append(x3.flatten(2) + self.level_embed.weight[2][None, :, None])  # b*c*dhw
        # src.append(x.flatten(2) + self.level_embed.weight.unsqueeze(1))
        # first_tensor_shape = src[0].shape
        # print(src[0].shape,pos[0].shape)

        # flatten NxCxDxHxW to DHWxNxC
        pos[2] = pos[2].permute(2, 0, 1)
        src[2] = src[2].permute(2, 0, 1)  # b*c*dhw--> dhw*b*c



        # QxNxC



        # if self.mask_classification:
        #     predictions_class.append(outputs_class)
        # predictions_mask.append(outputs_mask)
        #
        # for i in range(self.num_layers):
        #     level_index = i % self.num_feature_levels
        # test = (attn_mask.sum(-1) == attn_mask.shape[-1])
        # attn_mask[test] = False
        #attn_mask = None
        attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

        # print(src[0].shape,pos[0].shape)
        output = self.transformer_cross_attention_layers[3](
            output, src[2],
            memory_mask=attn_mask,
            memory_key_padding_mask=None,
            pos=pos[2] if not self.no_max_hw_pe else None, query_pos=query_embed
        )

        output = self.transformer_self_attention_layers[3](
            output, tgt_mask=None,
            tgt_key_padding_mask=None,
            query_pos=query_embed
        )

        # FFN
        output = self.transformer_ffn_layers[3](
            output
        )

        output = output.view(self.num_classes, self.num_prototype, bs, self.hidden_dim)
        outputs_class, outputs_mask, attn_mask, proto_logits, proto_target = self.forward_prediction_heads(output,num_queries, bs, mask_features,gt_seg,
                                                                               attn_mask_target_size=size_list[2],
                                                                              mask_classification=self.mask_classification)




        predictions_mask.append(outputs_mask)
        contrast_logits.append(proto_logits)
        contrast_target.append(proto_target)
# #x4
#         pos.append(self.pe_layer(x4, None).flatten(2))
#
#         src.append(x4.flatten(2) + self.level_embed.weight[3][None, :, None])  # b*c*dhw
#         # src.append(x.flatten(2) + self.level_embed.weight.unsqueeze(1))
#         # first_tensor_shape = src[0].shape
#         # print(src[0].shape,pos[0].shape)
#
#         # flatten NxCxDxHxW to DHWxNxC
#         pos[3] = pos[3].permute(2, 0, 1)
#         src[3] = src[3].permute(2, 0, 1)  # b*c*dhw--> dhw*b*c
#
#         # QxNxC
#
#         # if self.mask_classification:
#         #     predictions_class.append(outputs_class)
#         # predictions_mask.append(outputs_mask)
#         #
#         # for i in range(self.num_layers):
#         #     level_index = i % self.num_feature_levels
#         # test = (attn_mask.sum(-1) == attn_mask.shape[-1])
#         # attn_mask[test] = False
#         # attn_mask = None
#         attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
#         # print('output',output.shape)
#         # print('src',src[3].shape)
#         # print('attn_mask',attn_mask.shape)
#
#         # print(src[0].shape,pos[0].shape)
#         output = self.transformer_cross_attention_layers[3](
#             output, src[3],
#             memory_mask=attn_mask,
#             memory_key_padding_mask=None,
#             pos=pos[3] if not self.no_max_hw_pe else None, query_pos=query_embed
#         )
#
#         output = self.transformer_self_attention_layers[3](
#             output, tgt_mask=None,
#             tgt_key_padding_mask=None,
#             query_pos=query_embed
#         )
#
#         # FFN
#         output = self.transformer_ffn_layers[3](
#             output
#         )
#
#         outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features,
#                                                                                attn_mask_target_size=size_list[3],
#                                                                                mask_classification=self.mask_classification)
#
#         predictions_mask.append(outputs_mask)

        # print(predictions_mask[0].shape)
        # print(predictions_mask[1].shape)
        # print(predictions_mask[2].shape)





        #     if self.mask_classification:
        #         predictions_class.append(outputs_class)
        #     predictions_mask.append(outputs_mask)
        #
        # if self.mask_classification:
        #     assert len(predictions_class) == self.num_layers + 1

        # out = {
        #     'pred_logits': predictions_class[-1] if self.mask_classification else None,
        #     'pred_masks': predictions_mask[-1],
        #     'aux_outputs': self._set_aux_loss(
        #         predictions_class if self.mask_classification else None, predictions_mask
        #     )
        # }
        #out_m = [outputs_mask,]

        #return outputs_mask





        return predictions_mask, contrast_logits, contrast_target


    def forward_prediction_heads(self, output, num_queries, bs, mask_features, gt_seg, attn_mask_target_size, mask_classification=True):
        #print("1",output.shape)
        decoder_output = self.decoder_norm(output)  #mkbc       dxhxw
        #print("2", decoder_output.shape)
        decoder_output = decoder_output.permute(2, 0, 1, 3)#bmkc
        #print("3", decoder_output.shape)

        outputs_class = self.class_embed(decoder_output) if mask_classification else None  # (b, num_query, n_class+1)

        mask_embed = self.mask_embed(decoder_output)  # bmkc
        # print("4", mask_embed.shape)
        # print("mask_features", mask_features.shape)
        mask = torch.einsum("bmkc,bchw->bmkhw", mask_embed, mask_features)
        # print(mask.shape)

        permuted_features = mask.permute(0, 3, 4, 1, 2)  # [b, h, w, m, k]
        attn_mask = permuted_features.flatten(-2, -1)  # [b, h, w, m*k]
        attn_mask = attn_mask.permute(0, 3, 1, 2)  # [b, m*k, h, w]

        outputs_mask = torch.amax(mask, dim=2)
        outputs_mask = outputs_mask.permute(0,2,3,1)


        #print(outputs_mask.shape)
        outputs_mask = self.mask_norm(outputs_mask)#[4, 6, 256, 256]

        outputs_mask = outputs_mask.permute(0,3,1,2)

        #print("mask", outputs_mask.shape)
        c = self.proj_head(mask_features)
        #print("1",c.shape)
        #_c = rearrange(c, 'b c h w -> (b h w) c')
        _c = rearrange(c, 'b c h w -> b (h w) c')
        #print(_c.shape)
        _c = self.feat_norm(_c)
        _c = l2_normalize(_c)
        #print(_c.shape)

        if gt_seg is None:
            proto_logits, proto_target = None, None

        else:

            proto_logits, proto_target = self.prototype_learning(_c, num_queries, bs, output, gt_seg, mask)

        attn_mask = F.interpolate(attn_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        #print(attn_mask.shape)
        if self.is_masking_argmax:
            attn_mask = torch.argmax(attn_mask.flatten(2), dim=1)  # 4 1 1
            attn_mask = nn.functional.one_hot(attn_mask, num_classes=self.num_classes)
            attn_mask = attn_mask.permute((0, 2, 1)).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1).bool()
        else:
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                             1) < 0.5).bool()

        #print(attn_mask.shape)
        attn_mask = attn_mask.detach()




        return outputs_class, outputs_mask, attn_mask, proto_logits, proto_target
    # def forward_prediction_heads(self, output, mask_features,  attn_mask_target_size, mask_classification=True):
    #     #print("1",output.shape)
    #     decoder_output = self.decoder_norm(output)  #mkbc       dxhxw
    #     #print("2", decoder_output.shape)
    #     decoder_output = decoder_output.permute(2, 0, 1, 3)#bmkc
    #     #print("3", decoder_output.shape)
    #
    #     outputs_class = self.class_embed(decoder_output) if mask_classification else None  # (b, num_query, n_class+1)
    #
    #     mask_embed = self.mask_embed(decoder_output)  # bmkc
    #     # print("4", mask_embed.shape)
    #     # print("mask_features", mask_features.shape)
    #     mask = torch.einsum("bmkc,bchw->bmkhw", mask_embed, mask_features)
    #     # print(mask.shape)
    #
    #     permuted_features = mask.permute(0, 3, 4, 1, 2)  # [b, h, w, m, k]
    #     attn_mask = permuted_features.flatten(-2, -1)  # [b, h, w, m*k]
    #     attn_mask = attn_mask.permute(0, 3, 1, 2)  # [b, m*k, h, w]
    #
    #     outputs_mask = torch.amax(mask, dim=2)
    #     outputs_mask = outputs_mask.permute(0,2,3,1)
    #
    #
    #     #print(outputs_mask.shape)
    #     outputs_mask = self.mask_norm(outputs_mask)#[4, 6, 256, 256]
    #
    #     outputs_mask = outputs_mask.permute(0,3,1,2)
    #
    #
    #
    #     attn_mask = F.interpolate(attn_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
    #     #print(attn_mask.shape)
    #     if self.is_masking_argmax:
    #         attn_mask = torch.argmax(attn_mask.flatten(2), dim=1)  # 4 1 1
    #         attn_mask = nn.functional.one_hot(attn_mask, num_classes=self.num_classes)
    #         attn_mask = attn_mask.permute((0, 2, 1)).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1).bool()
    #     else:
    #         attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
    #                                                                                                          1) < 0.5).bool()
    #
    #     #print(attn_mask.shape)
    #     attn_mask = attn_mask.detach()
    #
    #     return outputs_class, outputs_mask, attn_mask

    def prototype_learning(self, _c, num_queries, bs, output, gt_seg, mask):
        """
        Args:
            _c: 特征张量，形状 [n, c]，其中 n 是像素总数 (b * h * w)，c 是特征维度。
            gt_seg: Ground Truth，形状 [b, h, w]。
            mask: 掩码张量，形状 [b, m, k, h, w]。

        Returns:
            proto_logits: 原型分布的对数几率，形状 [n, num_classes * num_prototype]。00
            proto_target: 联合类别和原型的目标索引，形状 [n]。
        """
        batch_size, num_classes, num_prototype, height, width = mask.shape
        # print(mask.shape)
        # print("1",gt_seg.shape)
        gt_seg = F.interpolate(
            gt_seg.unsqueeze(1).float(),  # 增加通道维度并转换为浮点数
            size=(height, width),  # 指定目标尺寸
            mode="nearest"  # 最近邻插值，适用于分类标签
        ).squeeze(1).long()  # 去掉通道维度并恢复为整数类型
        # print("2",gt_seg.shape)

        # 1. 将 gt_seg 展平到与 _c 对应的维度
        gt_seg_flat = gt_seg.flatten()  # [n]
        # print(gt_seg.shape)
        # print("3",gt_seg_flat.shape)

        # 2. 展平 mask 的空间维度
        mask_flat = mask.view(batch_size, num_classes, num_prototype, -1)  # [b, m, k, h*w]
        # print(mask.shape)
        # print(mask_flat.shape)
        mask_flat = mask_flat.permute(0, 3, 1, 2).reshape(-1, num_classes, num_prototype)  # [n, m, k]
        # print(mask_flat.shape)

        output = output.view(num_queries, bs, self.hidden_dim)
   
        result = []  # 用于存储每个 b 的计算结果

        for i in range(bs):  # 遍历 batch_size 维度
            _c_single = _c[i]  # 提取当前 batch 的 _c，尺寸为 [hw, c]
            output_single = output[:, i, :]  # 提取当前 batch 的 output，尺寸为 [q, c]

            # 转置 output_single 并进行计算
            # proto_logits_single = torch.einsum('nc,cm->nm', _c_single, output_single.T)  # [hw, q]
            proto_logits_single = torch.mm(_c_single, output_single.T)  # [hw, q]

            result.append(proto_logits_single)  # 存储计算结果
            # print(proto_logits_single.shape)
            # print(bs)

        # 将结果沿 batch_size 维度合并，形成 [b, hw, q]
        proto_logits = torch.stack(result, dim=0)
        # print("5",proto_logits.shape)
        proto_logits = proto_logits.view(-1, num_queries)#(bhw,c)
        #print("6",proto_logits.shape)
        """

        # 3. 计算原型 logits (像素特征与原型的相似度)
        prototypes = output  # [m*k,c]
   
        proto_logits = torch.einsum('nc,qbc->nm', _c, prototypes)  # [n, num_classes * num_prototype]
        """

        # 4. 初始化 proto_target
        proto_target = gt_seg_flat.clone().float()  # [n]
        #print("4",proto_target.shape)

        # 5. 遍历类别，更新 proto_target
        for k in range(num_classes):
            # 筛选属于类别 k 的像素
            class_mask = (gt_seg_flat == k)  # [n]
            # print("class_mask",class_mask.shape) #262144
            if class_mask.sum() == 0:
                continue

            # 提取类别 k 的 mask
            init_q = mask_flat[class_mask, k, :]# [n_k, k]
            # init_q = mask_flat[class_mask, :]
            # print(mask_flat.shape)

            # init_q = mask_flat[class_mask, ...]

            # print(init_q.shape)
            if init_q.shape[0] == 0:
                continue

            # 对类别 k 的特征执行 Sinkhorn 聚类
            q, indexs = distributed_sinkhorn(init_q)  # q: [k, n_k]，indexs: [n_k]
            # print("q",q.shape)
            # print("indexs",indexs.shape)

            # 更新 proto_target
            proto_target[class_mask] = indexs.float() + (self.num_prototype * k)
            #print("proto_target",proto_target)
        # print("proto_target", proto_target.shape)

        # 返回 logits 和 target


        return proto_logits, proto_target

        # return proto_logits
    #
    # def prototype_learning(self, _c, gt_seg, mask):
    #     # 1. 获取预测类别
    #
    #     # 2. 计算像素特征与所有原型的余弦相似度
    #     cosine_similarity = torch.mm(_c, self.prototypes.view(-1, self.prototypes.shape[-1]).t())
    #
    #     proto_logits = cosine_similarity
    #     proto_target = gt_seg.clone().float()
    #
    #     # 3. 遍历每个类别，计算原型目标
    #     for k in range(self.num_classes):
    #         # 获取类别 k 的掩码和特征
    #         init_q = mask[:, k, :, :, :]
    #         init_q = init_q[gt_seg == k, ...]
    #         if init_q.shape[0] == 0:
    #             continue
    #
    #         # 使用 Sinkhorn 聚类，获取分布和索引
    #         _, indexs = distributed_sinkhorn(init_q)
    #
    #         # 更新 proto_target（类别和原型的联合标签）
    #         proto_target[gt_seg == k] = indexs.float() + (self.num_prototype * k)
    #
    #     return proto_logits, proto_target

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # print(self.relative_position_bias_table.shape)
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        # print(coords_h, coords_w)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        # print(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        # print(coords_flatten)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        # print(relative_coords[0,7,:])
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        # print(relative_coords[:, :, 0], relative_coords[:, :, 1])
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        # print(B_,N,C)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        # print(attn.shape)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH

        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # print(relative_position_bias.unsqueeze(0))
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


class SwinTransformer(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        pretrain_img_size (int): Input image size for training the pretrained models,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=128,
                 depths=[2, 2, 18, 2],
                 num_heads=[4, 8, 16, 32],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.3,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features
        self.apply(self._init_weights)

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)
        # print('patch_embed', x.size())

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
                # print('layer{} out size {}'.format(i, out.size()))

        return tuple(outs)

    def train(self, mode=True):
        """Convert the models into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

# class SimpleCNNFeatureExtractor(nn.Module):
#     def __init__(self, input_channels=3):
#         super(SimpleCNNFeatureExtractor, self).__init__()
#
#         # 第一个卷积层：3x3卷积，步幅为2x2，输入空间尺寸减半 (768 -> 384)
#         self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.gelu1 = nn.GELU()
#
#
#         # 第二个卷积层：3x3卷积，大小不变 (384 -> 192)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(128)
#         self.gelu2 = nn.GELU()
#
#
#         # 第三个卷积层：3x3卷积，大小不变 (192 -> 96)
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm2d(256)
#         self.gelu3 = nn.GELU()
#         self.cfb3 = CFBlock(256, 256)
#
#         # 第四个卷积层：1x1卷积，将通道数变为256 (96 -> 96)
#         self.conv4 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
#         self.bn4 = nn.BatchNorm2d(128)
#         self.gelu4 = nn.GELU()
#
#         # 第五个卷积层：将空间尺寸变为256x256
#         self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
#         self.bn5 = nn.BatchNorm2d(256)
#         self.gelu5 = nn.GELU()
#
#         self.upsample = nn.Upsample(scale_factor=3, mode='bilinear', align_corners=False)
#
#         # 最后的卷积层，用于将通道数恢复到输入的通道数
#
#
#     def forward(self, x):
#         input_tensor = x
#
#         x = self.gelu1(self.bn1(self.conv1(x)))  # (3, 768, 768) -> (64, 384, 384)
#        # x = self.cfb1(x)
#         x = self.gelu2(self.bn2(self.conv2(x)))  # (64, 384, 384) -> (128, 384, 384)
#        # x = self.cfb2(x)
#         x = self.gelu3(self.bn3(self.conv3(x)))  # (128, 384, 384) -> (256, 384, 384)
#        # x = self.cfb3(x)
#         x = self.gelu4(self.bn4(self.conv4(x)))  # (256, 384, 384) -> (256, 384, 384)
#         #print(x.shape)
#
#
#
#         #x = F.interpolate(x, size=(512, 512), mode='bilinear',align_corners=False)
#         #print(x.shape)
#         #                   align_corners=False)  # (256, 384, 384) -> (256, 256, 256)
#        # x = self.gelu5(self.bn5(self.conv5(x)))  # (256, 256, 256) -> (256, 256, 256)
#
#
#         return x
class Mlp_decoder(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalLocalAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 relative_pos_embedding=True
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3 * dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1, padding=(window_size // 2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size // 2 - 1))

        self.relative_pos_embedding = relative_pos_embedding

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape

        local = self.local2(x) + self.local1(x)

        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)

        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v

        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]

        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))

        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        # print(out.size())
        out = out[:, :, :H, :W]

        return out


class Block(nn.Module):
    def __init__(self, dim=256, num_heads=16, mlp_ratio=4., qkv_bias=False, drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_decoder(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                               drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = ConvBN(in_channels, decode_channels, kernel_size=3)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class FeatureRefinementHead(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pre_conv = ConvBN(in_channels, decode_channels, kernel_size=3)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

        self.pa = nn.Sequential(
            nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
            nn.Sigmoid())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                Conv(decode_channels, decode_channels // 16, kernel_size=1),
                                nn.ReLU6(),
                                Conv(decode_channels // 16, decode_channels, kernel_size=1),
                                nn.Sigmoid())

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6()

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca
        x = self.proj(x) + shortcut
        x = self.act(x)

        return x


class AuxHead(nn.Module):

    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x, h, w):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
        return feat


class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=(64, 128, 256, 512),
                 decode_channels=64,
                 dropout=0.1,
                 window_size=8,
                 num_classes=7,
                 num_prototype=10):
        super(Decoder, self).__init__()

        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.b4 = Block(dim=decode_channels, num_heads=16, window_size=window_size)

        self.b3 = Block(dim=decode_channels, num_heads=16, window_size=window_size)
        self.p3 = WF(encoder_channels[-2], decode_channels)

        self.b2 = Block(dim=decode_channels, num_heads=16, window_size=window_size)
        self.p2 = WF(encoder_channels[-3], decode_channels)

        self.p1 = FeatureRefinementHead(encoder_channels[-4], decode_channels)

        self.p1 = WF(encoder_channels[-4], decode_channels)
        self.b1 = Block(dim=decode_channels, num_heads=16, window_size=window_size)

        self.f1 = FeatureRefinementHead(encoder_channels[-4], decode_channels)
        self.p0 = WF(encoder_channels[-4], decode_channels)

        # self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
        #                                        nn.Dropout2d(p=dropout, inplace=True),
        #                                        Conv(decode_channels, num_classes, kernel_size=1))
        self.segmentation_head = nn.Sequential(ConvBNReLU(6, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.decoder3d = MultiScaleMaskedTransformerDecoder3d(
            in_channels=1024,
            mask_classification=True,
            num_classes=7,
            hidden_dim=256,
            num_queries=60,
            num_prototype=10,
            nheads=8,
            dim_feedforward=1024,
            dec_layers=4,
            pre_norm=False,
            mask_dim=256,
            enforce_input_project=False,
            non_object=False,
            is_masking=True,
            is_masking_argmax=False,

            is_mhsa_float32=False,

            no_max_hw_pe=False,

            use_layer_scale=False,
            num_feature_levels=1,
            scale_dim=128

        )
        self.init_weight()

    def forward(self, scale, res1, res2, res3, res4, mask, h, w):

        x1 = self.b4(self.pre_conv(res4))
        #print(x1.shape)

        x = self.p3(x1, res3)
        #print(x.shape)
        x2 = self.b3(x)
        #print(x2.shape)

        x = self.p2(x2, res2)
        x3 = self.b2(x)
        # print(x.shape)
        # print(x3.shape)
        x = self.f1(x3, res1)


        # print(stem.shape)
        # print('1')
        #
        # print(x4.shape)



        # x = self.p0(x4, stem)

        #print(x4.shape)

        predictions_mask, contrast_logits, contrast_target = self.decoder3d(scale=scale, x1=x1, x2=x2, x3=x3,
                                                                            gt_seg=mask, mask_features=x)
        #print("-----------")
        #print(x.shape)




        return predictions_mask, contrast_logits, contrast_target



    def init_weight(self):
        for m in self.children():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class FTUNetFormer_Scale(nn.Module):

    def __init__(self,
                 decode_channels=256,
                 dropout=0.2,
                 embed_dim=96,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 freeze_stages=-1,
                 window_size=8,
                 num_classes=6,
                 num_prototype=10,
                 scale_encoder_weights='./scalemae-vitlarge-800.pth',
                 scale_encoder_download_url='https://github.com/bair-climate-initiative/scale-mae/releases/download/base-800/scalemae-vitlarge-800.pth',
                 input_bands=None,  # Define input bands as required
                 output_layers=None
                 ):
        super().__init__()

        self.backbone = SwinTransformer(embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                                        frozen_stages=freeze_stages)
        encoder_channels = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]
        self.decoder = Decoder(encoder_channels, decode_channels, dropout, window_size, num_classes, num_prototype)
        # self.cnnstem = SimpleCNNFeatureExtractor(3)
        self.scale_mae_encoder = LoRA_ScaleMAE_Encoder(
            encoder_weights=scale_encoder_weights,
            download_url=scale_encoder_download_url,
            embed_dim=embed_dim,
            input_size=1024,  # Adjust input size if necessary
            input_bands=input_bands if input_bands else {'optical': ['B4', 'B3', 'B2']},  # Example input bands
            output_layers=[7, 11, 15, 23],
            output_dim=1024,
            r=7

            # multi_temporal=False,
            # multi_temporal_output=False,
            # pyramid_output=False,
        )
    #     self.scale_mae_encoder = ScaleMAE_Encoder(
    #         encoder_weights=scale_encoder_weights,
    #         download_url=scale_encoder_download_url,
    #         embed_dim=embed_dim,
    #         input_size=1024,  # Adjust input size if necessary
    #         input_bands=input_bands if input_bands else {'optical': ['B4', 'B3', 'B2']},  # Example input bands
    #         output_layers=[7, 11, 15, 23],
    #         output_dim=1024,
    #         # multi_temporal=False,
    #         # multi_temporal_output=False,
    #         # pyramid_output=False,
    #     )
    #     self.freeze_encoder_parameters(self.scale_mae_encoder)
    #
    # def freeze_encoder_parameters(self, encoder):
    #     """
    #     Freezes the parameters of the given encoder (ScaleMAE_Encoder) so that they do not get updated during training.
    #     """
    #     for param in encoder.parameters():
    #         param.requires_grad = False

    def forward(self, x, mask=None):
        # print("input",x.shape)
        h, w = x.size()[-2:]
        scale = self.scale_mae_encoder(x)
        # print(scale[-1].shape)
        # stem = self.cnnstem(x)
        res1, res2, res3, res4 = self.backbone(x)
        #print(mask.shape)

        predictions_mask, contrast_logits, contrast_target = self.decoder(scale, res1, res2, res3, res4, mask, h, w)

        predictions_mask_out = []
        for x in predictions_mask:


            # 使用 F.interpolate 将张量调整到 (h, w) 的大小
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

            # 将处理后的张量添加到结果列表
            predictions_mask_out.append(x)
            # print(x.shape)



        if mask is not None:

            return predictions_mask_out, contrast_logits, contrast_target
        else:
            x = predictions_mask_out[3]
            #x = predictions_mask_out

            return x
        #print(x_list[0].shape,x_list[1].shape,x_list[2].shape,x_list[3].shape)

        # return x_list


def ft_unetformer_scale(pretrained=True, num_classes=7, num_prototype=10, freeze_stages=-1, decoder_channels=256,
                         weight_path='pretrain_weights/stseg_base.pth'):
    #def ft_unetformer(pretrained=True, num_classes=6, freeze_stages=-1, decoder_channels=256,
    #weight_path='pretrain_weights/stseg_base.pth'):
    model = FTUNetFormer_Scale(num_classes=num_classes,
                                num_prototype=num_prototype,
                                freeze_stages=freeze_stages,
                                embed_dim=128,
                                depths=(2, 2, 18, 2),
                                num_heads=(4, 8, 16, 32),
                                decode_channels=decoder_channels)

    if pretrained and weight_path is not None:
        old_dict = torch.load(weight_path)['state_dict']
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model
