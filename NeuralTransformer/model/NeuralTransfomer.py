import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops import rearrange
from .utils.util import Block


class PatchEmbed(nn.Module):
    """ EEG to Patch Embedding
    """
    def __init__(self, EEG_size=2000, patch_size=200, in_chans=1, embed_dim=200):
        super().__init__()
        # EEG_size = to_2tuple(EEG_size)
        # patch_size = to_2tuple(patch_size)
        num_patches = 62 * (EEG_size // patch_size)
        self.patch_shape = (1, EEG_size // patch_size)
        self.EEG_size = EEG_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(1, patch_size), stride=(1, patch_size))

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class TemporalConv(nn.Module):
    """ EEG to Patch Embedding
    """
    def __init__(self, in_chans=1, out_chans=8, emb_dim=200):
        '''
        in_chans: in_chans of nn.Conv2d()
        out_chans: out_chans of nn.Conv2d(), determing the output dimension
        '''
        super().__init__()
        self.in_chans = in_chans
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu3 = nn.GELU()


    def forward(self, x, **kwargs):
        x = rearrange(x, 'B N A T -> B (N A) T')
        B, NA, T = x.shape
        x = x.unsqueeze(1)
        x = x.expand(-1, self.in_chans, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # x: (B, in_chans, NA, T)
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        # x: (B, out_chans, NA, T/8)
        x = rearrange(x, 'B C NA T -> B NA (T C)')
        # x: (B, NA, T/8*out_chans)
        return x

class NeuralTransformer(nn.Module):
    def __init__(self, EEG_size=200, # a*t
                 patch_size=200, # t
                 in_chans=62,
                 num_classes=4,
                 embed_dim=200,
                 depth=3,
                 num_heads=10, mlp_ratio=4., qkv_bias=False, qk_norm=None, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,

                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                 use_mean_pooling=True, init_scale=1, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.out_chans = 8
        self.patch_embed = nn.Sequential(
            TemporalConv(out_chans=self.out_chans),
            nn.Linear(patch_size//8*self.out_chans, embed_dim)
        )
        self.time_window = EEG_size // patch_size       # a
        self.patch_size = patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, 128 + 1, embed_dim), requires_grad=True)   # n*a <= 128
        else:
            self.pos_embed = None
        self.time_embed = nn.Parameter(torch.zeros(1, 16, embed_dim), requires_grad=True)           # a < 16
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.rel_pos_bias = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=None)
            for i in range(depth)])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear( EEG_size*in_chans, num_classes) if num_classes > 0 else nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        if self.time_embed is not None:
            trunc_normal_(self.time_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.mask_token, std=.02)
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, input_chans=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        batch_size, n, a, t = x.shape
        input_time_window = a if t == self.patch_size else t
        x = self.patch_embed(x)
        # temper conv x: (b, na, t/8*out_chans)
        # liner x: (b, na, emb_dim)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # cls_tokens: (b, 1, emb_dim)
        x = torch.cat((cls_tokens, x), dim=1)
        # x: (b, na+1, emb_dim)
        input_chans_with_cls_token = np.append(0, input_chans)
        pos_embed_used = self.pos_embed[:, input_chans_with_cls_token]
        # self.pos_embed: (1, 129, emb)
        # pos_embed_used: (1, n+1, emb)
        if self.pos_embed is not None:
            pos_embed = pos_embed_used[:, 1:, :].unsqueeze(2).expand(batch_size, -1, input_time_window, -1).flatten(1, 2)
            # (1, n, 1, emb) -> (b, n, a, emb) -> (b, na, emb)
            pos_embed = torch.cat((pos_embed_used[:, 0:1, :].expand(batch_size, -1, -1), pos_embed), dim=1)
            x = x + pos_embed
        if self.time_embed is not None:
            time_embed = self.time_embed[:, 0:input_time_window, :].unsqueeze(1).expand(batch_size, n, -1, -1).flatten(1, 2)
            # (1, 16, emb) -> (1, a, emb) -> (b, n, a ,emb) -> (b, na, emb)
            x[:, 1:, :] += time_embed

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, rel_pos_bias=None)

        x = self.norm(x)
        if self.fc_norm is not None:
            if return_all_tokens:
                return self.fc_norm(x)
                #(b, na+1, t)
            t = x[:, 1:, :]
            if return_patch_tokens:
                return self.fc_norm(t)
            else:
                return self.fc_norm(t.mean(1))
        else:
            if return_all_tokens:
                return x
            elif return_patch_tokens:
                return x[:, 1:]
            else:
                return x[:, 0]

    def forward(self, x, input_chans=None, return_patch_tokens=True, return_all_tokens=False, **kwargs):
        '''
        x: [batch size, number of electrodes, number of patches, patch size]
        For example, for an EEG sample of 4 seconds with 64 electrodes, x will be [batch size, 64, 4, 200]
        '''
        x = self.forward_features(x, input_chans=input_chans, return_patch_tokens=return_patch_tokens,
                                  return_all_tokens=return_all_tokens, **kwargs)
        if return_patch_tokens:
            x = x.view(x.shape[0], -1)
        x = self.head(x)
        return x

    # def forward_intermediate(self, x, layer_id=12, norm_output=False):
    #     x = self.patch_embed(x)
    #     batch_size, seq_len, _ = x.size()
    #
    #     cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    #     x = torch.cat((cls_tokens, x), dim=1)
    #     if self.pos_embed is not None:
    #         pos_embed = self.pos_embed[:, 1:, :].unsqueeze(2).expand(batch_size, -1, self.time_window, -1).flatten(1, 2)
    #         pos_embed = torch.cat((self.pos_embed[:, 0:1, :].expand(batch_size, -1, -1), pos_embed), dim=1)
    #         x = x + pos_embed
    #     if self.time_embed is not None:
    #         time_embed = self.time_embed.unsqueeze(1).expand(batch_size, 62, -1, -1).flatten(1, 2)
    #         x[:, 1:, :] += time_embed
    #     x = self.pos_drop(x)
    #
    #     rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
    #     if isinstance(layer_id, list):
    #         output_list = []
    #         for l, blk in enumerate(self.blocks):
    #             x = blk(x, rel_pos_bias=rel_pos_bias)
    #             # use last norm for all intermediate layers
    #             if l in layer_id:
    #                 if norm_output:
    #                     x_norm = self.fc_norm(self.norm(x[:, 1:]))
    #                     output_list.append(x_norm)
    #                 else:
    #                     output_list.append(x[:, 1:])
    #         return output_list
    #     elif isinstance(layer_id, int):
    #         for l, blk in enumerate(self.blocks):
    #             if l < layer_id:
    #                 x = blk(x, rel_pos_bias=rel_pos_bias)
    #             elif l == layer_id:
    #                 x = blk.norm1(x)
    #             else:
    #                 break
    #         return x[:, 1:]
    #     else:
    #         raise NotImplementedError(f"Not support for layer id is {layer_id} now!")
    #
    # def get_intermediate_layers(self, x, use_last_norm=False):
    #     x = self.patch_embed(x)
    #     batch_size, seq_len, _ = x.size()
    #
    #     cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    #     x = torch.cat((cls_tokens, x), dim=1)
    #     if self.pos_embed is not None:
    #         pos_embed = self.pos_embed[:, 1:, :].unsqueeze(2).expand(batch_size, -1, self.time_window, -1).flatten(1, 2)
    #         pos_embed = torch.cat((self.pos_embed[:, 0:1, :].expand(batch_size, -1, -1), pos_embed), dim=1)
    #         x = x + pos_embed
    #     if self.time_embed is not None:
    #         time_embed = self.time_embed.unsqueeze(1).expand(batch_size, 62, -1, -1).flatten(1, 2)
    #         x[:, 1:, :] += time_embed
    #     x = self.pos_drop(x)
    #
    #     features = []
    #     rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
    #     for blk in self.blocks:
    #         x = blk(x, rel_pos_bias)
    #         if use_last_norm:
    #             features.append(self.norm(x))
    #         else:
    #             features.append(x)
    #
    #     return features


# b, n, a, t = 16, 64, 1, 160
# in_chans = np.arange(1,65)
# model = NeuralTransformer(EEG_size=a*t, patch_size=t)
# x = torch.randn(b, n, a, t)
# y = model(x,in_chans)
# print(y.shape)