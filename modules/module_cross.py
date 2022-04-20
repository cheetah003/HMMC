from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil

import torch
from torch import nn
import torch.nn.functional as F
from .file_utils import cached_path
from .until_config import PretrainedConfig
from .until_module import PreTrainedModel, LayerNorm, ACT2FN
from collections import OrderedDict
from modules.module_clip import build_model, CLIP


logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {}
CONFIG_NAME = 'cross_config.json'
WEIGHTS_NAME = 'cross_pytorch_model.bin'


class CrossConfig(PretrainedConfig):
    """Configuration class to store the configuration of a `CrossModel`.
    """
    pretrained_model_archive_map = PRETRAINED_MODEL_ARCHIVE_MAP
    config_name = CONFIG_NAME
    weights_name = WEIGHTS_NAME
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """Constructs CrossConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `CrossModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `CrossModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.n_head = n_head

    def attention(self, x: torch.Tensor, attn_mask: torch.Tensor):
        attn_mask_ = attn_mask.repeat(self.n_head, 1, 1)
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]

    def forward(self, para_tuple: tuple):
        # x: torch.Tensor, attn_mask: torch.Tensor
        # print(para_tuple)
        x, attn_mask = para_tuple
        x = x + self.attention(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, attn_mask)

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        # logger.info("x.shpae:{},attn_mask:{}".format(x.shape, attn_mask.shape))
        return self.resblocks((x, attn_mask))[0]


class VisualEncoder(nn.Module):
    def __init__(self, local_rank, cross_config):
        super().__init__()
        pretrained_clip_name = "ViT-B/32"
        if local_rank == 0:
            logger.info("pretrained_clip_name:{}".format(pretrained_clip_name))
        clip_state_dict = CLIP.get_config(pretrained_clip_name=pretrained_clip_name)
        clip = build_model(clip_state_dict, local_rank=local_rank)
        self.is_vit = clip.vit
        self.visual = clip.visual
        self.temporal_transformer = Transformer(width=cross_config.temporal_hidden_size,
                                          layers=cross_config.temporal_hidden_layers,
                                          heads=cross_config.temporal_attention_heads)
        self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings,
                                                      cross_config.temporal_hidden_size)
        # [512, 768]
        self.temporal_proj = nn.Linear(cross_config.temporal_hidden_size, cross_config.hidden_size)
        # use clip.transformer to initial temporal_transformer
        # for param_1, param_2 in zip(self.temporal_transformer.parameters(), clip.transformer.parameters()):
        #     param_1.data.copy_(param_2.data)  # initialize

    def forward(self, video, video_frames):
        # encode frames
        bs = video.size(0)
        visual_hidden_list = []
        for b in range(bs):
            # [frame, 3, 224, 224]
            video_frame = video_frames[b]
            video_b = video[b, 0:video_frame, :, :, :]
            # logger.info("video_b.shape:{}, dtype:{}".format(video_b.shape, video_b.dtype))
            # logger.info("video_frame[{}]:{}".format(b, video_frame))
            visual_hidden = self.encode_image(video_b, video_frame=video_frame)
            # [frame, hidden_size]
            # logger.info("visual_hidden.shape:{}".format(visual_hidden.shape))
            visual_hidden = visual_hidden.view(-1, visual_hidden.size(-1))
            # logger.info("visual_hidden1.shape:{}".format(visual_hidden.shape))
            # get temporal information
            visual_hidden_original = visual_hidden
            seq_length = visual_hidden.size(0)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_hidden.device)
            # logger.info("position_ids.shape:{}".format(position_ids.shape))
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            # logger.info("frame_position_embeddings.shape:{}".format(frame_position_embeddings.shape))
            visual_hidden = visual_hidden + frame_position_embeddings

            # visual_hidden = visual_hidden.permute(1, 0, 2)  # NLD -> LND
            video_mask = torch.zeros([video_frame, video_frame], device=visual_hidden.device)
            visual_hidden = self.temporal_transformer(visual_hidden, video_mask)
            # visual_hidden = visual_hidden.permute(1, 0, 2)  # LND -> NLD
            visual_hidden = visual_hidden + visual_hidden_original
            # proj hidder: [bs, frames,512] -> [bs, frames,768]
            visual_hidden = self.temporal_proj(visual_hidden)
            # [bs, frames,512] -> [bs, 1,768]
            # logger.info("visual_hidden.shape:{}".format(visual_hidden.shape))
            visual_hidden = torch.mean(visual_hidden, dim=0)
            # logger.info("visual_hidden mean.shape:{}".format(visual_hidden.shape))
            visual_hidden_list.append(visual_hidden)
        visual_output = torch.stack(visual_hidden_list, dim=0)
        # logger.info("visual encoder visual_output.shape:{}".format(visual_output.shape))
        return visual_output

    def encode_image(self, image, return_hidden=False, video_frame=-1):
        if self.is_vit:
            # logger.info("image.shape:{}".format(image.shape))
            hidden = self.visual(image, video_frame=video_frame)
            # logger.info("hidden1.shape:{}".format(hidden.shape))
            hidden = self.visual.ln_post(hidden) @ self.visual.proj
            # logger.info("hidden2.shape:{}".format(hidden.shape))
            x = hidden[:, 0, :]
            # x = hidden
        else:
            hidden = self.visual(image)
            x = hidden
        if return_hidden:
            return x, hidden
        return x