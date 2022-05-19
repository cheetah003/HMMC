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
import sys

import torch
from torch import nn
import torch.nn.functional as F
from .file_utils import cached_path
from .until_config import PretrainedConfig
from .until_module import PreTrainedModel, LayerNorm, ACT2FN
from collections import OrderedDict
from modules.module_clip import build_model, CLIP, convert_weights
from transformers import AutoConfig, AutoModel, BertTokenizer


logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {}
CONFIG_NAME = 'cross_config.json'
WEIGHTS_NAME = 'cross_pytorch_model.bin'


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

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
    def __init__(self, task_config, cross_config):
        super().__init__()
        pretrained_clip_name = cross_config.pretrained_clip_name
        if task_config.local_rank == 0:
            logger.info("pretrained_clip_name:{}".format(pretrained_clip_name))
        clip_state_dict = CLIP.get_config(pretrained_clip_name=pretrained_clip_name)
        clip = build_model(clip_state_dict, local_rank=task_config.local_rank)
        self.use_temp = task_config.use_temp
        self.is_vit = copy.deepcopy(clip.vit)
        self.visual = copy.deepcopy(clip.visual)

        if self.use_temp:
            self.temporal_transformer = Transformer(width=cross_config.temporal_hidden_size,
                                              layers=cross_config.temporal_hidden_layers,
                                              heads=cross_config.temporal_attention_heads)
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings,
                                                      cross_config.temporal_hidden_size)

            # use clip.transformer to initial temporal_transformer
            # for param_1, param_2 in zip(self.temporal_transformer.parameters(), clip.transformer.parameters()):
            #     param_1.data.copy_(param_2.data)  # initialize
    def forward(self, video, video_frames):
        # encode frames
        bs, frames, channel, h, w = video.shape
        # [bs*frame, 3, 224, 224]
        video = video.view(bs * frames, channel, h, w)
        # logger.info("video_b.shape:{}, dtype:{}".format(video_b.shape, video_b.dtype))
        # logger.info("video_frame[{}]:{}".format(b, video_frame))
        visual_hidden = self.encode_image(video, video_frame=frames)
        # [bs, frame, hidden_size]
        # logger.info("visual_hidden.shape:{}".format(visual_hidden.shape))
        visual_hidden = visual_hidden.view(bs, frames, visual_hidden.size(-1))
        # logger.info("visual_hidden1.shape:{}".format(visual_hidden.shape))
        # get temporal information
        visual_hidden_original = visual_hidden
        frame_output = visual_hidden_original
        if self.use_temp:
            seq_length = visual_hidden.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_hidden.device)
            # logger.info("position_ids.shape:{}".format(position_ids.shape))
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            # logger.info("frame_position_embeddings.shape:{}".format(frame_position_embeddings.shape))
            visual_hidden = visual_hidden + frame_position_embeddings

            video_mask = torch.ones([bs, frames], device=visual_hidden.device)
            extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
            extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
            visual_hidden = visual_hidden.permute(1, 0, 2)  # NLD -> LND
            visual_hidden = self.temporal_transformer(visual_hidden, extended_video_mask)
            visual_hidden = visual_hidden.permute(1, 0, 2)  # LND -> NLD
            visual_hidden = visual_hidden + visual_hidden_original

        # logger.info("visual_hidden.shape:{}".format(visual_hidden.shape))
        visual_output = visual_hidden / visual_hidden.norm(dim=-1, keepdim=True)
        # [bs, frames,512] -> [bs, 512]
        visual_output = torch.mean(visual_output, dim=1)
        # logger.info("visual_hidden mean.shape:{}".format(visual_hidden.shape))

        # logger.info("visual encoder visual_output.shape:{}".format(visual_output.shape))
        return visual_output, frame_output

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, return_hidden=False, video_frame=-1):
        if self.is_vit:
            # logger.info("image.shape:{}".format(image.shape))
            # hidden = self.visual(image, video_frame=video_frame)
            hidden = self.visual(image.type(self.dtype), video_frame=video_frame)
            # logger.info("hidden1.shape:{}".format(hidden.shape))
            hidden = self.visual.ln_post(hidden) @ self.visual.proj
            # logger.info("hidden2.shape:{}".format(hidden.shape))
            x = hidden[:, 0, :]
            # x = hidden
        else:
            hidden = self.visual(image)
            x = hidden
        if return_hidden:
            return x.float(), hidden.float()
        return x.float()


class TextEncoder(nn.Module):
    def __init__(self, task_config, cross_config):
        super().__init__()
        self.language = task_config.language
        pretrained_clip_name = cross_config.pretrained_clip_name
        if task_config.local_rank == 0:
            logger.info("pretrained_clip_name:{}".format(pretrained_clip_name))
        clip_state_dict = CLIP.get_config(pretrained_clip_name=pretrained_clip_name)
        clip = build_model(clip_state_dict, local_rank=task_config.local_rank)
        self.logit_scale = copy.deepcopy(clip.logit_scale)
        if self.language == "english":
            self.token_embedding = copy.deepcopy(clip.token_embedding)
            self.positional_embedding = copy.deepcopy(clip.positional_embedding)
            self.transformer = copy.deepcopy(clip.transformer)
            self.ln_final = copy.deepcopy(clip.ln_final)
            self.text_projection = copy.deepcopy(clip.text_projection)
            self.dtype = clip.visual.conv1.weight.dtype
        elif self.language == "chinese":
            pretrained = task_config.pretrained_text
            t_config = AutoConfig.from_pretrained(pretrained)
            if task_config.rank == 0:
                logger.info("name:{},chinesebert_config:{}".format(pretrained, t_config))
            self.chinese_encoder = AutoModel.from_pretrained(pretrained)
            self.text_proj = nn.Linear(cross_config.chinese_hidden_size, cross_config.temporal_hidden_size)
        else:
            raise NotImplementedError("wrong language")

    def forward(self, input_ids, attention_mask, return_hidden=False):
        bs_pair = input_ids.size(0)
        if self.language == "english":
            text_output, hidden = self.encode_text(input_ids, return_hidden=True)
        else:
            temp_output = self.chinese_encoder(input_ids, attention_mask=attention_mask)
            # logger.info("hidden:{},text_output:{}".format(temp_output[0].shape, temp_output[1].shape))
            hidden = self.text_proj(temp_output[0])
            text_output = self.text_proj(temp_output[1])


        text_output = text_output.view(bs_pair, text_output.size(-1))
        hidden = hidden.view(bs_pair, -1, hidden.size(-1))
        if return_hidden:
            return hidden
        else:
            return text_output

    def encode_text(self, text, return_hidden=False):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        pos_emd = self.positional_embedding[:x.size(1), :].type(self.dtype)
        x = x + pos_emd
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        hidden = self.ln_final(x).type(self.dtype) @ self.text_projection

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = hidden[torch.arange(hidden.shape[0]), text.argmax(dim=-1)]

        if return_hidden:
            return x.float(), hidden.float()

        return x.float()


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size,bias=False,)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (
            sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)
        ):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias