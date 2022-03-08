from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from abc import ABC

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from functools import partial
from diffdist import functional
from transformers import BertModel, BertConfig, AutoConfig, AutoModel, BertTokenizer

from modules.until_module import PreTrainedModel, AllGather, CrossEn, Dual_CrossEn
from modules.module_cross import CrossModel, CrossConfig, Transformer
from modules.module_vilbert import co_attention_model, BertLMPredictionHead
from modules.module_clip import CLIP, convert_weights, build_model
from modules.swin_transformer import SwinTransformer

logger = logging.getLogger(__name__)
allgather = AllGather.apply


def dist_collect(x):
    """ collect all tensor from all GPUs
    args:
        x: shape (mini_batch, ...)
    returns:
        shape (mini_batch * num_gpu, ...)
    """
    x = x.contiguous()
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype).contiguous()
                for _ in range(dist.get_world_size())]
    out_list = functional.all_gather(out_list, x)
    return torch.cat(out_list, dim=0).contiguous()


class CLIP4ClipPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, cross_config, *inputs, **kwargs):
        super(CLIP4ClipPreTrainedModel, self).__init__(cross_config)
        self.cross_config = cross_config
        self.clip = None
        self.cross = None
        self.chinese_bert = None

    @classmethod
    def from_pretrained(cls, cross_model_name, state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):

        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        if state_dict is None: state_dict = {}
        clip_state_dict = CLIP.get_config(pretrained_clip_name="ViT-B/32")
        # clip_state_dict = CLIP.get_config(pretrained_clip_name="ViT-B/16")
        # clip_state_dict = CLIP.get_config(pretrained_clip_name="ViT-L/14")
        # clip_state_dict = CLIP.get_config(pretrained_clip_name="RN50")
        # clip_state_dict = CLIP.get_config(pretrained_clip_name="RN101")
        # clip_state_dict = CLIP.get_config(pretrained_clip_name="RN50x4")
        # clip_state_dict = CLIP.get_config(pretrained_clip_name="RN50x16")
        # clip_state_dict = CLIP.get_config(pretrained_clip_name="RN50x64")

        for key, val in clip_state_dict.items():
            logger.info("key:{}".format(key))
            new_key = "clip." + key
            new_key_k = "clip_k." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()
                state_dict[new_key_k] = val.clone()

        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None,
                                                 task_config=task_config)

        model = cls(cross_config, clip_state_dict, *inputs, **kwargs)

        # ===> Initialization trick [HARD CODE]
        '''
        if model.sim_header == "seqLSTM" or model.sim_header == "seqTransf":
            contain_frame_position = False
            for key in state_dict.keys():
                if key.find("frame_position_embeddings") > -1:
                    contain_frame_position = True
                    break
            if contain_frame_position is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["frame_position_embeddings.weight"] = val.clone()
                        continue
                    if model.sim_header == "seqTransf" and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])
                        # cut from beginning
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                            continue
         '''
        # <=== End of initialization trick

        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)

        return model


def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)


def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config


def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]


class BirdPreTrainedModel(CLIP4ClipPreTrainedModel):
    def __init__(self, cross_config, clip_state_dict, task_config):
        super(BirdPreTrainedModel, self).__init__(cross_config)
        self.task_config = task_config
        self.ignore_video_index = -1

        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings

        self.rank = task_config.local_rank
        self.mlm_probability = 0.15
        self.weight_sum = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.contrast_momentum = task_config.contrast_momentum
        self.contrast_temperature = task_config.contrast_temperature
        self.contrast_num_negative = task_config.contrast_num_negative
        ################## begin of chinese text Encoder
        # pretrained = 'voidful/albert_chinese_base'
        pretrained = 'hfl/chinese-roberta-wwm-ext'
        # pretrained = 'hfl/chinese-roberta-wwm-ext-large'
        # pretrained = "nghuyong/ernie-1.0"
        self.tokenizer = BertTokenizer.from_pretrained(pretrained)
        logger.info("tokenizer:pad:{},cls:{},mask:{}".format(self.tokenizer.pad_token_id,
                                                             self.tokenizer.cls_token_id, self.tokenizer.mask_token_id))
        t_config = AutoConfig.from_pretrained(pretrained)
        self.chinese_bert_config = t_config
        logger.info("name:{},chinesebert_config:{}".format(pretrained, t_config))
        self.chinese_bert = AutoModel.from_pretrained(pretrained)
        self.chinese_bert_k = AutoModel.from_pretrained(pretrained)
        for param_q, param_k in zip(self.chinese_bert.parameters(), self.chinese_bert_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        ################### End of albert text Encoder
        self.cls = BertLMPredictionHead(t_config)
        ################## begin of co_attention_model
        self.co_attention_model = co_attention_model()
        self.co_attention_model_k = co_attention_model()
        for param_q, param_k in zip(self.co_attention_model.parameters(), self.co_attention_model_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        ################## end of co_attention_model
        # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>

        self.clip = build_model(clip_state_dict, local_rank=task_config.local_rank, embed_dim=t_config.hidden_size)
        self.clip_k = build_model(clip_state_dict, local_rank=task_config.local_rank, embed_dim=t_config.hidden_size)
        for param_q, param_k in zip(self.clip.parameters(), self.clip_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        ################## swin transformer
        '''
        enc = partial(
            SwinTransformer,
            img_size=224,
            patch_size=4,
            in_chans=3,
            embed_dim=96,
            depths=[2,2,6,2],
            num_heads=[3,6,12,24],
            window_size=7,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            norm_befor_mlp="ln",
        )
        self.v_encoder = enc(
            num_classes=0,
            drop_path_rate=0.2,
        )
        '''

        ################## tag model
        self.tag_model = Transformer(width=t_config.hidden_size,
                                     layers=self.task_config.tag_num_hidden_layers,
                                     heads=t_config.hidden_size // 64)

        ################## create queue
        self.K = int(
            self.task_config.train_length * 1. / dist.get_world_size() / self.task_config.batch_size) * self.task_config.epochs
        self.k = int(
            self.task_config.train_length * 1. / dist.get_world_size() / self.task_config.batch_size) * 0

        self.register_buffer("queue_video_in", torch.randn(768, self.contrast_num_negative))
        self.register_buffer("queue_tag_in", torch.randn(768, self.contrast_num_negative))
        self.register_buffer("queue_title_in", torch.randn(768, self.contrast_num_negative))
        self.register_buffer("queue_co_video", torch.randn(768, self.contrast_num_negative))
        self.register_buffer("queue_co_tag", torch.randn(768, self.contrast_num_negative))
        self.register_buffer("queue_co_title", torch.randn(768, self.contrast_num_negative))
        self.register_buffer("queue_co_video_title", torch.randn(768, self.contrast_num_negative))
        self.queue_video_in = F.normalize(self.queue_video_in, dim=0)
        self.queue_tag_in = F.normalize(self.queue_tag_in, dim=0)
        self.queue_title_in = F.normalize(self.queue_title_in, dim=0)
        self.queue_co_video = F.normalize(self.queue_co_video, dim=0)
        self.queue_co_tag = F.normalize(self.queue_co_tag, dim=0)
        self.queue_co_title = F.normalize(self.queue_co_title, dim=0)
        self.queue_co_video_title = F.normalize(self.queue_co_video_title, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        ################## loss function
        self.loss_fct = CrossEn()
        self.loss_fct_dual = Dual_CrossEn()

        self.apply(self.init_weights)

    def forward(self, video_data, video_mask, tag_ids, tag_mask, title_ids, title_mask):

        tag_ids = tag_ids.view(-1, tag_ids.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        tag_mask = tag_mask.view(-1, tag_mask.shape[-1])
        title_ids = title_ids.view(-1, title_ids.shape[-1])
        title_mask = title_mask.view(-1, title_mask.shape[-1])
        # T x 3 x H x W
        video = torch.as_tensor(video_data).float()
        b, pair, bs, ts, channel, h, w = video.shape
        video = video.view(b * pair * bs * ts, channel, h, w)
        video_frame = bs * ts

        # logger.info("input_ids.shape:{}".format(input_ids.shape))
        logger.info("video.shape:{}".format(video.shape))
        # logger.info("sequence_output.shape:{}".format(sequence_output.shape))
        # logger.info("visual_output.shape:{}".format(visual_output.shape))
        # logger.info("masked_title.shape:{}".format(masked_title.shape))
        if self.training:
            loss = 0.
            if self.task_config.stage == "stage1":
                video_in = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_frame)
                video_in_mpool = torch.sum(video_in, dim=1) / video_in.size(1)
                tag_in = self.get_sequence_output(tag_ids, tag_mask, shaped=True)
                title_in = self.get_sequence_output(title_ids, title_mask, shaped=True)
                if self.rank == 0:
                    logger.info("video_in.shape:{}".format(video_in.shape))
                    logger.info("title_in.shape:{}".format(title_in.shape))
                _, co_video_output = self.co_attention_model(video_in, video_in)
                co_video_output = torch.sum(co_video_output, dim=1) / video_in.size(1)
                co_tag_output, _ = self.co_attention_model(tag_in, tag_in)
                co_title_output, _ = self.co_attention_model(title_in, title_in)
                _, co_video_title_output = self.co_attention_model(title_in, video_in)
                co_video_title_output = torch.sum(co_video_title_output, dim=1) / video_in.size(1)
                # compute key features
                with torch.no_grad():  # no gradient to keys
                    self._momentum_update_key_encoder()  # update the key encoder
                    video_in_k = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_frame,
                                                        is_momentum=True)
                    video_in_mpool_k = torch.sum(video_in_k, dim=1) / video_in_k.size(1)
                    tag_in_k = self.get_sequence_output(tag_ids, tag_mask, shaped=True, is_momentum=True)
                    title_in_k = self.get_sequence_output(title_ids, title_mask, shaped=True, is_momentum=True)
                    _, co_video_output_k = self.co_attention_model(video_in_k, video_in_k)
                    co_video_output_k = torch.sum(co_video_output_k, dim=1) / video_in.size(1)
                    co_tag_output_k, _ = self.co_attention_model(tag_in_k, tag_in_k)
                    co_title_output_k, _ = self.co_attention_model(title_in_k, title_in_k)
                    _, co_video_title_output_k = self.co_attention_model(title_in_k, video_in_k)
                    co_video_title_output_k = torch.sum(co_video_title_output_k, dim=1) / video_in.size(1)

                # compute contrastive loss
                # video_in - tag_in
                logger.info("video_in_mpool.dtype:{},tag_in_k.dtype:{},tag_in.dtype:{}".format(video_in_mpool.dtype,
                                                                                        tag_in_k.dtype, tag_in.dtype))
                video_tag_in_loss = self.contrastive_loss(video_in_mpool, tag_in_k, self.queue_tag_in) \
                                    + self.contrastive_loss(tag_in, video_in_mpool_k, self.queue_video_in)
                loss += video_tag_in_loss
                # video_in - title_in
                video_title_in_loss = self.contrastive_loss(video_in_mpool, title_in_k, self.queue_title_in) \
                                      + self.contrastive_loss(title_in, video_in_mpool_k, self.queue_video_in)
                loss += video_title_in_loss
                # video - tag
                video_tag_loss = self.contrastive_loss(co_video_output, co_tag_output_k, self.queue_co_tag) \
                                 + self.contrastive_loss(co_tag_output, co_video_output_k, self.queue_co_video)
                loss += video_tag_loss
                # video - title
                video_title_loss = self.contrastive_loss(co_video_output, co_title_output_k, self.queue_co_title) \
                                   + self.contrastive_loss(co_title_output, co_video_output_k, self.queue_co_video)
                loss += video_title_loss
                # (video + title) - tag
                video_title_tag_loss = self.contrastive_loss(co_video_title_output, co_tag_output_k, self.queue_co_tag) \
                                       + self.contrastive_loss(co_tag_output, co_video_title_output_k,
                                                               self.queue_co_video_title)
                loss += video_title_tag_loss

                self._dequeue_and_enqueue(video_in_mpool_k, tag_in_k, title_in_k, co_video_output_k, co_tag_output_k,
                                          co_title_output_k, co_video_title_output_k)

                # for MLM
                to_mask_tag_ids = tag_ids.clone()
                tag_labels = to_mask_tag_ids.clone()
                to_mask_title_ids = title_ids.clone()
                title_labels = to_mask_title_ids.clone()

                tag_probability_matrix = torch.full(tag_labels.shape, self.mlm_probability)
                masked_tag_ids, tag_label = self.mask(to_mask_tag_ids, self.tokenizer.vocab_size,
                                                      video.device, targets=tag_labels,
                                                      probability_matrix=tag_probability_matrix)
                title_probability_matrix = torch.full(title_labels.shape, self.mlm_probability)
                masked_title_ids, title_label = self.mask(to_mask_title_ids, self.tokenizer.vocab_size,
                                                          video.device, targets=title_labels,
                                                          probability_matrix=title_probability_matrix)

                masked_tag_output = self.get_sequence_output(masked_tag_ids, tag_mask, return_hidden=True, shaped=True)
                masked_title_output = self.get_sequence_output(masked_title_ids, title_mask, return_hidden=True,
                                                               shaped=True)

                co_masked_tag_output, _ = self.co_attention_model(masked_tag_output, video_in)
                co_masked_title_output, _ = self.co_attention_model(masked_title_output, video_in)

                mlm_tag_loss = self.calculate_mlm_loss(masked_tag_output, tag_labels)
                loss += mlm_tag_loss
                mlm_co_tag_loss = self.calculate_mlm_loss(co_masked_tag_output, tag_labels)
                loss += mlm_co_tag_loss
                mlm_title_loss = self.calculate_mlm_loss(masked_title_output, title_labels)
                loss += mlm_title_loss
                mlm_co_title_loss = self.calculate_mlm_loss(co_masked_title_output, title_labels)
                loss += mlm_co_title_loss
                logger.info("mlm_tag_loss:{},mlm_co_tag_loss:{}".format(mlm_tag_loss, mlm_co_tag_loss))
                logger.info("mlm_title_loss:{}, mlm_co_title_loss:{}".format(mlm_title_loss, mlm_co_title_loss))
                return loss
            elif self.task_config.stage == "stage2":
                # add MMM and VTM
                visual_output = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_frame)
                tag_label_output = self.get_sequence_output(tag_ids, tag_mask, shaped=True)
                title_output = self.get_sequence_output(title_ids, title_mask, shaped=True)

                # get tag feature from video
                extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
                extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
                logger.info("visual_output.shape:{}".format(visual_output.shape))
                logger.info("video_mask.shape:{}".format(video_mask.shape))
                logger.info("extended_video_mask.shape:{}".format(extended_video_mask.shape))
                visual_output_temp = visual_output.permute(1, 0, 2)
                video_tag_output = self.tag_model(visual_output_temp, extended_video_mask)
                video_tag_output = video_tag_output.permute(1, 0, 2)
                logger.info("video_tag_output.shape:{}".format(video_tag_output.shape))

                # fusion tag feature and video feature
                _, co_visual_output = self.co_attention_model(video_tag_output, visual_output)

                # (video, video_tag) - title loss
                sim_matrix = self.get_similarity_logits(title_output, co_visual_output)
                sim_loss = self.loss_fct(sim_matrix) + self.loss_fct(sim_matrix.T)
                loss += sim_loss
                logger.info("sim_loss:{}".format(sim_loss))
                # video_tag - tag_label loss
                tag_loss = (tag_label_output - video_tag_output) ** 2
                tag_loss = tag_loss.mean(dim=-1)
                tag_loss = tag_loss.mean(dim=-1)
                logger.info("tag_loss.shape:{}".format(tag_loss.shape))
                tag_loss = tag_loss.sum()
                logger.info("tag_loss.sum:{}".format(tag_loss))
                loss += tag_loss
                return loss
        else:
            return None

    def calculate_mlm_loss(self, sequence_output_mlm, labels):
        mlm_scores = self.cls(sequence_output_mlm)
        # logger.info("mlm_scores.shape:{}".format(mlm_scores.shape))
        # logger.info("labels.shape:{}".format(labels.shape))
        mlm_loss = F.cross_entropy(mlm_scores.view(-1, self.chinese_bert_config.vocab_size),
                                   labels.view(-1), ignore_index=-100)
        return mlm_loss

    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        # logger.info("masked_indices:{}".format(masked_indices))
        # logger.info("masked_indices.shape:{}".format(masked_indices.shape))
        if targets is not None:
            targets[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

    def get_sequence_output(self, input_ids, attention_mask, return_hidden=False, shaped=False, is_momentum=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        bs_pair = input_ids.size(0)
        # logger.info("albert encoder")
        # logger.info("input_ids.shape:{}".format(input_ids.shape))
        if is_momentum:
            sequence_hidden = self.chinese_bert_k(input_ids, attention_mask=attention_mask)
        else:
            sequence_hidden = self.chinese_bert(input_ids, attention_mask=attention_mask)
        # logger.info("before sequence_hidden.shape:{}".format(sequence_hidden.shape))
        if return_hidden:
            sequence_hidden = sequence_hidden[0]
        else:
            sequence_hidden = sequence_hidden[1]

        sequence_hidden = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1))
        # logger.info("after sequence_hidden1.shape:{}".format(sequence_hidden.shape))
        return sequence_hidden

    def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1, is_momentum=False):
        if shaped is False:
            video = torch.as_tensor(video).float()
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts
        bs_pair = video.size(0) // video_frame
        # logger.info("video.shape:{}".format(video.shape))
        if is_momentum:
            visual_hidden = self.clip_k.encode_image(video, video_frame=video_frame)
        else:
            visual_hidden = self.clip.encode_image(video, video_frame=video_frame)
        # visual_hidden = self.v_encoder(video)
        # logger.info("visual_hidden.shape:{}".format(visual_hidden.shape))
        visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1))
        # logger.info("visual_hidden.shape:{}".format(visual_hidden.shape))
        return visual_hidden

    def mean_pooling_for_similarity_sequence(self, sequence_output, attention_mask):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)
        return text_out

    def mean_pooling_for_similarity_visual(self, visual_output):
        video_out = torch.sum(visual_output, dim=1) / visual_output.size(1)
        return video_out

    def mean_pooling_for_similarity(self, sequence_output, visual_output, attention_mask, video_mask, ):
        text_out = self.mean_pooling_for_similarity_sequence(sequence_output, attention_mask)
        video_out = self.mean_pooling_for_similarity_visual(visual_output, video_mask)

        return text_out, video_out

    def loose_similarity_for_text(self, query_text, input_text):
        query_text = query_text.contiguous()
        input_text = input_text.contiguous()

        if self.training:
            input_text = allgather(input_text, self.task_config)
            query_text = allgather(query_text, self.task_config)
            torch.distributed.barrier()

        input_text = input_text.squeeze(1)
        input_text = input_text / input_text.norm(dim=-1, keepdim=True)

        query_text = query_text.squeeze(1)
        query_text = query_text / query_text.norm(dim=-1, keepdim=True)

        logit_scale = self.clip.logit_scale.exp()
        # if self.rank == 0:
        #     logger.info("logit_scale_text:{}".format(logit_scale))
        retrieve_logits = logit_scale * torch.matmul(query_text, input_text.t())
        # logger.info("retrieve_logits.shape:{}".format(retrieve_logits.shape))
        return retrieve_logits

    def loose_similarity(self, sequence_output, visual_output):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        if self.training:
            visual_output = allgather(visual_output, self.task_config)
            sequence_output = allgather(sequence_output, self.task_config)
            torch.distributed.barrier()

        visual_output = visual_output.squeeze(1)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        visual_output = self.mean_pooling_for_similarity_visual(visual_output)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

        sequence_output = sequence_output.squeeze(1)
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)
        # logger.info("allgather sequence_output.shape:{}".format(sequence_output.shape))
        # logger.info("allgather visual_output.shape:{}".format(visual_output.shape))
        # sequence_output, visual_output = self.co_attention_model(sequence_output, visual_output)

        logit_scale = self.clip.logit_scale.exp()
        # if self.rank == 0:
        #     logger.info("logit_scale:{}".format(logit_scale))
        retrieve_logits = logit_scale * torch.matmul(sequence_output, visual_output.t())
        # logger.info("retrieve_logits.shape:{}".format(retrieve_logits.shape))
        return retrieve_logits

    def get_similarity_logits(self, sequence_output, visual_output):
        retrieve_logits = self.loose_similarity(sequence_output, visual_output)
        return retrieve_logits

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        _contrast_momentum = 1. - (1. - self.contrast_momentum) * (np.cos(np.pi * self.k / self.K) + 1) / 2.
        self.k = self.k + 1

        for param_q, param_k in zip(self.clip.parameters(), self.clip_k.parameters()):
            param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)

        for param_q, param_k in zip(self.chinese_bert.parameters(), self.chinese_bert_k.parameters()):
            param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)

        for param_q, param_k in zip(self.co_attention_model.parameters(), self.co_attention_model_k.parameters()):
            param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, video_in_k, tag_in_k, title_in_k, co_video_output_k, co_tag_output_k,
                             co_title_output_k, co_video_title_output_k):
        # gather keys before updating queue
        video_in_k = dist_collect(video_in_k).squeeze(1)
        tag_in_k = dist_collect(tag_in_k).squeeze(1)
        title_in_k = dist_collect(title_in_k).squeeze(1)
        co_video_output_k = dist_collect(co_video_output_k).squeeze(1)
        co_tag_output_k = dist_collect(co_tag_output_k).squeeze(1)
        co_title_output_k = dist_collect(co_title_output_k).squeeze(1)
        co_video_title_output_k = dist_collect(co_video_title_output_k).squeeze(1)

        batch_size = video_in_k.shape[0]

        ptr = int(self.queue_ptr)
        assert self.contrast_num_negative % batch_size == 0  # for simplicity
        print("keys1.shape:{}".format(video_in_k.shape))
        # replace the keys at ptr (dequeue and enqueue)
        self.queue_video_in[:, ptr:ptr + batch_size] = video_in_k.T
        self.queue_tag_in[:, ptr:ptr + batch_size] = tag_in_k.T
        self.queue_title_in[:, ptr:ptr + batch_size] = title_in_k.T
        self.queue_co_video[:, ptr:ptr + batch_size] = co_video_output_k.T
        self.queue_co_tag[:, ptr:ptr + batch_size] = co_tag_output_k.T
        self.queue_co_title[:, ptr:ptr + batch_size] = co_title_output_k.T
        self.queue_co_video_title[:, ptr:ptr + batch_size] = co_video_title_output_k.T
        ptr = (ptr + batch_size) % self.contrast_num_negative  # move pointer

        self.queue_ptr[0] = ptr

    def contrastive_loss(self, q, k, queue):

        q = q.squeeze(1)
        k = k.squeeze(1)
        logger.info("q.dtype:{},k.dtype:{}".format(q.dtype, k.dtype))
        # positive logits: Nx1
        # l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_pos = torch.matmul(q, k.T)
        # negative logits: NxK
        # l_neg = torch.einsum('nc,ck->nk', [q, queue.clone().detach()])
        l_neg = torch.matmul(q, queue.clone().detach())
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.contrast_temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        return F.cross_entropy(logits, labels)


class BirdModel(BirdPreTrainedModel):
    def forward(self, query_ids, query_mask, pos_video_data, pos_video_mask, hard_video_data, hard_video_mask, \
                pos_title_ids, pos_title_mask, hard_title_ids, hard_title_mask):
        query_ids = query_ids.view(-1, query_ids.shape[-1])

        query_mask = query_mask.view(-1, query_mask.shape[-1])
        pos_title_ids = pos_title_ids.view(-1, pos_title_ids.shape[-1])
        pos_title_mask = pos_title_mask.view(-1, pos_title_mask.shape[-1])
        hard_title_ids = hard_title_ids.view(-1, hard_title_ids.shape[-1])
        hard_title_mask = hard_title_mask.view(-1, hard_title_mask.shape[-1])
        pos_video_mask = pos_video_mask.view(-1, pos_video_mask.shape[-1])
        hard_video_mask = hard_video_mask.view(-1, hard_video_mask.shape[-1])

        # T x 3 x H x W
        pos_video = torch.as_tensor(pos_video_data).float()
        hard_video = torch.as_tensor(hard_video_data).float()
        b, pair, bs, ts, channel, h, w = pos_video.shape
        pos_video = pos_video.view(b * pair * bs * ts, channel, h, w)
        hard_video = hard_video.view(b * pair * bs * ts, channel, h, w)
        video_frame = bs * ts
        if self.training:
            loss = 0.0
            query_output = self.get_sequence_output(query_ids, query_mask, return_hidden=False, shaped=True)
            pos_visual_output = self.get_visual_output(pos_video, pos_video_mask, shaped=True, video_frame=video_frame)
            hard_visual_output = self.get_visual_output(hard_video, hard_video_mask, shaped=True,
                                                        video_frame=video_frame)
            pos_title_output = self.get_sequence_output(pos_title_ids, pos_title_mask, shaped=True)
            hard_title_output = self.get_sequence_output(hard_title_ids, hard_title_mask, shaped=True)
            if self.rank == 0:
                logger.info("pos_title_output.shape:{}".format(pos_title_output.shape))
                logger.info("pos_visual_output.shape:{}".format(pos_visual_output.shape))
            # _, pooled_output = self.get_cross_output(video_fea=visual_output, text_fea=title_output)

            _, pos_co_visual_output = self.co_attention_model(pos_title_output, pos_visual_output)
            _, hard_co_visual_output = self.co_attention_model(hard_title_output, hard_visual_output)

            # random negtive loss
            sim_matrix = self.get_similarity_logits(query_output, pos_co_visual_output)
            sim_loss = self.loss_fct(sim_matrix) + self.loss_fct(sim_matrix.T)
            loss += sim_loss
            if self.rank == 0:
                logger.info("sim_loss:{}".format(sim_loss))
            # logger.info("sim_matrix:{}".format(sim_matrix))
            # hard negtive loss
            single_positive_scores = torch.diagonal(sim_matrix, 0)
            hard_doc_num = len(single_positive_scores)
            positive_scores = single_positive_scores.reshape(-1, 1).repeat(1, hard_doc_num).reshape(-1)
            sim_matrix_hard = self.get_similarity_logits(query_output, hard_co_visual_output)
            hard_batch_scores = sim_matrix_hard.reshape(-1)
            hard_sim_matrix = torch.cat([positive_scores.unsqueeze(1),
                                         hard_batch_scores.unsqueeze(1)], dim=1)

            hard_lsm = F.log_softmax(hard_sim_matrix, dim=1)
            # logger.info("hard_sim_matrix:{}".format(hard_sim_matrix))
            # logger.info("hard_lsm:{}".format(hard_lsm))
            hard_loss = -1.0 * hard_lsm[:, 0]
            hard_loss = hard_loss.mean()
            if self.rank == 0:
                logger.info("hard_loss:{}".format(hard_loss))
            loss += hard_loss

            # logger.info("hard_sim_matrix:{}".format(hard_sim_matrix))
            return loss
        else:
            return None
