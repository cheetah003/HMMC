from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from abc import ABC
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from functools import partial
from diffdist import functional
from transformers import AutoConfig, AutoModel, BertTokenizer
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.until_module import PreTrainedModel, AllGather, CrossEn, Dual_CrossEn
from modules.module_cross import TextEncoder, VisualEncoder, CrossConfig, BertLMPredictionHead
# from modules.module_vilbert import co_attention_model, BertLMPredictionHead, BertConfig
from modules.module_clip import CLIP, convert_weights, build_model

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

    @classmethod
    def from_pretrained(cls, cross_model_name, state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):

        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None,
                                                 task_config=task_config)

        model = cls(cross_config, *inputs, **kwargs)

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
    def __init__(self, cross_config, task_config):
        super(BirdPreTrainedModel, self).__init__(cross_config)
        self.task_config = task_config
        self.rank = task_config.local_rank
        self.mlm_probability = cross_config.mlm_probability
        # self.weight_sum = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.float32), requires_grad=True)
        self.weight_v = cross_config.weight_sum_v
        self.weight_t = cross_config.weight_sum_t
        self.weight_cross = cross_config.weight_sum_cross
        self.logit_scale = torch.nn.Parameter(torch.tensor([np.log(1 / 0.07)], dtype=torch.float32), requires_grad=True)
        self.contrast_momentum = task_config.contrast_momentum
        self.contrast_temperature = task_config.contrast_temperature
        self.contrast_num_negative = task_config.contrast_num_negative
        ################## chinese text Encoder
        if self.task_config.language == "chinese":
            self.tokenizer = BertTokenizer.from_pretrained(self.task_config.pretrained_text)
        else:
            self.tokenizer = ClipTokenizer()
        if self.rank == 0:
            logger.info("tokenizer:pad:{},cls:{},mask:{}, voacb_size:{}".format(self.tokenizer.pad_token_id,
                                                                 type(self.tokenizer.cls_token_id),
                                                                 self.tokenizer.mask_token_id,
                                                                 self.tokenizer.vocab_size))
        t_config = AutoConfig.from_pretrained(self.task_config.pretrained_text)
        self.text_encoder = TextEncoder(self.task_config, cross_config)
        self.text_encoder_k = TextEncoder(self.task_config, cross_config)
        self.t_projector = MLP(num_layers=cross_config.proj_num_layers)
        self.t_projector_k = MLP(num_layers=cross_config.proj_num_layers)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.t_projector)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.t_projector_k)
        # for MLM
        t_config.hidden_size = cross_config.temporal_hidden_size
        t_config.vocab_size = self.tokenizer.vocab_size
        self.cls = BertLMPredictionHead(t_config)
        ################## visual_encoder
        self.visual_encoder = VisualEncoder(self.rank, cross_config)
        self.visual_encoder_k = VisualEncoder(self.rank, cross_config)
        self.v_projector = MLP(num_layers=cross_config.proj_num_layers)
        self.v_projector_k = MLP(num_layers=cross_config.proj_num_layers)
        self.v_predictor = MLP(num_layers=cross_config.pred_num_layers)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.v_projector)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.v_projector_k)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.v_predictor)
        ################# momemtun mdoel pairs
        self.model_pairs = [[self.visual_encoder, self.visual_encoder_k],
                            [self.text_encoder, self.text_encoder_k],
                            [self.v_projector, self.v_projector_k],
                            [self.t_projector, self.t_projector_k],
                            ]
        self.copy_params()
        ################## create queue
        self.register_buffer("queue_v1_self_ng", torch.randn(cross_config.temporal_hidden_size, self.contrast_num_negative))
        self.register_buffer("queue_v2_self_ng", torch.randn(cross_config.temporal_hidden_size, self.contrast_num_negative))
        self.register_buffer("queue_v1_cross_ng", torch.randn(cross_config.temporal_hidden_size, self.contrast_num_negative))
        self.register_buffer("queue_v2_cross_ng", torch.randn(cross_config.temporal_hidden_size, self.contrast_num_negative))
        self.register_buffer("queue_title_cross_ng", torch.randn(cross_config.temporal_hidden_size, self.contrast_num_negative))
        self.register_buffer("queue_tag_cross_ng", torch.randn(cross_config.temporal_hidden_size, self.contrast_num_negative))
        self.queue_v1_self_ng = F.normalize(self.queue_v1_self_ng, dim=0)
        self.queue_v2_self_ng = F.normalize(self.queue_v2_self_ng, dim=0)
        self.queue_v1_cross_ng = F.normalize(self.queue_v1_cross_ng, dim=0)
        self.queue_v2_cross_ng = F.normalize(self.queue_v2_cross_ng, dim=0)
        self.queue_title_cross_ng = F.normalize(self.queue_title_cross_ng, dim=0)
        self.queue_tag_cross_ng = F.normalize(self.queue_tag_cross_ng, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        ################## loss function
        self.loss_fct = CrossEn()
        self.loss_fct_dual = Dual_CrossEn()

        # self.apply(self.init_weights)

    def get_mlm_loss(self, input_ids, input_mask):
        to_mask_input_ids = input_ids.clone()
        input_labels = to_mask_input_ids.clone()
        input_probability_matrix = torch.full(input_labels.shape, self.mlm_probability)
        masked_input_ids, input_labels = self.mask(to_mask_input_ids, self.tokenizer.vocab_size,
                                                   input_mask.device, targets=input_labels,
                                                   probability_matrix=input_probability_matrix)
        masked_input_output = self.text_encoder(masked_input_ids, input_mask, return_hidden=True)
        mlm_input_loss = self.calculate_mlm_loss(masked_input_output, input_labels)
        return mlm_input_loss

    def calculate_mlm_loss(self, sequence_output_mlm, labels):

        mlm_scores = self.cls(sequence_output_mlm)
        # logger.info("sequence_output_mlm.shape:{}".format(sequence_output_mlm.shape))
        # logger.info("mlm_scores.shape:{}".format(mlm_scores.shape))
        # logger.info("labels.shape:{}".format(labels.shape))
        mlm_loss = F.cross_entropy(mlm_scores.view(-1, self.tokenizer.vocab_size),
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

    def loose_similarity(self, sequence_output, visual_output):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        visual_output = visual_output.squeeze()
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

        sequence_output = sequence_output.squeeze()
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logit_scale.data = torch.clamp(logit_scale.data, max=100)
        # if self.rank == 0:
        #     logger.info("logit_scale:{},dtype:{}".format(logit_scale, logit_scale.dtype))
        #     logger.info("sequence_output.shape:{}".format(sequence_output.shape))
        #     logger.info("visual_output.shape:{}".format(visual_output.shape))
        retrieve_logits = logit_scale * torch.matmul(sequence_output, visual_output.t())
        return retrieve_logits

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_k in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_k.data.copy_(param.data)  # initialize
                param_k.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_k in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_k.data = param_k.data * self.contrast_momentum + param.data * (1. - self.contrast_momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, v1_self_k, v2_self_k, v1_cross_k, v2_cross_k, tag_cross_k, title_cross_k):

        # gather keys before updating queue
        v1_self_k = dist_collect(v1_self_k).squeeze()
        v1_self_k = F.normalize(v1_self_k, dim=1)
        v2_self_k = dist_collect(v2_self_k).squeeze()
        v2_self_k = F.normalize(v2_self_k, dim=1)
        v1_cross_k = dist_collect(v1_cross_k).squeeze()
        v1_cross_k = F.normalize(v1_cross_k, dim=1)
        v2_cross_k = dist_collect(v2_cross_k).squeeze()
        v2_cross_k = F.normalize(v2_cross_k, dim=1)
        tag_cross_k = dist_collect(tag_cross_k).squeeze()
        tag_cross_k = F.normalize(tag_cross_k, dim=1)
        title_cross_k = dist_collect(title_cross_k).squeeze()
        title_cross_k = F.normalize(title_cross_k, dim=1)

        batch_size = v1_self_k.size(0)
        ptr = int(self.queue_ptr)
        if self.rank == 0:
            logger.info(
                "begin>>>>: ptr:{},batch_size:{},queue_size:{}".format(ptr, batch_size, self.contrast_num_negative))
            logger.info("v1_self_k.shape:{},tag_cross_k.shape:{}".format(v1_self_k.shape, tag_cross_k.shape))

        # replace the keys at ptr (dequeue and enqueue)
        self.queue_v1_self_ng[:, ptr:ptr + batch_size] = v1_self_k.T
        self.queue_v2_self_ng[:, ptr:ptr + batch_size] = v2_self_k.T
        self.queue_v1_cross_ng[:, ptr:ptr + batch_size] = v1_cross_k.T
        self.queue_v2_cross_ng[:, ptr:ptr + batch_size] = v2_cross_k.T
        self.queue_tag_cross_ng[:, ptr:ptr + batch_size] = tag_cross_k.T
        self.queue_title_cross_ng[:, ptr:ptr + batch_size] = title_cross_k.T

        # move pointer
        ptr = (ptr + batch_size) % self.contrast_num_negative
        if self.rank == 0:
            logger.info("end>>>>: ptr:{}".format(ptr))
        self.queue_ptr[0] = ptr

    def contrastive_loss(self, q, k, queue):

        q = q.squeeze()
        q = F.normalize(q, dim=1)
        k = k.squeeze()
        k = F.normalize(k, dim=1)

        bs = q.size(0)
        # logger.info("q.dtype:{},k.dtype:{}".format(q.dtype, k.dtype))
        # positive logits: Nx1
        # >>>>>>got error in apex:amp level=01!!!!!!!!!
        # l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_pos = torch.matmul(q, k.T)
        l_pos = torch.diag(l_pos).reshape([bs, -1])
        # negative logits: NxK
        # l_neg = torch.einsum('nc,ck->nk', [q, queue.clone().detach()])
        l_neg = torch.matmul(q, queue.clone().detach())
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        # if self.rank == 0:
        #     logger.info("logits.shape:{}".format(logits.shape))
        # apply temperature
        logits /= self.contrast_temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        return F.cross_entropy(logits, labels)

    def get_cross_fea(self, v1_fea, v2_fea, v1_proj, v2_proj, tag_fea, title_fea, is_momentum=False):
        if self.task_config.cross_MLP == "VT_MLP":
            v1_cross = v1_proj
            v2_cross = v2_proj
            if is_momentum:
                tag_cross = self.t_projector_k(tag_fea)
                title_cross = self.t_projector_k(title_fea)
            else:
                tag_cross = self.t_projector(tag_fea)
                title_cross = self.t_projector(title_fea)
        elif self.task_config.cross_MLP == "V_MLP":
            v1_cross = v1_proj
            v2_cross = v2_proj
            tag_cross = tag_fea
            title_cross = title_fea
        elif self.task_config.cross_MLP == "T_MLP":
            v1_cross = v1_fea
            v2_cross = v2_fea
            if is_momentum:
                tag_cross = self.t_projector_k(tag_fea)
                title_cross = self.t_projector_k(title_fea)
            else:
                tag_cross = self.t_projector(tag_fea)
                title_cross = self.t_projector(title_fea)
        else:
            v1_cross = v1_fea
            v2_cross = v2_fea
            tag_cross = tag_fea
            title_cross = title_fea
        return v1_cross, v2_cross, tag_cross, title_cross

    def forward(self, video_data1, video_data2, video_frame, tag_ids, tag_mask, title_ids, title_mask, global_step):
        tag_ids = tag_ids.view(-1, tag_ids.shape[-1])
        tag_mask = tag_mask.view(-1, tag_mask.shape[-1])
        title_ids = title_ids.view(-1, title_ids.shape[-1])
        title_mask = title_mask.view(-1, title_mask.shape[-1])
        # bs x frames x 3 x H x W
        video1 = torch.as_tensor(video_data1)
        video2 = torch.as_tensor(video_data2)

        if self.rank == 0:
            logger.info("video1.shape:{}, dtype:{}, device:{}".format(video1.shape, video1.dtype, video1.device))

        if self.training:
            # loss = 0.0
            v1_fea, v1_frame = self.visual_encoder(video1, video_frame)
            v2_fea, v2_frame = self.visual_encoder(video2, video_frame)
            tag_fea = self.text_encoder(tag_ids, tag_mask)
            title_fea = self.text_encoder(title_ids, title_mask)

            # for video self supervised learning
            # [bs,hidden_size]
            v1_proj = self.v_projector(v1_fea)
            v1_pred = self.v_predictor(v1_proj)
            # [bs,hidden_size]
            v2_proj = self.v_projector(v2_fea)
            v2_pred = self.v_predictor(v2_proj)
            # [bs,hidden_size]
            v1_cross, v2_cross, tag_cross, title_cross = self.get_cross_fea(v1_fea, v2_fea, v1_proj,
                                                                            v2_proj, tag_fea, title_fea)

            # if self.rank == 0:
            #     logger.info("video1_fea.shape:{}".format(v1_fea.shape))
            #     logger.info("v1_proj.shape:{}".format(v1_proj.shape))
            #     logger.info("v1_pred.shape:{}".format(v1_pred.shape))
            #     logger.info("title_fea.shape:{}".format(title_fea.shape))
            # compute key features
            with torch.no_grad():  # no gradient to keys
                self._momentum_update()  # update the key encoder

                tag_fea_k = self.text_encoder_k(tag_ids, tag_mask)
                title_fea_k = self.text_encoder_k(title_ids, title_mask)
                #
                v1_fea_k, v1_frame_k = self.visual_encoder_k(video1, video_frame)
                v2_fea_k, v2_frame_k = self.visual_encoder_k(video2, video_frame)
                v1_proj_k = self.v_projector_k(v1_fea_k)
                v2_proj_k = self.v_projector_k(v2_fea_k)
                v1_cross_k, v2_cross_k, tag_cross_k, title_cross_k = self.get_cross_fea(v1_fea_k, v2_fea_k,
                                                        v1_proj_k, v2_proj_k, tag_fea_k, title_fea_k, is_momentum=True)

            # compute loss
            if self.rank == 0:
                logger.info(
                    "dtype: v1_fea:{},v1_fea_k:{},title_fea:{}".format(v1_fea.dtype, v1_fea_k.dtype, title_fea.dtype))
            # single modality: video queue loss
            v_queue_loss = self.contrastive_loss(v1_pred, v2_proj_k, self.queue_v2_self_ng) \
                           + self.contrastive_loss(v2_pred, v1_proj_k, self.queue_v1_self_ng)

            # cross modality: queue loss
            v1_tag_queue_loss = self.contrastive_loss(v1_cross, tag_cross_k, self.queue_tag_cross_ng) \
                                + self.contrastive_loss(tag_cross, v1_cross_k, self.queue_v1_cross_ng)
            v1_title_queue_loss = self.contrastive_loss(v1_cross, title_cross_k, self.queue_title_cross_ng) \
                                  + self.contrastive_loss(title_cross, v1_cross_k, self.queue_v1_cross_ng)
            v2_tag_queue_loss = self.contrastive_loss(v2_cross, tag_cross_k, self.queue_tag_cross_ng) \
                                + self.contrastive_loss(tag_cross, v2_cross_k, self.queue_v2_cross_ng)
            v2_title_queue_loss = self.contrastive_loss(v2_cross, title_cross_k, self.queue_title_cross_ng) \
                                  + self.contrastive_loss(title_cross, v2_cross_k, self.queue_v2_cross_ng)
            cross_queue_loss = (v1_tag_queue_loss + v1_title_queue_loss + v2_tag_queue_loss + v2_title_queue_loss) / 4

            # dequeue_and_enqueue
            self._dequeue_and_enqueue(v1_proj_k, v2_proj_k, v1_cross_k, v2_cross_k, tag_cross_k, title_cross_k)

            # for MLM loss
            mlm_tag_loss = self.get_mlm_loss(tag_ids, tag_mask)
            mlm_title_loss = self.get_mlm_loss(title_ids, title_mask)
            mlm_loss = mlm_tag_loss + mlm_title_loss

            # total loss
            # loss += inbatch_loss + v_queue_loss + cross_queue_loss + mlm_loss
            loss = self.weight_v * v_queue_loss + self.weight_cross * cross_queue_loss + self.weight_t * mlm_loss
            if self.rank == 0:
                logger.info("v1_tag_queue_loss:{},v1_title_queue_loss:{},v2_tag_queue_loss:{},v2_title_queue_loss:{}"
                            "".format(v1_tag_queue_loss, v1_title_queue_loss, v2_tag_queue_loss, v2_title_queue_loss))
                logger.info("loss:{},v_queue_loss:{},cross_queue_loss:{},mlm_loss:{}"
                            "".format(loss, v_queue_loss, cross_queue_loss, mlm_loss))
                if self.task_config.logdir:
                    loss_item = {"loss": float(loss), "v_queue_loss": float(v_queue_loss),
                                 "cross_queue_loss": float(cross_queue_loss), "mlm_loss": float(mlm_loss)}
                    self.task_config.writer.add_scalars('loss', loss_item, global_step=global_step)
            return loss
        else:
            return None


class BirdModel(BirdPreTrainedModel):
    def __init__(self, cross_config, task_config):
        super(BirdPreTrainedModel, self).__init__(cross_config)
        self.task_config = task_config
        self.rank = task_config.local_rank
        self.weight_sum = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.float32), requires_grad=True)
        self.logit_scale = torch.nn.Parameter(torch.tensor([np.log(1 / 0.07)], dtype=torch.float32),
                                              requires_grad=True)
        ################## text Encoder
        self.text_encoder = TextEncoder(self.task_config, cross_config)
        ################## visual_encoder
        self.visual_encoder = VisualEncoder(self.rank, cross_config)
        ################## loss function
        self.loss_fct = CrossEn()
        self.loss_fct_dual = Dual_CrossEn()

    def frame_loss(self, query_output, frame_output):
        frame_num = frame_output.size(1)
        loss = 0.
        # for i in range(frame_num):
        #     frame_single = frame_output[:, i, :].squeeze()
        #     sim_matrix = self.loose_similarity(query_output, frame_single)
        #     sim_loss = self.loss_fct(sim_matrix) + self.loss_fct(sim_matrix.T)
        #     loss += sim_loss / frame_num
        frame_single, _ = torch.max(frame_output, dim=1)
        sim_matrix = self.loose_similarity(query_output, frame_single)
        sim_loss = self.loss_fct(sim_matrix) + self.loss_fct(sim_matrix.T)
        loss += sim_loss
        return loss


    def forward(self, query_ids, query_mask, video_data, video_frame, idx, global_step):
        query_ids = query_ids.view(-1, query_ids.shape[-1])
        query_mask = query_mask.view(-1, query_mask.shape[-1])
        # T x 3 x H x W
        video = torch.as_tensor(video_data)
        if self.rank == 0:
            logger.info("video.shape:{}, dtype:{}".format(video.shape, video.dtype))
        if self.training:
            loss = 0.0
            query_output = self.text_encoder(query_ids, query_mask)
            visual_output, frame_output = self.visual_encoder(video, video_frame)
            if self.rank == 0:
                logger.info("query_output.shape:{},dtype:{}".format(query_output.shape, query_output.dtype))
                logger.info("visual_output.shape:{},dtype:{}".format(visual_output.shape, visual_output.dtype))
                logger.info("frame_output.shape:{},dtype:{}".format(frame_output.shape, frame_output.dtype))

            visual_output = dist_collect(visual_output).squeeze(1)
            query_output = dist_collect(query_output).squeeze(1)
            frame_output = dist_collect(frame_output).squeeze(1)

            # in batch loss
            sim_matrix = self.loose_similarity(query_output, visual_output)
            sim_loss = self.loss_fct(sim_matrix) + self.loss_fct(sim_matrix.T)
            loss += sim_loss

            # frame loss
            frame_loss = self.frame_loss(query_output, frame_output)
            loss += frame_loss

            if self.rank == 0:
                logger.info(
                    "sim_loss:{},type:{},sim_matrix.shape:{}".format(sim_loss, sim_loss.dtype, sim_matrix.shape))

                if self.task_config.logdir:
                    self.task_config.writer.add_scalar('loss', float(loss), global_step=global_step)
            return loss
        else:
            return None


class BirdModel_VT(BirdPreTrainedModel):
    def __init__(self, cross_config, task_config):
        super(BirdPreTrainedModel, self).__init__(cross_config)
        self.task_config = task_config
        self.rank = task_config.local_rank
        self.weight_sum = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.float32), requires_grad=True)
        self.logit_scale = torch.nn.Parameter(torch.tensor([np.log(1 / 0.07)], dtype=torch.float32),
                                              requires_grad=True)
        ################## text Encoder
        self.text_encoder = TextEncoder(self.task_config, cross_config)
        ################## visual_encoder
        self.visual_encoder = VisualEncoder(self.rank, cross_config)

        ################## loss function
        self.loss_fct = CrossEn()
        self.loss_fct_dual = Dual_CrossEn()

    def forward(self, query_ids, query_mask, video_data, video_frame, title_ids, title_mask, idx, global_step):
        query_ids = query_ids.view(-1, query_ids.shape[-1])
        query_mask = query_mask.view(-1, query_mask.shape[-1])
        title_ids = title_ids.view(-1, title_ids.shape[-1])
        title_mask = title_mask.view(-1, title_mask.shape[-1])
        # T x 3 x H x W
        video = torch.as_tensor(video_data)
        if self.training:
            loss = 0.0
            query_output = self.text_encoder(query_ids, query_mask)
            title_output = self.text_encoder(title_ids, title_mask)
            visual_output, frame_output = self.visual_encoder(video, video_frame)

            visual_output = dist_collect(visual_output).squeeze(1)
            query_output = dist_collect(query_output).squeeze(1)
            title_output = dist_collect(title_output).squeeze(1)
            # in batch loss
            sim_matrix = self.loose_similarity(query_output, visual_output)
            sim_loss = self.loss_fct(sim_matrix) + self.loss_fct(sim_matrix.T)
            loss += sim_loss

            sim_matrix_title = self.loose_similarity(query_output, title_output)
            sim_loss_title = self.loss_fct(sim_matrix_title) + self.loss_fct(sim_matrix_title.T)
            loss += sim_loss_title

            if self.rank == 0:
                logger.info("sim_loss:{},sim_loss_title:{}".format(sim_loss, sim_loss_title))
                if self.task_config.logdir:
                    loss_item = {"loss": float(loss), "sim_loss": float(sim_loss),
                                 "sim_loss_title": float(sim_loss_title)}
                    # self.task_config.writer.add_scalars('loss', loss_item, global_step=global_step)
                    self.task_config.writer.add_scalar('loss', float(loss), global_step=global_step)
            return loss
        else:
            return None


class MLP(nn.Module):
    def __init__(self, in_dim=512, inner_dim=4096, out_dim=512, num_layers=2):
        super(MLP, self).__init__()

        # hidden layers
        linear_hidden = [nn.Identity()]
        for i in range(num_layers - 1):
            linear_hidden.append(nn.Linear(in_dim if i == 0 else inner_dim, inner_dim))
            linear_hidden.append(nn.BatchNorm1d(inner_dim))
            linear_hidden.append(nn.ReLU(inplace=True))
        self.linear_hidden = nn.Sequential(*linear_hidden)

        self.linear_out = nn.Linear(in_dim if num_layers == 1 else inner_dim,
                                    out_dim) if num_layers >= 1 else nn.Identity()

    def forward(self, x):
        x = self.linear_hidden(x)
        x = self.linear_out(x)

        return x
