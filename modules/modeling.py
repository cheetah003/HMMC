from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from abc import ABC

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, AutoConfig, AutoModel, BertTokenizer

from modules.until_module import PreTrainedModel, AllGather, CrossEn, Dual_CrossEn
from modules.module_cross import CrossModel, CrossConfig, Transformer
from modules.module_vilbert import co_attention_model, BertLMPredictionHead
from modules.module_clip import CLIP, convert_weights

logger = logging.getLogger(__name__)
allgather = AllGather.apply


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
        # clip_state_dict = CLIP.get_config(pretrained_clip_name="RN50")
        # clip_state_dict = CLIP.get_config(pretrained_clip_name="RN101")
        # clip_state_dict = CLIP.get_config(pretrained_clip_name="RN50x4")
        # clip_state_dict = CLIP.get_config(pretrained_clip_name="RN50x16")

        for key, val in clip_state_dict.items():
            new_key = "clip." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()

        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None,
                                                 task_config=task_config)

        model = cls(cross_config, clip_state_dict, *inputs, **kwargs)

        ## ===> Initialization trick [HARD CODE]
        if model.linear_patch == "3d":
            contain_conv2 = False
            for key in state_dict.keys():
                if key.find("visual.conv2.weight") > -1:
                    contain_conv2 = True
                    break
            if contain_conv2 is False and hasattr(model.clip.visual, "conv2"):
                cp_weight = state_dict["clip.visual.conv1.weight"].clone()
                kernel_size = model.clip.visual.conv2.weight.size(2)
                conv2_size = model.clip.visual.conv2.weight.size()
                conv2_size = list(conv2_size)

                left_conv2_size = conv2_size.copy()
                right_conv2_size = conv2_size.copy()
                left_conv2_size[2] = (kernel_size - 1) // 2
                right_conv2_size[2] = kernel_size - 1 - left_conv2_size[2]

                left_zeros, right_zeros = None, None
                if left_conv2_size[2] > 0:
                    left_zeros = torch.zeros(*tuple(left_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)
                if right_conv2_size[2] > 0:
                    right_zeros = torch.zeros(*tuple(right_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)

                cat_list = []
                if left_zeros != None: cat_list.append(left_zeros)
                cat_list.append(cp_weight.unsqueeze(2))
                if right_zeros != None: cat_list.append(right_zeros)
                cp_weight = torch.cat(cat_list, dim=2)

                state_dict["clip.visual.conv2.weight"] = cp_weight

        if model.sim_header == 'tightTransf':
            contain_cross = False
            for key in state_dict.keys():
                if key.find("cross.transformer") > -1:
                    contain_cross = True
                    break
            if contain_cross is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["cross.embeddings.position_embeddings.weight"] = val.clone()
                        continue
                    if key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])

                        # cut from beginning
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict["cross." + key] = val.clone()
                            continue

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
        ## <=== End of initialization trick

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


        self.loose_type = False
        if check_attr('loose_type', self.task_config):
            self.loose_type = True
            show_log(task_config, "Test retrieval by loose type.")
        self.mlm_probability = 0.15
        self.weight_sum = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        ################## begin of chinese text Encoder
        # pretrained = 'voidful/albert_chinese_base'
        pretrained = 'hfl/chinese-roberta-wwm-ext'
        # pretrained = 'hfl/chinese-roberta-wwm-ext-large'
        # pretrained = "nghuyong/ernie-1.0"
        self.tokenizer = BertTokenizer.from_pretrained(pretrained)
        logger.info("tokenizer:pad:{},cls:{},mask:{}".format(self.tokenizer.pad_token_id,
                                                             self.tokenizer.cls_token_id,self.tokenizer.mask_token_id))
        my_config = AutoConfig.from_pretrained(pretrained)
        self.chinese_bert_config = my_config
        logger.info("name:{},chinesebert_config:{}".format(pretrained, my_config))
        self.chinese_bert = AutoModel.from_pretrained(pretrained)
        ################### End of albert text Encoder
        self.cls = BertLMPredictionHead(my_config)
        ################## begin of co_attention_model
        self.co_attention_model = co_attention_model()
        ################## end of co_attention_model
        # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>
        vit = "visual.proj" in clip_state_dict
        # assert vit
        if vit:
            vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len(
                [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b
                            in [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == clip_state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32  # resolution 224
            # image_resolution = output_width * 32 * 2  # resolution 448
            # logger.info("shape:{}".format(clip_state_dict["visual.attnpool.positional_embedding"].shape))
        # logger.info("output_width:{}".format(output_width))

        # embed_dim = clip_state_dict["text_projection"].shape[1]
        embed_dim = my_config.hidden_size
        context_length = clip_state_dict["positional_embedding"].shape[0]
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        # vocab_size = my_config.vocab_size
        # transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_width = my_config.hidden_size
        transformer_heads = transformer_width // 64
        transformer_layers = len(
            set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

        show_log(task_config, "\t embed_dim: {}".format(embed_dim))
        show_log(task_config, "\t image_resolution: {}".format(image_resolution))
        show_log(task_config, "\t vision_layers: {}".format(vision_layers))
        show_log(task_config, "\t vision_width: {}".format(vision_width))
        show_log(task_config, "\t vision_patch_size: {}".format(vision_patch_size))
        show_log(task_config, "\t context_length: {}".format(context_length))
        show_log(task_config, "\t not used vocab_size: {}".format(vocab_size))
        show_log(task_config, "\t transformer_width: {}".format(transformer_width))
        show_log(task_config, "\t transformer_heads: {}".format(transformer_heads))
        show_log(task_config, "\t transformer_layers: {}".format(transformer_layers))

        self.linear_patch = '2d'
        if hasattr(task_config, "linear_patch"):
            self.linear_patch = task_config.linear_patch
            show_log(task_config, "\t\t linear_patch: {}".format(self.linear_patch))

        # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
        cut_top_layer = 0
        show_log(task_config, "\t cut_top_layer: {}".format(cut_top_layer))
        self.clip = CLIP(
            embed_dim,
            # image_resolution, vision_layers - cut_top_layer, vision_width, vision_patch_size,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers - cut_top_layer,
            linear_patch=self.linear_patch
        ).float()

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]

        convert_weights(self.clip)
        # <=== End of CLIP Encoders

        self.sim_header = 'meanP'
        if hasattr(task_config, "sim_header"):
            self.sim_header = task_config.sim_header
            show_log(task_config, "\t sim_header: {}".format(self.sim_header))
        if self.sim_header == "tightTransf": assert self.loose_type is False

        cross_config.max_position_embeddings = context_length
        cross_config = update_attr("cross_config", cross_config, "num_hidden_layers", self.task_config,
                                   "cross_num_hidden_layers")
        self.cross = CrossModel(cross_config)
        if self.loose_type is False:
            self.similarity_dense = nn.Linear(cross_config.hidden_size, 1)

        if self.sim_header == "seqLSTM" or self.sim_header == "seqTransf":
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings,
                                                          cross_config.hidden_size)
        if self.sim_header == "seqTransf":
            self.transformerClip = Transformer(width=transformer_width,
                                                   layers=self.task_config.cross_num_hidden_layers,
                                                   heads=transformer_heads, )
        if self.sim_header == "seqLSTM":
            self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                       batch_first=True, bidirectional=False, num_layers=1)

        ################## begin of tag model
        self.tag_model = Transformer(width=transformer_width,
                                                   layers=self.task_config.tag_num_hidden_layers,
                                                   heads=transformer_heads, )
        ################## end of tag model

        self.loss_fct = CrossEn()
        self.loss_fct_dual = Dual_CrossEn()

        self.apply(self.init_weights)

    def forward(self, video_data, video_mask, tag_ids, tag_mask, title_ids, title_mask, asr_ids=None, asr_mask=None):

        tag_ids = tag_ids.view(-1, tag_ids.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        tag_mask = tag_mask.view(-1, tag_mask.shape[-1])
        title_ids = title_ids.view(-1, title_ids.shape[-1])
        title_mask = title_mask.view(-1, title_mask.shape[-1])

        if self.task_config.stage == "stage1":
            asr_ids = asr_ids.view(-1, title_ids.shape[-1])
            asr_mask = asr_mask.view(-1, asr_mask.shape[-1])
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
                tag_output = self.get_sequence_output(tag_ids, tag_mask, shaped=True)
                visual_output = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_frame)
                title_output = self.get_sequence_output(title_ids, title_mask, shaped=True)
                asr_output = self.get_sequence_output(asr_ids, asr_mask, shaped=True)
                logger.info("visual_output.shape:{}".format(visual_output.shape))
                co_sequence_output, co_visual_output = self.co_attention_model(visual_output, visual_output)
                #for contrasive loss
                # video-tag
                sim_matrix = self.get_similarity_logits(tag_output, co_visual_output)
                sim_loss = self.loss_fct(sim_matrix) + self.loss_fct(sim_matrix.T)
                loss += sim_loss
                # video-title
                sim_video_title = self.get_similarity_logits(title_output, co_visual_output)
                sim_title_loss = self.loss_fct(sim_video_title) + self.loss_fct(sim_video_title.T)
                loss += sim_title_loss
                # video-asr
                sim_video_asr = self.get_similarity_logits(asr_output, co_visual_output)
                sim_asr_loss = self.loss_fct(sim_video_asr) + self.loss_fct(sim_video_asr.T)
                loss += sim_asr_loss
                logger.info("sim_loss:{},sim_title_loss:{},sim_asr_loss:{}".format(sim_loss, sim_title_loss, sim_asr_loss))
                # _, pooled_output = self.get_cross_output(text_fea=title_output)
                # # title - tag
                # sim_title_tag = self.loose_similarity_for_text(sequence_output, title_output)
                # sim_title_tag_loss = self.loss_fct(sim_title_tag) + self.loss_fct(sim_title_tag.T)
                # loss += sim_title_tag_loss
                # title_output = self.get_sequence_output(title_ids, input_mask, input_segment, shaped=True)
                co_sequence_output, co_visual_output = self.co_attention_model(title_output, visual_output)
                # (video,title) - tag
                sim_videotitle_tag = self.get_similarity_logits(tag_output, co_visual_output)
                sim_videotitle_tag_loss = self.loss_fct(sim_videotitle_tag) + self.loss_fct(sim_videotitle_tag.T)
                loss += sim_videotitle_tag_loss
                logger.info("sim_videotitle_tag_loss:{}".format(sim_videotitle_tag_loss))
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

                masked_tag_output = self.get_sequence_output(masked_tag_ids, tag_mask,return_hidden=True, shaped=True)
                masked_title_output = self.get_sequence_output(masked_title_ids, title_mask, return_hidden=True, shaped=True)

                co_masked_tag_output, _ = self.co_attention_model(masked_tag_output, visual_output)
                co_masked_title_output, _ = self.co_attention_model(masked_title_output, visual_output)

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

    def calculate_mfm_loss(self, visual_output_alm, video, video_mask, video_labels_index):
        afm_scores = self.cls_visual(visual_output_alm)
        afm_scores_tr = afm_scores.view(-1, afm_scores.shape[-1])

        video_tr = video.permute(2, 0, 1)
        video_tr = video_tr.view(video_tr.shape[0], -1)

        logits_matrix = torch.mm(afm_scores_tr, video_tr)
        video_mask_float = video_mask.to(dtype=torch.float)
        mask_matrix = torch.mm(video_mask_float.view(-1, 1), video_mask_float.view(1, -1))
        masked_logits = logits_matrix + (1. - mask_matrix) * -1e8

        logpt = F.log_softmax(masked_logits, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt

        video_labels_index_mask = (video_labels_index != self.ignore_video_index)
        nce_loss = nce_loss.masked_select(video_labels_index_mask.view(-1))
        nce_loss = nce_loss.mean()
        return nce_loss

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

    def get_sequence_output(self, input_ids, attention_mask, return_hidden=False, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        bs_pair = input_ids.size(0)
        # logger.info("albert encoder")
        # logger.info("input_ids.shape:{}".format(input_ids.shape))
        sequence_hidden = self.chinese_bert(input_ids, attention_mask=attention_mask)
        # logger.info("before sequence_hidden.shape:{}".format(sequence_hidden.shape))
        if return_hidden:
            sequence_hidden = sequence_hidden[0]
        else:
            sequence_hidden = sequence_hidden[1]

        sequence_hidden = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1))
        # logger.info("after sequence_hidden1.shape:{}".format(sequence_hidden.shape))
        return sequence_hidden

    def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            video = torch.as_tensor(video).float()
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts
        bs_pair = video.size(0) // video_frame
        # logger.info("video.shape:{}".format(video.shape))
        visual_hidden = self.clip.encode_image(video, video_frame=video_frame).float()
        # logger.info("visual_hidden.shape:{}".format(visual_hidden.shape))
        visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1))
        # logger.info("visual_hidden.shape:{}".format(visual_hidden.shape))
        return visual_hidden

    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, video, video_mask, shaped=False,
                                   video_frame=-1):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        sequence_output = self.get_sequence_output(input_ids, token_type_ids, attention_mask,return_hidden=False, shaped=True)
        visual_output = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_frame)
        return sequence_output, visual_output

    def get_cross_output(self, text_fea=None, video_fea=None):
        assert (text_fea is not None) or (video_fea is not None)

        if (text_fea is not None) and (video_fea is not None):
            # logger.info("text and video")
            # logger.info("text_fea.shape:{}".format(text_fea.shape))
            # logger.info("video_fea.shape:{}".format(video_fea.shape))
            concat_features = torch.cat((text_fea, video_fea), dim=1)  # concatnate tokens and frames
            # concat_mask = torch.cat((text_mask, video_mask), dim=1)
            video_type_ = torch.zeros_like(video_fea[:, :, 0])
            text_type_ = torch.ones_like(text_fea[:, :, 0])
            # logger.info("video_type_.shape:{}".format(video_type_.shape))
            # logger.info("text_type_.shape:{}".format(text_type_.shape))
            concat_type = torch.cat((text_type_, video_type_), dim=1)
        elif video_fea is not None:
            # logger.info("only video")
            concat_features = video_fea
            # concat_mask = video_mask
            concat_type = torch.zeros_like(video_fea[:, :, 0])
        elif text_fea is not None:
            # logger.info("only text")
            concat_features = text_fea
            # concat_mask = text_mask
            concat_type = torch.ones_like(text_fea[:, :, 0])

        logger.info("concat_features.shape:{}".format(concat_features.shape))
        # logger.info("concat_mask.shape:{}".format(concat_mask.shape))
        # logger.info("concat_type.shape:{}".format(concat_type.shape))
        cross_layers, pooled_output = self.cross(concat_features, concat_type)
        # logger.info("cross_layers.shape:{}".format(cross_layers.shape))
        # logger.info("pooled_output.shape:{}".format(pooled_output.shape))

        return cross_layers, pooled_output

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
        # logger.info("logit_scale_text:{}".format(logit_scale))
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
        # logger.info("logit_scale:{}".format(logit_scale))
        retrieve_logits = logit_scale * torch.matmul(sequence_output, visual_output.t())
        # logger.info("retrieve_logits.shape:{}".format(retrieve_logits.shape))
        return retrieve_logits

    def cross_similarity(self, sequence_output, visual_output, attention_mask, video_mask):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        if self.training:
            visual_output = allgather(visual_output, self.task_config)
            video_mask = allgather(video_mask, self.task_config)
            sequence_output = allgather(sequence_output, self.task_config)
            torch.distributed.barrier()

        b_text, s_text, h_text = sequence_output.size()

        b_visual, s_visual, h_visual = visual_output.size()

        retrieve_logits_list = []

        step_size = b_text  # set smaller to reduce memory cost
        split_size = [step_size] * (b_text // step_size)
        release_size = b_text - sum(split_size)
        if release_size > 0:
            split_size += [release_size]

        # due to clip text branch retrun the last hidden
        attention_mask = torch.ones(sequence_output.size(0), 1) \
            .to(device=attention_mask.device, dtype=attention_mask.dtype)

        sequence_output_splits = torch.split(sequence_output, split_size, dim=0)
        attention_mask_splits = torch.split(attention_mask, split_size, dim=0)
        logger.info("split_size:{},len(split_size):{}".format(split_size, len(split_size)))
        for i in range(len(split_size)):
            sequence_output_row = sequence_output_splits[i]
            attention_mask_row = attention_mask_splits[i]
            sequence_output_l = sequence_output_row.unsqueeze(1).repeat(1, b_visual, 1, 1)
            sequence_output_l = sequence_output_l.view(-1, s_text, h_text)
            attention_mask_l = attention_mask_row.unsqueeze(1).repeat(1, b_visual, 1)
            attention_mask_l = attention_mask_l.view(-1, s_text)
            logger.info("sequence_output_l:{}".format(sequence_output_l.shape))
            step_truth = sequence_output_row.size(0)
            visual_output_r = visual_output.unsqueeze(0).repeat(step_truth, 1, 1, 1)
            visual_output_r = visual_output_r.view(-1, s_visual, h_visual)
            video_mask_r = video_mask.unsqueeze(0).repeat(step_truth, 1, 1)
            video_mask_r = video_mask_r.view(-1, s_visual)
            logger.info("visual_output_r:{}".format(visual_output_r.shape))

            cross_output, pooled_output = \
                self.get_cross_output(sequence_output_l, visual_output_r, attention_mask_l, video_mask_r)

            logger.info("pooled_output:{}".format(pooled_output.shape))
            logger.info("cross_output:{}".format(cross_output.shape))
            retrieve_logits_row = self.similarity_dense(pooled_output).squeeze(-1).view(step_truth, b_visual)

            retrieve_logits_list.append(retrieve_logits_row)

        retrieve_logits = torch.cat(retrieve_logits_list, dim=0)
        return retrieve_logits

    def get_similarity_logits(self, sequence_output, visual_output):
        retrieve_logits = self.loose_similarity(sequence_output, visual_output)
        return retrieve_logits


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
            hard_visual_output = self.get_visual_output(hard_video, hard_video_mask, shaped=True, video_frame=video_frame)
            pos_title_output = self.get_sequence_output(pos_title_ids, pos_title_mask, shaped=True)
            hard_title_output = self.get_sequence_output(hard_title_ids, hard_title_mask, shaped=True)
            # logger.info("pos_title_output.shape:{}".format(pos_title_output.shape))
            # logger.info("pos_visual_output.shape:{}".format(pos_visual_output.shape))
            # _, pooled_output = self.get_cross_output(video_fea=visual_output, text_fea=title_output)

            _, pos_co_visual_output = self.co_attention_model(pos_title_output, pos_visual_output)
            _, hard_co_visual_output = self.co_attention_model(hard_title_output, hard_visual_output)

            # random negtive loss
            sim_matrix = self.get_similarity_logits(query_output, pos_co_visual_output)
            sim_loss = self.loss_fct(sim_matrix) + self.loss_fct(sim_matrix.T)
            loss += sim_loss
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
            logger.info("hard_loss:{}".format(hard_loss))
            loss += hard_loss


            # logger.info("hard_sim_matrix:{}".format(hard_sim_matrix))
            return loss
        else:
            return None