from typing import Callable, Any, Optional, Tuple, Callable, List, Dict, cast
import os
import json
import sys
import numpy as np

import lmdb
import torch
import random
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from dataloaders.randaugment import RandomAugment
from PIL import Image
from PIL import ImageFilter
import cv2
import logging

logger = logging.getLogger(__name__)

# global, number of frames in lmdb per video
g_lmdb_frames = 48
title_max_words = 45
tag_max_words = 25
query_max_words = 15


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def read_json(path):
    with open(path, 'r', encoding='utf8') as f:
        data = json.load(f)
    return data


def read_json_line(path):
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        data = []
        for line in lines:
            item = json.loads(line)
            data.append(item)
    return data


def get_flat_query_list(query_list):
    flat_query_list = list()
    for itm in query_list:
        query = itm['query']
        query_eng = itm["query_eng"]
        poslist = itm['videolist']
        for positem in poslist:
            item = dict()
            item["query"] = query
            item["query_eng"] = query_eng
            item["docid"] = positem["docid"]
            item["title"] = positem["title"]
            item["title_eng"] = positem["title_eng"]
            item["duration"] = positem["duration"]
            flat_query_list.append(item)

    return flat_query_list


class dataload_bird_pretrain(VisionDataset):
    def __init__(self, root: str, language: str, json_path: str, maxTxns: int = 1, tokenizer=None,
                 resolution=224, max_frames=12, frame_sample=None, frame_sample_len=None) -> None:
        super().__init__(root)
        self._maxTxns = maxTxns
        # env and txn is delay-loaded in ddp. They can't pickle
        self._env = None
        self._txn = None
        self.language = language
        self.resolution = resolution
        self.max_frames = max_frames
        self.frame_sample = frame_sample
        self.frame_sample_len = frame_sample_len
        self.title_max_words = title_max_words
        self.tag_max_words = tag_max_words
        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
        else:
            self.tokenizer = tokenizer
        # print("self.tokenizer:{}.".format(self.tokenizer.vocab))
        # Length is needed for DistributedSampler, but we can't use env to get it, env can't pickle.
        # So we decide to read from metadata placed in the same folder --- see src/misc/datasetCreate.py
        # with open(os.path.join(root, "metadata.json"), "r") as fp:
        #     metadata = json.load(fp)
        # self._length = metadata["length"]
        self.datalist = read_json_line(json_path)

        # for fast debug
        # self.datalist = self.datalist[::10]

        self._length = len(self.datalist)
        if self.language == "chinese":
            self.SPECIAL_TOKEN = {"CLS_TOKEN": "[CLS]", "SEP_TOKEN": "[SEP]",
                                  "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        else:
            self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                                  "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),  # 0.08-1
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.4578275], std=[0.26862954, 0.26130258, 0.27577711]),
        ])
        # self.transform = transforms.Compose([
        #     transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        #     transforms.CenterCrop(self.resolution),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        # ])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._txn is not None:
            self._txn.__exit__(exc_type, exc_val, exc_tb)
        if self._env is not None:
            self._env.close()

    def _initEnv(self):
        self._env = lmdb.open(self.root, map_size=1024 * 1024 * 1024 * 500, subdir=True, readonly=True, readahead=False,
                              meminit=False, max_spare_txns=self._maxTxns, lock=False)
        self._txn = self._env.begin(write=False, buffers=True)

    def _get_text(self, caption, max_words):
        words = self.tokenizer.tokenize(caption)
        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = max_words - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
        input_ids = self.tokenizer.convert_tokens_to_ids(words)

        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)
        while len(input_ids) < max_words:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        assert len(input_ids) == max_words
        assert len(input_mask) == max_words
        assert len(segment_ids) == max_words

        pairs_text = np.array(input_ids)
        pairs_mask = np.array(input_mask)
        pairs_segment = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment

    def _get_video(self, video_key, frames):
        video_list = list()
        global g_lmdb_frames
        # global writer
        # random sample start ##################################################
        if self.frame_sample == "uniform_random":
            # assert g_lmdb_frames % frames == 0
            video_index = list(np.arange(0, g_lmdb_frames))
            # print("video_index:{}".format(video_index))
            sample_slice = list()
            k = g_lmdb_frames // frames
            for i in np.arange(frames):
                index = random.sample(video_index[k * i:k * (i + 1)], 1)
                sample_slice.append(index[0])
        elif self.frame_sample == "random":
            # sample
            video_index = list(np.arange(0, g_lmdb_frames))
            sample_slice = random.sample(video_index, frames)
            sample_slice = sorted(sample_slice)
        else:
            sample_slice = np.linspace(0, g_lmdb_frames, frames, endpoint=False, dtype=int)
            # random sample end ##################################################
        for step, i in enumerate(sample_slice):
            video_key_new = video_key + "_%d" % i
            video_key_new = video_key_new.encode()
            video = self._txn.get(video_key_new)
            frame_buffer = np.frombuffer(video, dtype=np.uint8)
            # print("frame_buffer.shape:{}".format(frame_buffer.shape))
            frame_data = cv2.imdecode(frame_buffer, cv2.IMREAD_COLOR)
            # print("frame_data.shape:{}".format(frame_data.shape))
            frame_rgb = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
            # print("[{}]frame_rgb.shape:{}".format(step, frame_rgb.shape))
            frame_img = Image.fromarray(frame_rgb).convert("RGB")
            # print("frame_img.shape:{}".format(np.array(frame_img).shape))
            # writer.add_image('original_img', frame_rgb, global_step=step, dataformats='HWC')
            frame_data = self.transform(frame_img)
            # print("[{}]frame_data.shape:{}".format(step, frame_data.shape))
            video_list.append(frame_data)
        video_data = np.stack(video_list)
        video_data = video_data.copy()
        # video_data = video_data.astype('float64')
        video_data = video_data.reshape([frames, 3, self.resolution, self.resolution])
        if self.frame_sample_len == "dynamic":
            # dynamic frame needs pad
            if frames < self.max_frames:
                pad = np.zeros([self.max_frames - frames, 3, self.resolution, self.resolution],
                               dtype=np.float32)
                video_data = np.concatenate((video_data, pad), axis=0)

        return video_data

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if self._env is None:
            self._initEnv()
        item = self.datalist[index]
        if self.frame_sample_len == "dynamic":
            # dynamic frame
            frames = min(max(int(item["duration"] * 0.3), 3), self.max_frames)
        else:
            # fix frame
            frames = self.max_frames
        video_key = "Video" + item['docid']
        video_data = self._get_video(video_key, frames)
        if self.language == "chinese":
            tag_text = item['tag']
            title_text = item['title']
        elif self.language == "english":
            tag_text = item['tag_eng']
            title_text = item['title_eng']
        else:
            raise NotImplementedError("bilingual:not implemented!")
        # print("[{}] video_key:{}; title:{}; tag:{}".format(index, video_key, title_text, tag_text))
        tag_ids, tag_mask, _ = self._get_text(tag_text, self.tag_max_words)
        title_ids, title_mask, _ = self._get_text(title_text, self.title_max_words)

        return video_data, frames, tag_ids, tag_mask, title_ids, title_mask

    def __len__(self) -> int:
        return self._length


class dataload_bird_train(VisionDataset):
    def __init__(self, root: str, language: str, json_path: str, maxTxns: int = 1, tokenizer=None, resolution=224,
                 max_frames=24,
                 frame_sample=None, frame_sample_len=None, task="retrieval_VT") -> None:
        super().__init__(root)
        self._maxTxns = maxTxns
        # env and txn is delay-loaded in ddp. They can't pickle
        self._env = None
        self._txn = None
        self.language = language
        self.resolution = resolution
        self.max_frames = max_frames
        self.frame_sample = frame_sample
        self.frame_sample_len = frame_sample_len
        self.query_max_words = query_max_words
        self.title_max_words = title_max_words
        self.task = task
        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
        else:
            self.tokenizer = tokenizer
        querylist = read_json_line(json_path)
        self.datalist = get_flat_query_list(querylist)
        # for fast debug
        # self.datalist = self.datalist[0:45000:10]

        self._length = len(self.datalist)
        if self.language == "chinese":
            self.SPECIAL_TOKEN = {"CLS_TOKEN": "[CLS]", "SEP_TOKEN": "[SEP]",
                                  "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        else:
            self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                                  "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        # self.transform = transforms.Compose([
        #     transforms.RandomResizedCrop(224, scale=(0.5, 1.)),  # 0.08-1
        #     transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        #     transforms.RandomGrayscale(p=0.2),
        #     transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        # ])
        self.transform = transforms.Compose([
            transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.resolution),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._txn is not None:
            self._txn.__exit__(exc_type, exc_val, exc_tb)
        if self._env is not None:
            self._env.close()

    def _initEnv(self):
        self._env = lmdb.open(self.root, map_size=1024 * 1024 * 1024 * 80, subdir=True, readonly=True, readahead=False,
                              meminit=False, max_spare_txns=self._maxTxns, lock=False)
        self._txn = self._env.begin(write=False, buffers=True)

    def _get_text(self, caption, max_words):
        words = self.tokenizer.tokenize(caption)
        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = max_words - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
        input_ids = self.tokenizer.convert_tokens_to_ids(words)

        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)
        while len(input_ids) < max_words:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        assert len(input_ids) == max_words
        assert len(input_mask) == max_words
        assert len(segment_ids) == max_words

        pairs_text = np.array(input_ids)
        pairs_mask = np.array(input_mask)
        pairs_segment = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment

    def _get_video(self, video_key, frames):
        video_list = list()
        global g_lmdb_frames
        # global writer
        # random sample start ##################################################
        if self.frame_sample == "uniform_random":
            # assert g_lmdb_frames % frames == 0
            video_index = list(np.arange(0, g_lmdb_frames))
            # print("video_index:{}".format(video_index))
            sample_slice = list()
            k = g_lmdb_frames // frames
            for i in np.arange(frames):
                index = random.sample(video_index[k * i:k * (i + 1)], 1)
                sample_slice.append(index[0])
        elif self.frame_sample == "random":
            # sample
            video_index = list(np.arange(0, g_lmdb_frames))
            sample_slice = random.sample(video_index, frames)
            sample_slice = sorted(sample_slice)
        else:
            sample_slice = np.linspace(0, g_lmdb_frames, frames, endpoint=False, dtype=int)
            # random sample end ##################################################
        for step, i in enumerate(sample_slice):
            video_key_new = video_key + "_%d" % i
            video_key_new = video_key_new.encode()
            video = self._txn.get(video_key_new)
            frame_buffer = np.frombuffer(video, dtype=np.uint8)
            # print("frame_buffer.shape:{}".format(frame_buffer.shape))
            frame_data = cv2.imdecode(frame_buffer, cv2.IMREAD_COLOR)
            # print("frame_data.shape:{}".format(frame_data.shape))
            frame_rgb = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
            # print("[{}]frame_rgb.shape:{}".format(step, frame_rgb.shape))
            frame_img = Image.fromarray(frame_rgb).convert("RGB")
            # print("frame_img.shape:{}".format(np.array(frame_img).shape))
            # writer.add_image('original_img', frame_rgb, global_step=step, dataformats='HWC')
            frame_data = self.transform(frame_img)
            # print("[{}]frame_data.shape:{}".format(step, frame_data.shape))
            video_list.append(frame_data)
        video_data = np.stack(video_list)
        video_data = video_data.copy()
        # video_data = video_data.astype('float64')
        video_data = video_data.reshape([frames, 3, self.resolution, self.resolution])
        if self.frame_sample_len == "dynamic":
            # dynamic frame needs pad
            if frames < self.max_frames:
                pad = np.zeros([self.max_frames - frames, 3, self.resolution, self.resolution],
                               dtype=np.float32)
                video_data = np.concatenate((video_data, pad), axis=0)

        return video_data

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if self._env is None:
            self._initEnv()
        item = self.datalist[index]
        if self.frame_sample_len == "dynamic":
            # dynamic frame
            frames = min(max(int(item["duration"] * 0.5), 3), self.max_frames)
        else:
            frames = self.max_frames
        videoid = "Video" + item['docid']
        video_data = self._get_video(videoid, frames)
        if self.language == "chinese":
            query = item['query']
            title = item['title']
        elif self.language == "english":
            query = item['query_eng']
            title = item['title_eng']
        else:
            raise NotImplementedError("bilingual:not implemented!")

        query_ids, query_mask, _ = self._get_text(query, self.query_max_words)
        # logger.info("train:[{}]query:{},video:{}".format(index, query, videoid))
        if self.task == "retrieval_VT":
            title_ids, title_mask, _ = self._get_text(title, self.title_max_words)
            return query_ids, query_mask, video_data, frames, title_ids, title_mask, index
        else:
            return query_ids, query_mask, video_data, frames, index

    def __len__(self) -> int:
        return self._length


class dataload_bird_val(VisionDataset):
    def __init__(self, root: str, language: str, json_path: str, maxTxns: int = 1, tokenizer=None,
                 resolution=224, max_frames=24, frame_sample_len=None, task="retrieval_VT") -> None:
        super().__init__(root)
        self._maxTxns = maxTxns
        # env and txn is delay-loaded in ddp. They can't pickle
        self._env = None
        self._txn = None
        self.language = language
        self.resolution = resolution
        self.max_frames = max_frames
        self.frame_sample_len = frame_sample_len
        self.query_max_words = query_max_words
        self.title_max_words = title_max_words
        self.task = task
        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
        else:
            self.tokenizer = tokenizer

        self.datalist = read_json_line(json_path)
        # for fast debug
        # self.datalist = self.datalist[:100]

        self._length = len(self.datalist)
        if self.language == "chinese":
            self.SPECIAL_TOKEN = {"CLS_TOKEN": "[CLS]", "SEP_TOKEN": "[SEP]",
                                  "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        else:
            self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                                  "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        self.transform = transforms.Compose([
            transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.resolution),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._txn is not None:
            self._txn.__exit__(exc_type, exc_val, exc_tb)
        if self._env is not None:
            self._env.close()

    def _initEnv(self):
        self._env = lmdb.open(self.root, map_size=1024 * 1024 * 1024 * 80, subdir=True, readonly=True, readahead=False,
                              meminit=False, max_spare_txns=self._maxTxns, lock=False)
        self._txn = self._env.begin(write=False, buffers=True)

    def _get_pos_pair(self, item=None):
        if self.language == "chinese":
            query = item['query']
        elif self.language == "english":
            query = item['query_eng']
        else:
            raise NotImplementedError("bilingual:not implemented!")
        poslist = item['videolist']
        pos_item = poslist[0]
        return query, pos_item

    def _get_text(self, caption, max_words):
        words = self.tokenizer.tokenize(caption)
        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = max_words - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
        input_ids = self.tokenizer.convert_tokens_to_ids(words)

        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)
        while len(input_ids) < max_words:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        assert len(input_ids) == max_words
        assert len(input_mask) == max_words
        assert len(segment_ids) == max_words

        pairs_text = np.array(input_ids)
        pairs_mask = np.array(input_mask)
        pairs_segment = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment

    def _get_video(self, video_key, frames):
        video_list = list()
        global g_lmdb_frames
        # uniform sample start ##################################################
        # assert g_lmdb_frames % self.max_frames == 0
        # video_index = np.arange(0, g_lmdb_frames, g_lmdb_frames // self.max_frames)
        video_index = np.linspace(0, g_lmdb_frames, frames, endpoint=False, dtype=int)
        # uniform sample end ##################################################
        for i in video_index:
            video_key_new = video_key + "_%d" % i
            video_key_new = video_key_new.encode()
            video = self._txn.get(video_key_new)
            frame_buffer = np.frombuffer(video, dtype=np.uint8)
            # print("frame_buffer.shape:{}".format(frame_buffer.shape))
            frame_data = cv2.imdecode(frame_buffer, cv2.IMREAD_COLOR)
            # print("frame_data.shape:{}".format(frame_data.shape))
            frame_rgb = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
            frame_img = Image.fromarray(frame_rgb).convert("RGB")
            frame_data = self.transform(frame_img)
            video_list.append(frame_data)
        video_data = np.stack(video_list)
        video_data = video_data.copy()
        video_data = video_data.reshape([frames, 3, self.resolution, self.resolution])
        if self.frame_sample_len == "dynamic":
            # dynamic frame needs pad
            if frames < self.max_frames:
                pad = np.zeros([self.max_frames - frames, 3, self.resolution, self.resolution],
                               dtype=np.float32)
                video_data = np.concatenate((video_data, pad), axis=0)

        return video_data

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if self._env is None:
            self._initEnv()
        item = self.datalist[index]
        query, pos_item = self._get_pos_pair(item)
        videoid = "Video" + pos_item["docid"]
        if self.frame_sample_len == "dynamic":
            # dynamic frame
            frames = min(max(int(pos_item["duration"] * 0.5), 3), self.max_frames)
        else:
            frames = self.max_frames
        video_data = self._get_video(videoid, frames)
        # query = "关于 " + query + " 的视频"
        # logger.info("val:[{}]query:{},video:{}".format(index, query, videoid))
        # print("video[{}]:{}".format(index, item['video_id']))
        query_ids, query_mask, _ = self._get_text(query, self.query_max_words)

        if self.task == "retrieval_VT":
            if self.language == "chinese":
                title = pos_item['title']
            elif self.language == "english":
                title = pos_item['title_eng']
            title_ids, title_mask, _ = self._get_text(title, self.title_max_words)
            return query_ids, query_mask, video_data, frames, title_ids, title_mask
        else:
            return query_ids, query_mask, video_data, frames

    def __len__(self) -> int:
        return self._length


class dataload_bird_debug_test(VisionDataset):
    def __init__(self, root: str, language: str, json_path: str, maxTxns: int = 1, tokenizer=None,
                 resolution=224, max_frames=12, frame_sample=None, frame_sample_len=None) -> None:
        super().__init__(root)
        self._maxTxns = maxTxns
        # env and txn is delay-loaded in ddp. They can't pickle
        self._env = None
        self._txn = None
        self.language = language
        self.resolution = resolution
        self.max_frames = max_frames
        self.frame_sample = frame_sample
        self.frame_sample_len = frame_sample_len
        self.title_max_words = title_max_words
        self.tag_max_words = tag_max_words
        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
        else:
            self.tokenizer = tokenizer
        # print("self.tokenizer:{}.".format(self.tokenizer.vocab))
        # Length is needed for DistributedSampler, but we can't use env to get it, env can't pickle.
        # So we decide to read from metadata placed in the same folder --- see src/misc/datasetCreate.py
        # with open(os.path.join(root, "metadata.json"), "r") as fp:
        #     metadata = json.load(fp)
        # self._length = metadata["length"]
        self.datalist = read_json_line(json_path)

        # for fast debug
        self.datalist = self.datalist[:1000]

        self._length = len(self.datalist)
        if self.language == "chinese":
            self.SPECIAL_TOKEN = {"CLS_TOKEN": "[CLS]", "SEP_TOKEN": "[SEP]",
                                  "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        else:
            self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                                  "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        self.transform = transforms.Compose([
            transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.resolution),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._txn is not None:
            self._txn.__exit__(exc_type, exc_val, exc_tb)
        if self._env is not None:
            self._env.close()

    def _initEnv(self):
        self._env = lmdb.open(self.root, map_size=1024 * 1024 * 1024 * 500, subdir=True, readonly=True, readahead=False,
                              meminit=False, max_spare_txns=self._maxTxns, lock=False)
        self._txn = self._env.begin(write=False, buffers=True)

    def _get_text(self, caption, max_words):
        words = self.tokenizer.tokenize(caption)
        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = max_words - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
        input_ids = self.tokenizer.convert_tokens_to_ids(words)

        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)
        while len(input_ids) < max_words:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        assert len(input_ids) == max_words
        assert len(input_mask) == max_words
        assert len(segment_ids) == max_words

        pairs_text = np.array(input_ids)
        pairs_mask = np.array(input_mask)
        pairs_segment = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment

    def _get_video(self, video_key, frames):
        video_list = list()
        global g_lmdb_frames
        # global writer
        # random sample start ##################################################
        if self.frame_sample == "uniform_random":
            # assert g_lmdb_frames % frames == 0
            video_index = list(np.arange(0, g_lmdb_frames))
            # print("video_index:{}".format(video_index))
            sample_slice = list()
            k = g_lmdb_frames // frames
            for i in np.arange(frames):
                index = random.sample(video_index[k * i:k * (i + 1)], 1)
                sample_slice.append(index[0])
        elif self.frame_sample == "random":
            # sample
            video_index = list(np.arange(0, g_lmdb_frames))
            sample_slice = random.sample(video_index, frames)
            sample_slice = sorted(sample_slice)
        else:
            sample_slice = np.linspace(0, g_lmdb_frames, frames, endpoint=False, dtype=int)
            # random sample end ##################################################
        for step, i in enumerate(sample_slice):
            video_key_new = video_key + "_%d" % i
            video_key_new = video_key_new.encode()
            video = self._txn.get(video_key_new)
            frame_buffer = np.frombuffer(video, dtype=np.uint8)
            # print("frame_buffer.shape:{}".format(frame_buffer.shape))
            frame_data = cv2.imdecode(frame_buffer, cv2.IMREAD_COLOR)
            # print("frame_data.shape:{}".format(frame_data.shape))
            frame_rgb = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
            # print("[{}]frame_rgb.shape:{}".format(step, frame_rgb.shape))
            frame_img = Image.fromarray(frame_rgb).convert("RGB")
            # print("frame_img.shape:{}".format(np.array(frame_img).shape))
            # writer.add_image('original_img', frame_rgb, global_step=step, dataformats='HWC')
            frame_data = self.transform(frame_img)
            # print("[{}]frame_data.shape:{}".format(step, frame_data.shape))
            video_list.append(frame_data)
        video_data = np.stack(video_list)
        video_data = video_data.copy()
        # video_data = video_data.astype('float64')
        video_data = video_data.reshape([frames, 3, self.resolution, self.resolution])
        if self.frame_sample_len == "dynamic":
            # dynamic frame needs pad
            if frames < self.max_frames:
                pad = np.zeros([self.max_frames - frames, 3, self.resolution, self.resolution],
                               dtype=np.float32)
                video_data = np.concatenate((video_data, pad), axis=0)

        return video_data

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if self._env is None:
            self._initEnv()
        item = self.datalist[index]
        if self.frame_sample_len == "dynamic":
            # dynamic frame
            frames = min(max(int(item["duration"] * 0.3), 3), self.max_frames)
        else:
            # fix frame
            frames = self.max_frames
        video_key = "Video" + item['docid']
        video_data = self._get_video(video_key, frames)
        if self.language == "chinese":
            tag_text = item['tag']
            title_text = item['title']
        elif self.language == "english":
            tag_text = item['tag_eng']
            title_text = item['title_eng']
        else:
            raise NotImplementedError("bilingual:not implemented!")
        # print("[{}] video_key:{}; title:{}; tag:{}".format(index, video_key, title_text, tag_text))
        tag_ids, tag_mask, _ = self._get_text(tag_text, self.tag_max_words)
        title_ids, title_mask, _ = self._get_text(title_text, self.title_max_words)

        return title_ids, title_mask, video_data, frames

    def __len__(self) -> int:
        return self._length