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

#global, number of frames in lmdb per video
g_lmdb_frames = 48
max_dynamic_pretrain_frames = 12
max_dynamic_train_frames = 18
max_dynamic_val_frames = 30

_pretrain_info_path = {
    "chinese": "/ai/swxdisk/data/bird/videoinfo_chinese.json",
    "english": "/ai/swxdisk/data/bird/videoinfo_english.json",
    "bilingual": "/ai/swxdisk/data/bird/videoinfo_bilingual.json"
}
_train_info_path = {
    "chinese": "/ai/swxdisk/data/bird/query_data_train_chinese.json",
    "english": "/ai/swxdisk/data/bird/query_data_train_english.json",
    "bilingual": "/ai/swxdisk/data/bird/query_data_train_bilingual.json"
}
_val_info_path = {
    "chinese": "/ai/swxdisk/data/bird/query_data_val_chinese.json",
    "english": "/ai/swxdisk/data/bird/query_data_val_english.json",
    "bilingual": "/ai/swxdisk/data/bird/query_data_val_bilingual.json"
}


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
        poslist = itm['videolist']
        for positem in poslist:
            item = dict()
            item["query"] = query
            item["docid"] = positem["docid"]
            item["title"] = positem["title"]
            item["duration"] = positem["duration"]
            flat_query_list.append(item)

    return flat_query_list


"""load: video title tag asr"""
class dataload_bird_pretrain(VisionDataset):
    def __init__(self, root: str, language:str, maxTxns: int = 1, tokenizer=None,
                 resolution=224, max_words=32, max_frames=12, transform: Optional[Callable] = None,
                 is_valid_file: Optional[Callable[[str], bool]] = None) -> None:
        super().__init__(root, transform=transform)
        self._maxTxns = maxTxns
        # env and txn is delay-loaded in ddp. They can't pickle
        self._env = None
        self._txn = None
        self.resolution = resolution
        self.max_words = max_words
        self.max_frames = max_frames
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
        self.datalist = read_json_line(_pretrain_info_path[language])
        # self.datalist = self.datalist[:256]
        self._length = len(self.datalist)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "[CLS]", "SEP_TOKEN": "[SEP]",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),  # 0.08-1
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

    def _get_text(self, caption=None):
        words = self.tokenizer.tokenize(caption)
        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = self.max_words - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
        input_ids = self.tokenizer.convert_tokens_to_ids(words)

        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)
        while len(input_ids) < self.max_words:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        assert len(input_ids) == self.max_words
        assert len(input_mask) == self.max_words
        assert len(segment_ids) == self.max_words

        pairs_text = np.array(input_ids)
        pairs_mask = np.array(input_mask)
        pairs_segment = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment

    def _get_video(self, video_key, max_frames):
        video_list = list()
        global g_lmdb_frames
        # global writer
        # random sample start ##################################################
        video_index = np.arange(0, g_lmdb_frames)
        sample_slice = random.sample(list(video_index), max_frames * 2)
        sample_slice1 = sorted(sample_slice[0:max_frames])
        sample_slice2 = sorted(sample_slice[max_frames:2 * max_frames])
        sample_slice = sample_slice1 + sample_slice2
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
        # print("video data.shape:{}".format(video_data.shape))
        video_data = video_data.copy()
        # video_data = video_data.astype('float64')
        video_data = video_data.reshape([2*max_frames, 3, self.resolution, self.resolution])
        video_data1 = video_data[:max_frames, :, :, :]
        video_data2 = video_data[max_frames:2*max_frames, :, :, :]
        if self.max_frames == -1:
            # dynamic frame needs pad
            if max_frames < max_dynamic_pretrain_frames:
                pad = np.zeros([max_dynamic_pretrain_frames - max_frames, 3, self.resolution, self.resolution],
                               dtype=np.float32)
                video_data1 = np.concatenate((video_data1, pad), axis=0)
                video_data2 = np.concatenate((video_data2, pad), axis=0)

        return video_data1, video_data2

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
        if self.max_frames == -1:
            # dynamic frame
            max_frames = min(max(item["duration"] // 5, 3), max_dynamic_pretrain_frames)
        else:
            max_frames = self.max_frames
        video_key = "Video" + item['docid']
        video_data1, video_data2 = self._get_video(video_key, max_frames)

        tag_text = item['tag']
        title_text = item['title']
        # print("title[{}]:{}".format(index,title_text))
        # print("video[{}]:{}".format(index, item['video_id']))
        tag_ids, tag_mask, _ = self._get_text(tag_text)
        title_ids, title_mask, _ = self._get_text(title_text)
        return video_data1, video_data2, max_frames, tag_ids, tag_mask, title_ids, title_mask

    def __len__(self) -> int:
        return self._length


class dataload_bird_train(VisionDataset):
    def __init__(self, root: str, language:str, maxTxns: int = 1, tokenizer=None,
                 resolution=224, max_words=32, max_frames=24, task="retrieval_VT") -> None:
        super().__init__(root)
        self._maxTxns = maxTxns
        # env and txn is delay-loaded in ddp. They can't pickle
        self._env = None
        self._txn = None
        self.resolution = resolution
        self.max_words = max_words
        self.max_frames = max_frames
        self.task = task
        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
        else:
            self.tokenizer = tokenizer
        querylist = read_json_line(_train_info_path[language])
        self.datalist = get_flat_query_list(querylist)
        # self.datalist = self.datalist[:2048]
        self._length = len(self.datalist)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "[CLS]", "SEP_TOKEN": "[SEP]",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.)),  # 0.08-1
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

    def _get_text(self, caption=None):
        words = self.tokenizer.tokenize(caption)
        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = self.max_words - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
        input_ids = self.tokenizer.convert_tokens_to_ids(words)

        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)
        while len(input_ids) < self.max_words:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        assert len(input_ids) == self.max_words
        assert len(input_mask) == self.max_words
        assert len(segment_ids) == self.max_words

        pairs_text = np.array(input_ids)
        pairs_mask = np.array(input_mask)
        pairs_segment = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment

    def _get_video(self, video_key, max_frames):
        video_list = list()
        global g_lmdb_frames
        # global writer
        # random sample start ##################################################
        # assert g_lmdb_frames % self.max_frames == 0
        # video_index = np.arange(0, g_lmdb_frames)
        # # print("video_index:{}".format(video_index))
        # sample_slice = list()
        # k = g_lmdb_frames // self.max_frames
        # for i in np.arange(self.max_frames):
        #     index = random.choice(video_index[k * i:k * (i + 1)])
        #     sample_slice.append(index)
        # sample
        video_index = np.arange(0, g_lmdb_frames)
        sample_slice = random.sample(list(video_index), max_frames)
        sample_slice = sorted(sample_slice)
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
        video_data = video_data.reshape([max_frames, 3, self.resolution, self.resolution])
        if self.max_frames == -1:
            # dynamic frame needs pad
            if max_frames < max_dynamic_train_frames:
                pad = np.zeros([max_dynamic_train_frames - max_frames, 3, self.resolution, self.resolution],
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
        if self.max_frames == -1:
            # dynamic frame
            max_frames = min(max(int(item["duration"] * 0.3), 3), max_dynamic_train_frames)
        else:
            max_frames = self.max_frames
        # query, pos_item = self._get_pos_pair(item)
        query = item['query']
        videoid = "Video" + item['docid']
        video_data = self._get_video(videoid, max_frames)
        query_ids, query_mask, _ = self._get_text(query)

        if self.task == "retrieval_VT":
            title = item['title']
            title_ids, title_mask, _ = self._get_text(title)
            return query_ids, query_mask, video_data, max_frames, title_ids, title_mask, index
        else:
            return query_ids, query_mask, video_data, max_frames, index

    def __len__(self) -> int:
        return self._length


class dataload_bird_val(VisionDataset):
    def __init__(self, root: str, language:str, maxTxns: int = 1, tokenizer=None,
                 resolution=224, max_words=32, max_frames=24, task="retrieval_VT") -> None:
        super().__init__(root)
        self._maxTxns = maxTxns
        # env and txn is delay-loaded in ddp. They can't pickle
        self._env = None
        self._txn = None
        self.resolution = resolution
        self.max_words = max_words
        self.max_frames = max_frames
        self.task = task
        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
        else:
            self.tokenizer = tokenizer

        self.datalist = read_json_line(_val_info_path[language])
        # self.datalist = self.datalist[:256]
        self._length = len(self.datalist)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "[CLS]", "SEP_TOKEN": "[SEP]",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        self.transform = transforms.Compose([
            transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BICUBIC),
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
        query = item['query']
        poslist = item['videolist']
        pos_item = poslist[0]
        return query, pos_item

    def _get_text(self, caption=None):
        words = self.tokenizer.tokenize(caption)
        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = self.max_words - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
        input_ids = self.tokenizer.convert_tokens_to_ids(words)

        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)
        while len(input_ids) < self.max_words:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        assert len(input_ids) == self.max_words
        assert len(input_mask) == self.max_words
        assert len(segment_ids) == self.max_words

        pairs_text = np.array(input_ids)
        pairs_mask = np.array(input_mask)
        pairs_segment = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment

    def _get_video(self, video_key, max_frames):
        video_list = list()
        global g_lmdb_frames
        #uniform sample start ##################################################
        # assert g_lmdb_frames % self.max_frames == 0
        # video_index = np.arange(0, g_lmdb_frames, g_lmdb_frames // self.max_frames)
        video_index = np.linspace(0, g_lmdb_frames, max_frames, endpoint=False, dtype=int)
        #uniform sample end ##################################################
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
        video_data = video_data.reshape([max_frames, 3, self.resolution, self.resolution])
        if self.max_frames == -1:
            # dynamic frame needs pad
            if max_frames < max_dynamic_val_frames:
                pad = np.zeros([max_dynamic_val_frames - max_frames, 3, self.resolution, self.resolution],
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
        if self.max_frames == -1:
            # dynamic frame
            max_frames = min(max(pos_item["duration"] // 2, 3), max_dynamic_val_frames)
        else:
            max_frames = self.max_frames
        video_data = self._get_video(videoid, max_frames)
        # query = "关于 " + query + " 的视频"
        # print("[{}]query:{},title:{},video:{}".format(index, query, pos_title, videoid))
        # print("video[{}]:{}".format(index, item['video_id']))
        query_ids, query_mask, _ = self._get_text(query)

        if self.task == "retrieval_VT":
            title = pos_item['title']
            title_ids, title_mask, _ = self._get_text(title)
            return query_ids, query_mask, video_data, max_frames, title_ids, title_mask
        else:
            return query_ids, query_mask, video_data, max_frames

    def __len__(self) -> int:
        return self._length
