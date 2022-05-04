from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import lmdb
import os
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
import numpy as np
import pandas as pd
from collections import defaultdict
import json
from PIL import Image
import cv2
import random
from dataloaders.rawvideo_util import RawVideoExtractor
from torchvision import transforms
from dataloaders.randaugment import RandomAugment

# global, number of frames in lmdb per video
g_lmdb_frames = 48


class MSRVTT_DataLoader(VisionDataset):
    """MSRVTT dataset loader."""

    def __init__(
            self,
            tokenizer,
            csv_path="/ai/swxdisk/data/msrvtt/MSRVTT_JSFUSION_test.csv",
            features_path=None,
            max_words=32,
            feature_framerate=1.0,
            max_frames=12,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
    ):
        self._maxTxns = 1
        # env and txn is delay-loaded in ddp. They can't pickle
        self._env = None
        self._txn = None
        self.root = "/ai/swxdisk/data/msrvtt/msrvtt_lmdb"
        self.data = pd.read_csv(csv_path)
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.resolution = 224
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(self.resolution, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._txn is not None:
            self._txn.__exit__(exc_type, exc_val, exc_tb)
        if self._env is not None:
            self._env.close()

    def _initEnv(self):
        self._env = lmdb.open(self.root, map_size=1024 * 1024 * 1024 * 5, subdir=True, readonly=True, readahead=False,
                              meminit=False, max_spare_txns=self._maxTxns, lock=False)
        self._txn = self._env.begin(write=False, buffers=True)

    def __len__(self):
        return len(self.data)

    def _get_text(self, video_id, sentence):
        choice_video_ids = [video_id]
        words = self.tokenizer.tokenize(sentence)

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

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_rawvideo(self, choice_video_ids):
        video_list = list()
        global g_lmdb_frames
        video_id = choice_video_ids[0]
        # uniform sample start ##################################################
        # assert g_lmdb_frames % self.max_frames == 0
        # video_index = np.arange(0, g_lmdb_frames, g_lmdb_frames // self.max_frames)
        video_index = np.linspace(0, g_lmdb_frames, self.max_frames, endpoint=False, dtype=int)
        # uniform sample end ##################################################
        for i in video_index:
            video_key_new = video_id + "_%d" % i
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
        video_data = video_data.reshape([self.max_frames, 3, self.resolution, self.resolution])

        return video_data

    def __getitem__(self, idx):
        if self._env is None:
            self._initEnv()
        video_id = self.data['video_id'].values[idx]
        sentence = self.data['sentence'].values[idx]

        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, sentence)
        video = self._get_rawvideo(choice_video_ids)
        return pairs_text, pairs_mask, video, self.max_frames


class MSRVTT_TrainDataLoader(VisionDataset):
    """MSRVTT train dataset loader."""

    def __init__(
            self,
            tokenizer,
            csv_path="/ai/swxdisk/data/msrvtt/MSRVTT_train.9k.csv",
            json_path="/ai/swxdisk/data/msrvtt/MSRVTT_data.json",
            root="/ai/swxdisk/data/msrvtt/msrvtt_lmdb",
            max_words=32,
            max_frames=12,
            unfold_sentences=True,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
    ):
        self._maxTxns = 1
        # env and txn is delay-loaded in ddp. They can't pickle
        self._env = None
        self._txn = None
        self.root = root
        self.csv = pd.read_csv(csv_path)
        self.data = json.load(open(json_path, 'r'))
        self.max_words = max_words
        self.max_frames = max_frames
        self.resolution = image_resolution
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.unfold_sentences = unfold_sentences
        self.sample_len = 0
        if self.unfold_sentences:
            train_video_ids = list(self.csv['video_id'].values)
            self.sentences_dict = {}
            for itm in self.data['sentences']:
                if itm['video_id'] in train_video_ids:
                    self.sentences_dict[len(self.sentences_dict)] = (itm['video_id'], itm['caption'])
            self.sample_len = len(self.sentences_dict)
        else:
            num_sentences = 0
            self.sentences = defaultdict(list)
            s_video_id_set = set()
            for itm in self.data['sentences']:
                self.sentences[itm['video_id']].append(itm['caption'])
                num_sentences += 1
                s_video_id_set.add(itm['video_id'])

            # Use to find the clips in the same video
            self.parent_ids = {}
            self.children_video_ids = defaultdict(list)
            for itm in self.data['videos']:
                vid = itm["video_id"]
                url_posfix = itm["url"].split("?v=")[-1]
                self.parent_ids[vid] = url_posfix
                self.children_video_ids[url_posfix].append(vid)
            self.sample_len = len(self.csv)

        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(self.resolution, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._txn is not None:
            self._txn.__exit__(exc_type, exc_val, exc_tb)
        if self._env is not None:
            self._env.close()

    def _initEnv(self):
        self._env = lmdb.open(self.root, map_size=1024 * 1024 * 1024 * 5, subdir=True, readonly=True, readahead=False,
                              meminit=False, max_spare_txns=self._maxTxns, lock=False)
        self._txn = self._env.begin(write=False, buffers=True)

    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id, sentence):
        choice_video_ids = [video_id]
        words = self.tokenizer.tokenize(sentence)

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

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_single_text(self, video_id):
        rind = random.randint(0, len(self.sentences[video_id]) - 1)
        caption = self.sentences[video_id][rind]
        words = self.tokenizer.tokenize(caption)
        return words

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.ones((len(choice_video_ids), self.max_frames), dtype=np.long)
        video_list = list()
        global g_lmdb_frames
        video_id = choice_video_ids[0]
        # uniform sample start ##################################################
        video_index = list(np.arange(0, g_lmdb_frames))
        sample_slice = random.sample(video_index, self.max_frames)
        sample_slice = sorted(sample_slice)
        # uniform sample end ##################################################
        for i in sample_slice:
            video_key_new = video_id + "_%d" % i
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
        video_data = video_data.reshape([self.max_frames, 3, self.resolution, self.resolution])

        return video_data, video_mask

    def __getitem__(self, idx):
        if self._env is None:
            self._initEnv()
        if self.unfold_sentences:
            video_id, caption = self.sentences_dict[idx]
        else:
            video_id, caption = self.csv['video_id'].values[idx], None
        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, caption)
        video, video_mask = self._get_rawvideo(choice_video_ids)
        return pairs_text, pairs_mask, video, self.max_frames, idx


class MSRVTT_prerainDataLoader(VisionDataset):
    """MSRVTT pretrain dataset loader."""

    def __init__(
            self,
            tokenizer,
            csv_path="/ai/swxdisk/data/msrvtt/MSRVTT_train.9k.csv",
            json_path="/ai/swxdisk/data/msrvtt/MSRVTT_data.json",
            root="/ai/swxdisk/data/msrvtt/msrvtt_lmdb",
            max_words=32,
            max_frames=12,
            unfold_sentences=True,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
    ):
        self._maxTxns = 1
        # env and txn is delay-loaded in ddp. They can't pickle
        self._env = None
        self._txn = None
        self.root = root
        self.csv = pd.read_csv(csv_path)
        self.data = json.load(open(json_path, 'r'))
        self.max_words = max_words
        self.max_frames = max_frames
        self.resolution = image_resolution
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.unfold_sentences = unfold_sentences
        self.sample_len = 0
        if self.unfold_sentences:
            train_video_ids = list(self.csv['video_id'].values)
            self.sentences_dict = {}
            for itm in self.data['sentences']:
                if itm['video_id'] in train_video_ids:
                    self.sentences_dict[len(self.sentences_dict)] = (itm['video_id'], itm['caption'])
            self.sample_len = len(self.sentences_dict)
        else:
            num_sentences = 0
            self.sentences = defaultdict(list)
            s_video_id_set = set()
            for itm in self.data['sentences']:
                self.sentences[itm['video_id']].append(itm['caption'])
                num_sentences += 1
                s_video_id_set.add(itm['video_id'])

            # Use to find the clips in the same video
            self.parent_ids = {}
            self.children_video_ids = defaultdict(list)
            for itm in self.data['videos']:
                vid = itm["video_id"]
                url_posfix = itm["url"].split("?v=")[-1]
                self.parent_ids[vid] = url_posfix
                self.children_video_ids[url_posfix].append(vid)
            self.sample_len = len(self.csv)

        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(self.resolution, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._txn is not None:
            self._txn.__exit__(exc_type, exc_val, exc_tb)
        if self._env is not None:
            self._env.close()

    def _initEnv(self):
        self._env = lmdb.open(self.root, map_size=1024 * 1024 * 1024 * 5, subdir=True, readonly=True, readahead=False,
                              meminit=False, max_spare_txns=self._maxTxns, lock=False)
        self._txn = self._env.begin(write=False, buffers=True)

    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id, sentence):
        choice_video_ids = [video_id]
        words = self.tokenizer.tokenize(sentence)

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

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_single_text(self, video_id):
        rind = random.randint(0, len(self.sentences[video_id]) - 1)
        caption = self.sentences[video_id][rind]
        words = self.tokenizer.tokenize(caption)
        return words

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.ones((len(choice_video_ids), self.max_frames), dtype=np.long)
        video_list = list()
        global g_lmdb_frames
        video_id = choice_video_ids[0]
        # uniform sample start ##################################################
        video_index = list(np.arange(0, g_lmdb_frames))
        sample_slice = random.sample(video_index, self.max_frames)
        sample_slice = sorted(sample_slice)
        # uniform sample end ##################################################
        for i in sample_slice:
            video_key_new = video_id + "_%d" % i
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
        video_data = video_data.reshape([self.max_frames, 3, self.resolution, self.resolution])

        return video_data, video_mask

    def __getitem__(self, idx):
        if self._env is None:
            self._initEnv()
        if self.unfold_sentences:
            video_id, caption = self.sentences_dict[idx]
        else:
            video_id, caption = self.csv['video_id'].values[idx], None
        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, caption)
        video, video_mask = self._get_rawvideo(choice_video_ids)
        return video, self.max_frames, pairs_text, pairs_mask, pairs_text, pairs_mask
