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
import cv2

#global, number of frames in lmdb per video
g_lmdb_frames = 24

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

"""load: video title tag asr"""
class dataload_bird_pretrain(VisionDataset):
    def __init__(self, root: str, jsonpath_asr:str, maxTxns: int = 1, tokenizer=None,stage=None,
                 resolution=224, max_words=32, max_frames=24, transform: Optional[Callable] = None,
                 is_valid_file: Optional[Callable[[str], bool]] = None) -> None:
        super().__init__(root, transform=transform)
        self._maxTxns = maxTxns
        # env and txn is delay-loaded in ddp. They can't pickle
        self._env = None
        self._txn = None
        self.resolution = resolution
        self.max_words = max_words
        self.max_frames = max_frames
        self.stage = stage
        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext-large')
        else:
            self.tokenizer = tokenizer
        # print("self.tokenizer:{}.".format(self.tokenizer.vocab))
        # Length is needed for DistributedSampler, but we can't use env to get it, env can't pickle.
        # So we decide to read from metadata placed in the same folder --- see src/misc/datasetCreate.py
        # with open(os.path.join(root, "metadata.json"), "r") as fp:
        #     metadata = json.load(fp)
        # self._length = metadata["length"]
        datadict = read_json(jsonpath_asr)
        self.datalist = list(datadict.values())
        self._length = len(self.datalist)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "[CLS]", "SEP_TOKEN": "[SEP]",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(self.resolution, scale=(0.2, 1.0),
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
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
        self._env = lmdb.open(self.root, map_size=1024 * 1024 * 1024 * 8, subdir=True, readonly=True, readahead=False,
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

    def _get_video(self, video_key=None):
        video_list = list()
        global g_lmdb_frames
        for i in range(g_lmdb_frames):
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
        # video_data = video_data.astype('float64')
        video_data = video_data.reshape([-1, g_lmdb_frames, 1, 3, self.resolution, self.resolution])
        #random sample start ##################################################
        assert g_lmdb_frames % self.max_frames == 0
        video_index = np.arange(0, g_lmdb_frames)
        # print("video_index:{}".format(video_index))
        slice = list()
        k = g_lmdb_frames // self.max_frames
        for i in np.arange(self.max_frames):
            index = random.choice(video_index[k * i:k * (i + 1)])
            slice.append(index)
        # print("slice:{}".format(slice))
        video_data = video_data[:, slice, :, :, :, :]
        # print("video_data2.shpae:{}".format(video_data.shape))
        #random sample end ##################################################
        # print("video:{},shape:{},type:{},dtype:{}".format(sys.getsizeof(video_data), video_data.shape, type(video_data),
        #                                                   video_data.dtype))
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
        video_key = "Video" + item['docid']
        video_data = self._get_video(video_key)
        video_mask = np.ones(self.max_frames, dtype=np.long)
        ########### not used ocr ###############
        # title_ids = masked_title = masked_title_label = masked_ocr = masked_ocr_label = ocr_ids
        ######################################
        # query_text = item['query']
        tag_text = item['tag']
        title_text = item['title']
        # print("title[{}]:{}".format(index,title_text))
        # print("video[{}]:{}".format(index, item['video_id']))
        tag_ids, tag_mask, tag_segment = self._get_text(tag_text)
        title_ids, title_mask, _ = self._get_text(title_text)
        return video_data, video_mask, tag_ids, tag_mask, title_ids, title_mask

    def __len__(self) -> int:
        return self._length


class dataload_bird_train(VisionDataset):
    def __init__(self, root: str, jsonpath_query:str, maxTxns: int = 1, tokenizer=None,
                 resolution=224, max_words=32, max_frames=24, transform: Optional[Callable] = None,
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
            self.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext-large')
        else:
            self.tokenizer = tokenizer
        # print("self.tokenizer:{}.".format(self.tokenizer.vocab))
        # Length is needed for DistributedSampler, but we can't use env to get it, env can't pickle.
        # So we decide to read from metadata placed in the same folder --- see src/misc/datasetCreate.py
        # with open(os.path.join(root, "metadata.json"), "r") as fp:
        #     metadata = json.load(fp)
        # self._length = metadata["length"]
        # self.datadict = read_json(jsonpath_asr)
        self.datalist = read_json_line(jsonpath_query)

        self._length = len(self.datalist)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "[CLS]", "SEP_TOKEN": "[SEP]",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(self.resolution, scale=(0.5, 1.0),
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
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

    def _get_video(self, video_key=None):
        video_list = list()
        global g_lmdb_frames
        for i in range(g_lmdb_frames):
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
        # video_data = video_data.astype('float32')
        video_data = video_data.reshape([-1, g_lmdb_frames, 1, 3, self.resolution, self.resolution])
        #random sample start ##################################################
        assert g_lmdb_frames % self.max_frames == 0
        video_index = np.arange(0, g_lmdb_frames)
        # print("video_index:{}".format(video_index))
        slice = list()
        k = g_lmdb_frames // self.max_frames
        for i in np.arange(self.max_frames):
            index = random.choice(video_index[k * i:k * (i + 1)])
            slice.append(index)
        # print("slice:{}".format(slice))
        video_data = video_data[:, slice, :, :, :, :]
        # print("video_data2.shpae:{}".format(video_data.shape))
        #random sample end ##################################################
        # print("video:{},shape:{},type:{},dtype:{}".format(sys.getsizeof(video_data), video_data.shape, type(video_data),
        #                                                   video_data.dtype))
        return video_data

    def _get_pos_hardneg_pair(self, item=None):
        query = item['query']
        poslist = item['video_id']
        hardneglist = item['hardneg_video_id']
        pos_item = random.choice(poslist)
        hardneg_item = random.choice(hardneglist)
        return query, pos_item, hardneg_item

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
        query, pos_item, hard_item = self._get_pos_hardneg_pair(item)
        pos_videoid = "Video" + pos_item['docid']
        hard_videoid = "Video" + hard_item['docid']
        pos_video_data = self._get_video(pos_videoid)
        hard_video_data = self._get_video(hard_videoid)
        pos_title = pos_item['title']
        hard_title = hard_item['title']
        # print("title[{}]:{}".format(index,title_text))
        # print("video[{}]:{}".format(index, item['video_id']))
        query_ids, query_mask, _ = self._get_text(query)
        pos_title_ids, pos_title_mask, _ = self._get_text(pos_title)
        hard_title_ids, hard_title_mask, _ = self._get_text(hard_title)
        pos_video_mask = np.ones(self.max_frames, dtype=np.long)
        hard_video_mask = np.ones(self.max_frames, dtype=np.long)
        return query_ids, query_mask, pos_video_data, pos_video_mask, hard_video_data, hard_video_mask, \
               pos_title_ids, pos_title_mask, hard_title_ids, hard_title_mask

    def __len__(self) -> int:
        return self._length


class dataload_bird_val(VisionDataset):
    def __init__(self, root: str, jsonpath_query:str, maxTxns: int = 1, tokenizer=None,
                 resolution=224, max_words=32, max_frames=24, transform: Optional[Callable] = None,
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
            self.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext-large')
        else:
            self.tokenizer = tokenizer
        # print("self.tokenizer:{}.".format(self.tokenizer.vocab))
        # Length is needed for DistributedSampler, but we can't use env to get it, env can't pickle.
        # So we decide to read from metadata placed in the same folder --- see src/misc/datasetCreate.py
        # with open(os.path.join(root, "metadata.json"), "r") as fp:
        #     metadata = json.load(fp)
        # self._length = metadata["length"]
        # self.datadict = read_json(jsonpath_asr)
        self.datalist = read_json_line(jsonpath_query)
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
        self._env = lmdb.open(self.root, map_size=1024 * 1024 * 1024 * 500, subdir=True, readonly=True, readahead=False,
                              meminit=False, max_spare_txns=self._maxTxns, lock=False)
        self._txn = self._env.begin(write=False, buffers=True)

    def _get_pos_pair(self, item=None):
        query = item['query']
        poslist = item['video_id']
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

    def _get_video(self, video_key=None):
        video_list = list()
        global g_lmdb_frames
        for i in range(g_lmdb_frames):
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
        # video_data = video_data.astype('float64')
        video_data = video_data.reshape([-1, g_lmdb_frames, 1, 3, self.resolution, self.resolution])
        #uniform sample start ##################################################
        assert g_lmdb_frames % self.max_frames == 0
        video_index = np.arange(0, g_lmdb_frames, g_lmdb_frames//self.max_frames)
        video_data = video_data[:, video_index, :, :, :, :]
        # print("video_data2.shpae:{}".format(video_data.shape))
        #uniform sample end ##################################################
        # print("video:{},shape:{},type:{},dtype:{}".format(sys.getsizeof(video_data), video_data.shape, type(video_data),
        #                                                   video_data.dtype))
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
        pos_videoid = "Video" + pos_item["docid"]
        pos_video_data = self._get_video(pos_videoid)
        pos_title = pos_item['title']
        # print("title[{}]:{}".format(index,title_text))
        # print("video[{}]:{}".format(index, item['video_id']))
        query_ids, query_mask, _ = self._get_text(query)
        pos_title_ids, pos_title_mask, _ = self._get_text(pos_title)
        pos_video_mask = np.ones(self.max_frames, dtype=np.long)
        return query_ids, query_mask, pos_video_data, pos_video_mask, pos_title_ids, pos_title_mask

    def __len__(self) -> int:
        return self._length


# if __name__ == "__main__":
#     testdataset = BasicLMDB(root='database')
#     dataloader = DataLoader(
#         testdataset,
#         batch_size=1,
#         num_workers=0,
#         shuffle=False,
#         drop_last=False,
#     )
#     for bid, batch in enumerate(dataloader):
#         pairs_text, pairs_mask, pairs_segment, video_data, video_mask = batch
#         print("bid:{},video.shape:{},pairs_text:{}".format(bid, video_data.shape, pairs_text))
#         print("pairs_mask.shape:{},pairs_mask:{}".format(pairs_mask.shape,pairs_mask))
#         print("pairs_segment.shape:{},pairs_segment:{}".format(pairs_segment.shape, pairs_segment))
#         print("video_mask.shape:{},video_mask:{}".format(video_mask.shape, video_mask))
        # print("bid:{},caption:{}".format(bid, caption))
