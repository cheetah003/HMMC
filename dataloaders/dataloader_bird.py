from typing import Callable, Any, Optional, Tuple, Callable, List, Dict, cast
import os
import json
import sys
import numpy as np

import lmdb
import torch
import random
from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

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

    def _mask_tokens(self, words):
        token_labels = []
        masked_tokens = words.copy()

        for token_id, token in enumerate(masked_tokens):
            if token_id == 0 or token_id == len(masked_tokens) - 1:
                token_labels.append(-1)
                continue
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                if prob < 0.8:
                    masked_tokens[token_id] = "[MASK]"
                elif prob < 0.9:
                    masked_tokens[token_id] = random.choice(list(self.tokenizer.vocab.items()))[0]
                try:
                    token_labels.append(self.tokenizer.vocab[token])
                except KeyError:
                    token_labels.append(self.tokenizer.vocab["[UNK]"])
            else:
                token_labels.append(-1)

        return masked_tokens, token_labels

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
        video_key = video_key.encode()
        video = self._txn.get(video_key)
        video_data = np.frombuffer(video)
        # video.shape: (1, 12, 1, 3, 224, 224)
        video_data.dtype = 'float16'

        # print("data:{}".format(video_data))
        # print("caption:{}".format(caption))
        video_data = video_data.copy()
        video_data = video_data.astype('float64')
        video_data = video_data.reshape([-1, self.max_frames, 1, 3, self.resolution, self.resolution])
        #random sample start ##################################################
        # video_index = np.arange(0, 24, 2)
        # slice = list()
        # k = 24 // self.max_frames
        # for i in np.arange(self.max_frames):
        #     index = random.choice(video_index[k * i:k * (i+1)])
        #     slice.append(index)
        # print("video_data.shpae:{}".format(video_data.shape))
        # video_data = video_data[:, video_index, :, :, :, :]
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
        video_key = item['video_id']
        video_data = self._get_video(video_key)
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
        if self.stage == "stage1":
            asr_text = item['asr']
            asr_ids, asr_mask, _, = self._get_text(asr_text)
            return video_data, tag_ids, tag_mask, title_ids, title_mask, asr_ids, asr_mask
        else:
            return video_data, tag_ids, tag_mask, title_ids, title_mask

    def __len__(self) -> int:
        return self._length


class dataload_bird_train(VisionDataset):
    def __init__(self, root: str, jsonpath_asr:str, jsonpath_query:str, maxTxns: int = 1, tokenizer=None,
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
        self.datadict = read_json(jsonpath_asr)
        self.datalist = read_json_line(jsonpath_query)

        self._length = len(self.datalist)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "[CLS]", "SEP_TOKEN": "[SEP]",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

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

    def _mask_tokens(self, words):
        token_labels = []
        masked_tokens = words.copy()

        for token_id, token in enumerate(masked_tokens):
            if token_id == 0 or token_id == len(masked_tokens) - 1:
                token_labels.append(-1)
                continue
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                if prob < 0.8:
                    masked_tokens[token_id] = "[MASK]"
                elif prob < 0.9:
                    masked_tokens[token_id] = random.choice(list(self.tokenizer.vocab.items()))[0]
                try:
                    token_labels.append(self.tokenizer.vocab[token])
                except KeyError:
                    token_labels.append(self.tokenizer.vocab["[UNK]"])
            else:
                token_labels.append(-1)

        return masked_tokens, token_labels

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
        video_key = video_key.encode()
        video = self._txn.get(video_key)
        video_data = np.frombuffer(video)
        # video.shape: (1, 12, 1, 3, 224, 224)
        video_data.dtype = 'float16'

        # print("data:{}".format(video_data))
        # print("caption:{}".format(caption))
        video_data = video_data.copy()
        video_data = video_data.astype('float64')
        video_data = video_data.reshape([-1, self.max_frames, 1, 3, self.resolution, self.resolution])
        #random sample start ##################################################
        # video_index = np.arange(0, 24, 2)
        # slice = list()
        # k = 24 // self.max_frames
        # for i in np.arange(self.max_frames):
        #     index = random.choice(video_index[k * i:k * (i+1)])
        #     slice.append(index)
        # print("video_data.shpae:{}".format(video_data.shape))
        # video_data = video_data[:, video_index, :, :, :, :]
        # print("video_data2.shpae:{}".format(video_data.shape))
        #random sample end ##################################################
        # print("video:{},shape:{},type:{},dtype:{}".format(sys.getsizeof(video_data), video_data.shape, type(video_data),
        #                                                   video_data.dtype))
        return video_data

    def _get_pos_hardneg_pair(self, item=None):
        query = item['query']
        poslist = item['video_id']
        hardneglist = item['hardneg_video_id']
        pos_id = random.choice(poslist)
        hardneg_id = random.choice(hardneglist)
        return query, pos_id, hardneg_id

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
        query, pos_videoid, hard_videoid = self._get_pos_hardneg_pair(item)
        pos_video_data = self._get_video(pos_videoid)
        hard_video_data = self._get_video(hard_videoid)
        pos_item = self.datadict[pos_videoid]
        hard_item = self.datadict[hard_videoid]
        pos_title = pos_item['title']
        hard_title = hard_item['title']
        # print("title[{}]:{}".format(index,title_text))
        # print("video[{}]:{}".format(index, item['video_id']))
        query_ids, query_mask, _ = self._get_text(query)
        pos_title_ids, pos_title_mask, _ = self._get_text(pos_title)
        hard_title_ids, hard_title_mask, _ = self._get_text(hard_title)
        return query_ids, query_mask, pos_video_data, hard_video_data, \
               pos_title_ids, pos_title_mask, hard_title_ids, hard_title_mask

    def __len__(self) -> int:
        return self._length


class dataload_bird_val(VisionDataset):
    def __init__(self, root: str, jsonpath_asr:str, jsonpath_query:str, maxTxns: int = 1, tokenizer=None,
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
        self.datadict = read_json(jsonpath_asr)
        self.datalist = read_json_line(jsonpath_query)
        self._length = len(self.datalist)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "[CLS]", "SEP_TOKEN": "[SEP]",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}


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

    def _mask_tokens(self, words):
        token_labels = []
        masked_tokens = words.copy()

        for token_id, token in enumerate(masked_tokens):
            if token_id == 0 or token_id == len(masked_tokens) - 1:
                token_labels.append(-1)
                continue
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                if prob < 0.8:
                    masked_tokens[token_id] = "[MASK]"
                elif prob < 0.9:
                    masked_tokens[token_id] = random.choice(list(self.tokenizer.vocab.items()))[0]
                try:
                    token_labels.append(self.tokenizer.vocab[token])
                except KeyError:
                    token_labels.append(self.tokenizer.vocab["[UNK]"])
            else:
                token_labels.append(-1)

        return masked_tokens, token_labels

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
        video_key = video_key.encode()
        video = self._txn.get(video_key)
        video_data = np.frombuffer(video)
        # video.shape: (1, 24, 1, 3, 224, 224)
        video_data.dtype = 'float16'

        video_data = video_data.copy()
        video_data = video_data.astype('float64')
        video_data = video_data.reshape([-1, self.max_frames, 1, 3, self.resolution, self.resolution])

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
        query = item["query"]
        pos_videoid = item["video_id"]
        pos_video_data = self._get_video(pos_videoid)
        pos_item = self.datadict[pos_videoid]
        pos_title = pos_item['title']
        # print("title[{}]:{}".format(index,title_text))
        # print("video[{}]:{}".format(index, item['video_id']))
        query_ids, query_mask, _ = self._get_text(query)
        pos_title_ids, pos_title_mask, _ = self._get_text(pos_title)
        return query_ids, query_mask, pos_video_data, pos_title_ids, pos_title_mask

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
