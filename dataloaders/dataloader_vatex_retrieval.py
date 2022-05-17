#coding:utf-8
# @Time : 2021/6/19
# @Author : Han Fang
# @File: dataloader_vatexEnglish_frame.py
# @Version: version 1.0

import os
from torch.utils.data import Dataset
import numpy as np
import json
import random
import cv2
from PIL import Image
from torchvision import transforms
import lmdb
g_lmdb_frames = 24


class VATEX_multi_sentence_dataLoader(Dataset):
    """VATEX with English annotations dataset loader for multi-sentence

    Attributes:
        subset: indicate train or test or val
        data_path: path of data list
        features_path: frame directory
        tokenizer: tokenize the word
        max_words: the max number of word
        feature_framerate: frame rate for sampling video
        max_frames: the max number of frame
        image_resolution: resolution of images
    """

    def __init__(
            self,
            root,
            language,
            subset,
            json_path,
            tokenizer,
            frame_sample,
            max_words=32,
            feature_framerate=1.0,
            max_frames=12,
            image_resolution=224,
    ):
        self._env = None
        self._txn = None
        self.root = root
        self.subset = subset
        self.data_path = json_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.language = language
        self.frame_sample = frame_sample
        self.resolution = image_resolution
        # load the id of split list
        assert self.subset in ["train", "val", "test"]
        video_id_path_dict = {}
        video_id_path_dict["train"] = os.path.join(self.data_path, "train_list.txt")
        # video_id_path_dict["test"] = os.path.join(self.data_path, "test_list.txt")
        video_id_path_dict["test"] = os.path.join(self.data_path, "vatex_test_HGR.txt")

        # construct ids for data loader
        with open(video_id_path_dict[self.subset], 'r') as fp:
            video_ids = [itm.strip() for itm in fp.readlines()]
        print("video_ids:".format(video_ids))

        # load caption
        caption_file = os.path.join(self.data_path, "vatex_data.json")
        captions = json.load(open(caption_file, 'r'))

        # construct pairs
        self.sample_len = 0
        self.sentences_dict = {}
        self.cut_off_points = [] # used to tag the label when calculate the metric
        if self.language == "chinese":
            cap = "chCap"
        else:
            cap = "enCap"
        for video_id in video_ids:
            assert video_id in captions
            for cap_txt in captions[video_id][cap]:
                self.sentences_dict[len(self.sentences_dict)] = (video_id, cap_txt)
            self.cut_off_points.append(len(self.sentences_dict))

        # usd for multi-sentence retrieval
        self.multi_sentence_per_video = True # important tag for eval in multi-sentence retrieval
        if self.subset == "val" or self.subset == "test":
            self.sentence_num = len(self.sentences_dict) # used to cut the sentence representation
            self.video_num = len(video_ids) # used to cut the video representation
            assert len(self.cut_off_points) == self.video_num
            print("For {}, sentence number: {}".format(self.subset, self.sentence_num))
            print("For {}, video number: {}".format(self.subset, self.video_num))

        print("Video number: {}".format(len(video_ids)))
        print("Total Paire: {}".format(len(self.sentences_dict)))

        # length of dataloader for one epoch
        self.sample_len = len(self.sentences_dict)

        # start and end token
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
                              meminit=False, max_spare_txns=1, lock=False)
        self._txn = self._env.begin(write=False, buffers=True)

    def __len__(self):
        """length of data loader

        Returns:
            length: length of data loader
        """
        length = self.sample_len
        return length

    def _get_text(self, caption):
        """get tokenized word feature

        Args:
            caption: caption

        Returns:
            pairs_text: tokenized text
            pairs_mask: mask of tokenized text
            pairs_segment: type of tokenized text

        """
        # tokenize word
        words = self.tokenizer.tokenize(caption)

        # add cls token
        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = self.max_words - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]

        # add end token
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

        # convert token to id according to the vocab
        input_ids = self.tokenizer.convert_tokens_to_ids(words)

        # add zeros for feature of the same length
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)
        while len(input_ids) < self.max_words:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        # ensure the length of feature to be equal with max words
        assert len(input_ids) == self.max_words
        assert len(input_mask) == self.max_words
        assert len(segment_ids) == self.max_words
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
        video_data = video_data.reshape([self.max_frames, 3, self.resolution, self.resolution])

        return video_data

    def __getitem__(self, idx):
        """forward method
        Args:
            idx: id
        Returns:
            pairs_text: tokenized text
            pairs_mask: mask of tokenized text
            pairs_segment: type of tokenized text
            video: sampled frames
            video_mask: mask of sampled frames
        """
        if self._env is None:
            self._initEnv()

        video_id, caption = self.sentences_dict[idx]

        # obtain text data
        pairs_text, pairs_mask, pairs_segment = self._get_text(caption)

        #obtain video data
        video = self._get_video(video_id, self.max_frames)
        if self.subset == "train":
            return pairs_text, pairs_mask, video, self.max_frames, idx
        else:
            return pairs_text, pairs_mask, video, self.max_frames
