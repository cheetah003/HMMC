import json
import os
import shutil
import cv2
import tqdm
import numpy as np
import lmdb
import torch as th
import multiprocessing as mp
from PIL import Image
import time
import sys
import random
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

total_time = 0


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def video_to_tensor(video_file, num_frames=24):
    # time1 = time.time()
    # Samples a frame sample_fp X frames.
    # print("video_file:{}".format(video_file))
    if not os.path.exists(video_file):
        print("path not exist:{}".format(video_file))
        return None
    cap = cv2.VideoCapture(video_file)
    videoname = os.path.basename(video_file)
    videoname = videoname.split('.')[0]
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if 0 == fps or 0 == frameCount:
        return None
    total_duration = (frameCount + fps - 1) // fps
    # logger.info("frameCount:{},fps:{},total_duration:{}".format(frameCount, fps, total_duration))
    # start_sec, end_sec = 0, total_duration

    ret = True
    images = []
    # print("fps:{},framecount:{}".format(fps,frameCount))
    # print("gap:{}".format(gap))
    inds = np.linspace(0, frameCount - 10, num=num_frames, dtype=int)
    # inds = inds * fps
    retry_time = 3
    # print("inds:{}".format(inds))
    for k, ind in enumerate(inds):
        cap.set(cv2.CAP_PROP_POS_FRAMES, ind)
        ret, frame = cap.read()
        for i in range(retry_time):
            if not ret:
                increment = random.randint(-fps, fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, ind + increment)
                ret, frame = cap.read()
        if not ret:
            print("error ind:{}".format(ind))
            break

        # print("frame.size:{},shape:{}".format(sys.getsizeof(frame),frame.shape))
        # filename = "picture/" + videoname + "_%d_100.jpg" % k
        # print("filename:{}".format(filename))
        # cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        # encode_frame:array
        # print("video_name:{}".format(videoname))
        # encode_frame_95 = cv2.imencode('.jpg', frame, params=[cv2.IMWRITE_JPEG_QUALITY, 95])[1]
        # print("encode_frame_95.shape:{},type:{},dtype:{}".format(encode_frame_95.shape,type(encode_frame_95),encode_frame_95.dtype))

        # decode_frame_95 = cv2.imdecode(encode_frame_95, cv2.IMREAD_COLOR)
        # print("decode_frame_95.shape:{},type:{}".format(decode_frame_95.shape, type(decode_frame_95)))
        # print("{},decode_frame={}".format(k, decode_frame_95))
        # images.append(encode_frame_95)
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # print("frame.size:{},shape:{}".format(sys.getsizeof(frame), frame.shape))
        # print("frame_rgb.size:{},shape:{}".format(sys.getsizeof(frame_rgb), frame_rgb.shape))
        # preprocess = _transform(resolution)
        # images.append(preprocess(Image.fromarray(frame_rgb).convert("RGB")))
        # frame_img1 = Image.fromarray(frame_rgb).convert("RGB")
        # print("frame_img1:{},shape:{}".format(frame_img1, np.array(frame_img1).shape))
        # frame_img2 = frame_img1.convert("RGB")
        # print("frame_img2:{},shape:{}".format(frame_img2, np.array(frame_img2).shape))
        frame_jpg = cv2.imencode('.jpg', frame)[1]
        # print("frame_jpg.size:{},shape:{}".format(sys.getsizeof(frame_jpg), frame_jpg.shape))
        images.append(frame_jpg)
    cap.release()

    if len(images) != num_frames:
        return None
    # video_data = np.stack(images)
    video_data = images
    # tensor_size = video_data.shape
    # video_data = video_data.reshape(-1, 1, tensor_size[-3], tensor_size[-2], tensor_size[-1])
    # video = np.zeros((1, num_frames, 1, 3,
    #                   resolution, resolution), dtype=np.float16)
    # video[0][:, ...] = video_data
    # print("video.shape:{},type:{}".format(video_data.shape, type(video_data[0][0][0][0])))
    # print("video.size:{}".format(sys.getsizeof(video_data)))
    # video = video.reshape(1, -1)
    return video_data


# def write(txn, i, item, video_dir):
#     max_frames = 24
#     filename = item['video_id'] + '.mp4'
#     video_path = os.path.join(video_dir, filename)
#     video_data_list = video_to_tensor(video_path, num_frames=max_frames)
#     # print("video_data.shape:{}, size:{}".format(video_data_list[0].shape, sys.getsizeof(video_data_list)))
#     if video_data_list is None:
#         return 0
#     for i,video_data in enumerate(video_data_list):
#         video_key = item['video_id'] + '_%d' % i
#         # print("video_key={}".format(video_key))
#         video_key = video_key.encode()
#         txn.put(video_key, video_data)


def get_video_data_list(line):
    item = json.loads(line)
    filename = item['video_id'] + '.mp4'
    video_path = os.path.join(video_dir, filename)
    video_data_list = video_to_tensor(video_path, num_frames=max_frames)
    if video_data_list is None:
        return item['video_id'], None
    else:
        return item['video_id'], video_data_list


def main(targetDir, source_json_path, n_size, num_workers):
    if os.path.exists(targetDir):
        print("path exists:{}".format(targetDir))
        return
    shutil.rmtree(targetDir, ignore_errors=True)
    os.makedirs(targetDir, exist_ok=True)
    env = lmdb.Environment(targetDir, subdir=True, map_size=1024 * 1024 * 1024 * n_size)
    print("n_size:{}G".format(n_size))
    error_video = []
    txn = env.begin(write=True)
    with open(source_json_path, 'r', encoding='utf8') as jsonfile:
        lines = jsonfile.readlines()

        # 测试用
        # lines = lines[0:1]

        if num_workers > 1:
            with mp.Pool(num_workers) as pool, tqdm.tqdm(total=len(lines)) as pbar:
                for idx, (video_id, video_data_list) in enumerate(
                        pool.imap_unordered(
                            get_video_data_list, lines, chunksize=128)):
                    if video_data_list is None:
                        error_video.append(video_id)
                        continue
                    # print("video_key:{}".format(video_id))
                    for i, frame_data in enumerate(video_data_list):
                        video_key = video_id + "_%d" % i
                        video_key = video_key.encode()
                        txn.put(video_key, frame_data)
                    if idx % 1000 == 0:
                        txn.commit()
                        txn = env.begin(write=True)
                    pbar.update(1)
        else:
            for idx, line in tqdm.tqdm(enumerate(lines), total=len(lines)):
                video_id, video_data_list = get_video_data_list(line)
                if video_data_list is None:
                    error_video.append(video_id)
                    continue
                # print("video_key:{}".format(video_id))
                for i, frame_data in enumerate(video_data_list):
                    video_key = video_id + "_%d" % i
                    video_key = video_key.encode()
                    txn.put(video_key, frame_data)
                if idx % 1000 == 0:
                    txn.commit()
                    txn = env.begin(write=True)
        print("error_video:{}".format(error_video))
    txn.commit()
    env.close()

    with open(source_json_path, "w", encoding='utf8') as fp:
        for line in lines:
            item = json.loads(line)
            if item['video_id'] in error_video:
                continue
            fp.write(line)

    # Create metadata needed for dataset
    with open(os.path.join(targetDir, "metadata.json"), "w") as fp:
        json.dump({"length": len(lines) - len(error_video)}, fp)


video_dir = "cfm_msrvtt_videos"  # video directory
max_frames = 30  # frame nums of a video

if __name__ == "__main__":
    main(targetDir="msrvtt_lmdb", source_json_path="msrvtt.json", n_size=50, num_workers=20)
    # main(targetDir="pretrain_frame_lmdb", source_json_path="videoinfo_for_lmdb.json", n_size=1000, num_workers=20)
    # main(targetDir="videoinfo_lmdb", source_json_path="videolmdb.json", n_size=1850, num_workers=40)
