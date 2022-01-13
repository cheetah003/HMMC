from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torch.utils.data import (SequentialSampler)
import numpy as np
import random
from thop import profile


from metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim
import time
import argparse
from sklearn import preprocessing
from transformers import BertTokenizer, AutoTokenizer, AutoModel
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import CLIP4Clip
from modules.optimization import BertAdam
from modules.until_module import get_dual_matrix
from torch.utils.data import DataLoader
from util import parallel_apply, get_logger
from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_DataLoader
from dataloaders.mydataloader import BasicLMDB
from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_TrainDataLoader
from dataloaders.dataloader_msvd_retrieval import MSVD_DataLoader
from dataloaders.dataloader_lsmdc_retrieval import LSMDC_DataLoader

torch.distributed.init_process_group(backend="nccl")

global logger


def get_args(description='CLIP4Clip on Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_params", action='store_true', help="text the params of the model.")

    parser.add_argument('--train_csv', type=str, default='data/.train.csv', help='')
    parser.add_argument('--val_csv', type=str, default='data/.val.csv', help='')
    parser.add_argument('--data_path', type=str, default='data/caption.pickle', help='data pickle file path')
    parser.add_argument('--features_path', type=str, default='data/videos_feature.pickle', help='feature path')

    parser.add_argument('--num_thread_reader', type=int, default=1, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--chinese_lr', type=float, default=0.00001, help='chinese learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=3500, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=20, help='')
    parser.add_argument('--max_frames', type=int, default=100, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--task_type", default="retrieval", type=str, help="Point the task `retrieval` to finetune.")
    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset to finetune.")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument("--rank", default=0, type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=1., help='coefficient for bert branch.')
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=12, help="Layer NO. of visual.")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=6, help="Layer NO. of cross.")

    parser.add_argument('--loose_type', action='store_true', help="Default using tight type for retrieval.")
    parser.add_argument('--expand_msrvtt_sentences', action='store_true', help="")

    parser.add_argument('--train_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
    parser.add_argument('--eval_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")

    parser.add_argument('--freeze_layer_num', type=int, default=0, help="Layer NO. of CLIP need to freeze.")
    parser.add_argument('--slice_framepos', type=int, default=0, choices=[0, 1, 2],
                        help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.")
    parser.add_argument('--linear_patch', type=str, default="2d", choices=["2d", "3d"],
                        help="linear projection of flattened patches.")
    parser.add_argument('--sim_header', type=str, default="meanP",
                        choices=["meanP", "seqLSTM", "seqTransf", "tightTransf"],
                        help="choice a similarity header.")

    args = parser.parse_args()

    if args.sim_header == "tightTransf":
        args.loose_type = False

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval and not args.do_pretrain and not args.do_params:
        raise ValueError("At least one of `do_train` or `do_eval` or 'do_pretrain' 'do_params' must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args


def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)
    args.world_size = world_size
    rank = torch.distributed.get_rank()
    args.rank = rank

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args


def init_device(args, local_rank):
    global logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)

    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError(
            "Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
                args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu


def init_model(args, device, n_gpu, local_rank):
    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu')
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict,
                                      task_config=args)

    model.to(device)

    return model


def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):
    if hasattr(model, 'module'):
        model = model.module

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." in n]
    decay_chinesebert_param_tp = [(n, p) for n, p in decay_param_tp if "chinese_bert." in n]
    decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if ("clip." not in n) and ("chinese_bert." not in n)]

    no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." in n]
    no_decay_chinesebert_param_tp = [(n, p) for n, p in no_decay_param_tp if "chinese_bert." in n]
    no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if
                                ("clip." not in n) and ("chinese_bert." not in n)]

    weight_decay = 0.2
    optimizer_grouped_parameters = [
        {'params': [p for n, p in decay_clip_param_tp], 'weight_decay': weight_decay, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in decay_chinesebert_param_tp], 'weight_decay': weight_decay, 'lr': args.chinese_lr},
        {'params': [p for n, p in decay_noclip_param_tp], 'weight_decay': weight_decay},
        {'params': [p for n, p in no_decay_clip_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in no_decay_chinesebert_param_tp], 'weight_decay': 0.0, 'lr': args.chinese_lr},
        {'params': [p for n, p in no_decay_noclip_param_tp], 'weight_decay': 0.0}
    ]

    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                         t_total=num_train_optimization_steps, weight_decay=weight_decay,
                         max_grad_norm=1.0)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=True)

    return optimizer, scheduler, model


def dataloader_msrvtt_train(args, tokenizer):
    # msrvtt_dataset = MSRVTT_TrainDataLoader(
    #     csv_path=args.train_csv,
    #     json_path=args.data_path,
    #     features_path=args.features_path,
    #     max_words=args.max_words,
    #     feature_framerate=args.feature_framerate,
    #     tokenizer=tokenizer,
    #     max_frames=args.max_frames,
    #     unfold_sentences=args.expand_msrvtt_sentences,
    #     frame_order=args.train_frame_order,
    #     slice_framepos=args.slice_framepos,
    # )
    # msrvtt_dataset = BasicLMDB(root='/home/shenwenxue/data/dataset/bird/video_lmdb',
    #                            jsonpath='/home/shenwenxue/data/dataset/bird/all_data.json',
    #                            tokenizer=tokenizer, max_words=args.max_words, max_frames=args.max_frames)
    msrvtt_dataset = BasicLMDB(root='/home/shenwenxue/data/dataset/bird/test_array',
                               # jsonpath='/home/shenwenxue/data/dataset/bird/train_data_ocr.json',
                               jsonpath='/home/shenwenxue/data/dataset/bird/test_data.json',
                               tokenizer=tokenizer, max_words=args.max_words, max_frames=args.max_frames)
    train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=True,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msrvtt_dataset), train_sampler


def dataloader_msrvtt_test(args, tokenizer):
    # msrvtt_testset = MSRVTT_DataLoader(
    #     csv_path=args.val_csv,
    #     features_path=args.features_path,
    #     max_words=args.max_words,
    #     feature_framerate=args.feature_framerate,
    #     tokenizer=tokenizer,
    #     max_frames=args.max_frames,
    #     frame_order=args.eval_frame_order,
    #     slice_framepos=args.slice_framepos,
    # )
    # msrvtt_testset = BasicLMDB(root='/home/shenwenxue/data/dataset/bird/video_lmdb',
    #                            jsonpath='/home/shenwenxue/data/dataset/bird/all_data.json',
    #                            tokenizer=tokenizer, max_words=args.max_words, max_frames=args.max_frames)
    msrvtt_testset = BasicLMDB(root='/home/shenwenxue/data/dataset/bird/test_array',
                               jsonpath='/home/shenwenxue/data/dataset/bird/test_data.json',
                               # jsonpath='/home/shenwenxue/data/dataset/bird/val_data_asr.json',
                               tokenizer=tokenizer, max_words=args.max_words, max_frames=args.max_frames)
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvtt_testset)


def dataloader_msvd_train(args, tokenizer):
    msvd_dataset = MSVD_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msvd_dataset)
    dataloader = DataLoader(
        msvd_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msvd_dataset), train_sampler


def dataloader_msvd_test(args, tokenizer, subset="test"):
    msvd_testset = MSVD_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )
    dataloader_msrvtt = DataLoader(
        msvd_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msvd_testset)


def dataloader_lsmdc_train(args, tokenizer):
    lsmdc_dataset = LSMDC_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(lsmdc_dataset)
    dataloader = DataLoader(
        lsmdc_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(lsmdc_dataset), train_sampler


def dataloader_lsmdc_test(args, tokenizer, subset="test"):
    lsmdc_testset = LSMDC_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )
    dataloader_msrvtt = DataLoader(
        lsmdc_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(lsmdc_testset)


def save_model(epoch, args, model, type_name=""):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name == "" else type_name + ".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)
    return output_model_file


def load_model(epoch, args, n_gpu, device, model_file=None):
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        # Prepare model
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                       'distributed')
        model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict,
                                          task_config=args)

        model.to(device)
    else:
        model = None
    return model


def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, local_rank=0):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0
    load_start_time = time.time()
    for step, batch in enumerate(train_dataloader):
        load_finish_time = time.time()
        logger.info("data loader time:{}".format(load_finish_time - load_start_time))
        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
        video_data, video_mask, tag_ids, tag_mask, tag_segment, title_ids, title_mask, asr_ids, asr_mask = batch

        loss = model(video_data, video_mask, tag_ids, tag_mask, tag_segment, title_ids, title_mask, asr_ids, asr_mask)

        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()
        forward_and_backward_time = time.time()
        logger.info("forward_and_backward_time :{}".format(forward_and_backward_time - load_finish_time))
        total_loss += float(loss)
        if (step + 1) % args.gradient_accumulation_steps == 0:

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule

            optimizer.step()
            optimizer.zero_grad()

            # https://github.com/openai/CLIP/issues/46
            if hasattr(model, 'module'):
                torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
            else:
                torch.clamp_(model.clip.logit_scale.data, max=np.log(100))

            global_step += 1
            if global_step % log_step == 0 and local_rank == 0:
                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f", epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader),
                            "-".join([str('%.9f' % itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(loss),
                            (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                start_time = time.time()
        load_start_time = time.time()
    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step


def _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list,
                       batch_title_output_list):
    sim_matrix = []
    sim_matrix_title = []
    # logger.info("batch_sequence_output_list:{}".format(batch_sequence_output_list))
    # logger.info("batch_visual_output_list:{}".format(batch_visual_output_list))
    # logger.info("batch_ocr_output_list:{}".format(batch_ocr_output_list))
    # logger.info("batch_title_output_list:{}".format(batch_title_output_list))
    for idx1, b1 in enumerate(batch_list_t):
        input_mask, segment_ids, *_tmp = b1
        sequence_output = batch_sequence_output_list[idx1]
        each_row = []
        title_each_row = []
        for idx2, b2 in enumerate(batch_list_v):
            video_mask, *_tmp = b2
            visual_output = batch_visual_output_list[idx2]
            title_output = batch_title_output_list[idx2]
            # print("visual shape:{},type:{}".format(visual_output.shape,type(visual_output)))
            # print("ocr_output shape:{},type:{}".format(ocr_output.shape,type(ocr_output)))
            # co_visual_output = co_visual_output.view(co_visual_output.size(0), video_frame, -1, co_visual_output.size(-1))
            # co_visual_output = co_visual_output[:, :, 0, :]
            b1b2_logits, *_tmp = model.get_similarity_logits(sequence_output, visual_output, input_mask, video_mask,
                                                             loose_type=model.loose_type)
            title_logits = model.loose_similarity_for_text(sequence_output, title_output)
            # title_logits = b1b2_logits

            b1b2_logits = b1b2_logits.cpu().detach().numpy()
            title_logits = title_logits.cpu().detach().numpy()

            each_row.append(b1b2_logits)
            title_each_row.append(title_logits)

        each_row = np.concatenate(tuple(each_row), axis=-1)
        title_each_row = np.concatenate(tuple(title_each_row), axis=-1)

        # sim_matrix.append(preprocessing.scale(each_row, axis=1))
        # sim_matrix_ocr.append(preprocessing.scale(ocr_each_row, axis=1))
        # sim_matrix_title.append(preprocessing.scale(title_each_row, axis=1))
        sim_matrix.append(each_row)
        sim_matrix_title.append(title_each_row)
    # logger.info("sim_matrix:{}".format(sim_matrix))
    # logger.info("sim_matrix_ocr:{}".format(sim_matrix_ocr))
    # logger.info("sim_matrix_title:{}".format(sim_matrix_title))
    return sim_matrix, sim_matrix_title


def eval_epoch(args, model, test_dataloader, device, n_gpu):
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    # #################################################################
    ## below variables are used to multi-sentences retrieval
    # multi_sentence_: important tag for eval
    # cut_off_points: used to tag the label when calculate the metric
    # sentence_num: used to cut the sentence representation
    # video_num: used to cut the video representation
    # #################################################################
    multi_sentence_ = False
    cut_off_points_, sentence_num_, video_num_ = [], -1, -1
    if hasattr(test_dataloader.dataset, 'multi_sentence_per_video') \
            and test_dataloader.dataset.multi_sentence_per_video:
        multi_sentence_ = True
        cut_off_points_ = test_dataloader.dataset.cut_off_points
        sentence_num_ = test_dataloader.dataset.sentence_num
        video_num_ = test_dataloader.dataset.video_num
        cut_off_points_ = [itm - 1 for itm in cut_off_points_]

    logger.info("multi_sentence:{}".format(multi_sentence_))
    if multi_sentence_:
        logger.warning("Eval under the multi-sentence per video clip setting.")
        logger.warning("sentence num: {}, video num: {}".format(sentence_num_, video_num_))

    model.eval()
    with torch.no_grad():
        batch_list_t = []
        batch_list_v = []
        batch_sequence_output_list, batch_visual_output_list = [], []
        batch_title_output_list = []
        total_video_num = 0

        # ----------------------------
        # 1. cache the features
        # ----------------------------
        for bid, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            # input_ids, input_mask, segment_ids, video, video_mask = batch
            video, video_mask, tag_ids, tag_mask, tag_segment, title_ids, title_mask, asr_ids, asr_mask = batch

            logger.info("bid:{}/{}".format(bid, len(test_dataloader)))
            logger.info("video.shape:{}".format(video.shape))
            b, _, num_frame, *_t = video.shape
            # logger.info("video.shape:{}".format(video.shape))
            if multi_sentence_:
                # multi-sentences retrieval means: one clip has two or more descriptions.
                sequence_output = model.get_sequence_output(tag_ids, tag_segment, tag_mask)
                batch_sequence_output_list.append(sequence_output)
                batch_list_t.append((tag_mask, tag_segment,))

                s_, e_ = total_video_num, total_video_num + b
                filter_inds = [itm - s_ for itm in cut_off_points_ if itm >= s_ and itm < e_]

                if len(filter_inds) > 0:
                    video, video_mask = video[filter_inds, ...], video_mask[filter_inds, ...]
                    visual_output = model.get_visual_output(video, video_mask)
                    batch_visual_output_list.append(visual_output)
                    batch_list_v.append((video_mask,))
                total_video_num += b
            else:
                sequence_output, visual_output = model.get_sequence_visual_output(tag_ids, tag_segment, tag_mask,
                                                                                  video, video_mask)
                title_output = model.get_sequence_output(title_ids, tag_segment, tag_mask, shaped=True)
                # _, visual_output = model.get_cross_output(video_fea=visual_output, text_fea=title_output)
                title_output, visual_output = model.co_attention_model(title_output, visual_output)
                # co_sequence_output = co_sequence_output[:, 0, :]
                # co_sequence_output = co_sequence_output.view(co_sequence_output.size(0), -1, co_sequence_output.size(-1))
                #
                # co_visual_output = co_visual_output.view(co_visual_output.size(0), num_frame, -1, co_visual_output.size(-1))
                # co_visual_output = co_visual_output[:, :, 0, :]
                # visual_output = co_visual_output

                logger.info("sequence_output.shape:{}".format(sequence_output.shape))
                logger.info("visual_output.shape:{}".format(visual_output.shape))

                # logger.info("sequence_output.shape:{}".format(sequence_output.shape))
                # logger.info("visual_output.shape:{}".format(visual_output.shape))
                batch_sequence_output_list.append(sequence_output)
                batch_title_output_list.append(title_output)

                batch_list_t.append((tag_mask, tag_segment,))

                batch_visual_output_list.append(visual_output)
                batch_list_v.append((video_mask,))

            # logger.info("eval step:{}/{}".format(bid, len(test_dataloader)))
        # logger.info("batch_sequence_output_list.len:{},shape:{}".format(len(batch_sequence_output_list),
        #                                                                 batch_sequence_output_list[0].shape))
        # logger.info("batch_visual_output_list.shape:{},shape:{}".format(len(batch_visual_output_list),
        #                                                                 batch_visual_output_list[0].shape))
        # logger.info("batch_ocr_output_list.shape:{},shape:{}".format(len(batch_ocr_output_list),
        #                                                              batch_ocr_output_list[0].shape))
        # logger.info("batch_title_output_list.shape:{},shape:{}".format(len(batch_title_output_list),
        #                                                                batch_title_output_list[0].shape))
        # ----------------------------------
        # 2. calculate the similarity
        # ----------------------------------
        logger.info("n_gpu:{}".format(n_gpu))
        # logger.info("model.weight_sum:{}".format(model.weight_sum))
        if n_gpu > 1:
            device_ids = list(range(n_gpu))
            batch_list_t_splits = []
            batch_list_v_splits = []

            batch_title_output_splits = []
            batch_t_output_splits = []
            batch_v_output_splits = []
            bacth_len = len(batch_list_t)
            split_len = (bacth_len + n_gpu - 1) // n_gpu
            for dev_id in device_ids:
                s_, e_ = dev_id * split_len, (dev_id + 1) * split_len
                if dev_id == 0:
                    batch_list_t_splits.append(batch_list_t[s_:e_])
                    batch_list_v_splits.append(batch_list_v)

                    batch_t_output_splits.append(batch_sequence_output_list[s_:e_])
                    batch_v_output_splits.append(batch_visual_output_list)
                    batch_title_output_splits.append(batch_title_output_list)
                else:
                    devc = torch.device('cuda:{}'.format(str(dev_id)))
                    devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list_t[s_:e_]]
                    batch_list_t_splits.append(devc_batch_list)
                    devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list_v]
                    batch_list_v_splits.append(devc_batch_list)

                    devc_batch_list = [b.to(devc) for b in batch_sequence_output_list[s_:e_]]
                    batch_t_output_splits.append(devc_batch_list)
                    devc_batch_list = [b.to(devc) for b in batch_visual_output_list]
                    batch_v_output_splits.append(devc_batch_list)
                    devc_batch_list = [b.to(devc) for b in batch_title_output_list]
                    batch_title_output_splits.append(devc_batch_list)

            parameters_tuple_list = [(batch_list_t_splits[dev_id], batch_list_v_splits[dev_id],
                                      batch_t_output_splits[dev_id], batch_v_output_splits[dev_id],
                                      batch_title_output_splits[dev_id]) for dev_id in device_ids]
            parallel_outputs_tuple = parallel_apply(_run_on_single_gpu, model, parameters_tuple_list, device_ids)
            sim_matrix = []
            sim_matrix_title = []
            for idx in range(len(parallel_outputs_tuple)):
                parallel_outputs, parallel_outputs_title = parallel_outputs_tuple[idx]
                sim_matrix += parallel_outputs
                sim_matrix_title += parallel_outputs_title
            sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
            # sim_matrix_title = np.concatenate(tuple(sim_matrix_title), axis=0)
            # sim_matrix_ocr_frame = sim_matrix + sim_matrix_ocr
            # sim_matrix_title_frame = sim_matrix + sim_matrix_title
            # sim_matrix = sim_matrix + sim_matrix_title
            # sim_matrix = model.weight_sum[0].cpu() * sim_matrix + (1 - model.weight_sum[0].cpu()) * sim_matrix_title
        else:
            sim_matrix_tuple = _run_on_single_gpu(model, batch_list_t, batch_list_v,
                                                  batch_sequence_output_list, batch_visual_output_list,
                                                  batch_title_output_list)
            sim_matrix, sim_matrix_title = sim_matrix_tuple
            sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
            # sim_matrix_title = np.concatenate(tuple(sim_matrix_title), axis=0)
            # sim_matrix = sim_matrix + sim_matrix_title
            # sim_matrix = model.weight_sum[0].cpu() * sim_matrix + (1 - model.weight_sum[0].cpu()) * sim_matrix_title
    if multi_sentence_:
        logger.info("before reshape, sim matrix size: {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
        max_length = max([e_ - s_ for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_)])
        sim_matrix_new = []
        for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
            sim_matrix_new.append(np.concatenate((sim_matrix[s_:e_],
                                                  np.full((max_length - e_ + s_, sim_matrix.shape[1]), -np.inf)),
                                                 axis=0))
        sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
        logger.info("after reshape, sim matrix size: {} x {} x {}".
                    format(sim_matrix.shape[0], sim_matrix.shape[1], sim_matrix.shape[2]))

        tv_metrics = tensor_text_to_video_metrics(sim_matrix)
        vt_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix))
    else:
        logger.info("sim matrix size:  {}".format(np.array(sim_matrix).shape))
        # sim_matrix = get_dual_matrix(sim_matrix)
        tv_metrics = compute_metrics(sim_matrix)
        vt_metrics = compute_metrics(sim_matrix.T)
        # tv_metrics_title = compute_metrics(sim_matrix_title)
        # tv_metrics_all = compute_metrics(sim_matrix_all)
        # tv_metrics_title_frame = compute_metrics(sim_matrix_title_frame)
        # tv_metrics_title_ocr = compute_metrics(sim_matrix_title_ocr)
        logger.info('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))

    logger.info("Text-to-Video:")
    logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))
    logger.info("Video-to-Text:")
    logger.info(
        '\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
            format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))
    logger.info("sim_matrix:\n{}".format(sim_matrix))
    # logger.info('\tframe>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
    #             format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))
    # logger.info('\tocr>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
    #             format(tv_metrics_ocr['R1'], tv_metrics_ocr['R5'], tv_metrics_ocr['R10'], tv_metrics_ocr['MR'], tv_metrics_ocr['MeanR']))
    # logger.info('\ttitle>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
    #             format(tv_metrics_title['R1'], tv_metrics_title['R5'], tv_metrics_title['R10'], tv_metrics_title['MR'], tv_metrics_title['MeanR']))
    # logger.info('\tocr_frame>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
    #             format(tv_metrics_ocr_frame['R1'], tv_metrics_ocr_frame['R5'], tv_metrics_ocr_frame['R10'], tv_metrics_ocr_frame['MR'], tv_metrics_ocr_frame['MeanR']))
    # logger.info('\ttitle_frame>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
    #             format(tv_metrics_title_frame['R1'], tv_metrics_title_frame['R5'], tv_metrics_title_frame['R10'], tv_metrics_title_frame['MR'], tv_metrics_title_frame['MeanR']))
    # logger.info('\ttitle_ocr>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
    #             format(tv_metrics_title_ocr['R1'], tv_metrics_title_ocr['R5'], tv_metrics_title_ocr['R10'], tv_metrics_title_ocr['MR'], tv_metrics_title_ocr['MeanR']))
    # logger.info('\tall>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
    #             format(tv_metrics_all['R1'], tv_metrics_all['R5'], tv_metrics_all['R10'], tv_metrics_all['MR'], tv_metrics_all['MeanR']))

    R1 = tv_metrics['R1']
    return R1


DATALOADER_DICT = {}
DATALOADER_DICT["msrvtt"] = {"train": dataloader_msrvtt_train, "val": None, "test": dataloader_msrvtt_test}
DATALOADER_DICT["msvd"] = {"train": dataloader_msvd_train, "val": dataloader_msvd_test, "test": dataloader_msvd_test}
DATALOADER_DICT["lsmdc"] = {"train": dataloader_lsmdc_train, "val": dataloader_lsmdc_test,
                            "test": dataloader_lsmdc_test}


def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)

    if False:
        # 使用原来的tokenizer
        logger.info("ClipTokenizer")
        tokenizer = ClipTokenizer()
    else:
        # 使用albert的tokenizer
        # pretrained = 'voidful/albert_chinese_base'
        pretrained = 'hfl/chinese-roberta-wwm-ext'
        # pretrained = 'hfl/chinese-roberta-wwm-ext-large'
        # pretrained = "nghuyong/ernie-1.0"
        logger.info("tokenizer:{}".format(pretrained))
        tokenizer = BertTokenizer.from_pretrained(pretrained)

    assert args.task_type == "retrieval"
    model = init_model(args, device, n_gpu, args.local_rank)
    ## ####################################
    # freeze testing
    ## ####################################
    assert args.freeze_layer_num <= 12 and args.freeze_layer_num >= -1
    if hasattr(model, "clip") and args.freeze_layer_num > -1:
        for name, param in model.clip.named_parameters():
            # top layers always need to train
            if name.find("ln_final.") == 0 or name.find("text_projection") == 0 or name.find("logit_scale") == 0 \
                    or name.find("visual.ln_post.") == 0 or name.find("visual.proj") == 0:
                continue  # need to train
            elif name.find("visual.transformer.resblocks.") == 0 or name.find("transformer.resblocks.") == 0:
                layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                if layer_num >= args.freeze_layer_num:
                    continue  # need to train

            if args.linear_patch == "3d" and name.find("conv2."):
                continue
            else:
                # paramenters which < freeze_layer_num will be freezed
                param.requires_grad = False

    assert args.datatype in DATALOADER_DICT
    test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)

    if DATALOADER_DICT[args.datatype]["val"] is not None:
        val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, subset="val")
    else:
        val_dataloader, val_length = test_dataloader, test_length

    if args.local_rank == 0:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))
        logger.info("***** Running val *****")
        logger.info("  Num examples = %d", val_length)

    if args.do_train or args.do_pretrain:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs
        # logger.info("train_dataloader len = {}".format(len(train_dataloader)))
        # logger.info("gradient_accumulation_steps = {}".format(args.gradient_accumulation_steps))
        coef_lr = args.coef_lr
        optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu,
                                                     args.local_rank, coef_lr=coef_lr)
        if args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

        best_score = 0.00001
        best_output_model_file = "None"
        global_step = 0
        for epoch in range(args.epochs):
            train_sampler.set_epoch(epoch)
            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                                               scheduler, global_step, local_rank=args.local_rank)
            if args.local_rank == 0:
                logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)
                if epoch % 10 == 0:
                    ## Uncomment if want to save checkpoint
                    # save_model(epoch, args, model, type_name="")
                    if epoch == 100:
                        eval_epoch(args, model, test_dataloader, device, n_gpu)

                    ## Run on val dataset, this process is *TIME-consuming*.
                    # logger.info("Eval on val dataset")
                    # # R1 = eval_epoch(args, model, val_dataloader, device, n_gpu)
                    #
                    # R1 = eval_epoch(args, model, test_dataloader, device, n_gpu)
                    # if best_score <= R1:
                    #     best_score = R1
                    #     best_output_model_file = output_model_file
                    # logger.info("The best model is: {}, the R1 is: {:.4f}".format(best_output_model_file, best_score))

        ## Uncomment if want to test on the best checkpoint
        # if args.local_rank == 0:
        #     model = load_model(-1, args, n_gpu, device, model_file=best_output_model_file)
        #     eval_epoch(args, model, test_dataloader, device, n_gpu)

    elif args.do_eval:
        if args.local_rank == 0:
            eval_epoch(args, model, test_dataloader, device, n_gpu)
    elif args.do_params:
        logger.info("do_params begin!")
        # total = sum([param.nelement() for param in model.parameters()])
        total = sum(p.numel() for p in model.parameters())
        logger.info("Number of parameter: %.2fM" % (total / 1e6))
        for bid, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            video, video_mask, tag_ids, tag_mask, tag_segment, title_ids, title_mask, asr_ids, asr_mask = batch
            flops, params = profile(model, (video, video_mask, tag_ids, tag_mask, tag_segment,title_ids, title_mask, asr_ids, asr_mask,))
            print('flops: %.2f G, params: %.2f M' % (flops / 1e9, params / 1e6))
            break


if __name__ == "__main__":
    main()
