from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import os

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
from tensorboardX import SummaryWriter
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.modeling import BirdModel_VT, BirdPreTrainedModel, BirdModel
from modules.optimization import BertAdam
from modules.until_module import get_dual_matrix
from dataloaders.dataloader import DATALOADER_DICT
from util import parallel_apply, get_logger

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

torch.distributed.init_process_group(backend="nccl")

global logger


def get_args(description='CLIP4Clip on Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_params", action='store_true', help="text the params of the model.")
    parser.add_argument("--use_frame_fea", action='store_true', help="whether use frame feature matching text")
    parser.add_argument('--task', type=str, default="retrieval", choices=["retrieval_VT", "retrieval"],
                        help="choose downstream task.")
    parser.add_argument('--dataset', type=str, default="bird", choices=["bird", "msrvtt"],
                        help="choose dataset.")
    parser.add_argument('--num_thread_reader', type=int, default=1, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--text_lr', type=float, default=0.00001, help='text encoder learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=3500, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=32, help='')
    parser.add_argument('--max_frames', type=int, default=12, help='')
    parser.add_argument('--frame_sample', type=str, default="uniform", choices=["uniform", "random", "uniform_random"],
                        help='frame sample strategy')
    parser.add_argument('--frame_sample_len', type=str, default="fix", choices=["dynamic", "fix"],
                        help='use dynamic frame length of fix frame length')
    parser.add_argument('--contrast_num_negative', type=int, default=4096, help='Num of negative sample in queue')
    parser.add_argument('--contrast_momentum', type=float, default=0.99, help='momentum')
    parser.add_argument('--contrast_temperature', type=float, default=0.07, help='temperature')
    parser.add_argument('--language', type=str, default="chinese", choices=["chinese", "english"],
                        help='language for text encoder')
    parser.add_argument('--pretrain_path', type=str, default="/ai/swxdisk/data/bird/videoinfo_bilingual.json",
                        help='pretrain data path')
    parser.add_argument('--val_path', type=str, default="/ai/swxdisk/data/bird/query_data_val_bilingual.json",
                        help='val data path')

    parser.add_argument("--logdir", default=None, type=str, required=False, help="log dir for tensorboardX writer")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--warmup_proportion", default=0.2, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument("--rank", default=0, type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=1., help='coefficient for bert branch.')

    args = parser.parse_args()

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
        if args.logdir:
            args.writer = SummaryWriter(args.logdir)
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
    model = BirdPreTrainedModel.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict,
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

    decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "visual_encoder.visual." in n]
    decay_chinesebert_param_tp = [(n, p) for n, p in decay_param_tp if "text_encoder." in n]
    decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if
                             ("visual_encoder.visual." not in n) and ("text_encoder." not in n)]

    no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "visual_encoder.visual." in n]
    no_decay_text_param_tp = [(n, p) for n, p in no_decay_param_tp if "text_encoder." in n]
    no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if
                                ("visual_encoder.visual." not in n) and ("text_encoder." not in n)]

    weight_decay = args.weight_decay
    optimizer_grouped_parameters = [
        {'params': [p for n, p in decay_clip_param_tp], 'weight_decay': weight_decay, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in decay_chinesebert_param_tp], 'weight_decay': weight_decay, 'lr': args.text_lr},
        {'params': [p for n, p in decay_noclip_param_tp], 'weight_decay': weight_decay},
        {'params': [p for n, p in no_decay_clip_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in no_decay_text_param_tp], 'weight_decay': 0.0, 'lr': args.text_lr},
        {'params': [p for n, p in no_decay_noclip_param_tp], 'weight_decay': 0.0}
    ]

    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                         t_total=num_train_optimization_steps, weight_decay=weight_decay,
                         max_grad_norm=1.0)
    if args.fp16_opt_level != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=True)
    if args.local_rank == 0:
        for name, parameters in model.named_parameters():
            logger.info("name:{} requires_grad:{} size:{}".format(name, parameters.requires_grad, parameters.size()))
    return optimizer, scheduler, model


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
        model = BirdPreTrainedModel.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict,
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
        if args.local_rank == 0:
            logger.info("[{}]data loader time:{}".format(args.local_rank, load_finish_time - load_start_time))
        global_step += 1
        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        video_data, frames, tag_ids, tag_mask, title_ids, title_mask = batch
        loss = model(video_data, frames, tag_ids, tag_mask, title_ids, title_mask, global_step)

        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        if args.fp16_opt_level != "O0":
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
                loss = scaled_loss
        else:
            loss.backward()
        total_loss += float(loss)
        forward_and_backward_time = time.time()
        if args.local_rank == 0:
            logger.info("[{}]forward_and_backward_time :{}".format(args.local_rank,
                                                                   forward_and_backward_time - load_finish_time))
        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16_opt_level != "O0":
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.0)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule

            optimizer.step()
            optimizer.zero_grad()

            if global_step % log_step == 0 and local_rank == 0:
                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f", epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader),
                            "-".join([str('%.9f' % itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(loss),
                            (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                if args.logdir:
                    # args.writer.add_scalar('loss', loss.item(), global_step=global_step)
                    args.writer.add_scalars('lr', {"lr%d" % i: itm for i, itm in
                                                   enumerate(sorted(list(set(optimizer.get_lr()))))},
                                            global_step=global_step)
                start_time = time.time()
        load_start_time = time.time()
    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step


def _run_on_single_gpu(model, batch_query_output_list, batch_visual_output_list, batch_title_output_list,
                       batch_frame_output_list):
    sim_matrix = []
    sim_matrix_title = []
    sim_matrix_frame = []
    for idx1, query_output in enumerate(batch_query_output_list):
        each_row = []
        title_each_row = []
        frame_each_row = []
        for idx2, (visual_output, title_output, frame_output) in enumerate(zip(batch_visual_output_list,
                                                                               batch_title_output_list,
                                                                               batch_frame_output_list)):
            b1b2_logits = model.loose_similarity(query_output, visual_output)
            title_logits = model.loose_similarity(query_output, title_output)
            frame_logits = model.loose_similarity(query_output, frame_output)
            b1b2_logits = b1b2_logits.cpu().detach().numpy()
            title_logits = title_logits.cpu().detach().numpy()
            frame_logits = frame_logits.cpu().detach().numpy()
            each_row.append(b1b2_logits)
            title_each_row.append(title_logits)
            frame_each_row.append(frame_logits)

        each_row = np.concatenate(tuple(each_row), axis=-1)
        title_each_row = np.concatenate(tuple(title_each_row), axis=-1)
        frame_each_row = np.concatenate(tuple(frame_each_row), axis=-1)
        # sim_matrix.append(preprocessing.scale(each_row, axis=1))
        sim_matrix.append(each_row)
        sim_matrix_title.append(title_each_row)
        sim_matrix_frame.append(frame_each_row)
    # logger.info("sim_matrix:{}".format(sim_matrix))
    return sim_matrix, sim_matrix_title, sim_matrix_frame


def eval_epoch(args, model, test_dataloader, device, n_gpu):
    torch.cuda.empty_cache()
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    model.eval()
    logger.info("args.task:{}".format(args.task))
    with torch.no_grad():
        batch_query_output_list, batch_visual_output_list = [], []
        batch_title_output_list = []
        batch_frame_output_list = []
        # ----------------------------
        # 1. cache the features
        # ----------------------------
        for bid, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            if args.task == "retrieval_VT":
                query_ids, query_mask, video, video_frame, title_ids, title_mask = batch
            elif args.task == "retrieval":
                query_ids, query_mask, video, video_frame = batch
            else:
                raise ValueError("wrong task type:{}".format(args.task))

            logger.info("bid:{}/{}".format(bid, len(test_dataloader)))

            logger.info("eval video.shape:{}".format(video.shape))
            query_output = model.text_encoder(query_ids, query_mask)
            visual_output, frame_output = model.visual_encoder(video, video_frame)
            frame_output = torch.mean(frame_output, dim=1)
            if args.task == "retrieval_VT":
                title_output = model.text_encoder(title_ids, title_mask)
                logger.info("title_output.shape:{}".format(title_output.shape))
            elif args.task == "retrieval":
                title_output = torch.zeros_like(query_output)
            else:
                raise ValueError("wrong task type:{}".format(args.task))

            logger.info("query_output.shape:{}".format(query_output.shape))
            logger.info("visual_output.shape:{}".format(visual_output.shape))
            logger.info("frame_output.shape:{}".format(frame_output.shape))

            batch_query_output_list.append(query_output)
            batch_visual_output_list.append(visual_output)
            batch_title_output_list.append(title_output)
            batch_frame_output_list.append(frame_output)

        # ----------------------------------
        # 2. calculate the similarity
        # ----------------------------------
        logger.info("n_gpu:{}".format(n_gpu))
        # logger.info("model.weight_sum:{}".format(model.weight_sum))
        if n_gpu > 1:
            device_ids = list(range(n_gpu))
            batch_t_output_splits = []
            batch_v_output_splits = []
            batch_title_output_splits = []
            batch_frame_output_splits = []
            bacth_len = len(batch_query_output_list)
            split_len = (bacth_len + n_gpu - 1) // n_gpu
            for dev_id in device_ids:
                s_, e_ = dev_id * split_len, (dev_id + 1) * split_len
                if dev_id == 0:
                    batch_t_output_splits.append(batch_query_output_list[s_:e_])
                    batch_v_output_splits.append(batch_visual_output_list)
                    batch_title_output_splits.append(batch_title_output_list)
                    batch_frame_output_splits.append(batch_frame_output_list)
                else:
                    devc = torch.device('cuda:{}'.format(str(dev_id)))

                    devc_batch_list = [b.to(devc) for b in batch_query_output_list[s_:e_]]
                    batch_t_output_splits.append(devc_batch_list)
                    devc_batch_list = [b.to(devc) for b in batch_visual_output_list]
                    batch_v_output_splits.append(devc_batch_list)
                    devc_batch_list = [b.to(devc) for b in batch_title_output_list]
                    batch_title_output_splits.append(devc_batch_list)
                    devc_batch_list = [b.to(devc) for b in batch_frame_output_list]
                    batch_frame_output_splits.append(devc_batch_list)

            parameters_tuple_list = [(batch_t_output_splits[dev_id], batch_v_output_splits[dev_id],
                                      batch_title_output_splits[dev_id], batch_frame_output_splits[dev_id]) for dev_id
                                     in device_ids]
            parallel_outputs_tuple = parallel_apply(_run_on_single_gpu, model, parameters_tuple_list, device_ids)
            sim_matrix = []
            sim_matrix_title = []
            sim_matrix_frame = []
            for idx in range(len(parallel_outputs_tuple)):
                parallel_outputs, parallel_outputs_title, parallel_outputs_frame = parallel_outputs_tuple[idx]
                sim_matrix += parallel_outputs
                sim_matrix_title += parallel_outputs_title
                sim_matrix_frame += parallel_outputs_frame
            sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
            sim_matrix_title = np.concatenate(tuple(sim_matrix_title), axis=0)
            sim_matrix_frame = np.concatenate(tuple(sim_matrix_frame), axis=0)
        else:
            sim_matrix_tuple = _run_on_single_gpu(model, batch_query_output_list, batch_visual_output_list,
                                                  batch_title_output_list, batch_frame_output_list)
            sim_matrix, sim_matrix_title, sim_matrix_frame = sim_matrix_tuple
            sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
            sim_matrix_title = np.concatenate(tuple(sim_matrix_title), axis=0)
            sim_matrix_frame = np.concatenate(tuple(sim_matrix_frame), axis=0)

        logger.info("sim_matrix:{}".format(sim_matrix))
        if args.use_frame_fea:
            logger.info("sim_matrix_frame:{}".format(sim_matrix_frame))
            sim_matrix = sim_matrix + sim_matrix_frame

        if args.task == "retrieval_VT":
            logger.info("sim_matrix_title:{}".format(sim_matrix_title))
            sim_matrix = sim_matrix + sim_matrix_title

    logger.info("sim matrix size:  {}".format(np.array(sim_matrix).shape))
    # sim_matrix = get_dual_matrix(sim_matrix)
    tv_metrics = compute_metrics(sim_matrix)
    vt_metrics = compute_metrics(sim_matrix.T)
    logger.info('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))

    logger.info("Text-to-Video:")
    logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))
    logger.info("Video-to-Text:")
    logger.info(
        '\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
            format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))

    R1 = tv_metrics['R1']
    return tv_metrics


def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)

    # get text pretrained path
    pretrained_text = "hfl/chinese-roberta-wwm-ext"
    args.pretrained_text = pretrained_text
    if args.language == "chinese":
        tokenizer = BertTokenizer.from_pretrained(pretrained_text)
    else:
        tokenizer = ClipTokenizer()

    model = init_model(args, device, n_gpu, args.local_rank)

    assert args.dataset in DATALOADER_DICT
    test_dataloader, test_length = DATALOADER_DICT[args.dataset]["test"](args, tokenizer)
    if args.local_rank == 0:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))

    if args.do_pretrain:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.dataset]["pretrain"](args, tokenizer)
        # train_dataloader, train_length, train_sampler = dataloader_msrvtt_pretrain(args, tokenizer)
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
                if epoch % 1 == 0:
                    # Uncomment if want to save checkpoint
                    # save_model(epoch, args, model, type_name="")
                    # if epoch == 100:
                    metrics = eval_epoch(args, model, test_dataloader, device, n_gpu)
                    if args.logdir:
                        args.writer.add_scalars('metrics', {'R1': metrics["R1"], 'R5': metrics["R5"],
                                                            'R10': metrics["R10"]}, global_step=epoch)
        if args.local_rank == 0:
            save_model(epoch, args, model, type_name="")
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
            query_ids, query_mask, pos_video_data, pos_title_ids, pos_title_mask, = batch
            flops, params = profile(model, (query_ids, query_mask, pos_video_data, pos_title_ids, pos_title_mask,))
            print('flops: %.2f G, params: %.2f M' % (flops / 1e9, params / 1e6))
            break
    if args.local_rank == 0 and args.logdir:
        args.writer.close()


if __name__ == "__main__":
    main()
