import torch
from torch.utils.data import DataLoader

from dataloaders.dataloader_bird import dataload_bird_pretrain, dataload_bird_train, dataload_bird_val
from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_prerainDataLoader, MSRVTT_TrainDataLoader, MSRVTT_DataLoader
from dataloaders.dataloader_vatex_retrieval import dataload_vatex_train, dataload_vatex_val


def dataloader_bird_pretrain(args, tokenizer):
    bird_dataset = dataload_bird_pretrain(root='/ai/swxdisk/data/bird/videoinfo_lmdb', language=args.language,
                                          json_path="/ai/swxdisk/data/bird/videoinfo_bilingual.json",
                                          tokenizer=tokenizer, max_frames=args.max_frames,
                                          frame_sample=args.frame_sample, frame_sample_len=args.frame_sample_len)
    train_sampler = torch.utils.data.distributed.DistributedSampler(bird_dataset)
    dataloader = DataLoader(
        bird_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=True,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )
    return dataloader, len(bird_dataset), train_sampler


def dataloader_bird_train(args, tokenizer):
    bird_trainset = dataload_bird_train(root='/ai/swxdisk/data/bird/query_lmdb', language=args.language,
                                        json_path="/ai/swxdisk/data/bird/query_data_train_bilingual.json",
                                        tokenizer=tokenizer, max_frames=args.max_frames,
                                        frame_sample=args.frame_sample, frame_sample_len=args.frame_sample_len,
                                        task=args.task)
    train_sampler = torch.utils.data.distributed.DistributedSampler(bird_trainset)
    dataloader = DataLoader(
        bird_trainset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=True,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )
    return dataloader, len(bird_trainset), train_sampler


def dataloader_bird_test(args, tokenizer):
    bird_testset = dataload_bird_val(root='/ai/swxdisk/data/bird/query_lmdb', language=args.language,
                                     json_path="/ai/swxdisk/data/bird/query_data_val_bilingual.json", tokenizer=tokenizer, max_frames=args.max_frames,
                                     frame_sample_len=args.frame_sample_len, task=args.task)
    dataloader = DataLoader(
        bird_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader, len(bird_testset)


def dataloader_msrvtt_pretrain(args, tokenizer):
    msrvtt_trainset = MSRVTT_prerainDataLoader(tokenizer=tokenizer, root="/ai/swxdisk/data/msrvtt/msrvtt_lmdb",
                                               csv_path="/ai/swxdisk/data/msrvtt/MSRVTT_train.9k.csv",
                                               json_path="/ai/swxdisk/data/msrvtt/MSRVTT_data.json",
                                               max_frames=args.max_frames)
    train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_trainset)
    dataloader = DataLoader(
        msrvtt_trainset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=True,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )
    return dataloader, len(msrvtt_trainset), train_sampler


def dataloader_msrvtt_train(args, tokenizer):
    msrvtt_trainset = MSRVTT_TrainDataLoader(tokenizer=tokenizer, root="/ai/swxdisk/data/msrvtt/msrvtt_lmdb",
                                             csv_path="/ai/swxdisk/data/msrvtt/MSRVTT_train.9k.csv",
                                             json_path="/ai/swxdisk/data/msrvtt/MSRVTT_data.json",
                                             frame_sample=args.frame_sample, max_frames=args.max_frames)
    train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_trainset)
    dataloader = DataLoader(
        msrvtt_trainset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=True,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )
    return dataloader, len(msrvtt_trainset), train_sampler


def dataloader_msrvtt_test(args, tokenizer):
    msrvtt_testset = MSRVTT_DataLoader(tokenizer=tokenizer, root="/ai/swxdisk/data/msrvtt/msrvtt_lmdb",
                                       csv_path="/ai/swxdisk/data/msrvtt/MSRVTT_JSFUSION_test.csv",
                                       max_frames=args.max_frames)
    dataloader = DataLoader(
        msrvtt_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader, len(msrvtt_testset)


def dataloader_vatex_train(args, tokenizer):
    vatex_trainset = dataload_vatex_train(root='/ai/swxdisk/data/vatex/vatex_lmdb', language=args.language,
                                          json_path='/ai/swxdisk/data/vatex/vatex_train.json', tokenizer=tokenizer,
                                          max_frames=args.max_frames)
    train_sampler = torch.utils.data.distributed.DistributedSampler(vatex_trainset)
    dataloader = DataLoader(
        vatex_trainset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=True,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )
    return dataloader, len(vatex_trainset), train_sampler


def dataloader_vatex_test(args, tokenizer):
    vatex_testset = dataload_vatex_val(root='/ai/swxdisk/data/vatex/vatex_lmdb', language=args.language,
                                          # json_path='/ai/swxdisk/data/vatex/vatex_val.json',
                                       json_path='/ai/swxdisk/data/vatex/vatex_test_HGR.json',
                                       tokenizer=tokenizer, max_frames=args.max_frames)
    dataloader = DataLoader(
        vatex_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader, len(vatex_testset)


DATALOADER_DICT = {}
DATALOADER_DICT["bird"] = {"pretrain": dataloader_bird_pretrain, "train": dataloader_msrvtt_train,
                           "test": dataloader_msrvtt_test}
DATALOADER_DICT["msrvtt"] = {"pretrain": dataloader_msrvtt_pretrain, "train": dataloader_msrvtt_train,
                             "test": dataloader_msrvtt_test}
DATALOADER_DICT["vatex"] = {"train": dataloader_vatex_train, "test": dataloader_vatex_test}
