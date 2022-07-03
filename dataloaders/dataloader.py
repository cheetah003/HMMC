import torch
from torch.utils.data import DataLoader

from dataloaders.dataloader_bird import dataload_bird_pretrain, dataload_bird_train, dataload_bird_val, dataload_bird_debug_test
from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_TrainDataLoader, MSRVTT_DataLoader
from dataloaders.dataloader_vatex_retrieval import VATEX_multi_sentence_dataLoader


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
                                     json_path="/ai/swxdisk/data/bird/query_data_val_bilingual.json",
                                     tokenizer=tokenizer, max_frames=args.max_frames,
                                     frame_sample_len=args.frame_sample_len, task=args.task)
    dataloader = DataLoader(
        bird_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader, len(bird_testset)


def dataloader_bird_debug_test(args, tokenizer):
    bird_testset = dataload_bird_debug_test(root='/ai/swxdisk/data/bird/videoinfo_lmdb', language=args.language,
                                          json_path="/ai/swxdisk/data/bird/videoinfo_bilingual.json",
                                          tokenizer=tokenizer, max_frames=args.max_frames,
                                          frame_sample=args.frame_sample, frame_sample_len=args.frame_sample_len)
    dataloader = DataLoader(
        bird_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader, len(bird_testset)


def dataloader_msrvtt_train(args, tokenizer):
    msrvtt_trainset = MSRVTT_TrainDataLoader(tokenizer=tokenizer, root="/ai/swxdisk/data/msrvtt/cfm_msrvtt_lmdb",
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
    msrvtt_testset = MSRVTT_DataLoader(tokenizer=tokenizer, root="/ai/swxdisk/data/msrvtt/cfm_msrvtt_lmdb",
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


def dataloader_vatex_pretrain(args, tokenizer):
    vatex_pretrainset = VATEX_multi_sentence_dataLoader(root='/ai/swxdisk/data/vatex/vatex_lmdb', language=args.language,
                                          data_path='/ai/swxdisk/data/vatex', tokenizer=tokenizer, subset="pretrain",
                                          frame_sample=args.frame_sample, max_frames=args.max_frames)
    train_sampler = torch.utils.data.distributed.DistributedSampler(vatex_pretrainset)
    dataloader = DataLoader(
        vatex_pretrainset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=True,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )
    return dataloader, len(vatex_pretrainset), train_sampler


def dataloader_vatex_train(args, tokenizer):
    vatex_trainset = VATEX_multi_sentence_dataLoader(root='/ai/swxdisk/data/vatex/vatex_lmdb', language=args.language,
                                          data_path='/ai/swxdisk/data/vatex', tokenizer=tokenizer, subset="train",
                                          frame_sample=args.frame_sample, max_frames=args.max_frames)
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


def dataloader_vatex_val(args, tokenizer):
    vatex_testset = VATEX_multi_sentence_dataLoader(root='/ai/swxdisk/data/vatex/vatex_lmdb', language=args.language,
                                       subset="val", frame_sample="uniform",
                                       data_path='/ai/swxdisk/data/vatex',
                                       tokenizer=tokenizer, max_frames=args.max_frames)
    dataloader = DataLoader(
        vatex_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader, len(vatex_testset)


def dataloader_vatex_test(args, tokenizer):
    vatex_testset = VATEX_multi_sentence_dataLoader(root='/ai/swxdisk/data/vatex/vatex_lmdb', language=args.language,
                                       subset="test", frame_sample="uniform",
                                       data_path='/ai/swxdisk/data/vatex',
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
DATALOADER_DICT["bird"] = {"pretrain": dataloader_bird_pretrain, "train": dataloader_bird_train,
                           "test": dataloader_bird_test, "debug_test": dataloader_bird_debug_test}
DATALOADER_DICT["msrvtt"] = {"train": dataloader_msrvtt_train, "test": dataloader_msrvtt_test}
DATALOADER_DICT["vatex"] = {"pretrain": dataloader_vatex_pretrain, "train": dataloader_vatex_train,
                            "val": dataloader_vatex_val, "test": dataloader_vatex_test}
