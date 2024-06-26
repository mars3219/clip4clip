#coding:utf-8
# @Time : 2021/6/19
# @Author : Han Fang
# @File: infer_retrieval.py
# @Version: version 1.0

import torch
import numpy as np
import os
import random

from modules.module_embedding import prompt_embedding

from modules.modeling import CLIP2Video
from evaluation.eval import eval_epoch

from utils.config import get_args
from utils.utils import get_logger
from utils.dataloader import dataloader_msrvtt_train
from utils.dataloader import dataloader_msrvtt_test
from utils.dataloader import dataloader_msrvttfull_test
from utils.dataloader import dataloader_msvd_train
from utils.dataloader import dataloader_msvd_test
from utils.dataloader import dataloader_vatexEnglish_train
from utils.dataloader import dataloader_vatexEnglish_test


# define the dataloader
# new dataset can be added from import and inserted according to the following code
DATALOADER_DICT = {}
DATALOADER_DICT["msrvtt"] = {"train":dataloader_msrvtt_train, "test":dataloader_msrvtt_test}
DATALOADER_DICT["msrvttfull"] = {"train":dataloader_msrvtt_train, "val":dataloader_msrvttfull_test, "test":dataloader_msrvttfull_test}
DATALOADER_DICT["msvd"] = {"train":dataloader_msvd_train, "val":dataloader_msvd_test, "test":dataloader_msvd_test}
DATALOADER_DICT["vatexEnglish"] = {"train":dataloader_vatexEnglish_train, "test":dataloader_vatexEnglish_test}


def set_seed_logger(args):
    """Initialize the seed and environment variable

    Args:
        args: the hyper-parameters.

    Returns:
        args: the hyper-parameters modified by the random seed.

    """

    global logger

    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # get logger
    logger = get_logger(os.path.join(args.output_dir))

    return args

def init_device(args, local_rank):
    """Initialize device to determine CPU or GPU

     Args:
         args: the hyper-parameters
         local_rank: GPU id

     Returns:
         devices: cuda
         n_gpu: number of gpu

     """
    global logger
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size_val % args.n_gpu != 0:
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu


def init_model(args, device):
    """Initialize model.

    if location of args.init_model exists, model will be initialized from the pretrained model.
    if no model exists, the training will be initialized from CLIP's parameters.

    Args:
        args: the hyper-parameters
        devices: cuda

    Returns:
        model: the initialized model

    """

    # resume model if pre-trained model exist.
    model_file = os.path.join(args.checkpoint, "pytorch_model.bin.{}".format(args.model_num))
    # model_file = "/workspace/CLIP4Clip/ckpts/ckpt_msrvtt_retrieval_looseType/clip4clip_vit-base-p32-res224-clip-pre_8xb16-u12-5e_msrvtt-9k-rgb.pth"
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
    else:
        model_state_dict = None
        if args.local_rank == 0:
            logger.info("Model loaded fail %s", model_file)

    # Prepare model
    model = CLIP2Video.from_pretrained(args.cross_model, cache_dir=None, state_dict=model_state_dict,
                                       task_config=args)
    model.to(device)

    return model


def main():
    global logger

    # obtain the hyper-parameter
    args = get_args()

    # set the seed
    args = set_seed_logger(args)

    # setting the testing device
    device, n_gpu = init_device(args, args.local_rank)


    # init model
    model = init_model(args, device)

    # setting tokenizer and text_features
    text_features_masks = prompt_embedding(model, args.max_words, device)
    # torch.save(text_features_masks, "/workspace/CLIP4Clip/ckpts/ckpt_msrvtt_retrieval_looseType/text_features_masks.pt")

    # text_features_masks = torch.load("/workspace/CLIP4Clip/ckpts/ckpt_msrvtt_retrieval_looseType/text_features_masks.pt")
    text_features, text_mask = text_features_masks
                                                             
    # text_features_masks = torch.load("/workspace/CLIP4Clip/ckpts/ckpt_msrvtt_retrieval_looseType/text_features_t.pt")
    # text_features, text_mask = text_features_masks, 0

    # rtsp_url = "rtsp://192.168.10.32:8554/stream"
    rtsp_url = "rtsp://192.168.0.2:8554/stream"
    # rtsp_url = "rtsp://192.168.0.4:28550/h264_ulaw.sdp"

    # evaluation for text-to-video and video-to-text retrieval
    if args.local_rank == 0:
        eval_epoch(model, args.max_frames, rtsp_url, text_features, text_mask, device)
    
if __name__ == "__main__":
    main()
