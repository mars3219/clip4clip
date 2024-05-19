# Adapted from https://github.com/ArrowLuo/CLIP4Clip/blob/668334707c493a4eaee7b4a03b2dae04915ce170/main_task_retrieval.py#L457

import os
from utils.prompt_config import prompt
from itertools import product
import tqdm

import sys
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
import numpy as np

from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from evaluation.metrics import compute_metrics
from evaluation.metrics import tensor_text_to_video_metrics
from evaluation.metrics import tensor_video_to_text_sim
from utils.utils import parallel_apply
import torch


def _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list):
    """run similarity in one single gpu
    Args:
        model: CLIP2Video
        batch_list_t: id of text embedding
        batch_list_v: id of visual embedding
        batch_sequence_output_list: batch text embedding
        batch_visual_output_list: batch visual embedding
    Returns:
        sim_matrix: similarity

    """
    sim_matrix = []
    for idx1, b1 in enumerate(batch_list_t):
        input_mask, segment_ids, *_tmp = b1
        sequence_output = batch_sequence_output_list[idx1]
        each_row = []
        for idx2, b2 in enumerate(batch_list_v):
            video_mask, *_tmp = b2
            visual_output = batch_visual_output_list[idx2]
            # calculate the similarity
            b1b2_logits, *_tmp = model.get_inference_logits(sequence_output, visual_output, input_mask, video_mask)
            b1b2_logits = b1b2_logits.cpu().detach().numpy()
            each_row.append(b1b2_logits)
        each_row = np.concatenate(tuple(each_row), axis=-1)
        sim_matrix.append(each_row)
    return sim_matrix



def eval_epoch(model, rtsp_url, text_features, text_mask, device):
    """run similarity in one single gpu
    Args:
        model: CLIP2Video
        test_dataloader: data loader for test
        device: device to run model
        n_gpu: GPU number
        batch_sequence_output_list: batch text embedding
        batch_visual_output_list: batch visual embedding
    Returns:
        R1: rank 1 of text-to-video retrieval
    """

    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    model.eval()

    with torch.no_grad():
        visual_output = model.get_visual_output(input_ids, segment_ids, input_mask, video, video_mask)

        batch_sequence_output_list.append(sequence_output)
        batch_list_t.append((input_mask, segment_ids,))

        batch_visual_output_list.append(visual_output)
        batch_list_v.append((video_mask,))
        
        # calculate the similarity  in one GPU
        sim_matrix = _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list)
        sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)

        R1 = logging_rank(sim_matrix, multi_sentence_, cut_off_points_, logger)
        return R1

def logging_rank(sim_matrix, multi_sentence_, cut_off_points_, logger):
    """run similarity in one single gpu
    Args:
        sim_matrix: similarity matrix
        multi_sentence_: indicate whether the multi sentence retrieval
        cut_off_points_:  tag the label when calculate the metric
        logger: logger for metric
    Returns:
        R1: rank 1 of text-to-video retrieval

    """

    if multi_sentence_:
        # if adopting multi-sequence retrieval, the similarity matrix should be reshaped
        logger.info("before reshape, sim matrix size: {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
        max_length = max([e_-s_ for s_, e_ in zip([0]+cut_off_points2len_[:-1], cut_off_points2len_)])
        sim_matrix_new = []
        for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
            sim_matrix_new.append(np.concatenate((sim_matrix[s_:e_],
                                                  np.full((max_length-e_+s_, sim_matrix.shape[1]), -np.inf)), axis=0))
        sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
        logger.info("after reshape, sim matrix size: {} x {} x {}".
                    format(sim_matrix.shape[0], sim_matrix.shape[1], sim_matrix.shape[2]))

        # compute text-to-video retrieval
        tv_metrics = tensor_text_to_video_metrics(sim_matrix)

        # compute video-to-text retrieval
        vt_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix))
    else:
        logger.info("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))

        # compute text-to-video retrieval
        tv_metrics = compute_metrics(sim_matrix)

        # compute video-to-text retrieval
        vt_metrics = compute_metrics(sim_matrix.T)
        logger.info('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))


    # logging the result of text-to-video retrieval
    logger.info("Text-to-Video:")
    logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))

    # logging the result of video-to-text retrieval
    logger.info("Video-to-Text:")
    logger.info(
        '\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.format(
            vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))

    R1 = tv_metrics['R1']
    return R1
