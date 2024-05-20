# Adapted from https://github.com/ArrowLuo/CLIP4Clip/blob/668334707c493a4eaee7b4a03b2dae04915ce170/main_task_retrieval.py#L457

import os
import sys

import cv2
import torch
import threading
import queue
import tqdm, time
import numpy as np

sys.path.append(os.path.dirname(__file__) + os.sep + '../')


# initialize Queue and Variable 
FRAME_QUEUE = queue.Queue()
ANALYSIS_INTERVAL = 1

def capture_frames(rtsp_url, frame_queue, max_frames):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Error: Could not open RTSP stream")
        return
    
    frame_interval = 1 / max_frames
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        timestamp = time.time()
        frame_queue.put((timestamp, frame))
        time.sleep(frame_interval)

    cap.release()

def analyze_frames(frame_queue, analysis_interval, model, max_frames, text_features, text_mask, device):
    
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    model.eval()

    while True:
        time.sleep(analysis_interval)
        frames_to_analyze = []
        current_time = time.time()
        
        while not frame_queue.empty():
            timestamp, frame = frame_queue.queue[0]
            if current_time - timestamp <= analysis_interval:
                img = frame_queue.get()
                frame = cv2.resize(img[1], (244, 244))
                frame = torch.tensor(frame).permute(2, 0, 1)
                frames_to_analyze.append(frame)
                if len(frames_to_analyze) == max_frames * analysis_interval:
                    break
            else:
                frame_queue.get()  # 오래된 프레임을 버립니다

        if len(frames_to_analyze) >= max_frames:
            frame = np.stack(frames_to_analyze, axis=0)
            print(frame.shape)
            frame_mask = np.ones((1, max_frames), dtype=np.int64)
            frame = torch.tensor(frame).float().to(device)
            frame_mask = torch.tensor(frame_mask).to(device)

            with torch.no_grad():
                batch_list_t = []
                batch_list_v = []
                batch_sequence_output_list, batch_visual_output_list = [], []
                total_video_num = 0


                visual_output = model.get_visual_output(frame, frame_mask, shaped=True, video_frame=max_frames)

                batch_sequence_output_list.append(text_features)
                batch_list_t.append((text_mask, 0,))

                batch_visual_output_list.append(visual_output)
                batch_list_v.append((frame_mask,))
                
                # calculate the similarity  in one GPU
                sim_matrix = _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list)
                sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)




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



def eval_epoch(model, max_frames, rtsp_url, text_features, text_mask, device):
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

    capture_thread = threading.Thread(target=capture_frames, args=(rtsp_url, FRAME_QUEUE, max_frames))
    analyze_thread = threading.Thread(target=analyze_frames, args=(FRAME_QUEUE, ANALYSIS_INTERVAL, model, max_frames, text_features, text_mask, device))

    capture_thread.start()
    analyze_thread.start()

    capture_thread.join()
    analyze_thread.join()
