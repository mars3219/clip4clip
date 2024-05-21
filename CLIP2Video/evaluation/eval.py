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
FRAME_QUEUE = queue.Queue(maxsize=10)
RESULT_EVENT = threading.Event()
INFERENCE_RESULT = None
ANALYSIS_INTERVAL = 1

def capture_frame_and_display(rtsp_url, frame_queue, max_frames):
    global INFERENCE_RESULT
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Error: Could not open RTSP stream")
        return
    
    frame_interval = 1 / max_frames
    dresult = 0

    def add_frame_to_queue(frame):
        # if FRAME_QUEUE.full():
        #     FRAME_QUEUE.get()
        frame_queue.put(frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        threading.Timer(frame_interval, add_frame_to_queue, args=(frame,)).start()

        if RESULT_EVENT.is_set():
            result = INFERENCE_RESULT
            RESULT_EVENT.clear()
            dresult = result
        try:        
            cv2.putText(frame, f"event: {dresult}", (50,150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            # cv2.putText(frame, text, (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("inference", frame)
            cv2.waitKey(1)
        except:
            pass

    cap.release()

def analyze_frames(frame_queue, analysis_interval, model, max_frames, text_features, text_mask, device):
    global INFERENCE_RESULT

    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    model.eval()

    frames_to_analyze = []
    while True:
        # print(len(frame_queue.queue))

        while not frame_queue.empty():
            frame = frame_queue.get()
            frame = cv2.resize(frame, (244, 244))
            frame = torch.tensor(frame).permute(2, 0, 1)
            frames_to_analyze.append(frame)

            if len(frames_to_analyze) == max_frames:
                # while not frame_queue.empty():
                #     frame_queue.get()
                break

        if len(frames_to_analyze) >= max_frames:
            frames = np.stack(frames_to_analyze, axis=0)
            # print(frame.shape)
            frame_mask = np.ones((1, max_frames), dtype=np.int64)
            frames = torch.tensor(frames).float().to(device)
            frame_mask = torch.tensor(frame_mask).to(device)

            frames_to_analyze.clear()

            with torch.no_grad():

                visual_output = model.get_visual_output(frames, frame_mask, shaped=True, video_frame=max_frames)
                
                # calculate the similarity  in one GPU
                probs = _run_on_single_gpu(model, text_mask, frame_mask, text_features, visual_output)

                
                if probs[0] > 0.5:
                    text = f"prediction: {probs[0].item():.2f} >>>>>>>>>>>> Violence!!!!!!!{probs[1].item():.2f}"
                else:
                    text = f"prediction: {probs[0].item():.2f} >>>>>>>>>>>> No Event!!!!!{probs[1].item():.2f}"

                print(text)


                INFERENCE_RESULT = text
                RESULT_EVENT.set()

                # frame_queue.task_done()
                
              
def _run_on_single_gpu(model, input_mask, video_mask, sequence_output, visual_output):
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
    
    # calculate the similarity
    b1b2_logits, *_tmp = model.get_inference_logits(sequence_output, visual_output, input_mask, video_mask, shaped=True)
    # b1b2_logits = b1b2_logits.cpu().detach().numpy()
    probs = b1b2_logits.softmax(dim=0).cpu().numpy()
    
    return probs



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

    capture_thread = threading.Thread(target=capture_frame_and_display, args=(rtsp_url, FRAME_QUEUE, max_frames))
    analyze_thread = threading.Thread(target=analyze_frames, args=(FRAME_QUEUE, ANALYSIS_INTERVAL, model, max_frames, text_features, text_mask, device))

    capture_thread.start()
    analyze_thread.start()

    capture_thread.join()
    analyze_thread.join()
