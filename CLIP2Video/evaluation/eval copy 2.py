# Adapted from https://github.com/ArrowLuo/CLIP4Clip/blob/668334707c493a4eaee7b4a03b2dae04915ce170/main_task_retrieval.py#L457

import os
import sys
import threading
import queue

import numpy as np
import cv2
from PIL import Image
from datetime import datetime
import time, math

import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, Lambda


sys.path.append(os.path.dirname(__file__) + os.sep + '../')


# initialize Queue and Variable
NUM_SPLITS = 4 
FRAME_QUEUE = queue.Queue(maxsize=10)
RESULT_EVENT = threading.Event()
QUEUE_EVENT = threading.Event()
INFERENCE_RESULT = None
ANALYSIS_INTERVAL = 1

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def add_frame_to_queue(rtsp_url, frame_queue, max_frames):
    time_interval = 10 / max_frames
    st = time.time()
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Error: Could not open RTSP stream")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        if QUEUE_EVENT.is_set():
            if (time.time() - st) >= time_interval:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st = time.time()
                now = datetime.now()
                cv2.putText(frame, f"event: {now.strftime('%H:%M:%S')}", (10,180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                frame = Image.fromarray(frame)
                frame_queue.put(frame)


def capture_frame_and_display(rtsp_url, frame_queue, max_frames):
    global INFERENCE_RESULT
    QUEUE_EVENT.set()
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Error: Could not open RTSP stream")
        return
    
    dresult = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        if RESULT_EVENT.is_set():
            result = INFERENCE_RESULT
            RESULT_EVENT.clear()
            dresult = result
        try:        
            cv2.putText(frame, f"event: {dresult}", (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.imshow("inference", frame)
            cv2.waitKey(1)
        except:
            pass

    cap.release()

def multi_crop(image, n, preprocess):

    width, height = image.size
    num_rows = math.isqrt(n)  # n의 제곱근의 정수 부분 (행의 수)
    num_cols = (n + num_rows - 1) // num_rows  # n을 num_rows로 나눈 후 올림 (열의 수)
    
    crops = []
    crop_width = width // num_cols
    crop_height = height // num_rows
    
    for i in range(num_rows):
        for j in range(num_cols):
            left = j * crop_width
            upper = i * crop_height
            right = left + crop_width
            lower = upper + crop_height
            tmp_img = image.crop((left, upper, right, lower))

            # preprocessing frames
            img = preprocess(tmp_img)
            crops.append(img)
    
    return crops

def analyze_frames(num_splits, frame_queue, model, max_frames, text_features, text_mask, device):
    global INFERENCE_RESULT
    preprocess_fn = _transform(224)

    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    model.eval()

    frames_to_analyze = []
    
    origin_frames = []

    crop_lists = [[] for _ in range(num_splits)]  # screen_lists: [[],[],...,[]]
    trans_crop = Lambda(lambda img: multi_crop(img, num_splits, preprocess_fn))

    while True:
        if frame_queue.qsize() >= max_frames:
            QUEUE_EVENT.clear()

            ss = time.time()

            while not frame_queue.empty():
                frame = frame_queue.get()

                origin_frames.append(frame)  #   FrameExtractor: def get_video_data/ video_to_tensor(dataloaders/rawframe_util.py)

            for img0 in origin_frames:
                # # 파일 이름 생성
                # file_name = f"rtsp_image_{idx + 1}.jpg"
                # destination_path = os.path.join("/workspace/CLIP4Clip/CLIP2Video/test/output/real", file_name)
                
                # # NumPy 배열을 이미지로 저장
                # img0.save(destination_path)
                # print(f"Image saved to {destination_path}")

                cropped_img = trans_crop(img0)
                for i, crop in enumerate(cropped_img):
                    crop_lists[i].append(crop)
                
                # # preprocessing frames
                # img = preprocess_fn(img0)
                # frames_to_analyze.append(img)
                    
                    

            origin_frames.clear()

            frames = torch.tensor(np.stack(frames_to_analyze, axis=0))

            frame_mask = np.ones((1, max_frames), dtype=np.int64)
            frames = torch.tensor(frames).float().to(device)
            frame_mask = torch.tensor(frame_mask).to(device)

            frames_to_analyze.clear()
            QUEUE_EVENT.set()

            with torch.no_grad():

                visual_output = model.get_visual_output(frames, frame_mask, shaped=True, video_frame=max_frames)
                # torch.save(visual_output, "/workspace/CLIP4Clip/CLIP2Video/evaluation/real_visual_output.pt")
                
                # calculate the similarity  in one GPU
                probs = _run_on_single_gpu(model, text_mask, frame_mask, text_features, visual_output)

                if probs[0] > 0.65:
                    text = f"pred: {probs[0].item():.2f} >> Violence >> {probs[1].item():.2f} >>> interval: {time.time()-ss}"
                else:
                    text = f"pred: {probs[0].item():.2f} >> No Event >> {probs[1].item():.2f} >>> interval: {time.time()-ss}"


                print(text)

                INFERENCE_RESULT = text
                RESULT_EVENT.set()

              
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
    print(b1b2_logits)
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

    queueing = threading.Thread(target=add_frame_to_queue, args=(rtsp_url, FRAME_QUEUE, max_frames,))
    capture_thread = threading.Thread(target=capture_frame_and_display, args=(rtsp_url, FRAME_QUEUE, max_frames))
    analyze_thread = threading.Thread(target=analyze_frames, args=(NUM_SPLITS, FRAME_QUEUE, model, max_frames, text_features, text_mask, device))

    queueing.start()
    capture_thread.start()
    analyze_thread.start()

    queueing.join()
    capture_thread.join()
    analyze_thread.join()
