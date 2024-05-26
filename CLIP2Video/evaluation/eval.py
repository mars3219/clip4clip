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
MAX_FRAMES = 12
FRAME_QUEUE = queue.Queue(maxsize=MAX_FRAMES)
TIME_INTERVAL = 6 / MAX_FRAMES
RESULT_EVENT = threading.Event()
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
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Error: Could not open RTSP stream")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        now = datetime.now()
        cv2.putText(frame, f"event: {now.strftime('%H:%M:%S')}", (10,180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        frame = Image.fromarray(frame)

        try:
            frame_queue.put_nowait(frame)
        except:
            frame_queue.get()
            frame_queue.put_nowait(frame)
        time.sleep(TIME_INTERVAL)


def capture_frame_and_display(num_splits, rtsp_url, frame_queue, max_frames):
    global INFERENCE_RESULT

    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    if not cap.isOpened():
        print("Error: Could not open RTSP stream")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # 프레임의 높이와 너비 가져오기
        height, width = frame.shape[:2]

        split_count = int(math.sqrt(num_splits))
        # 분할된 이미지 생성
        split_height = height // split_count
        split_width = width // split_count
        split_images = []

        for i in range(split_count):
            for j in range(split_count):
                split_images.append(frame[i*split_height:(i+1)*split_height, j*split_width:(j+1)*split_width])
        
        if RESULT_EVENT.is_set():
            result = INFERENCE_RESULT
            RESULT_EVENT.clear()

        # 분할된 이미지를 디스플레이
        for idx, split_image in enumerate(split_images):
            try: 
                cv2.putText(split_image, f"event: {result[idx * 2]}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                cv2.imshow(f"window_{idx}", split_image)
                cv2.waitKey(1)
            except:
                pass
        
        # try:        
        #     cv2.putText(frame, f"event: {result}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        #     cv2.imshow(f"window", frame)
        #     cv2.waitKey(1)
        # except:
        #     pass



        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def visual_encoding(model, queue, frames, frame_mask, text_mask, text_features, shaped, video_frame):
    with torch.no_grad():

        visual_output = model.get_visual_output(frames, frame_mask, shaped, video_frame)
        # torch.save(visual_output, "/workspace/CLIP4Clip/CLIP2Video/evaluation/real_visual_output.pt")
        
        # calculate the similarity  in one GPU
        probs = _run_on_single_gpu(model, text_mask, frame_mask, text_features, visual_output)
        # print(f'probs: {probs}----- {datetime.now()}')
        queue.put(probs)

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

            if preprocess is not None:
                # preprocessing frames
                img = preprocess(tmp_img)
                crops.append(img)
            else:
                crops.append(tmp_img)
    
    return crops

def analyze_frames(num_splits, frame_queue, model, max_frames, text_features, text_mask, device):
    global INFERENCE_RESULT
    preprocess_fn = _transform(224)

    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    model.eval()

    # origin_frames = []
    frames_to_analyze = []
    frame_masks = []
    
    crop_lists = [[] for _ in range(num_splits)]  # screen_lists: [[],[],...,[]]
    trans_crop = Lambda(lambda img: multi_crop(img, num_splits, preprocess_fn))

    while True:
        if frame_queue.qsize() >= max_frames:
                                                                          
            # while not frame_queue.empty():
                # frame = frame_queue.get()
            with frame_queue.mutex:
                origin_frames = list(frame_queue.queue)
                # print(origin_frames)
                # origin_frames.append(frame)  #   FrameExtractor: def get_video_data/ video_to_tensor(dataloaders/rawframe_util.py)

            for img0 in origin_frames:
                cropped_img = trans_crop(img0)
                for i, crop in enumerate(cropped_img):
                    crop_lists[i].append(crop)
                
            for i in crop_lists:
                frames = torch.tensor(np.stack(i, axis=0))
                frames = torch.tensor(frames).float().to(device)
                frames_to_analyze.append(frames)

                frame_mask = np.ones((1, max_frames), dtype=np.int64)
                frame_mask = torch.tensor(frame_mask).to(device)
                frame_masks.append(frame_mask)

            # 결과를 저장할 큐
            output_queue = queue.Queue()

            # 쓰레드 생성 및 시작
            threads = []
            for crop_l, mask_l in zip(frames_to_analyze, frame_masks):
                t = threading.Thread(target=visual_encoding, args=(model, output_queue, crop_l, mask_l, text_mask, text_features, True, max_frames))
                t.start()
                threads.append(t)

            # 모든 쓰레드가 작업을 마치기를 기다림
            for t in threads:
                t.join()

            # 결과 취합
            all_results = []
            while not output_queue.empty():
                all_results.extend(output_queue.get())

            # # 결과 확인 (여기서는 각 탐지 결과를 출력)
            # for result in all_results:
            #     print(result)

            # if probs[0] > 0.65:
            #     text = f"pred: {probs[0].item():.2f} >> Violence >> {probs[1].item():.2f} >>> interval: {time.time()-ss}"
            # else:
            #     text = f"pred: {probs[0].item():.2f} >> No Event >> {probs[1].item():.2f} >>> interval: {time.time()-ss}"

            origin_frames.clear()
            frames_to_analyze.clear()
            frame_masks.clear()
            for i in range(len(crop_lists)):
                crop_lists[i].clear()

            INFERENCE_RESULT = all_results.copy()
            # print(f'all: {all_results[0]}----- {datetime.now()}')
            all_results.clear()
            
            RESULT_EVENT.set()
        time.sleep(TIME_INTERVAL*2)

              
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
    capture_thread = threading.Thread(target=capture_frame_and_display, args=(NUM_SPLITS, rtsp_url, FRAME_QUEUE, max_frames))
    analyze_thread = threading.Thread(target=analyze_frames, args=(NUM_SPLITS, FRAME_QUEUE, model, max_frames, text_features, text_mask, device))

    queueing.start()
    capture_thread.start()
    analyze_thread.start()

    queueing.join()
    capture_thread.join()
    analyze_thread.join()
