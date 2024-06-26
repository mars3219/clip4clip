from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import cv2
import os
from config.prompt_config import prompt
from itertools import product
from tqdm import tqdm
import clip

import torch
import numpy as np
import random
import os
from metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim
import time
import argparse
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import CLIP4Clip
from modules.optimization import BertAdam

from util import parallel_apply, get_logger
from dataloaders.data_dataloaders import DATALOADER_DICT

master_addr = "localhost"
master_port = "8888"

torch.distributed.init_process_group(backend="nccl", 
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=0,
        world_size=1)

global logger

def get_args(description='CLIP4Clip on Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_false', help="Whether to run eval on the dev set.")

    parser.add_argument('--train_csv', type=str, default='/workspace/CLIP4Clip/data/MSRVTT_train.9k.csv', help='')
    parser.add_argument('--val_csv', type=str, default='/workspace/CLIP4Clip/data/MSRVTT_JSFUSION_test.csv', help='')
    parser.add_argument('--data_path', type=str, default='/workspace/CLIP4Clip/data/MSRVTT_data.json', help='data pickle file path')
    parser.add_argument('--features_path', type=str, default='/workspace/CLIP4Clip/data/MSRVTT_Videos', help='feature path')

    parser.add_argument('--num_thread_reader', type=int, default=0, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=2, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=16, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=32, help='')
    parser.add_argument('--max_frames', type=int, default=90, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')

    parser.add_argument("--output_dir", default='/workspace/CLIP4Clip/ckpts/ckpt_msrvtt_retrieval_looseType', type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    # parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--init_model", default="/workspace/CLIP4Clip/ckpts/ckpt_msrvtt_retrieval_looseType/clip4clip_vit-base-p32-res224-clip-pre_8xb16-u12-5e_msrvtt-9k-rgb.pth", type=str, required=False, help="Initial model.")
    parser.add_argument("--resume_model", default=None, type=str, required=False, help="Resume train model.")
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
    parser.add_argument('--coef_lr', type=float, default=0.001, help='coefficient for bert branch.')
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=12, help="Layer NO. of visual.")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=4, help="Layer NO. of cross.")

    parser.add_argument('--loose_type', action='store_false', help="Default using tight type for retrieval.")
    parser.add_argument('--expand_msrvtt_sentences', action='store_false', help="")

    parser.add_argument('--train_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
    parser.add_argument('--eval_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")

    parser.add_argument('--freeze_layer_num', type=int, default=0, help="Layer NO. of CLIP need to freeze.")
    parser.add_argument('--slice_framepos', type=int, default=2, choices=[0, 1, 2],
                        help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.")
    parser.add_argument('--linear_patch', type=str, default="2d", choices=["2d", "3d"],
                        help="linear projection of flattened patches.")
    parser.add_argument('--sim_header', type=str, default="meanP",
                        choices=["meanP", "seqLSTM", "seqTransf", "tightTransf"],
                        help="choice a similarity header.")

    parser.add_argument("--pretrained_clip_name", default="ViT-B/32", type=str, help="Choose a CLIP version")

    args = parser.parse_args()

    if args.sim_header == "tightTransf":
        args.loose_type = False

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

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
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu

def init_model(args, device, n_gpu, local_rank):

    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu')
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

    model.to(device)

    return model

def load_model(epoch, args, n_gpu, device, model_file=None):
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')

        if isinstance(model_state_dict, dict):
            # mmaction2 clip4clip weight state_dict
            if 'meta' in model_state_dict.keys():
                model_state_dict = model_state_dict['state_dict']

        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        # Prepare model
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
        model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

        model.to(device)
    else:
        model = None
    return model

def _run_on_single_gpu(model, visual_features, text_features, sim_header='meanP', loose_type=True, device="cuda:0"):
    logits, logits_each, *_tmp = model.get_similarity_logits(text_features, visual_features, sim_header=sim_header, loose_type=loose_type, device=device)
    probs = logits.softmax(dim=-1).cpu().numpy()
    probs_each = logits_each.softmax(dim=-1).cpu().numpy()
    return probs, probs_each

# def prompt_config(pos, neg, model, device):
#         SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
#                               "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        
#         max_words = 77 

#         tokenizer = ClipTokenizer()
#         # # product 함수를 사용하여 가능한 모든 조합 생성
#         # positive_combinations = product(*pos)
#         # negative_combinations = product(*neg)

#         # Positive=[]
#         # for combination in tqdm(positive_combinations, desc="Positive prompt", mininterval=0.01):
#         #     result = " ".join(combination)
#         #     Positive.append(result)

#         # Negative=[]
#         # for combination in tqdm(negative_combinations, desc="Negative prompt", mininterval=0.01):
#         #     result = " ".join(combination)
#         #     Negative.append(result)

#         # "Two men are kicking each other", "People are fighting in the street", "Two men are throwing punches at each other"
#         Positive = ["Two men are kicking each other", "People are fighting in the street", 
#                     "Two men are throwing punches at each other", "People are wrestling",
#                     "People are hugging", "A person is choking another person"]
#         Negative = ["People are standing side by side", "Some people are running on the street happily", "People are walking on the street peacefully",
#                     ]

#         TextSet = Positive + Negative

#         encoding_text = []
#         for i, text in enumerate(TextSet):
#             words = tokenizer.tokenize(text)
#             words = [SPECIAL_TOKEN["CLS_TOKEN"]] + words
#             total_length_with_CLS = max_words - 1
#             if len(words) > total_length_with_CLS:
#                 words = words[:total_length_with_CLS]
#             words = words + [SPECIAL_TOKEN["SEP_TOKEN"]]

#             input_ids = tokenizer.convert_tokens_to_ids(words)

#             while len(input_ids) < max_words:
#                 input_ids.append(0)

#             assert len(input_ids) == max_words

#             encoding_text.append(np.array(input_ids))

#         with torch.no_grad():
#             # Positive/Negative Class Feature들로부터 평균 Feature를 구해 이를 대표로 사용
#             text = torch.tensor(encoding_text).to(device)
#             pos_features = model.clip.encode_text(text[0:1]).to(device) / float(len(Positive))
#             for pos in range(1,len(Positive)):
#                 pos_features = pos_features + model.clip.encode_text(text[pos:pos + 1]).to(device) / float(len(Positive))

#             neg_features = model.clip.encode_text(text[len(Positive):len(Positive) + 1]).to(device) / float(len(Negative))
#             for neg in range(len(Positive) + 1, len(text)):
#                 neg_features = neg_features + model.clip.encode_text(text[neg:neg + 1]).to(device) / float(len(Negative))

#             text_features = torch.cat((pos_features,neg_features),0)

#             return text_features

def prompt_config(pos, neg, model, device):
        # product 함수를 사용하여 가능한 모든 조합 생성
        positive_combinations = product(*pos)
        negative_combinations = product(*neg)

        Positive=[]
        for combination in tqdm(positive_combinations, desc="Positive prompt", mininterval=0.01):
            result = " ".join(combination)
            Positive.append(result)

        Negative=[]
        for combination in tqdm(negative_combinations, desc="Negative prompt", mininterval=0.01):
            result = " ".join(combination)
            Negative.append(result)

        TextSet = Positive + Negative
        text = clip.tokenize(TextSet).to(device)

        with torch.no_grad():
            # Positive/Negative Class Feature들로부터 평균 Feature를 구해 이를 대표로 사용
            text_features1 = model.encode_text(text[0:1]).to(device) / float(len(Positive))
            for ttt in range(1,len(Positive)):
                text_features1 = text_features1 + model.encode_text(text[ttt:ttt + 1]).to(device) / float(len(Positive))

            text_features2 = model.encode_text(text[len(Positive):len(Positive) + 1]).to(device) / float(len(Negative))
            for ttt in range(len(Positive) + 1, len(text)):
                text_features2 = text_features2 + model.encode_text(text[ttt:ttt+1]).to(device) / float(len(Negative))

            text_features = torch.cat((text_features1,text_features2),0)

            return text_features


def infer(args, model, rtspurl, device, n_gpu):

    prompt_pos=[prompt['outdoor']['start'], prompt['outdoor']['pos_words'], prompt['outdoor']['gender'], prompt['outdoor']['loc'], prompt['outdoor']['time_env']]
    prompt_neg=[prompt['outdoor']['start'], prompt['outdoor']['neg_words'], prompt['outdoor']['gender'], prompt['outdoor']['loc'], prompt['outdoor']['time_env']]


    text_features = prompt_config(prompt_pos, prompt_neg, model, device)
    # torch.save(text_features, "/workspace/CLIP4Clip/ckpts/ckpt_msrvtt_retrieval_looseType/text_features.pt")
    # text_features = torch.load("/workspace/CLIP4Clip/ckpts/ckpt_msrvtt_retrieval_looseType/text_features_p.pt")

    cap = cv2.VideoCapture(rtspurl)

    if not cap.isOpened():
        print("Error: Could not open RTSP stream.")
        exit()

    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    model.eval()

    fps = 25
    window_time = 2
    batch_size = fps * window_time
    frame_buffer = []
    # ori_buffer = []

    sim_header = 'meanP'
    loose_type = True

    with torch.no_grad():

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            if len(frame_buffer) < batch_size:
                resized_frame = cv2.resize(frame, (224, 224))
                # ori_buffer.append(resized_frame)
                tensor_frame = torch.tensor(resized_frame, device=device)
                frame_buffer.append(tensor_frame)

            if len(frame_buffer) >= batch_size:

                frames = torch.stack(frame_buffer).float()
                del frame_buffer[0]

                bs, h, w, c = frames.shape
                frames = frames.permute(0, 3, 1, 2)
                visual_features = model.encode_image(frames)
                # visual_features = model.clip.encode_image(frames, video_frame=bs).float()
                # print(f"visual encoding output: {visual_features}")

                # ----------------------------------
                # 2. calculate the similarity
                # ----------------------------------
                probs, probs_each = _run_on_single_gpu(model, visual_features, text_features, sim_header="meanP", loose_type=True, device=device)
                

                if probs[0] > 0.7:
                    text = f"prediction: {probs[0]:.2f} >>>>>>>>>>>> Violence!!!!!!!"
                else:
                    text = f"prediction: {probs[0]:.2f} >>>>>>>>>>>> No Event!!!!!"

                print(text)


                # for e, i in enumerate(ori_buffer):
                #     i = cv2.resize(i, (1280, 720))
                #     cv2.putText(i, f"each: {probs_each[e]} ---------- mean: {probs}", (5,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                #     cv2.imwrite(f'/workspace/CLIP4Clip/imgs/{e}.jpg', i)
                # del ori_buffer[0]


                cv2.putText(frame, f"each +: {probs_each[-1][0]:.2f} ---------- each -: {probs_each[-1][1]:.2f}", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                cv2.putText(frame, f"event: {text}", (50,150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

                # cv2.putText(frame, text, (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                cv2.imshow('Video', frame)
                cv2.waitKey(1)

def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)

    if args.local_rank == 0:
        # model = load_model(-1, args, n_gpu, device, model_file="/workspace/CLIP4Clip/ckpts/ckpt_msrvtt_retrieval_looseType/clip4clip_vit-base-p32-res224-clip-pre_8xb16-u12-5e_msrvtt-9k-rgb.pth")
        # model = init_model(args, device, n_gpu, args.local_rank)
        model = clip.load("/workspace/CLIP4Clip/modules/ViT-B-32.pt", device=device)

    # Uncomment if want to test on the best checkpoint
    rtsp_url = "rtsp://192.168.10.32:8554/stream"
    # rtsp_url = "rtsp://192.168.0.2:8554/stream"
    if args.do_eval:
        infer(args, model, rtsp_url, device, n_gpu)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
