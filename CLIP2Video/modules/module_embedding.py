import torch
from tqdm import tqdm
import numpy as np
from itertools import product
from utils.prompt_config import prompt

from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer


def get_text(caption, max_words):

    SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        
    tokenizer = ClipTokenizer()

    input_ids_list= []
    input_mask_list = []

    for i, text in enumerate(caption):
        # tokenize word
        words = tokenizer.tokenize(text)

        # add cls token
        words = [SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = max_words - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]

        # add end token
        words = words + [SPECIAL_TOKEN["SEP_TOKEN"]]

        # convert token to encoding id according to the vocab
        input_ids = tokenizer.convert_tokens_to_ids(words)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_words:
            input_ids.append(0)
            input_mask.append(0)

        # ensure the length of feature to be equal with max words
        assert len(input_ids) == max_words
        assert len(input_mask) == max_words

        input_ids_list.append(np.array(input_ids))
        input_mask_list.append(np.array(input_mask))

    return input_ids_list, input_mask_list


def get_text_embedding_output(model, text_l, mask_l, device, center_type="TAB", pos_len=1):
    bs_pair = 1
    neg_len = len(text_l) - pos_len
    
    text_l = torch.tensor(text_l).to(device)
    mask_l = torch.tensor(mask_l).to(device)

    pos_mask = torch.zeros(1, mask_l.size()[1], dtype=torch.int)
    neg_mask = torch.zeros(1, mask_l.size()[1], dtype=torch.int)
    
    pos_mask_mean = round(mask_l[:pos_len + 1].float().mean().item() * mask_l.size()[1])
    neg_mask_mean = round(mask_l[neg_len :].float().mean().item() * mask_l.size()[1])

    pos_mask[0, :pos_mask_mean] = 1
    neg_mask[0, :neg_mask_mean] = 1

            
    if center_type == 'TAB':
        
        with torch.no_grad():
            pos_features, pos_hidden = map(lambda x: x / float(pos_len), model.clip.encode_text(text_l[0:1], return_hidden=True))
            for pos in range(1, pos_len):
                pos_features_next, pos_hidden_next = map(lambda x: x / float(pos_len), model.clip.encode_text(text_l[pos:pos + 1], return_hidden=True))
                pos_features += pos_features_next
                pos_hidden += pos_hidden_next

            if neg_len != 0:
                neg_features, neg_hidden = map(lambda x: x / float(neg_len), model.clip.encode_text(text_l[pos_len:pos_len + 1], return_hidden=True))
                for neg in range(pos_len + 1, len(text_l)):
                    neg_features_next, neg_hidden_next = map(lambda x: x / float(pos_len), model.clip.encode_text(text_l[neg:neg + 1], return_hidden=True))
                    neg_features += neg_features_next
                    neg_hidden += neg_hidden_next

                text_features = torch.cat((pos_features, neg_features), 0)
                return_hidden = torch.cat((pos_hidden, neg_hidden), 0) 
                text_mask = torch.cat((pos_mask, neg_mask),0).to(device)
            else:
                text_features = pos_features
                return_hidden = pos_hidden
                text_mask = pos_mask.to(device)

            text_features = text_features.float()
            return_hidden = return_hidden.float()
            # return_hidden = return_hidden.view(bs_pair, -1, return_hidden.size(-1))


            return text_features, return_hidden, text_mask
    else:
        with torch.no_grad():
            pos_features = model.clip.encode_text(text_l[0:1]) / float(pos_len)
            for pos in range(1, pos_len):
                pos_features = pos_features + model.clip.encode_text(text_l[pos:pos + 1]) / float(pos_len)

            neg_features = model.clip.encode_text(text_l[pos_len:pos_len + 1]) / float(neg_len)
            for neg in range(pos_len + 1, len(text_l)):
                neg_features = neg_features + model.clip.encode_text(text_l[neg:neg + 1]) / float(neg_len)

            text_features = torch.cat((pos_features, neg_features), 0)
            text_features = text_features.float()

            return text_features, text_mask


def prompt_embedding(model, max_words=32, device="cuda:0", center_type="TAB"):

    # pos=[prompt['outdoor']['start'], prompt['outdoor']['pos_words'], prompt['outdoor']['gender'], prompt['outdoor']['loc'], prompt['outdoor']['time_env']]
    # neg=[prompt['outdoor']['start'], prompt['outdoor']['neg_words'], prompt['outdoor']['gender'], prompt['outdoor']['loc'], prompt['outdoor']['time_env']]

    # # product 함수를 사용하여 가능한 모든 조합 생성
    # positive_combinations = product(*pos)
    # negative_combinations = product(*neg)

    # Positive=[]
    # for combination in tqdm(positive_combinations, desc="Positive prompt", mininterval=0.01):
    #     result = " ".join(combination)
    #     Positive.append(result)

    # Negative=[]
    # for combination in tqdm(negative_combinations, desc="Negative prompt", mininterval=0.01):
    #     result = " ".join(combination)
    #     Negative.append(result)

    # Positive = ["men are doing wrestling", "some people are fighting",
    #             "people are punching each other", "people are kicking each other",
    #             "men are boxing", ]
    # Negative = ["cars are moving on the road", "players are playing a soccer on the playground",
    #             "some people are running in the streets", "some people are walking in the streets",
    #             "people are playing on the playground"]


    Positive = ["men are doing wrestling", "peple are fighting each other", "men are punching each other",
                "people are kicking each other",]
    Negative = ["people are walking on the street", "there is no people", "people are walking", "people are playing tennis",
                "people are play with racket"]
    # TextSet = Positive
    TextSet = Positive + Negative
    text_encoding_list, text_mask_list = get_text(TextSet, max_words)

    text_features = get_text_embedding_output(model, text_encoding_list, text_mask_list, device, center_type="TAB", pos_len=len(Positive))
    
    return (text_features[0], text_features[1]), text_features[2]