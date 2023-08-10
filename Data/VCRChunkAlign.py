import random
import os
from torch.utils.data import Dataset
import torch
import csv
from toolz.sandbox import unzip
from cytoolz import concat
import json
from PIL import Image
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from .data import (get_ids_and_lens, pad_tensors,
                   get_gather_index)
from utils.tsv_file import TSVFile
import base64
# import cv2
import pickle
from tqdm import tqdm
import clip

def _pad_ids(ids, max_len):
    if len(ids) >= max_len:
        return ids[:max_len]
    else:
        return ids + [0] * (max_len - len(ids))


def move_to_cuda(batch):
    if isinstance(batch, torch.Tensor):
        return batch.cuda(non_blocking=True)
    elif isinstance(batch, list):
        new_batch = [move_to_cuda(t) for t in batch]
    elif isinstance(batch, tuple):
        new_batch = tuple(move_to_cuda(t) for t in batch)
    elif isinstance(batch, dict):
        new_batch = {n: move_to_cuda(t) for n, t in batch.items()}
    else:
        return batch
    return new_batch


class VCR_ChunkAlign_Dataset_align(Dataset):
    def __init__(self, bert_tokenizer, GPT_tokenizer, VCR_example_file, VCR_chunk_mask_file,
                 flickr_feat_file,
                 max_img_seq_length=50,
                 is_train=True,heat_index=None):
        self.bert_toker = bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.chunk_mask_dict = pickle.load(open(VCR_chunk_mask_file, 'rb'))
        self.VCR_annot_dict = pickle.load(open(VCR_example_file, 'rb'))
        self.image_feat_dict = self.read_example(flickr_feat_file)
        self.add_od_labels = True
        self.is_train = is_train
        self.max_img_seq_length = max_img_seq_length
        self.cls = self.bert_toker.cls_token
        self.sep = self.bert_toker.sep_token

        if heat_index is not None:
            # 只保存当前样例
            self.VCR_annot_dict = self.VCR_annot_dict[heat_index:heat_index + 1]


    def read_example(self, path):
        data = pickle.load(open(path, 'rb'))
        return data

    def __len__(self):
        return len(self.VCR_annot_dict)

    def __getitem__(self, i):
        example = self.VCR_annot_dict[i]

        id = example['image_id']
        num_id = id.split("-")[1]

        str_id = "img-" + str(num_id)
        if self.is_train:
            answer_label = example['answer_label']
        else:
            answer_label = None

        image_feat = self.image_feat_dict[str_id]
        img_feat = image_feat['features'].cuda()
        img_mask = image_feat['img_mask'].cuda()

        que_tokens = example['sent'].lower()
        que_tokens = self.bert_toker.tokenize(que_tokens)
        outputs = []

        for ans_idx, ans in enumerate(example['answer_choices']):
            ans_tokens = self.bert_toker.tokenize(ans)
            input_tokens = [self.cls] + que_tokens + [self.sep] + ans_tokens + [self.sep]

            region_tokens = [0] * len(input_tokens)
            # region_tokens 为何物？
            # TODO
            for token_idx, token in enumerate(input_tokens):
                if '<|det' in token:
                    index = token[5:].split('|')[0]
                    region_tokens[token_idx] = int(index)
            region_tokens = torch.tensor(region_tokens).cuda()
            total_label = region_tokens
            align_pos = torch.where(total_label != 0, torch.ones_like(total_label).cuda(), total_label)

            input_ids = self.bert_toker.convert_tokens_to_ids(input_tokens)
            input_ids = torch.tensor(input_ids).cuda()
            mask_len = input_ids.size(0)
            input_mask = torch.ones(mask_len).cuda()

            segment_ids_ques = torch.zeros(len(que_tokens) + 2, dtype=torch.int64).cuda()
            segment_ids_ans = torch.ones(len(ans_tokens) + 1, dtype=torch.int64).cuda()
            segment_ids = torch.cat((segment_ids_ques, segment_ids_ans), 0)

            # chunk_mask_dict ?
            # TODO
            offsets = self.chunk_mask_dict[i][ans_idx]['offsets']
            chunk_mask = self.chunk_mask_dict[i][ans_idx]['mask'].cuda()

            gather_index = []
            for idx, set in enumerate(offsets):
                set = torch.tensor(set).cuda()
                gather_index.extend([idx] * set.size(0))
            gather_index = torch.tensor(gather_index).cuda()

            # 如果是答案的话 target为1，否则为0
            if isinstance(answer_label, list):
                if ans_idx in example['answer_label']:
                    target = torch.tensor(1).cuda()
                else:
                    target = torch.tensor(0).cuda()
            else:
                if ans_idx == example['answer_label']:
                    target = torch.tensor(1).cuda()
                else:
                    target = torch.tensor(0).cuda()

            outputs.append((example['annot_id'], input_ids, segment_ids, input_mask, img_feat, img_mask, target,
                            chunk_mask, gather_index, offsets, example['sent'],
                            example['answer_choices'][ans_idx], total_label, align_pos))
        return tuple(outputs)

    def SNLIGPT_gen_collate(self, inputs):

        (img_id, input_ids, segment_ids, input_mask, img_feat, img_mask, target, chunk_mask,
         gather_index, offsets, ques, ans, total_label, align_pos) = map(list, unzip(concat(inputs)))
        input_mask = pad_sequence(input_mask, batch_first=True, padding_value=0)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        total_label = pad_sequence(total_label, batch_first=True, padding_value=0)
        align_pos = pad_sequence(align_pos, batch_first=True, padding_value=0)

        segment_ids = pad_sequence(segment_ids, batch_first=True, padding_value=0)
        target = torch.stack(target, dim=0)

        img_mask = torch.stack(img_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)
        max_img = int(torch.max(torch.sum(img_mask, dim=-1)).item())
        img_mask = img_mask[:, :max_img]
        img_feat = img_feat[:, :max_img]

        input_mask = torch.cat((input_mask, img_mask), -1)

        max_hypo = input_ids.size(1)
        chunk_mask_padd = []
        for item in chunk_mask:
            padd_matrix = torch.zeros((max_hypo - item.size(0), item.size(1))).cuda()
            item = torch.cat((item, padd_matrix), 0)
            padd_matrix = torch.zeros((max_hypo, max_hypo - item.size(1))).cuda()
            item = torch.cat((item, padd_matrix), 1)
            chunk_mask_padd.append(item)
        chunk_mask_padd = torch.stack(chunk_mask_padd, 0)

        batch = {'img_id': img_id, 'input_ids': input_ids, 'token_type_ids': segment_ids,
                 'input_mask': input_mask, 'img_feat': img_feat, 'label': target, 'ques_str': ques, 'ans_str': ans,
                 'chunk_attention_mask': chunk_mask_padd, 'gather_index': gather_index, 'offsets': offsets,
                 'total_label': total_label, 'align_pos': align_pos
                 }

        return batch

class VCR_ChunkAlign_Dataset_align_ensemble(Dataset):
    def __init__(self, bert_tokenizer, GPT_tokenizer, roberta_tokenizer, VCR_example_file, VCR_chunk_mask_file,
                 flickr_feat_file,
                 roberta_example_file,
                 device,
                 max_img_seq_length=50,
                 is_train=True,heat_index=None):
        self.bert_toker = bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.roberta_toker = roberta_tokenizer

        self.chunk_mask_dict = pickle.load(open(VCR_chunk_mask_file, 'rb'))
        self.VCR_annot_dict = pickle.load(open(VCR_example_file, 'rb'))
        self.roberta_annot_dict = pickle.load((open(roberta_example_file, 'rb'))) # additional
        self.image_feat_dict = self.read_example(flickr_feat_file)
        self.add_od_labels = True
        self.is_train = is_train
        self.max_img_seq_length = max_img_seq_length
        self.cls = self.bert_toker.cls_token
        self.sep = self.bert_toker.sep_token

        self.rcls = self.roberta_toker.bos_token
        self.rsep = self.roberta_toker.eos_token
        self.device = device

        if heat_index is not None:
            # 只保存当前样例
            self.VCR_annot_dict = self.VCR_annot_dict[heat_index:heat_index + 1]


    def read_example(self, path):
        data = pickle.load(open(path, 'rb'))
        return data

    def __len__(self):
        return len(self.VCR_annot_dict)

    def __getitem__(self, i):
        example = self.VCR_annot_dict[i]

        roberta_example = self.roberta_annot_dict[i]

        id = example['image_id']
        num_id = id.split("-")[1]

        str_id = "img-" + str(num_id)

        answer_label = example['answer_label']

        image_feat = self.image_feat_dict[str_id]
        img_feat = image_feat['features'].cuda(self.device)
        img_mask = image_feat['img_mask'].cuda(self.device)

        que_tokens = example['sent'].lower()
        que_tokens = self.bert_toker.tokenize(que_tokens)
        outputs = []

        ## roberta
        r_que_tokens = roberta_example['sent'].lower()
        r_que_tokens = self.roberta_toker.tokenize(r_que_tokens)

        for ans_idx in range(len(example['answer_choices'])):

            r_ans = roberta_example['answer_choices'][ans_idx]
            r_ans_tokens = self.roberta_toker.tokenize(r_ans)
            r_input_tokens = [self.rcls] + r_que_tokens + [self.rsep] + r_ans_tokens + [self.rsep]
            r_input_ids = self.roberta_toker.convert_tokens_to_ids(r_input_tokens)
            r_input_ids = torch.tensor(r_input_ids).cuda(self.device)
            r_mask_len = r_input_ids.size(0)
            r_input_mask = torch.ones(r_mask_len).cuda(self.device)

            r_segment_ids_ques = torch.zeros(len(r_que_tokens) + 2, dtype=torch.int64).cuda(self.device)
            r_segment_ids_ans = torch.zeros(len(r_ans_tokens) + 1, dtype=torch.int64).cuda(self.device)
            r_segment_ids = torch.cat((r_segment_ids_ques, r_segment_ids_ans), 0)

            ans = example['answer_choices'][ans_idx]
            ans_tokens = self.bert_toker.tokenize(ans)
            input_tokens = [self.cls] + que_tokens + [self.sep] + ans_tokens + [self.sep]

            region_tokens = [0] * len(input_tokens)
            # region_tokens 为何物？
            # TODO
            for token_idx, token in enumerate(input_tokens):
                if '<|det' in token:
                    index = token[5:].split('|')[0]
                    region_tokens[token_idx] = int(index)
            region_tokens = torch.tensor(region_tokens).cuda(self.device)
            total_label = region_tokens
            align_pos = torch.where(total_label != 0, torch.ones_like(total_label).cuda(self.device), total_label)

            input_ids = self.bert_toker.convert_tokens_to_ids(input_tokens)
            input_ids = torch.tensor(input_ids).cuda(self.device)
            mask_len = input_ids.size(0)
            input_mask = torch.ones(mask_len).cuda(self.device)

            segment_ids_ques = torch.zeros(len(que_tokens) + 2, dtype=torch.int64).cuda(self.device)
            segment_ids_ans = torch.ones(len(ans_tokens) + 1, dtype=torch.int64).cuda(self.device)
            segment_ids = torch.cat((segment_ids_ques, segment_ids_ans), 0)

            # chunk_mask_dict ?
            # TODO
            offsets = self.chunk_mask_dict[i][ans_idx]['offsets']
            chunk_mask = self.chunk_mask_dict[i][ans_idx]['mask'].cuda(self.device)

            gather_index = []
            for idx, set in enumerate(offsets):
                set = torch.tensor(set).cuda(self.device)
                gather_index.extend([idx] * set.size(0))
            gather_index = torch.tensor(gather_index).cuda(self.device)

            # 如果是答案的话 target为1，否则为0
            if isinstance(answer_label, list):
                if ans_idx in example['answer_label']:
                    target = torch.tensor(1).cuda(self.device)
                else:
                    target = torch.tensor(0).cuda(self.device)
            else:
                if ans_idx == example['answer_label']:
                    target = torch.tensor(1).cuda(self.device)
                else:
                    target = torch.tensor(0).cuda(self.device)

            outputs.append((example['annot_id'],r_input_ids, r_segment_ids, r_input_mask, input_ids, segment_ids, input_mask, img_feat, img_mask, target,
                            chunk_mask, gather_index, offsets, example['sent'],
                            example['answer_choices'][ans_idx], total_label, align_pos))
        return tuple(outputs)

    def SNLIGPT_gen_collate(self, inputs):

        (img_id, r_input_ids, r_segment_ids, r_input_mask, input_ids, segment_ids, input_mask, img_feat, img_mask, target, chunk_mask,
         gather_index, offsets, ques, ans, total_label, align_pos) = map(list, unzip(concat(inputs)))

        r_input_ids = pad_sequence(r_input_ids, batch_first=True, padding_value=0)
        r_input_mask = pad_sequence(r_input_mask, batch_first=True, padding_value=0)
        r_segment_ids = pad_sequence(r_segment_ids, batch_first=True, padding_value=0)


        input_mask = pad_sequence(input_mask, batch_first=True, padding_value=0)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        total_label = pad_sequence(total_label, batch_first=True, padding_value=0)
        align_pos = pad_sequence(align_pos, batch_first=True, padding_value=0)

        segment_ids = pad_sequence(segment_ids, batch_first=True, padding_value=0)
        target = torch.stack(target, dim=0)
        target = target.type(torch.FloatTensor)
        target = target.cuda(self.device)

        img_mask = torch.stack(img_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)
        max_img = int(torch.max(torch.sum(img_mask, dim=-1)).item())
        img_mask = img_mask[:, :max_img]
        img_feat = img_feat[:, :max_img]

        input_mask = torch.cat((input_mask, img_mask), -1)

        max_hypo = input_ids.size(1)
        chunk_mask_padd = []
        for item in chunk_mask:
            padd_matrix = torch.zeros((max_hypo - item.size(0), item.size(1))).cuda(self.device)
            item = torch.cat((item, padd_matrix), 0)
            padd_matrix = torch.zeros((max_hypo, max_hypo - item.size(1))).cuda(self.device)
            item = torch.cat((item, padd_matrix), 1)
            chunk_mask_padd.append(item)
        chunk_mask_padd = torch.stack(chunk_mask_padd, 0)

        batch = {'img_id': img_id, 'r_input_ids':r_input_ids, "r_token_type_ids":r_segment_ids,
                 "r_attention_mask":r_input_mask,
                 'input_ids': input_ids, 'token_type_ids': segment_ids,
                 'input_mask': input_mask, 'img_feat': img_feat, 'label': target, 'ques_str': ques, 'ans_str': ans,
                 'chunk_attention_mask': chunk_mask_padd, 'gather_index': gather_index, 'offsets': offsets,
                 'total_label': total_label, 'align_pos': align_pos
                 }

        return batch


class VCR_ChunkAlign_Dataset_align_ensemble_gpt(Dataset):
    def __init__(self, bert_tokenizer, GPT_tokenizer, gpt2_tokenizer, VCR_example_file, VCR_chunk_mask_file,
                 flickr_feat_file,
                 roberta_example_file,
                 max_img_seq_length=50,
                 is_train=True,heat_index=None):
        self.bert_toker = bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.gpt2_toker = gpt2_tokenizer

        self.chunk_mask_dict = pickle.load(open(VCR_chunk_mask_file, 'rb'))
        self.VCR_annot_dict = pickle.load(open(VCR_example_file, 'rb'))
        self.roberta_annot_dict = pickle.load((open(roberta_example_file, 'rb'))) # additional
        self.image_feat_dict = self.read_example(flickr_feat_file)
        self.add_od_labels = True
        self.is_train = is_train
        self.max_img_seq_length = max_img_seq_length
        self.cls = self.bert_toker.cls_token
        self.sep = self.bert_toker.sep_token

        self.rcls = self.gpt2_toker.bos_token
        self.rsep = self.gpt2_toker.eos_token

        if heat_index is not None:
            # 只保存当前样例
            self.VCR_annot_dict = self.VCR_annot_dict[heat_index:heat_index + 1]


    def read_example(self, path):
        data = pickle.load(open(path, 'rb'))
        return data

    def __len__(self):
        return len(self.VCR_annot_dict)

    def __getitem__(self, i):
        example = self.VCR_annot_dict[i]

        roberta_example = self.roberta_annot_dict[i]

        id = example['image_id']
        num_id = id.split("-")[1]

        str_id = "img-" + str(num_id)

        answer_label = example['answer_label']

        image_feat = self.image_feat_dict[str_id]
        img_feat = image_feat['features'].cuda()
        img_mask = image_feat['img_mask'].cuda()

        que_tokens = example['sent'].lower()
        que_tokens = self.bert_toker.tokenize(que_tokens)
        outputs = []

        ## roberta
        r_que_tokens = roberta_example['sent'].lower()
        r_que_tokens = self.roberta_toker.tokenize(r_que_tokens)

        for ans_idx in range(len(example['answer_choices'])):

            r_ans = roberta_example['answer_choices'][ans_idx]
            r_ans_tokens = self.roberta_toker.tokenize(r_ans)
            r_input_tokens = [self.rcls] + r_que_tokens + [self.rsep] + r_ans_tokens + [self.rsep]
            r_input_ids = self.roberta_toker.convert_tokens_to_ids(r_input_tokens)
            r_input_ids = torch.tensor(r_input_ids).cuda()
            r_mask_len = r_input_ids.size(0)
            r_input_mask = torch.ones(r_mask_len).cuda()

            r_segment_ids_ques = torch.zeros(len(r_que_tokens) + 2, dtype=torch.int64).cuda()
            r_segment_ids_ans = torch.zeros(len(r_ans_tokens) + 1, dtype=torch.int64).cuda()
            r_segment_ids = torch.cat((r_segment_ids_ques, r_segment_ids_ans), 0)

            ans = example['answer_choices'][ans_idx]
            ans_tokens = self.bert_toker.tokenize(ans)
            input_tokens = [self.cls] + que_tokens + [self.sep] + ans_tokens + [self.sep]

            region_tokens = [0] * len(input_tokens)
            # region_tokens 为何物？
            # TODO
            for token_idx, token in enumerate(input_tokens):
                if '<|det' in token:
                    index = token[5:].split('|')[0]
                    region_tokens[token_idx] = int(index)
            region_tokens = torch.tensor(region_tokens).cuda()
            total_label = region_tokens
            align_pos = torch.where(total_label != 0, torch.ones_like(total_label).cuda(), total_label)

            input_ids = self.bert_toker.convert_tokens_to_ids(input_tokens)
            input_ids = torch.tensor(input_ids).cuda()
            mask_len = input_ids.size(0)
            input_mask = torch.ones(mask_len).cuda()

            segment_ids_ques = torch.zeros(len(que_tokens) + 2, dtype=torch.int64).cuda()
            segment_ids_ans = torch.ones(len(ans_tokens) + 1, dtype=torch.int64).cuda()
            segment_ids = torch.cat((segment_ids_ques, segment_ids_ans), 0)

            # chunk_mask_dict ?
            # TODO
            offsets = self.chunk_mask_dict[i][ans_idx]['offsets']
            chunk_mask = self.chunk_mask_dict[i][ans_idx]['mask'].cuda()

            gather_index = []
            for idx, set in enumerate(offsets):
                set = torch.tensor(set).cuda()
                gather_index.extend([idx] * set.size(0))
            gather_index = torch.tensor(gather_index).cuda()

            # 如果是答案的话 target为1，否则为0
            if isinstance(answer_label, list):
                if ans_idx in example['answer_label']:
                    target = torch.tensor(1).cuda()
                else:
                    target = torch.tensor(0).cuda()
            else:
                if ans_idx == example['answer_label']:
                    target = torch.tensor(1).cuda()
                else:
                    target = torch.tensor(0).cuda()

            outputs.append((example['annot_id'],r_input_ids, r_segment_ids, r_input_mask, input_ids, segment_ids, input_mask, img_feat, img_mask, target,
                            chunk_mask, gather_index, offsets, example['sent'],
                            example['answer_choices'][ans_idx], total_label, align_pos))
        return tuple(outputs)

    def SNLIGPT_gen_collate(self, inputs):

        (img_id, r_input_ids, r_segment_ids, r_input_mask, input_ids, segment_ids, input_mask, img_feat, img_mask, target, chunk_mask,
         gather_index, offsets, ques, ans, total_label, align_pos) = map(list, unzip(concat(inputs)))

        r_input_ids = pad_sequence(r_input_ids, batch_first=True, padding_value=0)
        r_input_mask = pad_sequence(r_input_mask, batch_first=True, padding_value=0)
        r_segment_ids = pad_sequence(r_segment_ids, batch_first=True, padding_value=0)


        input_mask = pad_sequence(input_mask, batch_first=True, padding_value=0)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        total_label = pad_sequence(total_label, batch_first=True, padding_value=0)
        align_pos = pad_sequence(align_pos, batch_first=True, padding_value=0)

        segment_ids = pad_sequence(segment_ids, batch_first=True, padding_value=0)
        target = torch.stack(target, dim=0)
        target = target.type(torch.FloatTensor)
        target = target.cuda()

        img_mask = torch.stack(img_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)
        max_img = int(torch.max(torch.sum(img_mask, dim=-1)).item())
        img_mask = img_mask[:, :max_img]
        img_feat = img_feat[:, :max_img]

        input_mask = torch.cat((input_mask, img_mask), -1)

        max_hypo = input_ids.size(1)
        chunk_mask_padd = []
        for item in chunk_mask:
            padd_matrix = torch.zeros((max_hypo - item.size(0), item.size(1))).cuda()
            item = torch.cat((item, padd_matrix), 0)
            padd_matrix = torch.zeros((max_hypo, max_hypo - item.size(1))).cuda()
            item = torch.cat((item, padd_matrix), 1)
            chunk_mask_padd.append(item)
        chunk_mask_padd = torch.stack(chunk_mask_padd, 0)

        batch = {'img_id': img_id, 'r_input_ids':r_input_ids, "r_token_type_ids":r_segment_ids,
                 "r_attention_mask":r_input_mask,
                 'input_ids': input_ids, 'token_type_ids': segment_ids,
                 'input_mask': input_mask, 'img_feat': img_feat, 'label': target, 'ques_str': ques, 'ans_str': ans,
                 'chunk_attention_mask': chunk_mask_padd, 'gather_index': gather_index, 'offsets': offsets,
                 'total_label': total_label, 'align_pos': align_pos
                 }

        return batch

class PMR_ChunkAlign_Dataset_align_ensemble_T(Dataset):
    def __init__(self, bert_tokenizer, GPT_tokenizer, roberta_tokenizer, VCR_example_file, VCR_chunk_mask_file,
                 flickr_feat_file,
                 roberta_example_file,
                 preprocess,
                 clip_example_file,
                 device,
                 max_img_seq_length=50,
                 is_train=True, heat_index=None):
        self.bert_toker = bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.roberta_toker = roberta_tokenizer
        self.chunk_mask_dict = pickle.load(open(VCR_chunk_mask_file, 'rb'))
        self.VCR_annot_dict = pickle.load(open(VCR_example_file, 'rb'))
        self.roberta_annot_dict = pickle.load((open(roberta_example_file, 'rb'))) # additional
        self.image_feat_dict = self.read_example(flickr_feat_file)
        self.add_od_labels = True
        self.is_train = is_train
        self.max_img_seq_length = max_img_seq_length
        self.cls = self.bert_toker.cls_token
        self.sep = self.bert_toker.sep_token
        self.rcls = self.roberta_toker.bos_token
        self.rsep = self.roberta_toker.eos_token
        self.preprocess = preprocess
        self.device = device

        #with open(clip_example_file, 'r') as file:
        #    lines = file.readlines()
        #    self.clip_annot_dict = [json.loads(line) for line in lines]

        if heat_index is not None:

            self.VCR_annot_dict = self.VCR_annot_dict[heat_index:heat_index + 1]


    def read_example(self, path):
        data = pickle.load(open(path, 'rb'))
        return data

    def __len__(self):
        return len(self.VCR_annot_dict)

    def __getitem__(self, i):

        image_ori_path = "pmr_data/images/"


        example = self.VCR_annot_dict[i]
        roberta_example = self.roberta_annot_dict[i]
        #clip_example = self.clip_annot_dict[i]

        # --------------CLIP---------------
        #image_path = image_ori_path + clip_example["img_fn"]
        #image = Image.open(image_path).convert('RGB')
        #image_input = self.preprocess(image)

        # --------------CALeC---------------
        id = example['image_id']
        num_id = id.split("-")[1]
        str_id = "img-" + str(num_id)
        try:
            answer_label = example['answer_label']
        except:
            answer_label = 0


        image_feat = self.image_feat_dict[str_id]
        img_feat = image_feat['features'].cuda(self.device)
        img_mask = image_feat['img_mask'].cuda(self.device)
        que_tokens = example['sent'].lower()
        premise_tokens = self.bert_toker.tokenize(que_tokens)
        que_tokens = image_feat['objects']

        outputs = []

        ## roberta
        #prompt_text = 'Hypothesis Text is correct or wrong? Conditions: Image Description is <mask>, ' + \
        #              'Bridge between Image and the following texts is <mask>. Premise Text is '
        prompt_text = 'Is Answer correct or wrong based on the Conditions? Conditions: Image Description is <mask>, ' + \
                      'Bridge between Image and the following texts is <mask>, Premise Text is '
        # prompt_text = 'Is Answer correct or wrong based on the Conditions? Conditions: Image Description is <mask>, ' + \
        #               'Premise Text is '
        r_que_tokens = roberta_example['sent'].lower()
        # r_que_tokens = self.roberta_toker.tokenize(r_que_tokens) # promptfuse
        r_que_tokens = self.roberta_toker.tokenize(prompt_text + r_que_tokens)

        for ans_idx in range(len(example['answer_choices'])):
            # ------------CLIP------------
            #clip_ans = clip_example['answer_choices'][ans_idx]
            #str_ans = ""
            #for x in clip_ans:
            #    if isinstance(x, list):
            #        id = x[0]
            #        str_ans += clip_example['objects'][id]
            #    else:
            #        str_ans += x + " "
            #clip_ans_tokens = clip.tokenize(str_ans)
            # ------------Roberta----------
            r_ans = roberta_example['answer_choices'][ans_idx]
            r_ans_tokens = self.roberta_toker.tokenize('Answer is ' + ' '.join(r_ans.split(' , ')))
            #r_ans_tokens = self.roberta_toker.tokenize(' '.join(r_ans.split(' , ')))
            r_input_tokens = [self.rcls] + r_que_tokens + [self.rsep] + r_ans_tokens + [self.rsep]
            r_input_ids = self.roberta_toker.convert_tokens_to_ids(r_input_tokens)
            r_input_ids = torch.tensor(r_input_ids).cuda(self.device)
            r_mask_len = r_input_ids.size(0)
            r_input_mask = torch.ones(r_mask_len).cuda(self.device)
            r_segment_ids_ques = torch.zeros(len(r_que_tokens) + 2, dtype=torch.int64).cuda(self.device)
            r_segment_ids_ans = torch.zeros(len(r_ans_tokens) + 1, dtype=torch.int64).cuda(self.device)
            r_segment_ids = torch.cat((r_segment_ids_ques, r_segment_ids_ans), 0)

            ans = example['answer_choices'][ans_idx]
            ans_tokens = self.bert_toker.tokenize(ans)
            input_tokens = [self.cls] + premise_tokens + [self.sep] + ans_tokens + [self.sep]
            #input_tokens = [self.cls] + ans_tokens + [self.sep]
            region_tokens = [0] * len(input_tokens)
            #
            # TODO
            for token_idx, token in enumerate(input_tokens):
                if '<|det' in token:
                    index = token[5:].split('|')[0]
                    region_tokens[token_idx] = int(index)
            region_tokens = torch.tensor(region_tokens).cuda(self.device)
            total_label = region_tokens
            align_pos = torch.where(total_label != 0, torch.ones_like(total_label).cuda(self.device), total_label)
            input_ids = self.bert_toker.convert_tokens_to_ids(input_tokens)
            input_ids = torch.tensor(input_ids).cuda(self.device)
            mask_len = input_ids.size(0)
            input_mask = torch.ones(mask_len).cuda(self.device)
            segment_ids_ques = torch.zeros(len(premise_tokens) + 2, dtype=torch.int64).cuda(self.device)
            segment_ids_ans = torch.ones(len(ans_tokens) + 1, dtype=torch.int64).cuda(self.device)
            segment_ids = torch.cat((segment_ids_ques, segment_ids_ans), 0)
            #segment_ids = torch.zeros(len(ans_tokens) + 2, dtype=torch.int64).cuda(self.device)
            #segment_ids[-len(que_tokens) - 1:] = 1
            # chunk_mask_dict ?
            # TODO
            offsets = self.chunk_mask_dict[i][ans_idx]['offsets']
            chunk_mask = self.chunk_mask_dict[i][ans_idx]['mask'].cuda(self.device)
            gather_index = []
            for idx, set in enumerate(offsets):
                set = torch.tensor(set).cuda(self.device)
                gather_index.extend([idx] * set.size(0))
            gather_index = torch.tensor(gather_index).cuda(self.device)

            if isinstance(answer_label, list):
                if ans_idx in example['answer_label']:
                    target = torch.tensor(1).cuda(self.device)
                else:
                    target = torch.tensor(0).cuda(self.device)
            else:
                if ans_idx == example['answer_label']:
                    target = torch.tensor(1).cuda(self.device)
                else:
                    target = torch.tensor(0).cuda(self.device)
            image_input =""
            clip_ans_tokens = ""
            outputs.append((example['annot_id'], image_input, clip_ans_tokens,
                            r_input_ids, r_segment_ids, r_input_mask, input_ids, segment_ids, input_mask, img_feat, img_mask, target,
                            chunk_mask, gather_index, offsets, example['sent'],
                            example['answer_choices'][ans_idx], total_label, align_pos))
        return tuple(outputs)

    def SNLIGPT_gen_collate(self, inputs):

        (img_id, image, text, r_input_ids, r_segment_ids, r_input_mask, input_ids, segment_ids, input_mask, img_feat, img_mask, target, chunk_mask,
         gather_index, offsets, ques, ans, total_label, align_pos) = map(list, unzip(concat(inputs)))

        r_input_ids = pad_sequence(r_input_ids, batch_first=True, padding_value=0)
        r_input_mask = pad_sequence(r_input_mask, batch_first=True, padding_value=0)
        r_segment_ids = pad_sequence(r_segment_ids, batch_first=True, padding_value=0)

        input_mask = pad_sequence(input_mask, batch_first=True, padding_value=0)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        total_label = pad_sequence(total_label, batch_first=True, padding_value=0)
        align_pos = pad_sequence(align_pos, batch_first=True, padding_value=0)

        segment_ids = pad_sequence(segment_ids, batch_first=True, padding_value=0)
        target = torch.stack(target, dim=0)
        target = target.type(torch.FloatTensor)
        target = target.cuda(self.device)
        img_mask = torch.stack(img_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)
        max_img = int(torch.max(torch.sum(img_mask, dim=-1)).item())
        img_mask = img_mask[:, :max_img]
        img_feat = img_feat[:, :max_img]
        input_mask = torch.cat((input_mask, img_mask), -1)
        max_hypo = input_ids.size(1)
        chunk_mask_padd = []
        for item in chunk_mask:
            padd_matrix = torch.zeros((max_hypo - item.size(0), item.size(1))).cuda(self.device)
            item = torch.cat((item, padd_matrix), 0)
            padd_matrix = torch.zeros((max_hypo, max_hypo - item.size(1))).cuda(self.device)
            item = torch.cat((item, padd_matrix), 1)
            chunk_mask_padd.append(item)
        chunk_mask_padd = torch.stack(chunk_mask_padd, 0)

        # ---------------CLIP
        #r_image = []
        #for i in range(0, len(image), 4):
        #    r_image.append(image[i])
        #image = torch.stack(r_image, dim=0).cuda(self.device)
        #text = torch.stack(text, dim=0).cuda(self.device)
        # ---------------CLIP
        
        batch = {'img_id': img_id, "image" : None, "text":None, 
                 'r_input_ids':r_input_ids, "r_token_type_ids":r_segment_ids,
                 "r_attention_mask":r_input_mask,
                 'input_ids': input_ids, 'token_type_ids': segment_ids,
                 'input_mask': input_mask, 'img_feat': img_feat, 'label': target, 'ques_str': ques, 'ans_str': ans,
                 'chunk_attention_mask': chunk_mask_padd, 'gather_index': gather_index, 'offsets': offsets,
                 'total_label': total_label, 'align_pos': align_pos
                 }

        return batch


class VCR_only_ChunkAlign_Dataset_align_ensemble_T(Dataset):
    def __init__(self, bert_tokenizer, GPT_tokenizer, roberta_tokenizer, VCR_example_file, VCR_chunk_mask_file,
                 flickr_feat_file,
                 roberta_example_file,
                 preprocess,
                 clip_example_file,
                 device,
                 max_img_seq_length=50,
                 is_train=True, heat_index=None):
        self.bert_toker = bert_tokenizer
        self.gpt_toker = GPT_tokenizer
        self.roberta_toker = roberta_tokenizer
        self.chunk_mask_dict = pickle.load(open(VCR_chunk_mask_file, 'rb'))
        self.VCR_annot_dict = pickle.load(open(VCR_example_file, 'rb'))
        self.roberta_annot_dict = pickle.load((open(roberta_example_file, 'rb'))) # additional
        self.image_feat_dict = self.read_example(flickr_feat_file)
        self.add_od_labels = True
        self.is_train = is_train
        self.max_img_seq_length = max_img_seq_length
        self.cls = self.bert_toker.cls_token
        self.sep = self.bert_toker.sep_token
        self.rcls = self.roberta_toker.bos_token
        self.rsep = self.roberta_toker.eos_token
        self.preprocess = preprocess
        self.device = device

        #with open(clip_example_file, 'r') as file:
        #    lines = file.readlines()
        #    self.clip_annot_dict = [json.loads(line) for line in lines]

        if heat_index is not None:
            self.VCR_annot_dict = self.VCR_annot_dict[heat_index:heat_index + 1]


    def read_example(self, path):
        data = pickle.load(open(path, 'rb'))
        return data

    def __len__(self):
        return len(self.VCR_annot_dict)

    def __getitem__(self, i):
        image_ori_path = "vcr_data/vcr1images/"
        # image_ori_path = 'pmr_data/images/'
        example = self.VCR_annot_dict[i]
        roberta_example = self.roberta_annot_dict[i]
        clip_example = self.clip_annot_dict[i]

        # --------------CLIP---------------
        #image_path = image_ori_path + clip_example["img_fn"]
        #image = Image.open(image_path).convert('RGB')
        #image_input = self.preprocess(image)

        # --------------CALeC---------------
        id = example['image_id']
        num_id = id.split("-")[1]
        str_id = "img-" + str(num_id)
        try:
            answer_label = example['answer_label']
        except:
            answer_label = 0


        image_feat = self.image_feat_dict[str_id]
        img_feat = image_feat['features'].cuda(self.device)
        img_mask = image_feat['img_mask'].cuda(self.device)
        que_tokens = example['sent'].lower()
        premise_tokens = self.bert_toker.tokenize(que_tokens)
        que_tokens = image_feat['objects']

        outputs = []

        ## roberta
        #prompt_text = 'Hypothesis Text is correct or wrong? Conditions: Image Description is <mask>, ' + \
        #              'Bridge between Image and the following texts is <mask>. Premise Text is '
        prompt_text = 'Is Answer correct or wrong based on the Conditions? Conditions: Image Description is <mask>, ' + \
                      'Bridge between Image and the following texts is <mask>, Premise Text is '
        # prompt_text = 'Is Answer correct or wrong based on the Conditions? Conditions: Image Description is <mask>, ' + \
        #              'Premise Text is '
        r_que_tokens = roberta_example['sent'].lower()
        r_que_tokens = self.roberta_toker.tokenize(prompt_text + r_que_tokens)
        #r_que_tokens = self.roberta_toker.tokenize(r_que_tokens)
        for ans_idx in range(len(example['answer_choices'])):
            # ------------CLIP------------
            #clip_ans = clip_example['answer_choices'][ans_idx]
            #str_ans = ""
            #for x in clip_ans:
            #    if isinstance(x, list):
            #        id = x[0]
            #        str_ans += clip_example['objects'][id]
            #    else:
            #        str_ans += x + " "
            #clip_ans_tokens = clip.tokenize(str_ans)

            # ------------Roberta----------
            r_ans = roberta_example['answer_choices'][ans_idx]
            r_ans_tokens = self.roberta_toker.tokenize('Answer is ' + r_ans)
            r_input_tokens = [self.rcls] + r_que_tokens + [self.rsep] + r_ans_tokens + [self.rsep]
            r_input_ids = self.roberta_toker.convert_tokens_to_ids(r_input_tokens)
            r_input_ids = torch.tensor(r_input_ids).cuda(self.device)
            r_mask_len = r_input_ids.size(0)
            r_input_mask = torch.ones(r_mask_len).cuda(self.device)
            r_segment_ids_ques = torch.zeros(len(r_que_tokens) + 2, dtype=torch.int64).cuda(self.device)
            r_segment_ids_ans = torch.zeros(len(r_ans_tokens) + 1, dtype=torch.int64).cuda(self.device)
            r_segment_ids = torch.cat((r_segment_ids_ques, r_segment_ids_ans), 0)

            ans = example['answer_choices'][ans_idx]
            if r_ans in ans:
                ans_tmp = ans.split(r_ans)[1]
                ans = r_ans + ' '.join(ans_tmp.split()[:10])
            ans_tokens = self.bert_toker.tokenize(ans)
            input_tokens = [self.cls] + premise_tokens + [self.sep] + ans_tokens + [self.sep]
            #input_tokens = [self.cls] + ans_tokens + [self.sep]
            region_tokens = [0] * len(input_tokens)
            # TODO
            for token_idx, token in enumerate(input_tokens):
                if '<|det' in token:
                    index = token[5:].split('|')[0]
                    region_tokens[token_idx] = int(index)
            region_tokens = torch.tensor(region_tokens).cuda(self.device)
            total_label = region_tokens
            align_pos = torch.where(total_label != 0, torch.ones_like(total_label).cuda(self.device), total_label)
            input_ids = self.bert_toker.convert_tokens_to_ids(input_tokens)
            input_ids = torch.tensor(input_ids).cuda(self.device)
            mask_len = input_ids.size(0)
            input_mask = torch.ones(mask_len).cuda(self.device)
            segment_ids_ques = torch.zeros(len(premise_tokens) + 2, dtype=torch.int64).cuda(self.device)
            segment_ids_ans = torch.ones(len(ans_tokens) + 1, dtype=torch.int64).cuda(self.device)
            segment_ids = torch.cat((segment_ids_ques, segment_ids_ans), 0)
            #segment_ids = torch.zeros(len(ans_tokens) + 2, dtype=torch.int64).cuda(self.device)
            #segment_ids[-len(que_tokens) - 1:] = 1
            # chunk_mask_dict ?
            # TODO
            offsets = self.chunk_mask_dict[i][ans_idx]['offsets']
            chunk_mask = self.chunk_mask_dict[i][ans_idx]['mask'].cuda(self.device)
            gather_index = []
            for idx, set in enumerate(offsets):
                set = torch.tensor(set).cuda(self.device)
                gather_index.extend([idx] * set.size(0))
            gather_index = torch.tensor(gather_index).cuda(self.device)
            #
            if isinstance(answer_label, list):
                if ans_idx in example['answer_label']:
                    target = torch.tensor(1).cuda(self.device)
                else:
                    target = torch.tensor(0).cuda(self.device)
            else:
                if ans_idx == example['answer_label']:
                    target = torch.tensor(1).cuda(self.device)
                else:
                    target = torch.tensor(0).cuda(self.device)
            image_input = ""
            clip_ans_tokens = ""
            outputs.append((example['annot_id'], image_input, clip_ans_tokens,
                            r_input_ids, r_segment_ids, r_input_mask, input_ids, segment_ids, input_mask, img_feat, img_mask, target,
                            chunk_mask, gather_index, offsets, example['sent'],
                            example['answer_choices'][ans_idx], total_label, align_pos))
        return tuple(outputs)

    def SNLIGPT_gen_collate(self, inputs):

        (img_id, image, text, r_input_ids, r_segment_ids, r_input_mask, input_ids, segment_ids, input_mask, img_feat, img_mask, target, chunk_mask,
         gather_index, offsets, ques, ans, total_label, align_pos) = map(list, unzip(concat(inputs)))

        r_input_ids = pad_sequence(r_input_ids, batch_first=True, padding_value=0)
        r_input_mask = pad_sequence(r_input_mask, batch_first=True, padding_value=0)
        r_segment_ids = pad_sequence(r_segment_ids, batch_first=True, padding_value=0)

        input_mask = pad_sequence(input_mask, batch_first=True, padding_value=0)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        total_label = pad_sequence(total_label, batch_first=True, padding_value=0)
        align_pos = pad_sequence(align_pos, batch_first=True, padding_value=0)

        segment_ids = pad_sequence(segment_ids, batch_first=True, padding_value=0)
        target = torch.stack(target, dim=0)
        target = target.type(torch.FloatTensor)
        target = target.cuda(self.device)
        img_mask = torch.stack(img_mask, dim=0)
        img_feat = torch.stack(img_feat, dim=0)
        max_img = int(torch.max(torch.sum(img_mask, dim=-1)).item())
        img_mask = img_mask[:, :max_img]
        img_feat = img_feat[:, :max_img]
        input_mask = torch.cat((input_mask, img_mask), -1)
        max_hypo = input_ids.size(1)
        chunk_mask_padd = []
        for item in chunk_mask:
            padd_matrix = torch.zeros((max_hypo - item.size(0), item.size(1))).cuda(self.device)
            item = torch.cat((item, padd_matrix), 0)
            padd_matrix = torch.zeros((max_hypo, max_hypo - item.size(1))).cuda(self.device)
            item = torch.cat((item, padd_matrix), 1)
            chunk_mask_padd.append(item)
        chunk_mask_padd = torch.stack(chunk_mask_padd, 0)

        # ---------------CLIP
        #r_image = []
        #for i in range(0, len(image), 4):
        #    r_image.append(image[i])
        #image = torch.stack(r_image, dim=0).cuda(self.device)
        #text = torch.stack(text, dim=0).cuda(self.device)

        batch = {'img_id': img_id, "image" : image, "text":text, 'r_input_ids':r_input_ids, "r_token_type_ids":r_segment_ids,
                 "r_attention_mask":r_input_mask,
                 'input_ids': input_ids, 'token_type_ids': segment_ids,
                 'input_mask': input_mask, 'img_feat': img_feat, 'label': target, 'ques_str': ques, 'ans_str': ans,
                 'chunk_attention_mask': chunk_mask_padd, 'gather_index': gather_index, 'offsets': offsets,
                 'total_label': total_label, 'align_pos': align_pos
                 }

        return batch



