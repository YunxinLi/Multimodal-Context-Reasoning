
import logging
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from a_transformers.modeling_bert import (BertEmbeddings,
                                              BertSelfAttention, BertAttention, BertEncoder, BertLayer,
                                              BertSelfOutput, BertIntermediate, BertOutput,
                                              BertPooler, BertPreTrainedModel)

from torch.nn.utils.rnn import pad_sequence
from collections import UserDict
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, LogitsProcessorList, MinLengthLogitsProcessor,
                          BeamScorer)
from transformers.generation_stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation_logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from modeling.modeling_vcr_chunkalign_v10 import ChunkAlign_CLS_enc4_align_ensemble
from local_transformers.adapter_transformers.models.roberta import RobertaModel


# CALeC + RoBERTa
class dual_ensemble_model(nn.Module):
    def __init__(self, roberta_model, calec_model, num_labels=4):
        super(dual_ensemble_model, self).__init__()
        self.num_labels = num_labels

        self.calec = calec_model
        self.roberta = roberta_model

        self.classifier = nn.Linear(1024+768, 1)
        # self.cls_loss_fct = CrossEntropyLoss()

    def forward(self, roberta_input_ids, roberta_token_type_ids, roberta_attention_mask, input_ids, img_feat,
                input_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                encoder_history_states=None, offsets=None, chunk_attention_mask=None, gather_index=None, label=None
                ,align_pos=None, total_label=None):

        CALeC_encoder_output, align_loss = self.calec(input_ids=input_ids,img_feat=img_feat,input_mask=input_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, encoder_history_states=encoder_history_states, offsets=offsets, chunk_attention_mask=chunk_attention_mask, gather_index=gather_index, align_pos=align_pos, total_label=total_label)

        roberta_encoder_output = self.roberta(input_ids=roberta_input_ids, token_type_ids=roberta_token_type_ids,
                                              attention_mask=roberta_attention_mask)

        roberta_encoder_output = roberta_encoder_output[1]

        ensemble_encoder_output = torch.concat((CALeC_encoder_output,roberta_encoder_output),dim=-1)

        logits = self.classifier(ensemble_encoder_output)
        reshaped_logits = logits.view(-1, self.num_labels)

        loss = None
        if label is not None:
            loss_fct = CrossEntropyLoss()
            label = label.view(reshaped_logits.size())
            loss = loss_fct(reshaped_logits, label)

        return loss, align_loss, reshaped_logits

# CALeC + RoBERTa with add
class dual_ensemble_model_add(nn.Module):
    def __init__(self, roberta_model, calec_model, num_labels=4):
        super(dual_ensemble_model_add, self).__init__()
        self.num_labels = num_labels

        self.calec = calec_model
        self.roberta = roberta_model

        self.classifier_c = nn.Linear(768, 1)
        self.classifier_r = nn.Linear(1024, 1)
        # self.cls_loss_fct = CrossEntropyLoss()

    def forward(self, roberta_input_ids, roberta_token_type_ids, roberta_attention_mask, input_ids, img_feat,
                input_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                encoder_history_states=None, offsets=None, chunk_attention_mask=None, gather_index=None, label=None
                ,align_pos=None, total_label=None):

        CALeC_encoder_output, align_loss = self.calec(input_ids=input_ids,img_feat=img_feat,input_mask=input_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, encoder_history_states=encoder_history_states, offsets=offsets, chunk_attention_mask=chunk_attention_mask, gather_index=gather_index, align_pos=align_pos, total_label=total_label)

        roberta_encoder_output = self.roberta(input_ids=roberta_input_ids, token_type_ids=roberta_token_type_ids,
                                              attention_mask=roberta_attention_mask)

        roberta_encoder_output = roberta_encoder_output[1]

        # ensemble_encoder_output = torch.concat((CALeC_encoder_output,roberta_encoder_output),dim=-1)

        # logits = self.classifier(ensemble_encoder_output)

        logits_r = self.classifier_r(roberta_encoder_output)
        logits_c = self.classifier_c(CALeC_encoder_output)
        logits = logits_r + logits_c
        reshaped_logits = logits.view(-1, self.num_labels)

        loss = None
        if label is not None:
            loss_fct = CrossEntropyLoss()
            label = label.view(reshaped_logits.size())
            loss = loss_fct(reshaped_logits, label)

        return loss, align_loss, reshaped_logits

# CALeC + RoBERTa with pairwise loss
class dual_ensemble_model_pairwise(nn.Module):
    def __init__(self, roberta_model, calec_model, num_labels=4):
        super(dual_ensemble_model_pairwise, self).__init__()
        self.num_labels = num_labels

        self.calec = calec_model
        self.roberta = roberta_model
        self.margin = 0.5 ## 阈值
        self.classifier = nn.Linear(1024+768, 1)
        self.relu = nn.ReLU()
        # self.cls_loss_fct = CrossEntropyLoss()

    def forward(self, roberta_input_ids, roberta_token_type_ids, roberta_attention_mask, input_ids, img_feat,
                input_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                encoder_history_states=None, offsets=None, chunk_attention_mask=None, gather_index=None, label=None
                ,align_pos=None, total_label=None):

        CALeC_encoder_output, align_loss = self.calec(input_ids=input_ids,img_feat=img_feat,input_mask=input_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, encoder_history_states=encoder_history_states, offsets=offsets, chunk_attention_mask=chunk_attention_mask, gather_index=gather_index, align_pos=align_pos, total_label=total_label)

        roberta_encoder_output = self.roberta(input_ids=roberta_input_ids, token_type_ids=roberta_token_type_ids,
                                              attention_mask=roberta_attention_mask)

        roberta_encoder_output = roberta_encoder_output[1]

        ensemble_encoder_output = torch.concat((CALeC_encoder_output,roberta_encoder_output),dim=-1)

        logits = self.classifier(ensemble_encoder_output)
        reshaped_logits = logits.view(-1, self.num_labels)

        loss = None
        if label is not None:
            # loss_fct = CrossEntropyLoss()
            # label = label.view(reshaped_logits.size())
            # loss = loss_fct(reshaped_logits, label)
            # 正样本的位置
            position = torch.nonzero(label)
            # 正样本的得分
            right_answer = logits[position]
            # 重复
            right_answer = right_answer.repeat(1,1,4)
            # 拉平
            right_answer = right_answer.reshape(-1)
            logits = logits.reshape(-1)
            # 阈值
            m = torch.tensor([self.margin]).to(logits.device)
            m = m.repeat(right_answer.size())
            # 相减
            hinge_loss = m + logits - right_answer
            # 去负
            hinge_loss = self.relu(hinge_loss)
            # 求和
            loss = torch.sum(hinge_loss)



        return loss, align_loss, reshaped_logits


# CALeC + RoBERTa with pairwise loss
class dual_ensemble_model_doubleloss(nn.Module):
    def __init__(self, roberta_model, calec_model, num_labels=4):
        super(dual_ensemble_model_doubleloss, self).__init__()
        self.num_labels = num_labels

        self.calec = calec_model
        self.roberta = roberta_model
        self.margin = 0.5 ## 阈值
        self.classifier = nn.Linear(1024+768, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        # self.cls_loss_fct = CrossEntropyLoss()

    def forward(self, roberta_input_ids, roberta_token_type_ids, roberta_attention_mask, input_ids, img_feat,
                input_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                encoder_history_states=None, offsets=None, chunk_attention_mask=None, gather_index=None, label=None
                ,align_pos=None, total_label=None):

        CALeC_encoder_output, align_loss = self.calec(input_ids=input_ids,img_feat=img_feat,input_mask=input_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, encoder_history_states=encoder_history_states, offsets=offsets, chunk_attention_mask=chunk_attention_mask, gather_index=gather_index, align_pos=align_pos, total_label=total_label)

        roberta_encoder_output = self.roberta(input_ids=roberta_input_ids, token_type_ids=roberta_token_type_ids,
                                              attention_mask=roberta_attention_mask)

        roberta_encoder_output = roberta_encoder_output[1]

        ensemble_encoder_output = torch.concat((CALeC_encoder_output,roberta_encoder_output),dim=-1)

        logits = self.classifier(ensemble_encoder_output)
        reshaped_logits = logits.view(-1, self.num_labels)


        loss = None
        if label is not None:
            # 正样本的位置
            logits = logits.view(-1, self.num_labels)
            logits = self.softmax(logits)
            logits = logits.view(-1, 1)

            position = torch.nonzero(label)
            # 正样本的得分
            right_answer = logits[position]
            # 重复
            right_answer = right_answer.repeat(1,1,4)
            # 拉平
            right_answer = right_answer.reshape(-1)
            logits = logits.reshape(-1)
            # 阈值
            m = torch.tensor([self.margin]).to(logits.device)
            m = m.repeat(right_answer.size())
            # 相减
            hinge_loss = m + logits - right_answer
            # 去负
            hinge_loss = self.relu(hinge_loss)
            # 求和
            hinge_loss = torch.sum(hinge_loss)

            loss_fct = CrossEntropyLoss()
            label = label.view(reshaped_logits.size())
            normal_loss = loss_fct(reshaped_logits, label)

            loss = hinge_loss + normal_loss

        return loss, align_loss, reshaped_logits


# CALeC + gpt2
class dual_ensemble_model_gpt(nn.Module):
    def __init__(self, gpt_model, calec_model, num_labels=4):
        super(dual_ensemble_model_gpt, self).__init__()
        self.num_labels = num_labels

        self.calec = calec_model
        self.gpt = gpt_model

        self.classifier = nn.Linear(768+768, 1)
        # self.cls_loss_fct = CrossEntropyLoss()

    def forward(self, gpt_input_ids, gpt_token_type_ids, gpt_attention_mask, input_ids, img_feat,
                input_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                encoder_history_states=None, offsets=None, chunk_attention_mask=None, gather_index=None, label=None
                ,align_pos=None, total_label=None):

        CALeC_encoder_output, align_loss = self.calec(input_ids=input_ids,img_feat=img_feat,input_mask=input_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, encoder_history_states=encoder_history_states, offsets=offsets, chunk_attention_mask=chunk_attention_mask, gather_index=gather_index, align_pos=align_pos, total_label=total_label)

        gpt_output = self.gpt(input_ids=gpt_input_ids, token_type_ids=gpt_token_type_ids,
                                              attention_mask=gpt_attention_mask)

        gpt_output = gpt_output[0]

        gpt_output = gpt_output[:,0,:]

        ensemble_encoder_output = torch.concat((CALeC_encoder_output,gpt_output),dim=-1)

        logits = self.classifier(ensemble_encoder_output)
        reshaped_logits = logits.view(-1, self.num_labels)

        loss = None
        if label is not None:
            loss_fct = CrossEntropyLoss()
            label = label.view(reshaped_logits.size())
            loss = loss_fct(reshaped_logits, label)

        return loss, align_loss, reshaped_logits


# clip + RoBERTa
# 使用概率乘的方式进行
class dual_ensemble_model_clip(nn.Module):
    def __init__(self, roberta_model, clip_model, num_labels=4):
        super(dual_ensemble_model_clip, self).__init__()
        self.num_labels = num_labels

        self.clip_model = clip_model
        self.roberta = roberta_model

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        # self.classifier = nn.Linear(768, 1)
        # self.cls_loss_fct = CrossEntropyLoss()

    def forward(self, input_ids, token_type_ids, attention_mask, image, text, label=None):


        # roberta_encoder_output = self.roberta(input_ids=input_ids, token_type_ids=token_type_ids,
        #                                       attention_mask=attention_mask)
        # roberta_encoder_output = roberta_encoder_output[1]
        # logits = self.classifier(roberta_encoder_output)
        # logits = logits.view(-1, 4)

        logits = self.roberta(input_ids=input_ids, token_type_ids=token_type_ids,
                                              attention_mask=attention_mask)[0]

        image_features_ = self.clip_model.encode_image(image)
        text_features_ = self.clip_model.encode_text(text.squeeze(1)).view(-1, 4, 512)
        image_features = image_features_ / image_features_.norm(dim=-1, keepdim=True)
        text_features = text_features_ / text_features_.norm(dim=-1, keepdim=True)

        if label is not None:
            # similarity = (text_features @ image_features.unsqueeze(2)).squeeze() * self.clip_model.logit_scale.exp()
            similarity = (text_features @ image_features.unsqueeze(2)).squeeze()
        else:
            similarity = (text_features @ image_features.unsqueeze(2)).squeeze()

        # 相加，计算得分
        # logits_ = self.softmax(logits)
        # logits = self.relu(logits)
        # similarity_ = self.softmax(similarity)
        # similarity = self.relu(similarity)


        scores = (logits + similarity)/2



        loss = None
        if label is not None:
            loss_fct = CrossEntropyLoss()
            label = label.view(scores.size())
            loss = loss_fct(scores, label)
        return loss, scores
        # reshaped_logits = logits.view(-1, self.num_labels)
        #
        # loss = None
        # if label is not None:
        #     loss_fct = CrossEntropyLoss()
        #     label = label.view(reshaped_logits.size())
        #     loss = loss_fct(reshaped_logits, label)
        #
        # return loss, align_loss, reshaped_logits
# with model CALeC, RoBERTa and CLIP
class ensemble_model_t(nn.Module):
    def __init__(self, roberta_model, calec_model, clip_model, num_labels=4):
        super(ensemble_model_t, self).__init__()
        self.num_labels = num_labels

        self.calec = calec_model
        self.roberta = roberta_model
        self.clip_model = clip_model

        self.classifier = nn.Linear(1024+768+512, 1)
        # self.cls_loss_fct = CrossEntropyLoss()

    def forward(self, image, text, roberta_input_ids, roberta_token_type_ids, roberta_attention_mask, input_ids, img_feat,
                input_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                encoder_history_states=None, offsets=None, chunk_attention_mask=None, gather_index=None, label=None
                ,align_pos=None, total_label=None):

        # CALeC model
        CALeC_encoder_output, align_loss = self.calec(input_ids=input_ids,img_feat=img_feat,input_mask=input_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, encoder_history_states=encoder_history_states, offsets=offsets, chunk_attention_mask=chunk_attention_mask, gather_index=gather_index, align_pos=align_pos, total_label=total_label)

        # RoBERTa model
        roberta_encoder_output = self.roberta(input_ids=roberta_input_ids, token_type_ids=roberta_token_type_ids,
                                              attention_mask=roberta_attention_mask)
        roberta_encoder_output = roberta_encoder_output[1]

        # CLIP model
        image_features_ = self.clip_model.encode_image(image)
        text_features_ = self.clip_model.encode_text(text.squeeze(1)).view(-1, 4, 512)
        image_features = image_features_ / image_features_.norm(dim=-1, keepdim=True)
        text_features = text_features_ / text_features_.norm(dim=-1, keepdim=True)

        image_features = image_features.repeat(1,4)
        image_features = image_features.view(image_features_.size(0), 4, -1)

        # 放大1000倍
        clip_encoder_output = image_features * text_features * 1000

        clip_encoder_output = clip_encoder_output.view(-1, text_features.size(-1))


        # 将CALeC模型和RoBERTa模型的cls端特征拼接，以增强多模态模型的推理能力
        ensemble_encoder_output = torch.concat((CALeC_encoder_output, roberta_encoder_output, clip_encoder_output),dim=-1)
        logits = self.classifier(ensemble_encoder_output)
        reshaped_logits = logits.view(-1, self.num_labels)

        loss = None
        if label is not None:
            loss_fct = CrossEntropyLoss()
            label = label.view(reshaped_logits.size())
            loss = loss_fct(reshaped_logits, label)

        return loss, align_loss, reshaped_logits


class PrefixMLP(nn.Module):
    def __init__(self, feature_size, hidden_size, prefix_len, hidden_dropout_prob):
        super(PrefixMLP, self).__init__()

        self.dense0 = nn.Linear(feature_size, (hidden_size * prefix_len) // 2)
        self.dense1 = nn.Linear((hidden_size * prefix_len) // 2, hidden_size * prefix_len)
        self.tanh = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor):
        # 512 -> 10240/2
        hidden_states = self.dense0(hidden_states)
        hidden_states = self.tanh(hidden_states)
        # 10240/2 -> 10240
        hidden_states = self.dense1(hidden_states)
        return hidden_states

# with model CALeC, RoBERTa
class Abstract_Specific(nn.Module):
    def __init__(self, roberta_model, calec_model, clip_model=None, num_labels=4):
        super(Abstract_Specific, self).__init__()
        self.num_labels = num_labels

        self.calec = calec_model
        self.roberta = roberta_model
        if clip_model is not None:
            self.clip_model = clip_model
            self.classifier = nn.Linear(1024 + 768 + 512, 1)
        else:
            self.classifier = nn.Linear(768 + 768, 1)

        self.abst_confidence_scorer = nn.Linear(1024, 1)
        self.confidence_scorer = nn.Linear(768, 1)
        self.mapping_network_alignment = nn.Sequential(
            nn.Dropout(p=0.1),
            torch.nn.Linear(768, 768 * 5, bias=True),
            torch.nn.Tanh(),
            nn.Dropout(p=0.1),
            torch.nn.Linear(768 * 5, 1024 * 5, bias=True)
            # torch.nn.ReLU(),
            # nn.Dropout(p=0.1),
            # torch.nn.Linear(512*2, 1, bias=True),
            # torch.nn.Sigmoid()
        )

        self.mapping_network_vision = nn.Sequential(
            nn.Dropout(p=0.1),
            torch.nn.Linear(768, 768 * 5, bias=True),
            torch.nn.Tanh(),
            nn.Dropout(p=0.1),
            torch.nn.Linear(768 * 5, 1024 * 5, bias=True)
        )
        self.promptfuse = torch.nn.Embedding(2, 1024)
    def forward(self, image, text, roberta_input_ids, roberta_token_type_ids, roberta_attention_mask, input_ids, img_feat,
                input_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                encoder_history_states=None, offsets=None, chunk_attention_mask=None, gather_index=None, label=None
                ,align_pos=None, total_label=None):


        # vision representations
        with torch.no_grad():
            img_attention_mask = torch.cat([input_mask[:, :1], input_mask[:, -img_feat.size(1):]], dim=-1)
            image_features_ = self.calec.global_enc(input_ids[:, :1], img_feats=img_feat, attention_mask=img_attention_mask,
                                      position_ids=None, token_type_ids=None,
                                      head_mask=None,
                                      encoder_history_states=None)

        prefix_vision = self.mapping_network_vision(image_features_[0][:, 0, :])
        prefix_vision = prefix_vision.reshape(input_ids.size(0), 5, 1024)
        vision_mask = input_mask[:, :1].repeat(1, 5)

        #### PromptFuse
        # align_embeddings = self.promptfuse(torch.arange(2).to(input_ids.device)).unsqueeze(0).repeat(input_ids.size(0), 1, 1)
        # align_mask = input_mask[:, :2]
        # prefix_emb = torch.cat([align_embeddings, prefix_vision], dim=1)
        # prompt_mask = torch.cat([align_mask, vision_mask], dim=1)

        # Alignment model with mapping network
        CALeC_encoder_output, align_loss, specific_alignment = self.calec(input_ids=input_ids, img_feat=img_feat,
                                                                          input_mask=input_mask, token_type_ids=token_type_ids,
                                                                          position_ids=position_ids, head_mask=head_mask,
                                                                          encoder_history_states=encoder_history_states, offsets=offsets,
                                                                          chunk_attention_mask=chunk_attention_mask, gather_index=gather_index,
                                                                          align_pos=align_pos, total_label=total_label,
                                                                          abstract_hidden_states=None)


        Alignment_prompt = self.mapping_network_alignment(CALeC_encoder_output).unsqueeze(1).view(input_ids.size(0), 5, 1024)
        align_mask = input_mask[:, :1].repeat(1, 5)
        
        ##### visual + align
        prefix_emb = torch.cat([prefix_vision, Alignment_prompt], dim=1)
        prompt_mask = torch.cat([vision_mask, align_mask], dim=1)

        # RoBERTa model
        roberta_encoder_outputs = self.roberta(input_ids=roberta_input_ids, token_type_ids=roberta_token_type_ids,
                                              attention_mask=roberta_attention_mask, prompt_embeddings=prefix_emb, input_mask=prompt_mask)
        roberta_encoder_output = roberta_encoder_outputs[1]

        ########## multi-view reasoning path
        abstract_level = roberta_encoder_output
        #chunk_level = CALeC_encoder_output
        #ensemble_encoder_output = torch.concat((CALeC_encoder_output, specific_alignment[1]), dim=-1)
        #logits = self.classifier(ensemble_encoder_output)
        #specific_logits = self.confidence_scorer(chunk_level)
        abst_logit = self.abst_confidence_scorer(abstract_level)

        '''
        # CLIP model
        image_features_ = self.clip_model.encode_image(image)
        text_features_ = self.clip_model.encode_text(text.squeeze(1)).view(-1, 4, 512)
        image_features = image_features_ / image_features_.norm(dim=-1, keepdim=True)
        text_features = text_features_ / text_features_.norm(dim=-1, keepdim=True)

        image_features = image_features.repeat(1,4)
        image_features = image_features.view(image_features_.size(0), 4, -1)

        # 放大1000倍
        clip_encoder_output = image_features * text_features * 1000

        clip_encoder_output = clip_encoder_output.view(-1, text_features.size(-1))
        '''
        reshaped_logits = abst_logit.view(-1, self.num_labels)
        loss = None
        loss_specific = None
        loss_abstract = None
        align_f_loss = None
        if label is not None:
            loss_fct = CrossEntropyLoss()
            label = label.view(reshaped_logits.size())
            loss = loss_fct(reshaped_logits, label)
            loss_abstract = loss_fct(abst_logit.view(reshaped_logits.size()), label)
            #loss_specific = loss_fct(specific_logits.view(reshaped_logits.size()), label)
        return loss, (None, loss_specific, loss_abstract, align_f_loss), reshaped_logits



class ensemble_model_t1(nn.Module):
    def __init__(self, roberta_model, calec_model, clip_model, num_labels=4):
        super(ensemble_model_t1, self).__init__()
        self.num_labels = num_labels

        self.calec = calec_model
        self.roberta = roberta_model
        self.clip_model = clip_model

        self.classifier = nn.Linear(1024 + 768, 1)
        # self.cls_loss_fct = CrossEntropyLoss()

    def forward(self, image, text, roberta_input_ids, roberta_token_type_ids, roberta_attention_mask, input_ids, img_feat,
                input_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                encoder_history_states=None, offsets=None, chunk_attention_mask=None, gather_index=None, label=None
                ,align_pos=None, total_label=None):

        # CALeC model
        CALeC_encoder_output, align_loss = self.calec(input_ids=input_ids,img_feat=img_feat,input_mask=input_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, encoder_history_states=encoder_history_states, offsets=offsets, chunk_attention_mask=chunk_attention_mask, gather_index=gather_index, align_pos=align_pos, total_label=total_label)

        # RoBERTa model
        roberta_encoder_output = self.roberta(input_ids=roberta_input_ids, token_type_ids=roberta_token_type_ids,
                                              attention_mask=roberta_attention_mask)
        roberta_encoder_output = roberta_encoder_output[1]

        # CLIP model
        image_features_ = self.clip_model.encode_image(image)
        text_features_ = self.clip_model.encode_text(text.squeeze(1)).view(-1, 4, 512)
        image_features = image_features_ / image_features_.norm(dim=-1, keepdim=True)
        text_features = text_features_ / text_features_.norm(dim=-1, keepdim=True)
        # CLIP余弦相似度计算
        similarity = (text_features @ image_features.unsqueeze(2)).squeeze()

        score, position = similarity.topk(k=2, dim=-1)
        score = score.mean(dim=-1)
        position = position.type(torch.IntTensor)
        position = position.cpu().numpy().tolist()
        clip_info = torch.ones([text_features.size(0), text_features.size(1)]).type(torch.FloatTensor)

        for idx in range(len(position)):
            poi = position[idx]
            for x in poi:
                clip_info[idx][x]=score[idx]
        clip_info = clip_info.view(clip_info.size(0)*clip_info.size(1),-1).to(similarity.device)
        # 将CALeC模型和RoBERTa模型的cls端特征拼接，以增强多模态模型的推理能力
        ensemble_encoder_output = torch.concat((CALeC_encoder_output, roberta_encoder_output),dim=-1)

        ensemble_encoder_output = clip_info*ensemble_encoder_output
        logits = self.classifier(ensemble_encoder_output)
        reshaped_logits = logits.view(-1, self.num_labels)

        loss = None
        if label is not None:
            loss_fct = CrossEntropyLoss()
            label = label.view(reshaped_logits.size())
            loss = loss_fct(reshaped_logits, label)

        return loss, align_loss, reshaped_logits

# 第三种融合方法，相加
class ensemble_model_t2(nn.Module):
    def __init__(self, roberta_model, calec_model, clip_model, num_labels=4):
        super(ensemble_model_t2, self).__init__()
        self.num_labels = num_labels

        self.calec = calec_model
        self.roberta = roberta_model
        self.clip_model = clip_model

        self.classifier = nn.Linear(1024+768,1)
        # self.cls_loss_fct = CrossEntropyLoss()

    def forward(self, image, text, roberta_input_ids, roberta_token_type_ids, roberta_attention_mask, input_ids, img_feat,
                input_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                encoder_history_states=None, offsets=None, chunk_attention_mask=None, gather_index=None, label=None
                ,align_pos=None, total_label=None):

        # CALeC model
        CALeC_encoder_output, align_loss = self.calec(input_ids=input_ids,img_feat=img_feat,input_mask=input_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, encoder_history_states=encoder_history_states, offsets=offsets, chunk_attention_mask=chunk_attention_mask, gather_index=gather_index, align_pos=align_pos, total_label=total_label)

        # RoBERTa model
        roberta_encoder_output = self.roberta(input_ids=roberta_input_ids, token_type_ids=roberta_token_type_ids,
                                              attention_mask=roberta_attention_mask)
        roberta_encoder_output = roberta_encoder_output[1]

        # CLIP model
        image_features_ = self.clip_model.encode_image(image)
        text_features_ = self.clip_model.encode_text(text.squeeze(1)).view(-1, 4, 512)
        image_features = image_features_ / image_features_.norm(dim=-1, keepdim=True)
        text_features = text_features_ / text_features_.norm(dim=-1, keepdim=True)
        # CLIP余弦相似度计算
        similarity = (text_features @ image_features.unsqueeze(2)).squeeze()

        score, position = similarity.topk(k=2, dim=-1)
        score = score.mean(dim=-1)
        position = position.type(torch.IntTensor)
        position = position.cpu().numpy().tolist()
        clip_info = torch.zeros([text_features.size(0), text_features.size(1)]).type(torch.FloatTensor)
        for idx in range(len(position)):
            poi = position[idx]
            for x in poi:
                clip_info[idx][x] = score[idx]
        clip_info = clip_info.view(clip_info.size(0)*clip_info.size(1), -1).to(similarity.device)
        # 将CALeC模型和RoBERTa模型的cls端特征拼接，以增强多模态模型的推理能力
        ensemble_encoder_output = torch.concat((CALeC_encoder_output, roberta_encoder_output),dim=-1)

        logits = self.classifier(ensemble_encoder_output)
        # 与clip模态的融合
        logits = logits + clip_info
        reshaped_logits = logits.view(-1, self.num_labels)

        loss = None
        if label is not None:
            loss_fct = CrossEntropyLoss()
            label = label.view(reshaped_logits.size())
            loss = loss_fct(reshaped_logits, label)

        return loss, align_loss, reshaped_logits

# 第四种融合方法
class ensemble_model_t3(nn.Module):
    def __init__(self, roberta_model, calec_model, clip_model, num_labels=4):
        super(ensemble_model_t3, self).__init__()
        self.num_labels = num_labels

        self.calec = calec_model
        self.roberta = roberta_model
        self.clip_model = clip_model

        self.classifier = nn.Linear(1024+768,1)
        self.adder = nn.Linear(2,1)
        # self.cls_loss_fct = CrossEntropyLoss()

    def forward(self, image, text, roberta_input_ids, roberta_token_type_ids, roberta_attention_mask, input_ids, img_feat,
                input_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                encoder_history_states=None, offsets=None, chunk_attention_mask=None, gather_index=None, label=None
                ,align_pos=None, total_label=None):

        # CALeC model
        CALeC_encoder_output, align_loss = self.calec(input_ids=input_ids,img_feat=img_feat,input_mask=input_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, encoder_history_states=encoder_history_states, offsets=offsets, chunk_attention_mask=chunk_attention_mask, gather_index=gather_index, align_pos=align_pos, total_label=total_label)

        # RoBERTa model
        roberta_encoder_output = self.roberta(input_ids=roberta_input_ids, token_type_ids=roberta_token_type_ids,
                                              attention_mask=roberta_attention_mask)
        roberta_encoder_output = roberta_encoder_output[1]

        # CLIP model
        image_features_ = self.clip_model.encode_image(image)
        text_features_ = self.clip_model.encode_text(text.squeeze(1)).view(-1, 4, 512)
        image_features = image_features_ / image_features_.norm(dim=-1, keepdim=True)
        text_features = text_features_ / text_features_.norm(dim=-1, keepdim=True)
        # CLIP余弦相似度计算
        similarity = (text_features @ image_features.unsqueeze(2)).squeeze()

        score, position = similarity.topk(k=2, dim=-1)
        score = score.mean(dim=-1)
        position = position.type(torch.IntTensor)
        position = position.cpu().numpy().tolist()
        clip_info = torch.zeros([text_features.size(0), text_features.size(1)]).type(torch.FloatTensor)
        for idx in range(len(position)):
            poi = position[idx]
            for x in poi:
                clip_info[idx][x] = score[idx]
        clip_info = clip_info.view(clip_info.size(0)*clip_info.size(1), -1).to(similarity.device)
        # 将CALeC模型和RoBERTa模型的cls端特征拼接，以增强多模态模型的推理能力
        ensemble_encoder_output = torch.concat((CALeC_encoder_output, roberta_encoder_output),dim=-1)

        logits = self.classifier(ensemble_encoder_output)

        logits = torch.concat((logits, clip_info), dim=-1)

        # 给来自CLIP的分类结果和来自 融合模型的分类结果各自学习一个权重
        logits = self.adder(logits)

        reshaped_logits = logits.view(-1, self.num_labels)

        loss = None
        if label is not None:
            loss_fct = CrossEntropyLoss()
            label = label.view(reshaped_logits.size())
            loss = loss_fct(reshaped_logits, label)

        return loss, align_loss, reshaped_logits

# 第四种融合方法
class ensemble_model_t4(nn.Module):
    def __init__(self, roberta_model, calec_model, clip_model, num_labels=4):
        super(ensemble_model_t4, self).__init__()
        self.num_labels = num_labels

        self.calec = calec_model
        self.roberta = roberta_model
        self.clip_model = clip_model

        self.classifier = nn.Linear(1024+768,1)
        # self.adder = nn.Linear(2,1)
        # self.cls_loss_fct = CrossEntropyLoss()

    def forward(self, image, text, roberta_input_ids, roberta_token_type_ids, roberta_attention_mask, input_ids, img_feat,
                input_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                encoder_history_states=None, offsets=None, chunk_attention_mask=None, gather_index=None, label=None
                ,align_pos=None, total_label=None):

        # CALeC model
        CALeC_encoder_output, align_loss = self.calec(input_ids=input_ids,img_feat=img_feat,input_mask=input_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, encoder_history_states=encoder_history_states, offsets=offsets, chunk_attention_mask=chunk_attention_mask, gather_index=gather_index, align_pos=align_pos, total_label=total_label)

        # RoBERTa model
        roberta_encoder_output = self.roberta(input_ids=roberta_input_ids, token_type_ids=roberta_token_type_ids,
                                              attention_mask=roberta_attention_mask)
        roberta_encoder_output = roberta_encoder_output[1]

        # CLIP model
        image_features_ = self.clip_model.encode_image(image)
        text_features_ = self.clip_model.encode_text(text.squeeze(1)).view(-1, 4, 512)
        image_features = image_features_ / image_features_.norm(dim=-1, keepdim=True)
        text_features = text_features_ / text_features_.norm(dim=-1, keepdim=True)
        # CLIP余弦相似度计算
        similarity = (text_features @ image_features.unsqueeze(2)).squeeze()

        # score, position = similarity.topk(k=2, dim=-1)
        # score = score.mean(dim=-1)
        # position = position.type(torch.IntTensor)
        # position = position.cpu().numpy().tolist()
        # clip_info = torch.zeros([text_features.size(0), text_features.size(1)]).type(torch.FloatTensor)
        # for idx in range(len(position)):
        #     poi = position[idx]
        #     for x in poi:
        #         clip_info[idx][x] = score[idx]
        # clip_info = clip_info.view(clip_info.size(0)*clip_info.size(1), -1).to(similarity.device)
        # 将CALeC模型和RoBERTa模型的cls端特征拼接，以增强多模态模型的推理能力
        ensemble_encoder_output = torch.concat((CALeC_encoder_output, roberta_encoder_output),dim=-1)

        logits = self.classifier(ensemble_encoder_output)

        logits = logits + similarity.view(logits.size())

        # 给来自CLIP的分类结果和来自 融合模型的分类结果各自学习一个权重
        # logits = self.adder(logits)

        reshaped_logits = logits.view(-1, self.num_labels)

        loss = None
        if label is not None:
            loss_fct = CrossEntropyLoss()
            label = label.view(reshaped_logits.size())
            loss = loss_fct(reshaped_logits, label)

        return loss, align_loss, reshaped_logits

# clip_model with another way to fusion image and text
class clip_model(nn.Module):
    def __init__(self, clip_, num_labels=4):
        super(clip_model, self).__init__()
        self.num_labels = num_labels
        self.clip = clip_
        self.easy_fusion = nn.Linear(1024, 512)
        self.classifier = nn.Linear(512, 1)


    def forward(self, image, text, label=None):

        image_features = self.clip.encode_image(image)
        text_features = self.clip.encode_text(text.squeeze(1))

        image_features = image_features.repeat(1, 4)
        image_features = image_features.view(image_features.size(0)*self.num_labels, -1)

        fusion_feature = torch.concat([image_features, text_features],dim=-1)
        fusion_feature = fusion_feature.type(torch.FloatTensor).to(image_features.device)
        fusion_feature = self.easy_fusion(fusion_feature)
        logits = self.classifier(fusion_feature)
        reshaped_logits = logits.view(-1, self.num_labels)

        # loss = None
        # if label is not None:
        #     loss_fct = CrossEntropyLoss()
        #     label = label.view(reshaped_logits.size())
        #     loss = loss_fct(reshaped_logits, label)
        #
        return reshaped_logits

class clip_model_r(nn.Module):
    def __init__(self, clip_, num_labels=4):
        super(clip_model_r, self).__init__()
        self.num_labels = num_labels
        self.clip = clip_
        self.classifier = nn.Linear(512, 1)


    def forward(self, image, text, label=None):

        image_features_ = self.clip.encode_image(image)
        text_features_ = self.clip.encode_text(text.squeeze(1)).view(-1, 4, 512)
        image_features = image_features_ / image_features_.norm(dim=-1, keepdim=True)
        text_features = text_features_ / text_features_.norm(dim=-1, keepdim=True)

        image_features = image_features.repeat(1, 4)
        image_features = image_features.view(image_features_.size(0), 4, -1)

        # 放大1000倍
        clip_encoder_output = image_features * text_features * 1000

        clip_encoder_output = clip_encoder_output.view(-1, text_features.size(-1))

        clip_encoder_output = clip_encoder_output.type(torch.FloatTensor).to(image_features.device)
        logits = self.classifier(clip_encoder_output)
        reshaped_logits = logits.view(-1, self.num_labels)
        # loss = None
        # if label is not None:
        #     loss_fct = CrossEntropyLoss()
        #     label = label.view(reshaped_logits.size())
        #     loss = loss_fct(reshaped_logits, label)
        #
        return reshaped_logits


class model_vote(nn.Module):
    def __init__(self):
        super(model_vote, self).__init__()
        self.vote = nn.Linear(8,1)

    def forward(self, input_ids):
        # input_ids  bs * 8 * 4
        logits = self.vote(input_ids)

        return logits

