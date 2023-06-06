from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import math
import torch
from torch import nn
from typing import Optional, Tuple, List
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

BertLayerNorm = torch.nn.LayerNorm
logger = logging.getLogger(__name__)


class CaptionBertSelfAttention(BertSelfAttention):
    """
    Modified from BertSelfAttention to add support for output_hidden_states.
    """

    def __init__(self, config):
        super(CaptionBertSelfAttention, self).__init__(config)
        self.output_attentions = config.output_attentions
        self.hidden_size = config.hidden_size

    def forward(self, hidden_states, attention_mask, head_mask=None,
                history_state=None, do_chunk_cross=False, offsets=None, gather_index=None):
        if history_state is not None:
            x_states = torch.cat([history_state, hidden_states], dim=1)
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(x_states)
            mixed_value_layer = self.value(x_states)
        else:
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)
        if do_chunk_cross == True:
            mixed_query_layer_new = mixed_query_layer.clone()
            # 用chunk的平均值作为查询
            for ba, offset in enumerate(offsets):
                sent_len = gather_index[ba].size(0)
                chunk = torch.zeros((len(offset), self.hidden_size)).cuda(hidden_states.device)
                chunk_hidden = mixed_query_layer[ba, 1:sent_len + 1]
                chunk = torch.index_add(chunk, 0, gather_index[ba], chunk_hidden)
                chunk_len = torch.tensor([len(item) for item in offset]).cuda(hidden_states.device)
                chunk_mean = chunk / chunk_len.unsqueeze(-1)
                mixed_query_layer_new[ba, 1:sent_len + 1] = torch.gather(chunk_mean, 0,
                                                                         gather_index[ba].unsqueeze(-1).repeat(1,self.hidden_size))
            mixed_query_layer = mixed_query_layer_new
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class CaptionBertAttention(BertAttention):
    """
    Modified from BertAttention to add support for output_hidden_states.
    """

    def __init__(self, config):
        super(CaptionBertAttention, self).__init__(config)
        self.self = CaptionBertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, head_mask=None,
                history_state=None, do_chunk_cross=False, offsets=None, gather_index=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask=head_mask, history_state=history_state,
                                 do_chunk_cross=do_chunk_cross, offsets=offsets, gather_index=gather_index)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class CaptionBertLayer(BertLayer):
    """
    Modified from BertLayer to add support for output_hidden_states.
    """

    def __init__(self, config):
        super(CaptionBertLayer, self).__init__(config)
        self.attention = CaptionBertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None,
                history_state=None, do_chunk_cross=False, offsets=None, gather_index=None):
        attention_outputs = self.attention(hidden_states, attention_mask,
                                           head_mask=head_mask, history_state=history_state,
                                           do_chunk_cross=do_chunk_cross, offsets=offsets, gather_index=gather_index)
        attention_output = attention_outputs[0]

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


class CaptionBertEncoder(BertEncoder):
    """
    Modified from BertEncoder to add support for output_hidden_states.
    """

    def __init__(self, config):
        super(CaptionBertEncoder, self).__init__(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([CaptionBertLayer(config) for _ in range(config.num_hidden_layers)])
        self.num_hidden_layers = config.num_hidden_layers
        self.add_residual = config.add_residual
        self.add_local_residual = config.add_local_residual
        self.chunk_attention_layers = [0, 1, 2, ]
        self.cross_chunk_attention_layers = [3, 4, 5, 6, 7, 8]
        self.cross_modal_layers = [9, 10, 11]
        self.max_hypo = config.max_hypo

    def forward(self, hidden_states, chunk_attention_mask, gather_index, img_mask, input_mask, hypo_len,
                img_len,
                head_mask=None, encoder_history_states=None,
                offsets=None):
        all_hidden_states = ()
        all_attentions = ()
        do_chunk_cross = False
        # 初始attention，只看到chunk内部和image
        attention_mask = input_mask.repeat(1, 1, hypo_len + img_len, 1)
        attention_mask[:, :, :hypo_len, :hypo_len] = chunk_attention_mask
        # attention_mask[:, :, :hypo_len, hypo_len:] = -10000.0
        # image只看到image之间
        attention_mask[:, :, hypo_len:, :hypo_len] = -10000.0
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            history_state = None if encoder_history_states is None else encoder_history_states[i]

            if i == self.cross_chunk_attention_layers[0]:
                attention_mask = input_mask
                # do_chunk_cross = True
            elif i in self.cross_modal_layers:
                # cross modal时取mean chunk
                do_chunk_cross = True
                if i == self.cross_modal_layers[0]:
                    chunk_hidden_states = hidden_states
                    # 图片部分仅自己可见
                    img_mask = torch.eye(img_len)
                    img_mask = img_mask.unsqueeze(0).repeat(attention_mask.size(0), 1, 1)
                    img_mask = torch.cat((torch.zeros(attention_mask.size(0), img_len, hypo_len), img_mask), -1)
                    img_mask = (1.0 - img_mask) * -10000.0
                    attention_mask = attention_mask.repeat(1, 1, hypo_len + img_len, 1)
                    attention_mask[:, :, hypo_len:, ] = img_mask.unsqueeze(1)
                    # 文本部分仅可见chunk 内部 和image
                    attention_mask[:, :, :hypo_len, :hypo_len] = chunk_attention_mask

            layer_outputs = layer_module(
                hidden_states, attention_mask,
                head_mask=head_mask[i], history_state=history_state, do_chunk_cross=do_chunk_cross, offsets=offsets,
                gather_index=gather_index)
            if self.add_local_residual and i in self.cross_modal_layers:
                # cross阶段加入残差
                former_hidden = hidden_states
                hidden_states = layer_outputs[0] + former_hidden
            else:
                hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if self.add_residual:
            # cross attention阶段可能会损失文本信息
            hidden_states = hidden_states + chunk_hidden_states
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs, chunk_hidden_states  # outputs, (hidden states), (attentions)


class SeqBertImgModel(BertPreTrainedModel):
    """ Expand from BertModel to handle image region features as input
    """

    def __init__(self, config):
        super(SeqBertImgModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = CaptionBertEncoder(config)
        self.pooler = BertPooler(config)
        self.img_dim = config.img_feature_dim
        logger.info('BertImgModel Image Dimension: {}'.format(self.img_dim))
        self.img_feature_type = config.img_feature_type
        if hasattr(config, 'use_img_layernorm'):
            self.use_img_layernorm = config.use_img_layernorm
        else:
            self.use_img_layernorm = None

        self.img_embedding = nn.Linear(self.img_dim, self.config.hidden_size, bias=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if self.use_img_layernorm:
            self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.img_layer_norm_eps)

        self.init_weights()
        self.max_hypo = config.max_hypo
        self.edge_dense = nn.Embedding(1, config.hidden_size)

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, input_mask=None,
                position_ids=None, head_mask=None, img_feats=None, img_mask=None,
                encoder_history_states=None, offsets=None, gather_index=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if img_mask.dim() == 2:
            extended_img_mask = img_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_img_mask = img_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        extended_img_mask = (1.0 - extended_img_mask) * -10000.0

        if input_mask.dim() == 2:
            extended_input_mask = input_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_input_mask = input_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        extended_input_mask = (1.0 - extended_input_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                    -1)  # We can specify head_mask for each layer
            # switch to float if needed + fp16 compatibility
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids,
                                           token_type_ids=token_type_ids)
        if encoder_history_states:
            assert img_feats is None, "Cannot take image features while using encoder history states"

        img_embedding_output = self.img_embedding(img_feats)
        if self.use_img_layernorm:
            img_embedding_output = self.LayerNorm(img_embedding_output)
        # add dropout on image embedding
        img_embedding_output = self.dropout(img_embedding_output)

        embedding_output = torch.cat((embedding_output, img_embedding_output), 1)
        hypo_len = input_ids.size(1)
        img_len = img_feats.size(1)
        encoder_outputs, chunk_hidden_states = self.encoder(embedding_output, extended_attention_mask,
                                                            gather_index=gather_index,
                                                            hypo_len=hypo_len, img_len=img_len,
                                                            img_mask=extended_img_mask,
                                                            input_mask=extended_input_mask,
                                                            head_mask=head_mask, offsets=offsets,
                                                            encoder_history_states=encoder_history_states)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        return outputs, chunk_hidden_states


def binary_to_mp(logit, num_labels=4):
    """
    convert binary vcr logits to 4-way multiple choice classification logits
    """
    sm = nn.Softmax(dim=1)

    logit = sm(logit)
    logit = logit[:, 1]  # get the values for answer being true of all pairs
    logit = logit.reshape(-1, num_labels)  # group them into 4's

    return logit


class BaseLine_cls_xe(nn.Module):
    def __init__(self, oscar, num_labels):
        super(BaseLine_cls_xe, self).__init__()
        self.oscar = oscar
        self.dropout = nn.Dropout(self.oscar.config.hidden_dropout_prob)
        self.num_labels = num_labels
        self.classifier = nn.Linear(self.oscar.config.hidden_size, 2)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids, img_feat, input_mask=None, label=None,
                token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                ):
        outputs = self.oscar(input_ids, img_feats=img_feat, attention_mask=input_mask,
                             position_ids=position_ids, token_type_ids=token_type_ids,
                             head_mask=head_mask,
                             encoder_history_states=encoder_history_states)
        oscar_CLS = outputs[1]

        logits = self.classifier(oscar_CLS)
        loss_cls_0 = self.loss_fct(logits.view(-1, 2), label)
        logit_expl = binary_to_mp(logits)
        pre = logit_expl.max(dim=-1)[1]
        label = label.reshape(-1, 4)
        label = torch.argmax(label, -1)
        matched_0 = pre == label

        return loss_cls_0, matched_0

    def evaluate(self, input_ids, img_feat, input_mask=None, label=None, attn_mask=None,
                 token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                 ):
        outputs = self.oscar(input_ids, img_feats=img_feat, attention_mask=input_mask,
                             position_ids=position_ids, token_type_ids=token_type_ids,
                             head_mask=head_mask,
                             encoder_history_states=encoder_history_states)

        oscar_CLS = outputs[1]
        logits = self.classifier(oscar_CLS)
        logit_expl = binary_to_mp(logits)
        pre = logit_expl.max(dim=-1)[1]
        label = label.reshape(-1, 4)
        label = torch.argmax(label, -1)
        matched_0 = pre == label

        return matched_0, pre


class BaseLine(nn.Module):
    def __init__(self, global_enc, dec, dec_toker, num_labels):
        super(BaseLine, self).__init__()
        self.oscar = global_enc
        self.dec_toker = dec_toker
        self.dec = dec

        self.num_labels = num_labels
        self.classifier = nn.Linear(self.oscar.config.hidden_size, 2)
        self.cls_loss_fct = nn.CrossEntropyLoss()

        self.lm_head = nn.Linear(self.dec.config.n_embd, self.dec.config.vocab_size, bias=False)
        self.gen_criterion = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=self.dec_toker.pad_token_id)
        self.max_gen_len = 50
        self.b_rtnl = torch.tensor(self.dec_toker.encode("<|b_rtnl|>")).cuda()

    def forward(self, input_ids, img_feat, expl_ids, input_mask=None, label=None, attn_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                ):
        outputs = self.oscar(input_ids, img_feats=img_feat, attention_mask=input_mask,
                             position_ids=position_ids, token_type_ids=token_type_ids,
                             head_mask=head_mask,
                             encoder_history_states=encoder_history_states)
        global_output = outputs[0]
        global_CLS = outputs[1]

        logits = self.classifier(global_CLS)
        loss_cls_0 = self.cls_loss_fct(logits.view(-1, 2), label)
        logit_expl = binary_to_mp(logits)
        pre = logit_expl.max(dim=-1)[1]
        label = label.reshape(-1, 4)
        label = torch.argmax(label, -1)
        matched_0 = pre == label

        encoder_hs = global_output.detach()
        encoder_mask = input_mask

        ques_num = label.size(0)
        add_tensor = torch.arange(ques_num, dtype=torch.int64).cuda()
        add_tensor = add_tensor * 4
        gather_index = add_tensor + label

        encoder_mask_chosen = torch.gather(encoder_mask, 0, gather_index.unsqueeze(-1).repeat(1, encoder_mask.size(-1)))
        encoder_hs_chosen = torch.gather(encoder_hs, 0,
                                         gather_index.unsqueeze(-1).unsqueeze(-1).repeat(1, encoder_hs.size(-2),
                                                                                         encoder_hs.size(-1)))

        expl_ids = expl_ids.reshape(ques_num, 4, -1)[:, 0, :]
        attn_mask = attn_mask.reshape(ques_num, 4, -1)[:, 0, :]

        gpt_out = self.dec(input_ids=expl_ids, attention_mask=attn_mask, encoder_hidden_states=encoder_hs_chosen,
                           encoder_attention_mask=encoder_mask_chosen)[0]
        lm_logits = self.lm_head(gpt_out)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = expl_ids[..., 1:].contiguous()
        gen_loss = self.gen_criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return gen_loss, loss_cls_0, matched_0

    def evaluate(self, input_ids, img_feat, expl_ids, input_mask=None, label=None,
                 token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None):
        outputs = self.oscar(input_ids, img_feats=img_feat, attention_mask=input_mask,
                             position_ids=position_ids, token_type_ids=token_type_ids,
                             head_mask=head_mask,
                             encoder_history_states=encoder_history_states)
        global_output = outputs[0]
        global_CLS = outputs[1]

        logits = self.classifier(global_CLS)
        logit_expl = binary_to_mp(logits)
        pre = logit_expl.max(dim=-1)[1]
        label = label.reshape(-1, 4)
        label = torch.argmax(label, -1)
        matched_0 = pre == label
        encoder_hs = global_output
        encoder_mask = input_mask
        ques_num = label.size(0)
        add_tensor = torch.arange(ques_num, dtype=torch.int64).cuda()
        add_tensor = add_tensor * 4
        gather_index = add_tensor + label
        encoder_mask_chosen = torch.gather(encoder_mask, 0, gather_index.unsqueeze(-1).repeat(1, encoder_mask.size(-1)))
        encoder_hs_chosen = torch.gather(encoder_hs, 0,
                                         gather_index.unsqueeze(-1).unsqueeze(-1).repeat(1, encoder_hs.size(-2),
                                                                                         encoder_hs.size(-1)))
        outputs = torch.full((encoder_hs_chosen.size(0), self.max_gen_len), fill_value=self.dec_toker.pad_token_id,
                             dtype=int).cuda()

        past_key_values = None
        cur_unfinished = outputs.new(outputs.size(0)).fill_(1)
        cur_len = 0
        tokens_to_add = torch.full((encoder_hs_chosen.size(0), 1),
                                   fill_value=self.dec_toker.bos_token_id).cuda().squeeze(-1)
        for index in range(self.max_gen_len - 1):
            gpt_out = self.dec(input_ids=tokens_to_add.unsqueeze(-1), encoder_hidden_states=encoder_hs_chosen,
                               encoder_attention_mask=encoder_mask_chosen, use_cache=True,
                               past_key_values=past_key_values)
            past_key_values = gpt_out.past_key_values
            gpt_out = gpt_out[0][:, -1:, :]
            # 只取最后一个作为当前步的输出
            lm_logits = self.lm_head(gpt_out)
            gen_label = torch.argmax(lm_logits, dim=-1).squeeze()
            tokens_to_add = gen_label * cur_unfinished + self.dec_toker.pad_token_id * (1 - cur_unfinished)
            outputs[:, index] = tokens_to_add
            cur_len += 1
            cur_unfinished = cur_unfinished.mul(tokens_to_add.ne(self.dec_toker.eos_token_id).long())
            if cur_unfinished.max() == 0:
                break
        if cur_len == self.max_gen_len:
            outputs[:, -1].masked_fill_(cur_unfinished.to(dtype=torch.bool), self.dec_toker.eos_token_id)
        return outputs, matched_0, pre


class Base_freeze(nn.Module):
    def __init__(self, global_enc, dec, dec_toker, num_labels):
        super(Base_freeze, self).__init__()
        self.oscar = global_enc
        self.dec_toker = dec_toker
        self.dec = dec

        self.num_labels = num_labels
        self.classifier = nn.Linear(self.oscar.config.hidden_size, 2)
        self.cls_loss_fct = nn.CrossEntropyLoss()

        self.vocab_num = self.dec.vocab_size
        self.lm_head = nn.Linear(self.dec.config.n_embd, self.dec.vocab_size, bias=False)
        self.gen_criterion = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=self.dec_toker.pad_token_id)
        self.max_gen_len = 50
        self.b_rtnl = torch.tensor(self.dec_toker.encode("<|b_rtnl|>")).cuda()

    def forward(self, input_ids, img_feat, expl_ids, input_mask=None, label=None, attn_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                ):
        hypo_len = input_ids.size(1)
        outputs = self.oscar(input_ids, img_feats=img_feat, attention_mask=input_mask,
                             position_ids=position_ids, token_type_ids=token_type_ids,
                             head_mask=head_mask,
                             encoder_history_states=encoder_history_states)
        global_output = outputs[0]
        global_CLS = outputs[1]

        logits = self.classifier(global_CLS)
        loss_cls_0 = self.cls_loss_fct(logits.view(-1, 2), label)
        logit_expl = binary_to_mp(logits)
        pre = logit_expl.max(dim=-1)[1]
        label = label.reshape(-1, 4)
        label = torch.argmax(label, -1)
        matched_0 = pre == label

        encoder_hs = global_output[:, 1:hypo_len].detach()
        encoder_mask = input_mask[:, 1:hypo_len]

        ques_num = label.size(0)
        add_tensor = torch.arange(ques_num, dtype=torch.int64).cuda()
        add_tensor = add_tensor * 4
        gather_index = add_tensor + label

        encoder_mask_chosen = torch.gather(encoder_mask, 0, gather_index.unsqueeze(-1).repeat(1, encoder_mask.size(-1)))
        encoder_hs_chosen = torch.gather(encoder_hs, 0,
                                         gather_index.unsqueeze(-1).unsqueeze(-1).repeat(1, encoder_hs.size(-2),
                                                                                         encoder_hs.size(-1)))

        expl_ids = expl_ids.reshape(ques_num, 4, -1)[:, 0, :]
        attn_mask = attn_mask.reshape(ques_num, 4, -1)[:, 0, :]

        gpt_out = self.dec(input_ids=expl_ids, attention_mask=attn_mask, encoder_hidden_states=encoder_hs_chosen,
                           encoder_attention_mask=encoder_mask_chosen)[0]
        lm_logits = self.lm_head(gpt_out)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = expl_ids[..., 1:].contiguous()
        gen_loss = self.gen_criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return gen_loss, loss_cls_0, matched_0

    def evaluate(self, input_ids, img_feat, expl_ids, input_mask=None, label=None,
                 token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None):
        hypo_len = input_ids.size(1)
        outputs = self.oscar(input_ids, img_feats=img_feat, attention_mask=input_mask,
                             position_ids=position_ids, token_type_ids=token_type_ids,
                             head_mask=head_mask,
                             encoder_history_states=encoder_history_states)
        global_output = outputs[0]
        global_CLS = outputs[1]

        logits = self.classifier(global_CLS)
        logit_expl = binary_to_mp(logits)
        pre = logit_expl.max(dim=-1)[1]
        label = label.reshape(-1, 4)
        label = torch.argmax(label, -1)
        matched_0 = pre == label
        encoder_hs = global_output[:, 1:hypo_len].detach()
        encoder_mask = input_mask[:, 1:hypo_len]
        ques_num = label.size(0)
        add_tensor = torch.arange(ques_num, dtype=torch.int64).cuda()
        add_tensor = add_tensor * 4
        gather_index = add_tensor + label
        encoder_mask_chosen = torch.gather(encoder_mask, 0, gather_index.unsqueeze(-1).repeat(1, encoder_mask.size(-1)))
        encoder_hs_chosen = torch.gather(encoder_hs, 0,
                                         gather_index.unsqueeze(-1).unsqueeze(-1).repeat(1, encoder_hs.size(-2),
                                                                                         encoder_hs.size(-1)))
        expl_ids = expl_ids.reshape(ques_num, 4, -1)[:, 0, :]
        outputs = torch.full((encoder_hs_chosen.size(0), self.max_gen_len), fill_value=self.dec_toker.pad_token_id,
                             dtype=int).cuda()

        for i in range(matched_0.size(0)):
            if matched_0[i] == torch.tensor(True):
                decoder_input = expl_ids[i]
                b_rtnl_index = torch.nonzero((decoder_input == self.b_rtnl.unsqueeze(0)).to(torch.int64))[0, -1]
                prompt_decoder_input = decoder_input[:b_rtnl_index]
                gpt_out = self.dec(input_ids=prompt_decoder_input.unsqueeze(0),
                                   encoder_hidden_states=encoder_hs_chosen[i].unsqueeze(0),
                                   encoder_attention_mask=encoder_mask_chosen[i].unsqueeze(0), use_cache=True,
                                   past_key_values=None)
                past_key_values = gpt_out[1]
                cur_unfinished = outputs.new(1).fill_(1)
                cur_len = 0
                tokens_to_add = decoder_input[b_rtnl_index].unsqueeze(-1)
                for index in range(self.max_gen_len - 1):
                    gpt_out = self.dec(input_ids=tokens_to_add.unsqueeze(0),
                                       encoder_hidden_states=encoder_hs_chosen[i].unsqueeze(0),
                                       encoder_attention_mask=encoder_mask_chosen[i].unsqueeze(0), use_cache=True,
                                       past_key_values=past_key_values)
                    past_key_values = gpt_out[1]
                    gpt_out = gpt_out[0][:, -1:, :]
                    # 只取最后一个作为当前步的输出
                    lm_logits = self.lm_head(gpt_out)
                    gen_label = torch.argmax(lm_logits, dim=-1).squeeze(-1)
                    tokens_to_add = gen_label * cur_unfinished + self.dec_toker.pad_token_id * (1 - cur_unfinished)
                    outputs[i, index] = tokens_to_add
                    cur_len += 1
                    cur_unfinished = cur_unfinished.mul(tokens_to_add.ne(self.dec_toker.eos_token_id).long())
                    if cur_unfinished.max() == 0:
                        break
                if cur_len == self.max_gen_len:
                    outputs[i, -1].masked_fill_(cur_unfinished.to(dtype=torch.bool), self.dec_toker.eos_token_id)
        return outputs, matched_0, pre


class cross_attention_lyx(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super(cross_attention_lyx, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder
        self.relu = torch.nn.ReLU()
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            neg_type: bool = False,
            tau=1.0,
            prior_score=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()
        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            # attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attention_mask = attention_mask.expand(-1, self.num_heads, -1, -1)
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(attention_mask, -10000.0)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if neg_type:
            attn_weights = 1.0 - torch.nn.functional.softmax(attn_weights / tau, dim=-1)
        else:
            attn_weights = torch.nn.functional.softmax(attn_weights / tau, dim=-1)
        if prior_score is not None:
            prior_score = prior_score.repeat(self.num_heads, 1, 1)
            attn_weights = attn_weights + prior_score
        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len,
                                                                                 src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights_reshaped, past_key_value



class ClsLayer2(BertLayer):
    """
    Modified from BertLayer to add support for output_hidden_states.
    """

    def __init__(self, config):
        super(ClsLayer2, self).__init__(config)
        self.cls_q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.align_k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, self_chunk_align, cls, word_mask, neg=False, tau=1.0):
        cls_q = self.cls_q_proj(cls.unsqueeze(1))
        self_chunk_align_k = self.align_k_proj(self_chunk_align)
        self_chunk_align_v = self_chunk_align_k.clone()
        self_chunk_align_k = self_chunk_align_k.permute(0, 2, 1)

        attn_weight = torch.matmul(cls_q, self_chunk_align_k)
        attn_weight = attn_weight + word_mask
        if neg:
            attn_weight = 1 - nn.Softmax(dim=-1)(attn_weight / tau)
        else:
            attn_weight = nn.Softmax(dim=-1)(attn_weight / tau)
        attn_weight = self.dropout(attn_weight)
        cls_attn_output = torch.matmul(attn_weight, self_chunk_align_v).squeeze(1)

        cls_attn_output = self.dense(cls_attn_output)
        cls_attn_output = self.dropout(cls_attn_output)
        cls_with_align = self.LayerNorm(cls_attn_output + cls)
        #cls_with_align = self.LayerNorm(cls_attn_output)
        intermediate_output = self.intermediate(cls_with_align)
        layer_output = self.output(intermediate_output, cls_with_align)
        return layer_output, attn_weight


class ClsLayer_lyx(BertLayer):
    """
    Modified from BertLayer to add support for output_hidden_states.
    """

    def __init__(self, config):
        super(ClsLayer_lyx, self).__init__(config)
        self.ensemble = nn.Linear(config.hidden_size * 2, 1)
        self.cross_attention = cross_attention_lyx(config.hidden_size, 8, dropout=0.1, is_decoder=True)
        #self.cross_attention_abs = cross_attention_lyx(config.hidden_size, 8, dropout=0.1, is_decoder=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, self_chunk_align, cls, word_mask=None, prior_score=None, cls_2=None):
        CLS_ensem_1 = self.cross_attention(cls.unsqueeze(1), self_chunk_align, tau=1.0, neg_type=False, prior_score=prior_score)
        #CLS_ensem_2 = self.cross_attention_abs(cls_2.unsqueeze(1), self_chunk_align, tau=1.0, neg_type=False, prior_score=prior_score)
        #CLS_ensem_F = self.cross_attention(cls.unsqueeze(1), self_chunk_align, tau=0.5, neg_type=True)
        #CLS_ensem_T_F = self.ensemble(torch.cat([CLS_ensem_T[0].squeeze(1), CLS_ensem_F[0].squeeze(1)], dim=-1))
        #cls_sig_1 = torch.sigmoid(self.ensemble(torch.cat([CLS_ensem_1[0].squeeze(1), cls], dim=-1)))
        #cls_sig_2 = torch.sigmoid(self.ensemble(torch.cat([CLS_ensem_2[0].squeeze(1), cls], dim=-1)))
        #cls_attn_output = self.dropout(cls_attn_output)
        #cls_with_align = self.LayerNorm(cls_attn_output + cls)
        #cls_with_align = cls_sig_1 * CLS_ensem_1[0].squeeze(1) + cls_sig_2 * CLS_ensem_2[0].squeeze(1) + cls
        cls_with_align = self.dropout(CLS_ensem_1[0].squeeze(1))
        cls_with_align = self.LayerNorm(cls_with_align + cls)
        intermediate_output = self.intermediate(cls_with_align)
        layer_output = self.output(intermediate_output, cls_with_align)
        return layer_output

class ChunkAlign_CLS_enc4_align_ensemble(nn.Module):
    def __init__(self, global_enc, seq_enc, num_labels):
        super(ChunkAlign_CLS_enc4_align_ensemble, self).__init__()
        self.global_enc = global_enc
        self.seq_enc = seq_enc
        self.cls_ensemble_1 = nn.Linear(self.global_enc.config.hidden_size + self.seq_enc.config.hidden_size,
                                       self.global_enc.config.hidden_size)
        #self.cls_ensemble_2 = nn.Linear(1024 + self.global_enc.config.hidden_size, 1)
        self.num_labels = num_labels
        self.cls_layer_num = 2
        self.cls_layer = nn.ModuleList([ClsLayer2(self.global_enc.config) for _ in range(self.cls_layer_num)])
        self.cls_layer_lyx = nn.ModuleList([ClsLayer_lyx(self.global_enc.config) for _ in range(self.cls_layer_num)])
        #self.cls_layer_lyx_2 = nn.ModuleList([ClsLayer_lyx(self.global_enc.config) for _ in range(self.cls_layer_num)])
        self.classifier = nn.Linear(self.global_enc.config.hidden_size, 2)
        self.fusion_align = nn.Linear(self.global_enc.config.hidden_size * 2, 1024)

        self.prior = nn.Linear(self.global_enc.config.hidden_size, 1)
        self.cls_loss_fct = CrossEntropyLoss()

    def forward(self, input_ids, img_feat, input_mask=None, label=None, token_type_ids=None, position_ids=None,
                head_mask=None, encoder_history_states=None, offsets=None, chunk_attention_mask=None,
                gather_index=None, align_pos=None, total_label=None, abstract_hidden_states=None):
        hypo_len = input_ids.size(1)
        with torch.no_grad():
            outputs = self.global_enc(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                      position_ids=position_ids, token_type_ids=token_type_ids,
                                      head_mask=head_mask,
                                      encoder_history_states=encoder_history_states)
            global_output = outputs[0]
            global_CLS = outputs[1]
            img_mask = input_mask[:, hypo_len:]
            seq_outputs, chunk_hidden_states = self.seq_enc(input_ids, img_feats=img_feat, img_mask=img_mask,
                                                            input_mask=input_mask,
                                                            attention_mask=chunk_attention_mask,
                                                            position_ids=position_ids, token_type_ids=token_type_ids,
                                                            head_mask=head_mask, offsets=offsets, gather_index=gather_index)
            chunk_CLS = seq_outputs[1]
            chunk_align = seq_outputs[0][:, 1:hypo_len]
            global_hypo = global_output[:, 1:hypo_len]
            chunk_hidden_states = chunk_hidden_states[:, 1:hypo_len]
        CLS_ensem = self.cls_ensemble_1(torch.cat((global_CLS, chunk_CLS), -1))
        self_chunk_align_ = torch.cat((global_hypo, chunk_align, chunk_hidden_states), dim=1)

        ###################### image-text pari representations
        '''
        global_hypo_ = global_hypo
        global_hypo_norm = global_hypo_ / global_hypo_.norm(dim=-1, keepdim=True)
        global_img_output = global_output[:, hypo_len:]
        global_img_output_ = global_img_output
        global_img_output_norm = global_img_output_ / global_img_output_.norm(dim=-1, keepdim=True)
        score_matrix = global_hypo_norm @ global_img_output_norm.transpose(1, 2)
        score_matrix_ = F.softmax(score_matrix / 0.25, dim=-1)
        seq_alignment_global_ = torch.cat([global_hypo, torch.bmm(score_matrix_, global_img_output)], dim=-1)
        seq_alignment_global = self.fusion_align(seq_alignment_global_)
        chunk_seq = torch.cat([chunk_align], dim=1)
        chunk_seq_ = chunk_seq
        chunk_seq_norm = chunk_seq_ / chunk_seq_.norm(dim=-1, keepdim=True)
        chunk_img_features = seq_outputs[0][:, hypo_len:]
        chunk_img_features_ = chunk_img_features
        chunk_img_features_norm = chunk_img_features_ / chunk_img_features_.norm(dim=-1, keepdim=True)
        score_matrix = chunk_seq_norm @ chunk_img_features_norm.transpose(1, 2)
        score_matrix_ = F.softmax(score_matrix / 0.25, dim=-1)
        chunk_alignment_info_ = torch.cat([chunk_seq, torch.bmm(score_matrix_, chunk_img_features)], dim=-1)
        chunk_alignment_info = self.fusion_align(chunk_alignment_info_)
        '''

        #prior_score = torch.sigmoid(self.prior(self_chunk_align_)).squeeze(-1).unsqueeze(1)
        #prior_score_ori = torch.sigmoid(self.prior(self_chunk_align_)).squeeze(-1) / 0.5
        #self_chunk_align = torch.cat((seq_alignment_global, chunk_alignment_info), dim=1)
        #self_chunk_align = torch.cat((chunk_align, chunk_hidden_states), dim=1)
        #self_token_align = global_hypo
        word_mask = input_mask[:, 1:hypo_len].unsqueeze(1)
        word_mask = (1.0 - word_mask) * -10000.0
        word_mask = torch.cat((word_mask, word_mask, word_mask), -1)  # batch,word*3,hidden
        #toekn_word_mask = word_mask # batch,word*1,hidden
        #chunk_word_mask = torch.cat((word_mask, word_mask), -1)  # batch,word*2,hidden
        '''
        padd_embedding = self.padding_embedding(torch.arange(1).to(chunk_CLS.device))
        chunk_level = chunk_align
        chunk_level_f = []
        max_length = 0
        for bs in range(len(offsets)):
            current = []
            cur_offset = offsets[bs]
            for _ in range(len(cur_offset)):
                if len(cur_offset[_]) > 1:
                    current.append(torch.mean(chunk_level[bs][cur_offset[_]], dim=0, keepdim=True))
            if len(current) > 0:
                chunk_level_f.append(torch.cat(current, dim=0))
            else:
                chunk_level_f.append(padd_embedding)
            if len(current) > max_length:
                max_length = len(current)
        chunk_level_f_tmp = []
        for _ in range(len(chunk_level_f)):
            if chunk_level_f[_].size(0) < max_length:
                chunk_level_f_tmp.append(torch.cat([chunk_level_f[_]] + (max_length - chunk_level_f[_].size(0)) *[padd_embedding], dim=0))
            else:
                chunk_level_f_tmp.append(chunk_level_f[_])
        CLS_ensem = torch.stack(chunk_level_f_tmp)
        '''

        Reasoning_path = []
        CLS_ensem_new = None
        for i, layer_module in enumerate(self.cls_layer_lyx):
            CLS_ensem = layer_module(self_chunk_align_, CLS_ensem, word_mask, None, None)

        align_loss = None
        loss_prior = None
        if total_label is not None:
            attn_weight = torch.stack(seq_outputs[2][-3:], dim=1).sum(1).sum(1)[:, :hypo_len, hypo_len:]
            attn_weight = attn_weight.masked_fill(attn_weight == 0, -1e5)
            attn_weight = nn.Softmax(dim=-1)(attn_weight)
            total_label_align = total_label[align_pos == 1].to(dtype=torch.int64)
            attn_weight_align = attn_weight[align_pos == 1, :]
            align_loss = self.cls_loss_fct(attn_weight_align, total_label_align)
            # init_label = torch.zeros_like(prior_score.squeeze(1)[:, :prior_score.size(2) // 3])
            # for _ in range(len(offsets)):
            #     for o_i in offsets[_]:
            #         if len(o_i) > 1:
            #             for o_i_index in o_i:
            #                 init_label[_][o_i_index - 1] = 1.0
            # init_label = torch.cat(3 * [init_label], dim=-1)
            # loss_mse = nn.MSELoss()
            # loss_prior = loss_mse(prior_score.squeeze(1), init_label)
        return CLS_ensem, align_loss, (Reasoning_path, None)
        # logits = self.classifier(CLS_ensem)
        # loss_cls_0 = self.cls_loss_fct(logits.view(-1, 2), label)
        # logit_expl = binary_to_mp(logits, self.num_labels)
        # pre = logit_expl.max(dim=-1)[1]
        # label = label.reshape(-1, self.num_labels)
        # label = torch.argmax(label, -1)
        # matched_0 = pre == label
        #

        #
        # correct = 0
        # total_sum = 0
        # total_sum += total_label_align.size(0)
        # correct += (torch.argmax(attn_weight_align, -1) == total_label_align).sum().item()
        #
        # return loss_cls_0, matched_0, align_loss, correct, total_sum


class ChunkAlign_CLS_enc4_align(nn.Module):
    def __init__(self, global_enc, seq_enc, num_labels):
        super(ChunkAlign_CLS_enc4_align, self).__init__()
        self.global_enc = global_enc
        self.seq_enc = seq_enc
        self.cls_ensemble = nn.Linear(self.global_enc.config.hidden_size + self.seq_enc.config.hidden_size,
                                      self.global_enc.config.hidden_size)
        self.num_labels = num_labels
        self.cls_layer_num = 3
        self.cls_layer = nn.ModuleList([ClsLayer2(self.global_enc.config) for _ in range(self.cls_layer_num)])
        self.classifier = nn.Linear(self.global_enc.config.hidden_size, 2)
        self.cls_loss_fct = CrossEntropyLoss()

    def forward(self, input_ids, img_feat, input_mask=None, label=None,
                token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                offsets=None, chunk_attention_mask=None, gather_index=None, align_pos=None, total_label=None):
        hypo_len = input_ids.size(1)

        outputs = self.global_enc(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        global_output = outputs[0]
        global_CLS = outputs[1]

        img_mask = input_mask[:, hypo_len:]
        seq_outputs, chunk_hidden_states = self.seq_enc(input_ids, img_feats=img_feat, img_mask=img_mask,
                                                        input_mask=input_mask,
                                                        attention_mask=chunk_attention_mask,
                                                        position_ids=position_ids, token_type_ids=token_type_ids,
                                                        head_mask=head_mask, offsets=offsets, gather_index=gather_index)
        chunk_CLS = seq_outputs[1]
        CLS_ensem = self.cls_ensemble(torch.cat((global_CLS, chunk_CLS), -1))

        chunk_align = seq_outputs[0][:, 1:hypo_len]
        global_hypo = global_output[:, 1:hypo_len]
        chunk_hidden_states = chunk_hidden_states[:, 1:hypo_len]
        self_chunk_align = torch.cat((global_hypo, chunk_align, chunk_hidden_states), dim=1)

        word_mask = input_mask[:, 1:hypo_len].unsqueeze(1)
        word_mask = (1.0 - word_mask) * -10000.0
        word_mask = torch.cat((word_mask, word_mask, word_mask), -1)  # batch,word*2,hidden
        for i, layer_module in enumerate(self.cls_layer):

            # (batch_size * num_choices, 768)
            CLS_ensem, _ = layer_module(self_chunk_align, CLS_ensem, word_mask)

        logits = self.classifier(CLS_ensem)
        loss_cls_0 = self.cls_loss_fct(logits.view(-1, 2), label)
        logit_expl = binary_to_mp(logits, self.num_labels)
        pre = logit_expl.max(dim=-1)[1]
        label = label.reshape(-1, self.num_labels)
        label = torch.argmax(label, -1)
        matched_0 = pre == label

        attn_weight = torch.stack(seq_outputs[2][-3:], dim=1).sum(1).sum(1)[:, :hypo_len, hypo_len:]
        attn_weight = attn_weight.masked_fill(attn_weight == 0, -1e5)
        attn_weight = nn.Softmax(dim=-1)(attn_weight)

        total_label_align = total_label[align_pos == 1].to(dtype=torch.int64)
        attn_weight_align = attn_weight[align_pos == 1, :]
        align_loss = self.cls_loss_fct(attn_weight_align, total_label_align)

        correct = 0
        total_sum = 0
        total_sum += total_label_align.size(0)
        correct += (torch.argmax(attn_weight_align, -1) == total_label_align).sum().item()

        return loss_cls_0, matched_0, align_loss, correct, total_sum

    def evaluate(self, input_ids, img_feat, input_mask=None, label=None,
                 token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                 offsets=None, chunk_attention_mask=None, gather_index=None):
        hypo_len = input_ids.size(1)

        outputs = self.global_enc(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        global_output = outputs[0]
        global_CLS = outputs[1]

        img_mask = input_mask[:, hypo_len:]
        seq_outputs, chunk_hidden_states = self.seq_enc(input_ids, img_feats=img_feat, img_mask=img_mask,
                                                        input_mask=input_mask,
                                                        attention_mask=chunk_attention_mask,
                                                        position_ids=position_ids, token_type_ids=token_type_ids,
                                                        head_mask=head_mask, offsets=offsets, gather_index=gather_index)
        chunk_CLS = seq_outputs[1]
        CLS_ensem = self.cls_ensemble(torch.cat((global_CLS, chunk_CLS), -1))

        chunk_align = seq_outputs[0][:, 1:hypo_len]
        global_hypo = global_output[:, 1:hypo_len]
        chunk_hidden_states = chunk_hidden_states[:, 1:hypo_len]
        self_chunk_align = torch.cat((global_hypo, chunk_align, chunk_hidden_states), dim=1)

        word_mask = input_mask[:, 1:hypo_len].unsqueeze(1)
        word_mask = (1.0 - word_mask) * -10000.0
        word_mask = torch.cat((word_mask, word_mask, word_mask), -1)  # batch,word*2,hidden
        for i, layer_module in enumerate(self.cls_layer):
            CLS_ensem, _ = layer_module(self_chunk_align, CLS_ensem, word_mask)
        logits = self.classifier(CLS_ensem)
        logit_expl = binary_to_mp(logits,self.num_labels)

        pre = logit_expl.max(dim=-1)[1]
        label = label.reshape(-1, self.num_labels)
        label = torch.argmax(label, -1)
        matched_0 = pre == label

        return matched_0, pre, logit_expl

    def save_heat(self, input_ids, img_feat, input_mask=None, label=None,
                  token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                  offsets=None, chunk_attention_mask=None, gather_index=None):
        hypo_len = input_ids.size(1)

        outputs = self.global_enc(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        global_output = outputs[0]
        global_CLS = outputs[1]

        img_mask = input_mask[:, hypo_len:]
        seq_outputs, chunk_hidden_states = self.seq_enc(input_ids, img_feats=img_feat, img_mask=img_mask,
                                                        input_mask=input_mask,
                                                        attention_mask=chunk_attention_mask,
                                                        position_ids=position_ids, token_type_ids=token_type_ids,
                                                        head_mask=head_mask, offsets=offsets, gather_index=gather_index)
        chunk_CLS = seq_outputs[1]
        CLS_ensem = self.cls_ensemble(torch.cat((global_CLS, chunk_CLS), -1))

        chunk_align = seq_outputs[0][:, 1:hypo_len]
        global_hypo = global_output[:, 1:hypo_len]
        chunk_hidden_states = chunk_hidden_states[:, 1:hypo_len]
        self_chunk_align = torch.cat((global_hypo, chunk_align, chunk_hidden_states), dim=1)

        word_mask = input_mask[:, 1:hypo_len].unsqueeze(1)
        word_mask = (1.0 - word_mask) * -10000.0
        word_mask = torch.cat((word_mask, word_mask, word_mask), -1)  # batch,word*2,hidden
        for i, layer_module in enumerate(self.cls_layer):
            CLS_ensem, _ = layer_module(self_chunk_align, CLS_ensem, word_mask)
        logits = self.classifier(CLS_ensem)
        logit_expl = binary_to_mp(logits)
        pre = logit_expl.max(dim=-1)[1]
        label = label.reshape(-1, 4)
        label = torch.argmax(label, -1)
        matched_0 = pre == label

        return matched_0, pre, torch.stack(seq_outputs[2])[:, label:label + 1, :, :hypo_len, hypo_len:].squeeze(1)


class ChunkAlign_CLS_enc4_align_wo_reasoning(nn.Module):
    def __init__(self, global_enc, seq_enc, num_labels):
        super(ChunkAlign_CLS_enc4_align_wo_reasoning, self).__init__()
        self.global_enc = global_enc
        self.seq_enc = seq_enc
        self.cls_ensemble = nn.Linear(self.global_enc.config.hidden_size + self.seq_enc.config.hidden_size,
                                      self.global_enc.config.hidden_size)
        self.num_labels = num_labels
        self.classifier = nn.Linear(self.global_enc.config.hidden_size, 2)
        self.cls_loss_fct = CrossEntropyLoss()

    def forward(self, input_ids, img_feat, input_mask=None, label=None,
                token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                offsets=None, chunk_attention_mask=None, gather_index=None, align_pos=None, total_label=None):
        hypo_len = input_ids.size(1)

        outputs = self.global_enc(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        global_CLS = outputs[1]

        img_mask = input_mask[:, hypo_len:]
        seq_outputs, chunk_hidden_states = self.seq_enc(input_ids, img_feats=img_feat, img_mask=img_mask,
                                                        input_mask=input_mask,
                                                        attention_mask=chunk_attention_mask,
                                                        position_ids=position_ids, token_type_ids=token_type_ids,
                                                        head_mask=head_mask, offsets=offsets, gather_index=gather_index)
        chunk_CLS = seq_outputs[1]
        CLS_ensem = self.cls_ensemble(torch.cat((global_CLS, chunk_CLS), -1))

        logits = self.classifier(CLS_ensem)
        loss_cls_0 = self.cls_loss_fct(logits.view(-1, 2), label)
        logit_expl = binary_to_mp(logits)
        pre = logit_expl.max(dim=-1)[1]
        label = label.reshape(-1, 4)
        label = torch.argmax(label, -1)
        matched_0 = pre == label

        attn_weight = torch.stack(seq_outputs[2][-3:], dim=1).sum(1).sum(1)[:, :hypo_len, hypo_len:]
        attn_weight = attn_weight.masked_fill(attn_weight == 0, -1e5)
        attn_weight = nn.Softmax(dim=-1)(attn_weight)

        total_label_align = total_label[align_pos == 1].to(dtype=torch.int64)
        attn_weight_align = attn_weight[align_pos == 1, :]
        align_loss = self.cls_loss_fct(attn_weight_align, total_label_align)

        correct = 0
        total_sum = 0
        total_sum += total_label_align.size(0)
        correct += (torch.argmax(attn_weight_align, -1) == total_label_align).sum().item()

        return loss_cls_0, matched_0, align_loss, correct, total_sum

    def evaluate(self, input_ids, img_feat, input_mask=None, label=None,
                 token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                 offsets=None, chunk_attention_mask=None, gather_index=None):
        hypo_len = input_ids.size(1)

        outputs = self.global_enc(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        global_CLS = outputs[1]

        img_mask = input_mask[:, hypo_len:]
        seq_outputs, chunk_hidden_states = self.seq_enc(input_ids, img_feats=img_feat, img_mask=img_mask,
                                                        input_mask=input_mask,
                                                        attention_mask=chunk_attention_mask,
                                                        position_ids=position_ids, token_type_ids=token_type_ids,
                                                        head_mask=head_mask, offsets=offsets, gather_index=gather_index)
        chunk_CLS = seq_outputs[1]
        CLS_ensem = self.cls_ensemble(torch.cat((global_CLS, chunk_CLS), -1))

        logits = self.classifier(CLS_ensem)
        logit_expl = binary_to_mp(logits)
        pre = logit_expl.max(dim=-1)[1]
        label = label.reshape(-1, 4)
        label = torch.argmax(label, -1)
        matched_0 = pre == label

        return matched_0, pre


class ChunkAlign_CLS_enc4_align_wo_chual(nn.Module):
    def __init__(self, global_enc, num_labels):
        super(ChunkAlign_CLS_enc4_align_wo_chual, self).__init__()
        self.global_enc = global_enc

        self.num_labels = num_labels
        self.cls_layer_num = 3
        self.cls_layer = nn.ModuleList([ClsLayer2(self.global_enc.config) for _ in range(self.cls_layer_num)])
        self.classifier = nn.Linear(self.global_enc.config.hidden_size, 2)
        self.cls_loss_fct = CrossEntropyLoss()

    def forward(self, input_ids, img_feat, input_mask=None, label=None,
                token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                offsets=None, chunk_attention_mask=None, gather_index=None, align_pos=None, total_label=None):
        img_len = img_feat.size(1)
        hypo_len = input_ids.size(1)

        outputs = self.global_enc(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        global_output = outputs[0]
        CLS_ensem = outputs[1]
        self_chunk_align = global_output[:, 1:hypo_len]
        word_mask = input_mask[:, 1:hypo_len].unsqueeze(1)
        word_mask = (1.0 - word_mask) * -10000.0

        for i, layer_module in enumerate(self.cls_layer):
            CLS_ensem, attn_weight_layer = layer_module(self_chunk_align, CLS_ensem, word_mask)

        logits = self.classifier(CLS_ensem)
        loss_cls_0 = self.cls_loss_fct(logits.view(-1, 2), label)
        logit_expl = binary_to_mp(logits)
        pre = logit_expl.max(dim=-1)[1]
        label = label.reshape(-1, 4)
        label = torch.argmax(label, -1)
        matched_0 = pre == label

        return loss_cls_0, matched_0

    def evaluate(self, input_ids, img_feat, input_mask=None, label=None,
                 token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                 offsets=None, chunk_attention_mask=None, gather_index=None):
        hypo_len = input_ids.size(1)

        outputs = self.global_enc(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        global_output = outputs[0]
        CLS_ensem = outputs[1]
        self_chunk_align = global_output[:, 1:hypo_len]
        word_mask = input_mask[:, 1:hypo_len].unsqueeze(1)
        word_mask = (1.0 - word_mask) * -10000.0

        for i, layer_module in enumerate(self.cls_layer):
            CLS_ensem, attn_weight_layer = layer_module(self_chunk_align, CLS_ensem, word_mask)
        logits = self.classifier(CLS_ensem)
        logit_expl = binary_to_mp(logits)
        pre = logit_expl.max(dim=-1)[1]
        label = label.reshape(-1, 4)
        label = torch.argmax(label, -1)
        matched_0 = pre == label

        return matched_0, pre


class ChunkAlign_CLS_dec5_4(nn.Module):
    def __init__(self, global_enc, seq_enc, dec, dec_toker, num_labels):
        super(ChunkAlign_CLS_dec5_4, self).__init__()
        self.global_enc = global_enc
        self.seq_enc = seq_enc
        self.dec_toker = dec_toker
        self.dec = dec

        self.cls_ensemble = nn.Linear(self.global_enc.config.hidden_size * 2, self.global_enc.config.hidden_size)
        self.num_labels = num_labels
        self.cls_layer_num = 3
        self.cls_layer = nn.ModuleList([ClsLayer2(self.global_enc.config) for _ in range(self.cls_layer_num)])
        self.classifier = nn.Linear(self.global_enc.config.hidden_size, 2)
        self.cls_loss_fct = CrossEntropyLoss()

        self.vocab_num = self.dec.vocab_size
        self.lm_head = nn.Linear(self.dec.config.n_embd, self.dec.vocab_size, bias=False)
        self.gen_criterion = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=self.dec_toker.pad_token_id)
        self.max_gen_len = 30
        self.b_rtnl = torch.tensor(self.dec_toker.encode("<|b_rtnl|>")).cuda()
        self.e_rtnl = self.dec_toker.encode("<|e_rtnl|>")[0]

    def forward(self, input_ids, img_feat, expl_ids, input_mask=None, label=None, attn_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                offsets=None, chunk_attention_mask=None, gather_index=None, gpt_labels=None):
        img_len = img_feat.size(1)
        hypo_len = input_ids.size(1)

        outputs = self.global_enc(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        global_output = outputs[0]
        global_CLS = outputs[1]

        img_mask = input_mask[:, hypo_len:]
        seq_outputs, chunk_hidden_states = self.seq_enc(input_ids, img_feats=img_feat, img_mask=img_mask,
                                                        input_mask=input_mask,
                                                        attention_mask=chunk_attention_mask,
                                                        position_ids=position_ids, token_type_ids=token_type_ids,
                                                        head_mask=head_mask, offsets=offsets, gather_index=gather_index)
        chunk_CLS = seq_outputs[1]
        CLS_ensem = self.cls_ensemble(torch.cat((global_CLS, chunk_CLS), -1))

        chunk_align = seq_outputs[0][:, 1:hypo_len]
        global_hypo = global_output[:, 1:hypo_len]
        chunk_hidden_states = chunk_hidden_states[:, 1:hypo_len]
        self_chunk_align = torch.cat((global_hypo, chunk_align, chunk_hidden_states), dim=1)

        word_mask = input_mask[:, 1:hypo_len].unsqueeze(1)
        word_mask = (1.0 - word_mask) * -10000.0
        word_mask = torch.cat((word_mask, word_mask, word_mask), -1)  # batch,word*2,hidden
        for i, layer_module in enumerate(self.cls_layer):
            CLS_ensem, _ = layer_module(self_chunk_align, CLS_ensem, word_mask)

        logits = self.classifier(CLS_ensem)
        loss_cls_0 = self.cls_loss_fct(logits.view(-1, 2), label)
        logit_expl = binary_to_mp(logits)
        pre = logit_expl.max(dim=-1)[1]
        label = label.reshape(-1, 4)
        label = torch.argmax(label, -1)
        matched_0 = pre == label

        encoder_hs = torch.cat(
            (seq_outputs[0][:, 1:hypo_len], global_output[:, 1:hypo_len], chunk_hidden_states),
            1).detach()
        encoder_mask = torch.cat((input_mask[:, 1:hypo_len], input_mask[:, 1:hypo_len], input_mask[:, 1:hypo_len]), 1)

        ques_num = label.size(0)
        add_tensor = torch.arange(ques_num, dtype=torch.int64).cuda()
        add_tensor = add_tensor * 4
        gather_index = add_tensor + label

        encoder_mask_chosen = torch.gather(encoder_mask, 0, gather_index.unsqueeze(-1).repeat(1, encoder_mask.size(-1)))
        encoder_hs_chosen = torch.gather(encoder_hs, 0,
                                         gather_index.unsqueeze(-1).unsqueeze(-1).repeat(1, encoder_hs.size(-2),
                                                                                         encoder_hs.size(-1)))
        gpt_labels = gpt_labels.reshape(ques_num, 4, -1)[:, 0, :]
        expl_ids = expl_ids.reshape(ques_num, 4, -1)[:, 0, :]
        attn_mask = attn_mask.reshape(ques_num, 4, -1)[:, 0, :]

        gpt_out = self.dec(input_ids=expl_ids, attention_mask=attn_mask, encoder_hidden_states=encoder_hs_chosen,
                           encoder_attention_mask=encoder_mask_chosen)[0]
        lm_logits = self.lm_head(gpt_out)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = gpt_labels[..., 1:].contiguous()
        gen_loss = self.gen_criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return gen_loss, loss_cls_0, matched_0

    def evaluate(self, input_ids, img_feat, expl_ids, input_mask=None, label=None,
                 token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                 offsets=None, chunk_attention_mask=None, gather_index=None):
        hypo_len = input_ids.size(1)

        outputs = self.global_enc(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        global_output = outputs[0]
        global_CLS = outputs[1]

        img_mask = input_mask[:, hypo_len:]
        seq_outputs, chunk_hidden_states = self.seq_enc(input_ids, img_feats=img_feat, img_mask=img_mask,
                                                        input_mask=input_mask,
                                                        attention_mask=chunk_attention_mask,
                                                        position_ids=position_ids, token_type_ids=token_type_ids,
                                                        head_mask=head_mask, offsets=offsets, gather_index=gather_index)
        chunk_CLS = seq_outputs[1]
        CLS_ensem = self.cls_ensemble(torch.cat((global_CLS, chunk_CLS), -1))

        chunk_align = seq_outputs[0][:, 1:hypo_len]
        global_hypo = global_output[:, 1:hypo_len]
        chunk_hidden_states = chunk_hidden_states[:, 1:hypo_len]
        self_chunk_align = torch.cat((global_hypo, chunk_align, chunk_hidden_states), dim=1)

        word_mask = input_mask[:, 1:hypo_len].unsqueeze(1)
        word_mask = (1.0 - word_mask) * -10000.0
        word_mask = torch.cat((word_mask, word_mask, word_mask), -1)  # batch,word*2,hidden
        for i, layer_module in enumerate(self.cls_layer):
            CLS_ensem, _ = layer_module(self_chunk_align, CLS_ensem, word_mask)
        logits = self.classifier(CLS_ensem)
        logit_expl = binary_to_mp(logits)
        pre = logit_expl.max(dim=-1)[1]
        label = label.reshape(-1, 4)
        label = torch.argmax(label, -1)
        matched_0 = pre == label

        encoder_hs = torch.cat(
            (seq_outputs[0][:, 1:hypo_len], global_output[:, 1:hypo_len], chunk_hidden_states), 1)
        encoder_mask = torch.cat((input_mask[:, 1:hypo_len], input_mask[:, 1:hypo_len], input_mask[:, 1:hypo_len]), 1)

        ques_num = label.size(0)
        add_tensor = torch.arange(ques_num, dtype=torch.int64).cuda()
        add_tensor = add_tensor * 4
        gather_index = add_tensor + label
        encoder_mask_chosen = torch.gather(encoder_mask, 0, gather_index.unsqueeze(-1).repeat(1, encoder_mask.size(-1)))
        encoder_hs_chosen = torch.gather(encoder_hs, 0,
                                         gather_index.unsqueeze(-1).unsqueeze(-1).repeat(1, encoder_hs.size(-2),
                                                                                         encoder_hs.size(-1)))
        expl_ids = expl_ids.reshape(ques_num, 4, -1)[:, 0, :]

        outputs = torch.full((encoder_hs_chosen.size(0), self.max_gen_len), fill_value=self.dec_toker.pad_token_id,
                             dtype=int).cuda()
        for i in range(matched_0.size(0)):
            if matched_0[i] == torch.tensor(True):
                decoder_input = expl_ids[i]
                b_rtnl_index = torch.nonzero((decoder_input == self.b_rtnl.unsqueeze(0)).to(torch.int64))[0, -1]
                prompt_decoder_input = decoder_input[:b_rtnl_index]
                gpt_out = self.dec(input_ids=prompt_decoder_input.unsqueeze(0),
                                   encoder_hidden_states=encoder_hs_chosen[i].unsqueeze(0),
                                   encoder_attention_mask=encoder_mask_chosen[i].unsqueeze(0), use_cache=True,
                                   past_key_values=None)
                past_key_values = gpt_out[1]
                cur_unfinished = outputs.new(1).fill_(1)
                cur_len = 0
                tokens_to_add = decoder_input[b_rtnl_index].unsqueeze(-1)
                for index in range(self.max_gen_len - 1):
                    gpt_out = self.dec(input_ids=tokens_to_add.unsqueeze(0),
                                       encoder_hidden_states=encoder_hs_chosen[i].unsqueeze(0),
                                       encoder_attention_mask=encoder_mask_chosen[i].unsqueeze(0), use_cache=True,
                                       past_key_values=past_key_values)
                    past_key_values = gpt_out[1]
                    gpt_out = gpt_out[0][:, -1:, :]
                    # 只取最后一个作为当前步的输出
                    lm_logits = self.lm_head(gpt_out)
                    gen_label = torch.argmax(lm_logits, dim=-1).squeeze(-1)
                    tokens_to_add = gen_label * cur_unfinished + self.dec_toker.pad_token_id * (1 - cur_unfinished)
                    outputs[i, index] = tokens_to_add
                    cur_len += 1
                    cur_unfinished = cur_unfinished.mul(tokens_to_add.ne(self.e_rtnl).long())
                    if cur_unfinished.max() == 0:
                        break
                if cur_len == self.max_gen_len:
                    outputs[i, -1].masked_fill_(cur_unfinished.to(dtype=torch.bool), self.dec_toker.eos_token_id)
        return outputs, matched_0, pre


class ChunkAlign_CLS_dec5_4_wo_reasoning(nn.Module):
    def __init__(self, global_enc, seq_enc, dec, dec_toker, num_labels):
        super(ChunkAlign_CLS_dec5_4_wo_reasoning, self).__init__()
        self.global_enc = global_enc
        self.seq_enc = seq_enc
        self.dec_toker = dec_toker
        self.dec = dec

        self.cls_ensemble = nn.Linear(self.global_enc.config.hidden_size * 2, self.global_enc.config.hidden_size)
        self.num_labels = num_labels
        self.classifier = nn.Linear(self.global_enc.config.hidden_size, 2)
        self.cls_loss_fct = CrossEntropyLoss()

        self.vocab_num = self.dec.vocab_size
        self.lm_head = nn.Linear(self.dec.config.n_embd, self.dec.vocab_size, bias=False)
        self.gen_criterion = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=self.dec_toker.pad_token_id)
        self.max_gen_len = 30
        self.b_rtnl = torch.tensor(self.dec_toker.encode("<|b_rtnl|>")).cuda()

    def forward(self, input_ids, img_feat, expl_ids, input_mask=None, label=None, attn_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                offsets=None, chunk_attention_mask=None, gather_index=None):
        img_len = img_feat.size(1)
        hypo_len = input_ids.size(1)

        outputs = self.global_enc(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        global_output = outputs[0]
        global_CLS = outputs[1]

        img_mask = input_mask[:, hypo_len:]
        seq_outputs, chunk_hidden_states = self.seq_enc(input_ids, img_feats=img_feat, img_mask=img_mask,
                                                        input_mask=input_mask,
                                                        attention_mask=chunk_attention_mask,
                                                        position_ids=position_ids, token_type_ids=token_type_ids,
                                                        head_mask=head_mask, offsets=offsets, gather_index=gather_index)
        chunk_CLS = seq_outputs[1]
        CLS_ensem = self.cls_ensemble(torch.cat((global_CLS, chunk_CLS), -1))

        logits = self.classifier(CLS_ensem)
        loss_cls_0 = self.cls_loss_fct(logits.view(-1, 2), label)
        logit_expl = binary_to_mp(logits)
        pre = logit_expl.max(dim=-1)[1]
        label = label.reshape(-1, 4)
        label = torch.argmax(label, -1)
        matched_0 = pre == label

        encoder_hs = torch.cat(
            (seq_outputs[0][:, 1:hypo_len], global_output[:, 1:hypo_len], chunk_hidden_states[:, 1:hypo_len]),
            1).detach()
        encoder_mask = torch.cat((input_mask[:, 1:hypo_len], input_mask[:, 1:hypo_len], input_mask[:, 1:hypo_len]), 1)

        ques_num = label.size(0)
        add_tensor = torch.arange(ques_num, dtype=torch.int64).cuda()
        add_tensor = add_tensor * 4
        gather_index = add_tensor + label

        encoder_mask_chosen = torch.gather(encoder_mask, 0, gather_index.unsqueeze(-1).repeat(1, encoder_mask.size(-1)))
        encoder_hs_chosen = torch.gather(encoder_hs, 0,
                                         gather_index.unsqueeze(-1).unsqueeze(-1).repeat(1, encoder_hs.size(-2),
                                                                                         encoder_hs.size(-1)))

        expl_ids = expl_ids.reshape(ques_num, 4, -1)[:, 0, :]
        attn_mask = attn_mask.reshape(ques_num, 4, -1)[:, 0, :]

        gpt_out = self.dec(input_ids=expl_ids, attention_mask=attn_mask, encoder_hidden_states=encoder_hs_chosen,
                           encoder_attention_mask=encoder_mask_chosen)[0]
        lm_logits = self.lm_head(gpt_out)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = expl_ids[..., 1:].contiguous()
        gen_loss = self.gen_criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return gen_loss, loss_cls_0, matched_0

    def evaluate(self, input_ids, img_feat, expl_ids, input_mask=None, label=None,
                 token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                 offsets=None, chunk_attention_mask=None, gather_index=None):
        hypo_len = input_ids.size(1)

        outputs = self.global_enc(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        global_output = outputs[0]
        global_CLS = outputs[1]

        img_mask = input_mask[:, hypo_len:]
        seq_outputs, chunk_hidden_states = self.seq_enc(input_ids, img_feats=img_feat, img_mask=img_mask,
                                                        input_mask=input_mask,
                                                        attention_mask=chunk_attention_mask,
                                                        position_ids=position_ids, token_type_ids=token_type_ids,
                                                        head_mask=head_mask, offsets=offsets, gather_index=gather_index)
        chunk_CLS = seq_outputs[1]
        CLS_ensem = self.cls_ensemble(torch.cat((global_CLS, chunk_CLS), -1))

        logits = self.classifier(CLS_ensem)
        logit_expl = binary_to_mp(logits)
        pre = logit_expl.max(dim=-1)[1]
        label = label.reshape(-1, 4)
        label = torch.argmax(label, -1)
        matched_0 = pre == label

        encoder_hs = torch.cat(
            (seq_outputs[0][:, 1:hypo_len], global_output[:, 1:hypo_len], chunk_hidden_states[:, 1:hypo_len]), 1)
        encoder_mask = torch.cat((input_mask[:, 1:hypo_len], input_mask[:, 1:hypo_len], input_mask[:, 1:hypo_len]), 1)

        ques_num = label.size(0)
        add_tensor = torch.arange(ques_num, dtype=torch.int64).cuda()
        add_tensor = add_tensor * 4
        gather_index = add_tensor + label
        encoder_mask_chosen = torch.gather(encoder_mask, 0, gather_index.unsqueeze(-1).repeat(1, encoder_mask.size(-1)))
        encoder_hs_chosen = torch.gather(encoder_hs, 0,
                                         gather_index.unsqueeze(-1).unsqueeze(-1).repeat(1, encoder_hs.size(-2),
                                                                                         encoder_hs.size(-1)))
        expl_ids = expl_ids.reshape(ques_num, 4, -1)[:, 0, :]

        outputs = torch.full((encoder_hs_chosen.size(0), self.max_gen_len), fill_value=self.dec_toker.pad_token_id,
                             dtype=int).cuda()
        for i in range(matched_0.size(0)):
            if matched_0[i] == torch.tensor(True):
                decoder_input = expl_ids[i]
                b_rtnl_index = torch.nonzero((decoder_input == self.b_rtnl.unsqueeze(0)).to(torch.int64))[0, -1]
                prompt_decoder_input = decoder_input[:b_rtnl_index]
                gpt_out = self.dec(input_ids=prompt_decoder_input.unsqueeze(0),
                                   encoder_hidden_states=encoder_hs_chosen[i].unsqueeze(0),
                                   encoder_attention_mask=encoder_mask_chosen[i].unsqueeze(0), use_cache=True,
                                   past_key_values=None)
                past_key_values = gpt_out[1]
                cur_unfinished = outputs.new(1).fill_(1)
                cur_len = 0
                tokens_to_add = decoder_input[b_rtnl_index].unsqueeze(-1)
                for index in range(self.max_gen_len - 1):
                    gpt_out = self.dec(input_ids=tokens_to_add.unsqueeze(0),
                                       encoder_hidden_states=encoder_hs_chosen[i].unsqueeze(0),
                                       encoder_attention_mask=encoder_mask_chosen[i].unsqueeze(0), use_cache=True,
                                       past_key_values=past_key_values)
                    past_key_values = gpt_out[1]
                    gpt_out = gpt_out[0][:, -1:, :]
                    # 只取最后一个作为当前步的输出
                    lm_logits = self.lm_head(gpt_out)
                    gen_label = torch.argmax(lm_logits, dim=-1).squeeze(-1)
                    tokens_to_add = gen_label * cur_unfinished + self.dec_toker.pad_token_id * (1 - cur_unfinished)
                    outputs[i, index] = tokens_to_add
                    cur_len += 1
                    cur_unfinished = cur_unfinished.mul(tokens_to_add.ne(self.dec_toker.eos_token_id).long())
                    if cur_unfinished.max() == 0:
                        break
                if cur_len == self.max_gen_len:
                    outputs[i, -1].masked_fill_(cur_unfinished.to(dtype=torch.bool), self.dec_toker.eos_token_id)
        return outputs, matched_0, pre


class ChunkAlign_CLS_dec5_4_wo_chual(nn.Module):
    def __init__(self, global_enc, dec, dec_toker, num_labels):
        super(ChunkAlign_CLS_dec5_4_wo_chual, self).__init__()
        self.global_enc = global_enc
        self.dec_toker = dec_toker
        self.dec = dec

        self.num_labels = num_labels
        self.cls_layer_num = 3
        self.cls_layer = nn.ModuleList([ClsLayer2(self.global_enc.config) for _ in range(self.cls_layer_num)])
        self.classifier = nn.Linear(self.global_enc.config.hidden_size, 2)
        self.cls_loss_fct = CrossEntropyLoss()

        self.vocab_num = self.dec.vocab_size
        self.lm_head = nn.Linear(self.dec.config.n_embd, self.dec.vocab_size, bias=False)
        self.gen_criterion = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=self.dec_toker.pad_token_id)
        self.max_gen_len = 30
        self.b_rtnl = torch.tensor(self.dec_toker.encode("<|b_rtnl|>")).cuda()

    def forward(self, input_ids, img_feat, expl_ids, input_mask=None, label=None, attn_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                offsets=None, chunk_attention_mask=None, gather_index=None):
        img_len = img_feat.size(1)
        hypo_len = input_ids.size(1)

        outputs = self.global_enc(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        global_output = outputs[0]
        CLS_ensem = outputs[1]
        self_chunk_align = global_output[:, 1:hypo_len]
        word_mask = input_mask[:, 1:hypo_len].unsqueeze(1)
        word_mask = (1.0 - word_mask) * -10000.0

        for i, layer_module in enumerate(self.cls_layer):
            CLS_ensem, attn_weight_layer = layer_module(self_chunk_align, CLS_ensem, word_mask)

        logits = self.classifier(CLS_ensem)
        loss_cls_0 = self.cls_loss_fct(logits.view(-1, 2), label)
        logit_expl = binary_to_mp(logits)
        pre = logit_expl.max(dim=-1)[1]
        label = label.reshape(-1, 4)
        label = torch.argmax(label, -1)
        matched_0 = pre == label

        encoder_hs = global_output[:, 1:hypo_len].contiguous().detach()
        encoder_mask = input_mask[:, 1:hypo_len]
        ques_num = label.size(0)
        add_tensor = torch.arange(ques_num, dtype=torch.int64).cuda()
        add_tensor = add_tensor * 4
        gather_index = add_tensor + label

        encoder_mask_chosen = torch.gather(encoder_mask, 0, gather_index.unsqueeze(-1).repeat(1, encoder_mask.size(-1)))
        encoder_hs_chosen = torch.gather(encoder_hs, 0,
                                         gather_index.unsqueeze(-1).unsqueeze(-1).repeat(1, encoder_hs.size(-2),
                                                                                         encoder_hs.size(-1)))

        expl_ids = expl_ids.reshape(ques_num, 4, -1)[:, 0, :]
        attn_mask = attn_mask.reshape(ques_num, 4, -1)[:, 0, :]

        gpt_out = self.dec(input_ids=expl_ids, attention_mask=attn_mask, encoder_hidden_states=encoder_hs_chosen,
                           encoder_attention_mask=encoder_mask_chosen)[0]
        lm_logits = self.lm_head(gpt_out)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = expl_ids[..., 1:].contiguous()
        gen_loss = self.gen_criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return gen_loss, loss_cls_0, matched_0

    def evaluate(self, input_ids, img_feat, expl_ids, input_mask=None, label=None,
                 token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                 offsets=None, chunk_attention_mask=None, gather_index=None):
        hypo_len = input_ids.size(1)

        outputs = self.global_enc(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        global_output = outputs[0]
        CLS_ensem = outputs[1]
        self_chunk_align = global_output[:, 1:hypo_len]
        word_mask = input_mask[:, 1:hypo_len].unsqueeze(1)
        word_mask = (1.0 - word_mask) * -10000.0

        for i, layer_module in enumerate(self.cls_layer):
            CLS_ensem, attn_weight_layer = layer_module(self_chunk_align, CLS_ensem, word_mask)

        logits = self.classifier(CLS_ensem)
        logit_expl = binary_to_mp(logits)
        pre = logit_expl.max(dim=-1)[1]
        label = label.reshape(-1, 4)
        label = torch.argmax(label, -1)
        matched_0 = pre == label

        encoder_hs = global_output[:, 1:hypo_len].contiguous().detach()
        encoder_mask = input_mask[:, 1:hypo_len]
        ques_num = label.size(0)
        add_tensor = torch.arange(ques_num, dtype=torch.int64).cuda()
        add_tensor = add_tensor * 4
        gather_index = add_tensor + label
        encoder_mask_chosen = torch.gather(encoder_mask, 0, gather_index.unsqueeze(-1).repeat(1, encoder_mask.size(-1)))
        encoder_hs_chosen = torch.gather(encoder_hs, 0,
                                         gather_index.unsqueeze(-1).unsqueeze(-1).repeat(1, encoder_hs.size(-2),
                                                                                         encoder_hs.size(-1)))
        expl_ids = expl_ids.reshape(ques_num, 4, -1)[:, 0, :]

        outputs = torch.full((encoder_hs_chosen.size(0), self.max_gen_len), fill_value=self.dec_toker.pad_token_id,
                             dtype=int).cuda()
        for i in range(matched_0.size(0)):
            if matched_0[i] == torch.tensor(True):
                decoder_input = expl_ids[i]
                b_rtnl_index = torch.nonzero((decoder_input == self.b_rtnl.unsqueeze(0)).to(torch.int64))[0, -1]
                prompt_decoder_input = decoder_input[:b_rtnl_index]
                gpt_out = self.dec(input_ids=prompt_decoder_input.unsqueeze(0),
                                   encoder_hidden_states=encoder_hs_chosen[i].unsqueeze(0),
                                   encoder_attention_mask=encoder_mask_chosen[i].unsqueeze(0), use_cache=True,
                                   past_key_values=None)
                past_key_values = gpt_out[1]
                cur_unfinished = outputs.new(1).fill_(1)
                cur_len = 0
                tokens_to_add = decoder_input[b_rtnl_index].unsqueeze(-1)
                for index in range(self.max_gen_len - 1):
                    gpt_out = self.dec(input_ids=tokens_to_add.unsqueeze(0),
                                       encoder_hidden_states=encoder_hs_chosen[i].unsqueeze(0),
                                       encoder_attention_mask=encoder_mask_chosen[i].unsqueeze(0), use_cache=True,
                                       past_key_values=past_key_values)
                    past_key_values = gpt_out[1]
                    gpt_out = gpt_out[0][:, -1:, :]
                    # 只取最后一个作为当前步的输出
                    lm_logits = self.lm_head(gpt_out)
                    gen_label = torch.argmax(lm_logits, dim=-1).squeeze(-1)
                    tokens_to_add = gen_label * cur_unfinished + self.dec_toker.pad_token_id * (1 - cur_unfinished)
                    outputs[i, index] = tokens_to_add
                    cur_len += 1
                    cur_unfinished = cur_unfinished.mul(tokens_to_add.ne(self.dec_toker.eos_token_id).long())
                    if cur_unfinished.max() == 0:
                        break
                if cur_len == self.max_gen_len:
                    outputs[i, -1].masked_fill_(cur_unfinished.to(dtype=torch.bool), self.dec_toker.eos_token_id)
        return outputs, matched_0, pre


class BeamHypotheses:
    def __init__(self, num_beams: int, max_length: int, length_penalty: float, early_stopping: bool):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp: torch.LongTensor, sum_logprobs: float):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs: float, cur_len: int) -> bool:
        """
        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
        one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret


class BeamSearchScorer_constrained(BeamScorer):

    def __init__(
            self,
            batch_size: int,
            max_length: int,
            num_beams: int,
            device: torch.device,
            length_penalty=1.0,
            do_early_stopping=False,
            num_beam_hyps_to_keep=1,
            num_beam_groups=1,
            constrained=1.0
    ):
        self.max_length = max_length
        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.num_beam_groups = num_beam_groups
        self.group_size = self.num_beams // self.num_beam_groups
        self.constrained = constrained
        self._is_init = False
        self._beam_hyps = [
            BeamHypotheses(
                num_beams=self.num_beams,
                max_length=self.max_length,
                length_penalty=self.length_penalty,
                early_stopping=self.do_early_stopping,
            )
            for _ in range(batch_size)
        ]
        self._done = torch.tensor([False for _ in range(batch_size)], dtype=torch.bool, device=self.device)

        if not isinstance(num_beams, int) or num_beams <= 1:
            raise ValueError(
                f"`num_beams` has to be an integer strictly greater than 1, but is {num_beams}. For `num_beams` == 1, one should make use of `greedy_search` instead."
            )

        if not isinstance(num_beam_groups, int) or (num_beam_groups > num_beams) or (num_beams % num_beam_groups != 0):
            raise ValueError(
                f"`num_beam_groups` has to be an integer smaller or equal than `num_beams` and `num_beams` "
                f"has to be divisible by `num_beam_groups`, but is {num_beam_groups} with `num_beams` being {num_beams}."
            )

    @property
    def is_done(self) -> bool:
        return self._done.all()

    def process(
            self,
            input_ids: torch.LongTensor,
            next_scores: torch.FloatTensor,
            next_tokens: torch.LongTensor,
            next_indices: torch.LongTensor,
            pad_token_id=None,
            eos_token_id=None,
            add_score_ids=None
    ):
        cur_len = input_ids.shape[-1]
        batch_size = len(self._beam_hyps)
        assert batch_size == (input_ids.shape[0] // self.group_size)

        device = input_ids.device
        next_beam_scores = torch.zeros((batch_size, self.group_size), dtype=next_scores.dtype, device=device)
        next_beam_tokens = torch.zeros((batch_size, self.group_size), dtype=next_tokens.dtype, device=device)
        next_beam_indices = torch.zeros((batch_size, self.group_size), dtype=next_indices.dtype, device=device)

        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                assert (
                        len(beam_hyp) >= self.num_beams
                ), f"Batch can only be done if at least {self.num_beams} beams have been generated"
                assert (
                        eos_token_id is not None and pad_token_id is not None
                ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                # pad the batch
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue

            # next tokens for this sentence
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                    zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
            ):
                batch_beam_idx = batch_idx * self.group_size + next_index
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (next_token.item() == eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.group_size
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    beam_hyp.add(
                        input_ids[batch_beam_idx].clone(),
                        next_score.item(),
                    )
                else:
                    # add next predicted token since it is not eos_token
                    if int(next_token) in add_score_ids:
                        next_score *= self.constrained
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1

                # once the beam for next step is full, don't add more tokens to it.
                if beam_idx == self.group_size:
                    break

            if beam_idx < self.group_size:
                raise ValueError(
                    f"At most {self.group_size} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id: {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected."
                )

            # Check if we are done so that we can save a pad step if all(done)
            self._done[batch_idx] = self._done[batch_idx] or beam_hyp.is_done(
                next_scores[batch_idx].max().item(), cur_len
            )

        return UserDict(
            {
                "next_beam_scores": next_beam_scores.view(-1),
                "next_beam_tokens": next_beam_tokens.view(-1),
                "next_beam_indices": next_beam_indices.view(-1),
            }
        )

    def finalize(
            self,
            input_ids: torch.LongTensor,
            final_beam_scores: torch.FloatTensor,
            final_beam_tokens: torch.LongTensor,
            final_beam_indices: torch.LongTensor,
            pad_token_id=None,
            eos_token_id=None,
    ):
        batch_size = len(self._beam_hyps)

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                continue

            # all open beam hypotheses are added to the beam hypothesis
            # beam hypothesis class automatically keeps the best beams
            for beam_id in range(self.num_beams):
                batch_beam_idx = batch_idx * self.num_beams + beam_id
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                beam_hyp.add(final_tokens, final_score)

        # select the best hypotheses
        sent_lengths = input_ids.new(batch_size * self.num_beam_hyps_to_keep)
        best = []
        best_scores = torch.zeros(batch_size * self.num_beam_hyps_to_keep, device=self.device, dtype=torch.float32)
        sorted_ids = []
        # retrieve best hypotheses
        for i, beam_hyp in enumerate(self._beam_hyps):
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
            score_list = [x[0] for x in beam_hyp.beams]
            sorted_id = sorted(range(len(score_list)), key=lambda k: score_list[k])[0]
            sorted_ids.append(sorted_id)
            for j in range(self.num_beam_hyps_to_keep):
                best_hyp_tuple = sorted_hyps.pop()
                best_score = best_hyp_tuple[0]
                best_hyp = best_hyp_tuple[1]
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)

                # append to lists
                best.append(best_hyp)
                best_scores[i * self.num_beam_hyps_to_keep + j] = best_score

        # prepare for adding eos
        sent_max_len = min(sent_lengths.max().item() + 1, self.max_length)
        decoded: torch.LongTensor = input_ids.new(batch_size * self.num_beam_hyps_to_keep, sent_max_len)
        # shorter batches are padded if needed
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`pad_token_id` has to be defined"
            decoded.fill_(pad_token_id)

        # fill with hypotheses and eos_token_id if the latter fits in
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < self.max_length:
                decoded[i, sent_lengths[i]] = eos_token_id
        return UserDict(
            {
                "sequences": decoded,
                "sequence_scores": best_scores,
                "sorted_ids": sorted_ids
            }
        )


class ChunkAlign_CLS_dec5_4_beam(nn.Module):
    def __init__(self, global_enc, seq_enc, dec, enc_toker, dec_toker, num_labels, repeat_penalty, length_penalty,
                 constrained):
        super(ChunkAlign_CLS_dec5_4_beam, self).__init__()
        self.global_enc = global_enc
        self.seq_enc = seq_enc
        self.enc_toker = enc_toker
        self.dec_toker = dec_toker
        self.dec = dec

        self.cls_ensemble = nn.Linear(self.global_enc.config.hidden_size * 2, self.global_enc.config.hidden_size)
        self.num_labels = num_labels
        self.cls_layer_num = 3
        self.cls_layer = nn.ModuleList([ClsLayer2(self.global_enc.config) for _ in range(self.cls_layer_num)])
        self.classifier = nn.Linear(self.global_enc.config.hidden_size, 2)
        self.cls_loss_fct = CrossEntropyLoss()

        self.vocab_num = self.dec.vocab_size
        self.lm_head = nn.Linear(self.dec.config.n_embd, self.dec.vocab_size, bias=False)
        self.gen_criterion = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=self.dec_toker.pad_token_id)

        self.beam_size = 5
        self.b_rtnl = torch.tensor(self.dec_toker.encode("<|b_rtnl|>")).cuda()
        self.e_rtnl = self.dec_toker.encode("<|e_rtnl|>")[0]

        self.repeat_penalty = repeat_penalty
        self.length_penalty = length_penalty
        self.constrained = constrained
        self.stop_words = ['a', 'is', 'am', 'are', 'the', 'an', 'it', 'has', 'very', 'and', 'his', 'her', 'there',
                           'she', 'they', 'what', 'this', 'that', 'these', 'those', 'do', 'does', 'why', 'for',
                           'with', 'how', 'when', 'where', 'woman', 'man', 'women', 'men', 'some', 'while',
                           'have', 'them', 'him', 'than', '[SEP]', '[PAD]', 'yes', 'about']

    def test_beam(self, input_ids, img_feat, expl_ids, input_mask=None, label=None,
                  token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                  offsets=None, chunk_attention_mask=None, gather_index=None):
        hypo_len = input_ids.size(1)
        batch_size = input_ids.size(0)

        outputs = self.global_enc(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        global_output = outputs[0]
        global_CLS = outputs[1]

        img_mask = input_mask[:, hypo_len:]
        seq_outputs, chunk_hidden_states = self.seq_enc(input_ids, img_feats=img_feat, img_mask=img_mask,
                                                        input_mask=input_mask,
                                                        attention_mask=chunk_attention_mask,
                                                        position_ids=position_ids, token_type_ids=token_type_ids,
                                                        head_mask=head_mask, offsets=offsets, gather_index=gather_index)
        chunk_CLS = seq_outputs[1]
        CLS_ensem = self.cls_ensemble(torch.cat((global_CLS, chunk_CLS), -1))

        chunk_align = seq_outputs[0][:, 1:hypo_len]
        global_hypo = global_output[:, 1:hypo_len]
        chunk_hidden_states = chunk_hidden_states[:, 1:hypo_len]
        self_chunk_align = torch.cat((global_hypo, chunk_align, chunk_hidden_states), dim=1)

        word_mask = input_mask[:, 1:hypo_len].unsqueeze(1)
        word_mask = (1.0 - word_mask) * -10000.0
        word_mask = torch.cat((word_mask, word_mask, word_mask), -1)  # batch,word*2,hidden
        attn_weight = []
        for i, layer_module in enumerate(self.cls_layer):
            CLS_ensem, attn_weight_layer = layer_module(self_chunk_align, CLS_ensem, word_mask)
            attn_weight.append(attn_weight_layer)

        # 找到CLS最关注的词语
        attn_weight = torch.stack(attn_weight).cuda()
        attn_weight = attn_weight.sum(dim=0).squeeze(1)[:, :hypo_len - 1]
        constraint_scores, constraint_index = torch.sort(attn_weight, descending=True, dim=-1)
        constraint_ids = input_ids[:, 1:].unsqueeze(1).repeat(1, 3, 1).reshape(input_ids.size(0), -1)
        constraint_ids = torch.gather(constraint_ids, -1, constraint_index)
        total_token = []
        for i in range(batch_size):
            e_qn_index = torch.nonzero(input_ids[i] == self.enc_toker.sep_token_id)[0, 0]
            total_token.append(e_qn_index * 3)
        total_token = torch.stack(total_token)
        mid_index = total_token // 2
        tokens = [[] for i in range(batch_size)]
        add_score_ids = [[] for i in range(batch_size)]

        for idx in range(batch_size):
            ids = constraint_ids[idx].tolist()
            for id_index, id in enumerate(ids):
                if id_index >= mid_index[idx]:
                    break
                token = self.enc_toker.decode(id, skip_special_tokens=True)
                token = token.replace('.', '').replace(',', '')
                if '#' not in token and token not in self.stop_words and len(token) > 2 and token not in tokens[idx]:
                    tokens[idx].append(token)
                    token = self.dec_toker.tokenize(' ' + token)
                    token = self.dec_toker.encode(token)[0]
                    add_score_ids[idx].append(token)

        logits = self.classifier(CLS_ensem)
        logit_expl = binary_to_mp(logits)
        pre = logit_expl.max(dim=-1)[1]
        label = label.reshape(-1, 4)
        label = torch.argmax(label, -1)
        matched_0 = pre == label

        add_score_ids = [add_score_ids[i:i + 4] for i in range(0, len(add_score_ids), 4)]
        add_score_ids_chosen = []
        for index, ids in enumerate(add_score_ids):
            add_score_ids_chosen.append(ids[label[index]])

        encoder_hs = torch.cat(
            (seq_outputs[0][:, 1:hypo_len], global_output[:, 1:hypo_len], chunk_hidden_states), 1)
        encoder_mask = torch.cat((input_mask[:, 1:hypo_len], input_mask[:, 1:hypo_len], input_mask[:, 1:hypo_len]), 1)

        ques_num = label.size(0)
        add_tensor = torch.arange(ques_num, dtype=torch.int64).cuda()
        add_tensor = add_tensor * 4
        gather_index = add_tensor + label
        encoder_mask_chosen = torch.gather(encoder_mask, 0, gather_index.unsqueeze(-1).repeat(1, encoder_mask.size(-1)))
        encoder_hs_chosen = torch.gather(encoder_hs, 0,
                                         gather_index.unsqueeze(-1).unsqueeze(-1).repeat(1, encoder_hs.size(-2),
                                                                                         encoder_hs.size(-1)))
        expl_ids = expl_ids.reshape(ques_num, 4, -1)[:, 0, :]

        outputs = [''] * matched_0.size(0)
        batch_size = 1

        for i in range(matched_0.size(0)):
            if matched_0[i] == torch.tensor(True):
                logits_processor = LogitsProcessorList(
                    [RepetitionPenaltyLogitsProcessor(penalty=self.repeat_penalty),
                     # MinLengthLogitsProcessor(6, eos_token_id=self.e_rtnl)
                     ])
                # instantiate logits processors
                logits_warper = LogitsProcessorList([TopKLogitsWarper(32)])

                decoder_input = expl_ids[i]
                b_rtnl_index = torch.nonzero((decoder_input == self.b_rtnl.unsqueeze(0)).to(torch.int64))[0, -1]
                prompt_decoder_input = decoder_input[:b_rtnl_index]
                cur_len = prompt_decoder_input.size(0)
                max_gen_len = cur_len + 50
                beam_scorer = BeamSearchScorer_constrained(
                    batch_size=batch_size,
                    num_beams=self.beam_size,
                    max_length=max_gen_len,
                    device=torch.device('cuda'),
                    length_penalty=self.length_penalty,
                    constrained=self.constrained)
                beam_output = self.beam_sample(input_ids=prompt_decoder_input.unsqueeze(0), beam_scorer=beam_scorer,
                                               encoder_hidden_states=encoder_hs_chosen[i].unsqueeze(0),
                                               encoder_mask=encoder_mask_chosen[i].unsqueeze(0),
                                               logits_processor=logits_processor,
                                               logits_warper=logits_warper, add_score_ids=add_score_ids_chosen[i])
                outputs[i] = beam_output[0, b_rtnl_index:]

        return outputs, matched_0, pre

    def beam_sample(
            self,
            input_ids: torch.LongTensor,
            beam_scorer,
            encoder_hidden_states,
            encoder_mask,
            add_score_ids=None,
            logits_processor=None,
            stopping_criteria=None,
            logits_warper=None,
            max_length=None,
            pad_token_id=None,
            eos_token_id=None,
            output_attentions=None,
            output_hidden_states=None,
            output_scores=None,
            return_dict_in_generate=None,
            **model_kwargs,

    ):
        # init values
        batch_beam_size, cur_len = input_ids.shape
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        max_length = cur_len + 50
        validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.dec_toker.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.e_rtnl
        # eos_token_id = eos_token_id if eos_token_id is not None else self.gpt_toker.convert_tokens_to_ids('<|e_exp|>')
        output_scores = True
        output_attentions = False
        output_hidden_states = (output_hidden_states)
        return_dict_in_generate = (return_dict_in_generate)

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        batch_size = encoder_hidden_states.size(0)
        num_beams = beam_scorer.num_beams

        input_ids = input_ids.unsqueeze(1).repeat(1, num_beams, 1).view(batch_size * num_beams, -1)

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores = beam_scores.view((batch_size * num_beams,))
        encoder_hidden_states_ex = encoder_hidden_states.unsqueeze(1).repeat(1, num_beams, 1, 1)
        encoder_hidden_states_ex = encoder_hidden_states_ex.reshape(
            (batch_size * num_beams, -1, encoder_hidden_states_ex.size(-1)))
        encoder_mask_ex = encoder_mask.unsqueeze(1).repeat(1, num_beams, 1)
        encoder_mask_ex = encoder_mask_ex.reshape(
            (batch_size * num_beams, -1))

        while cur_len < max_length:
            outputs = self.dec(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states_ex,
                               encoder_attention_mask=encoder_mask_ex)

            next_token_logits = self.lm_head(outputs.last_hidden_state[:, -1, :])

            next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

            next_token_scores = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            probs = F.softmax(next_token_scores, dim=-1)

            next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)
            next_token_scores = torch.gather(next_token_scores, -1, next_tokens)

            next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
            next_tokens = torch.gather(next_tokens, -1, _indices)

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                add_score_ids=add_score_ids,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            if beam_scorer.is_done:
                break

            if stopping_criteria(input_ids, scores):
                break

        sequence_outputs = beam_scorer.finalize(
            input_ids, beam_scores, next_tokens, next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id
        )

        return sequence_outputs["sequences"]


class ChunkAlign_CLS_dec5_4_wo_chual_beam(nn.Module):
    def __init__(self, global_enc, dec, enc_toker, dec_toker, num_labels, repeat_penalty, length_penalty,
                 constrained):
        super(ChunkAlign_CLS_dec5_4_wo_chual_beam, self).__init__()
        self.global_enc = global_enc
        self.enc_toker = enc_toker
        self.dec_toker = dec_toker
        self.dec = dec

        self.num_labels = num_labels
        self.cls_layer_num = 3
        self.cls_layer = nn.ModuleList([ClsLayer2(self.global_enc.config) for _ in range(self.cls_layer_num)])
        self.classifier = nn.Linear(self.global_enc.config.hidden_size, 2)
        self.cls_loss_fct = CrossEntropyLoss()

        self.vocab_num = self.dec.vocab_size
        self.lm_head = nn.Linear(self.dec.config.n_embd, self.dec.vocab_size, bias=False)
        self.gen_criterion = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=self.dec_toker.pad_token_id)

        self.beam_size = 5
        self.b_rtnl = torch.tensor(self.dec_toker.encode("<|b_rtnl|>")).cuda()
        self.repeat_penalty = repeat_penalty
        self.length_penalty = length_penalty
        self.constrained = constrained
        self.stop_words = ['a', 'is', 'am', 'are', 'the', 'an', 'it', 'has', 'very', 'and', 'his', 'her', 'there',
                           'she', 'they', 'what', 'this', 'that', 'these', 'those', 'do', 'does', 'why', 'for',
                           'with', 'how', 'when', 'where', 'woman', 'man', 'women', 'men', 'some', 'while',
                           'have', 'them', 'him', 'than']  # 1

    def test_beam(self, input_ids, img_feat, expl_ids, input_mask=None, label=None,
                  token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                  offsets=None, chunk_attention_mask=None, gather_index=None):
        hypo_len = input_ids.size(1)
        batch_size = input_ids.size(0)

        outputs = self.global_enc(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        global_output = outputs[0]
        CLS_ensem = outputs[1]
        self_chunk_align = global_output[:, 1:hypo_len]
        word_mask = input_mask[:, 1:hypo_len].unsqueeze(1)
        word_mask = (1.0 - word_mask) * -10000.0

        attn_weight = []
        for i, layer_module in enumerate(self.cls_layer):
            CLS_ensem, attn_weight_layer = layer_module(self_chunk_align, CLS_ensem, word_mask)
            attn_weight.append(attn_weight_layer)

        # 找到CLS最关注的词语
        attn_weight = torch.stack(attn_weight).cuda()
        attn_weight = attn_weight.sum(dim=0).squeeze(1)[:, :hypo_len - 1]
        constraint_scores, constraint_index = torch.sort(attn_weight, descending=True, dim=-1)
        constraint_ids = input_ids[:, 1:].unsqueeze(1).repeat(1, 3, 1).reshape(input_ids.size(0), -1)
        constraint_ids = torch.gather(constraint_ids, -1, constraint_index)
        # 作为add_score_ids
        add_score_ids = [[] for i in range(batch_size)]
        tokens = [[] for i in range(batch_size)]
        for idx in range(batch_size):
            ids = constraint_ids[idx].tolist()
            for id in ids:
                token = self.enc_toker.decode(id, skip_special_tokens=True)
                token = token.replace('.', '').replace(',', '')
                if '#' not in token and token not in self.stop_words and len(token) > 2:
                    tokens[idx].append(token)
                    token = self.dec_toker.tokenize(' ' + token)
                    token = self.dec_toker.encode(token)[0]
                    add_score_ids[idx].append(token)
                    if len(add_score_ids[idx]) == 3:
                        break

        logits = self.classifier(CLS_ensem)
        logit_expl = binary_to_mp(logits)
        pre = logit_expl.max(dim=-1)[1]
        label = label.reshape(-1, 4)
        label = torch.argmax(label, -1)
        matched_0 = pre == label

        add_score_ids = [add_score_ids[i:i + 4] for i in range(0, len(add_score_ids), 4)]
        add_score_ids_chosen = []
        for index, ids in enumerate(add_score_ids):
            add_score_ids_chosen.append(ids[label[index]])

        encoder_hs = global_output[:, 1:hypo_len].contiguous()
        encoder_mask = input_mask[:, 1:hypo_len]
        ques_num = label.size(0)
        add_tensor = torch.arange(ques_num, dtype=torch.int64).cuda()
        add_tensor = add_tensor * 4
        gather_index = add_tensor + label
        encoder_mask_chosen = torch.gather(encoder_mask, 0, gather_index.unsqueeze(-1).repeat(1, encoder_mask.size(-1)))
        encoder_hs_chosen = torch.gather(encoder_hs, 0,
                                         gather_index.unsqueeze(-1).unsqueeze(-1).repeat(1, encoder_hs.size(-2),
                                                                                         encoder_hs.size(-1)))
        expl_ids = expl_ids.reshape(ques_num, 4, -1)[:, 0, :]

        outputs = [''] * matched_0.size(0)
        batch_size = 1

        for i in range(matched_0.size(0)):
            if matched_0[i] == torch.tensor(True):
                logits_processor = LogitsProcessorList(
                    [RepetitionPenaltyLogitsProcessor(penalty=self.repeat_penalty),
                     # MinLengthLogitsProcessor(6, eos_token_id=self.e_rtnl)
                     ])
                # instantiate logits processors
                logits_warper = LogitsProcessorList([TopKLogitsWarper(32)])

                decoder_input = expl_ids[i]
                b_rtnl_index = torch.nonzero((decoder_input == self.b_rtnl.unsqueeze(0)).to(torch.int64))[0, -1]
                prompt_decoder_input = decoder_input[:b_rtnl_index]
                cur_len = prompt_decoder_input.size(0)
                max_gen_len = cur_len + 50
                beam_scorer = BeamSearchScorer_constrained(
                    batch_size=batch_size,
                    num_beams=self.beam_size,
                    max_length=max_gen_len,
                    device=torch.device('cuda'),
                    length_penalty=self.length_penalty,
                    constrained=self.constrained)
                beam_output = self.beam_sample(input_ids=prompt_decoder_input.unsqueeze(0), beam_scorer=beam_scorer,
                                               encoder_hidden_states=encoder_hs_chosen[i].unsqueeze(0),
                                               encoder_mask=encoder_mask_chosen[i].unsqueeze(0),
                                               logits_processor=logits_processor,
                                               logits_warper=logits_warper, add_score_ids=add_score_ids_chosen[i])
                outputs[i] = beam_output[0, b_rtnl_index + 1:]

        return outputs, matched_0, pre

    def beam_sample(
            self,
            input_ids: torch.LongTensor,
            beam_scorer,
            encoder_hidden_states,
            encoder_mask,
            add_score_ids=None,
            logits_processor=None,
            stopping_criteria=None,
            logits_warper=None,
            max_length=None,
            pad_token_id=None,
            eos_token_id=None,
            output_attentions=None,
            output_hidden_states=None,
            output_scores=None,
            return_dict_in_generate=None,
            **model_kwargs,

    ):
        # init values
        batch_beam_size, cur_len = input_ids.shape
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        max_length = cur_len + 50
        validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.dec_toker.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.dec_toker.eos_token_id
        # eos_token_id = eos_token_id if eos_token_id is not None else self.gpt_toker.convert_tokens_to_ids('<|e_exp|>')
        output_scores = True
        output_attentions = False
        output_hidden_states = (output_hidden_states)
        return_dict_in_generate = (return_dict_in_generate)

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        batch_size = encoder_hidden_states.size(0)
        num_beams = beam_scorer.num_beams

        input_ids = input_ids.unsqueeze(1).repeat(1, num_beams, 1).view(batch_size * num_beams, -1)

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores = beam_scores.view((batch_size * num_beams,))
        encoder_hidden_states_ex = encoder_hidden_states.unsqueeze(1).repeat(1, num_beams, 1, 1)
        encoder_hidden_states_ex = encoder_hidden_states_ex.reshape(
            (batch_size * num_beams, -1, encoder_hidden_states_ex.size(-1)))
        encoder_mask_ex = encoder_mask.unsqueeze(1).repeat(1, num_beams, 1)
        encoder_mask_ex = encoder_mask_ex.reshape(
            (batch_size * num_beams, -1))

        while cur_len < max_length:
            outputs = self.dec(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states_ex,
                               encoder_attention_mask=encoder_mask_ex)

            next_token_logits = self.lm_head(outputs.last_hidden_state[:, -1, :])

            next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

            next_token_scores = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            probs = F.softmax(next_token_scores, dim=-1)

            next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)
            next_token_scores = torch.gather(next_token_scores, -1, next_tokens)

            next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
            next_tokens = torch.gather(next_tokens, -1, _indices)

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                add_score_ids=add_score_ids,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            if beam_scorer.is_done:
                break

            if stopping_criteria(input_ids, scores):
                break

        sequence_outputs = beam_scorer.finalize(
            input_ids, beam_scores, next_tokens, next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id
        )

        return sequence_outputs["sequences"]


class ChunkAlign_CLS_dec5_4_wo_reasoning_beam(nn.Module):
    def __init__(self, global_enc, seq_enc, dec, enc_toker, dec_toker, num_labels, repeat_penalty, length_penalty,
                 constrained):
        super(ChunkAlign_CLS_dec5_4_wo_reasoning_beam, self).__init__()
        self.global_enc = global_enc
        self.seq_enc = seq_enc
        self.enc_toker = enc_toker
        self.dec_toker = dec_toker
        self.dec = dec

        self.cls_ensemble = nn.Linear(self.global_enc.config.hidden_size * 2, self.global_enc.config.hidden_size)
        self.num_labels = num_labels

        self.classifier = nn.Linear(self.global_enc.config.hidden_size, 2)
        self.cls_loss_fct = CrossEntropyLoss()

        self.vocab_num = self.dec.vocab_size
        self.lm_head = nn.Linear(self.dec.config.n_embd, self.dec.vocab_size, bias=False)
        self.gen_criterion = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=self.dec_toker.pad_token_id)

        self.beam_size = 5
        self.b_rtnl = torch.tensor(self.dec_toker.encode("<|b_rtnl|>")).cuda()
        self.repeat_penalty = repeat_penalty
        self.length_penalty = length_penalty
        self.constrained = constrained
        self.stop_words = ['a', 'is', 'am', 'are', 'the', 'an', 'it', 'has', 'very', 'and', 'his', 'her', 'there',
                           'she', 'they', 'what', 'this', 'that', 'these', 'those', 'do', 'does', 'why', 'for',
                           'with', 'how', 'when', 'where', 'woman', 'man', 'women', 'men', 'some', 'while',
                           'have', 'them', 'him', 'than']  # 1

    def test_beam(self, input_ids, img_feat, expl_ids, input_mask=None, label=None,
                  token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                  offsets=None, chunk_attention_mask=None, gather_index=None):
        hypo_len = input_ids.size(1)
        batch_size = input_ids.size(0)

        outputs = self.global_enc(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        global_output = outputs[0]
        global_CLS = outputs[1]

        img_mask = input_mask[:, hypo_len:]
        seq_outputs, chunk_hidden_states = self.seq_enc(input_ids, img_feats=img_feat, img_mask=img_mask,
                                                        input_mask=input_mask,
                                                        attention_mask=chunk_attention_mask,
                                                        position_ids=position_ids, token_type_ids=token_type_ids,
                                                        head_mask=head_mask, offsets=offsets, gather_index=gather_index)
        chunk_CLS = seq_outputs[1]
        CLS_ensem = self.cls_ensemble(torch.cat((global_CLS, chunk_CLS), -1))

        logits = self.classifier(CLS_ensem)

        logit_expl = binary_to_mp(logits)
        pre = logit_expl.max(dim=-1)[1]
        label = label.reshape(-1, 4)
        label = torch.argmax(label, -1)
        matched_0 = pre == label

        add_score_ids = [[] for i in range(batch_size)]

        add_score_ids = [add_score_ids[i:i + 4] for i in range(0, len(add_score_ids), 4)]
        add_score_ids_chosen = []
        for index, ids in enumerate(add_score_ids):
            add_score_ids_chosen.append(ids[label[index]])

        encoder_hs = torch.cat(
            (seq_outputs[0][:, 1:hypo_len], global_output[:, 1:hypo_len], chunk_hidden_states[:, 1:hypo_len]), 1)
        encoder_mask = torch.cat((input_mask[:, 1:hypo_len], input_mask[:, 1:hypo_len], input_mask[:, 1:hypo_len]), 1)

        ques_num = label.size(0)
        add_tensor = torch.arange(ques_num, dtype=torch.int64).cuda()
        add_tensor = add_tensor * 4
        gather_index = add_tensor + label
        encoder_mask_chosen = torch.gather(encoder_mask, 0, gather_index.unsqueeze(-1).repeat(1, encoder_mask.size(-1)))
        encoder_hs_chosen = torch.gather(encoder_hs, 0,
                                         gather_index.unsqueeze(-1).unsqueeze(-1).repeat(1, encoder_hs.size(-2),
                                                                                         encoder_hs.size(-1)))
        expl_ids = expl_ids.reshape(ques_num, 4, -1)[:, 0, :]

        outputs = [''] * matched_0.size(0)
        batch_size = 1

        for i in range(matched_0.size(0)):
            if matched_0[i] == torch.tensor(True):
                logits_processor = LogitsProcessorList(
                    [RepetitionPenaltyLogitsProcessor(penalty=self.repeat_penalty),
                     # MinLengthLogitsProcessor(6, eos_token_id=self.e_rtnl)
                     ])
                # instantiate logits processors
                logits_warper = LogitsProcessorList([TopKLogitsWarper(32)])

                decoder_input = expl_ids[i]
                b_rtnl_index = torch.nonzero((decoder_input == self.b_rtnl.unsqueeze(0)).to(torch.int64))[0, -1]
                prompt_decoder_input = decoder_input[:b_rtnl_index]
                cur_len = prompt_decoder_input.size(0)
                max_gen_len = cur_len + 50
                beam_scorer = BeamSearchScorer_constrained(
                    batch_size=batch_size,
                    num_beams=self.beam_size,
                    max_length=max_gen_len,
                    device=torch.device('cuda'),
                    length_penalty=self.length_penalty,
                    constrained=self.constrained)
                beam_output = self.beam_sample(input_ids=prompt_decoder_input.unsqueeze(0), beam_scorer=beam_scorer,
                                               encoder_hidden_states=encoder_hs_chosen[i].unsqueeze(0),
                                               encoder_mask=encoder_mask_chosen[i].unsqueeze(0),
                                               logits_processor=logits_processor,
                                               logits_warper=logits_warper, add_score_ids=add_score_ids_chosen[i])
                outputs[i] = beam_output[0, b_rtnl_index + 1:]

        return outputs, matched_0, pre

    def beam_sample(
            self,
            input_ids: torch.LongTensor,
            beam_scorer,
            encoder_hidden_states,
            encoder_mask,
            add_score_ids=None,
            logits_processor=None,
            stopping_criteria=None,
            logits_warper=None,
            max_length=None,
            pad_token_id=None,
            eos_token_id=None,
            output_attentions=None,
            output_hidden_states=None,
            output_scores=None,
            return_dict_in_generate=None,
            **model_kwargs,

    ):
        # init values
        batch_beam_size, cur_len = input_ids.shape
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        max_length = cur_len + 50
        validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.dec_toker.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.dec_toker.eos_token_id
        # eos_token_id = eos_token_id if eos_token_id is not None else self.gpt_toker.convert_tokens_to_ids('<|e_exp|>')
        output_scores = True
        output_attentions = False
        output_hidden_states = (output_hidden_states)
        return_dict_in_generate = (return_dict_in_generate)

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        batch_size = encoder_hidden_states.size(0)
        num_beams = beam_scorer.num_beams

        input_ids = input_ids.unsqueeze(1).repeat(1, num_beams, 1).view(batch_size * num_beams, -1)

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores = beam_scores.view((batch_size * num_beams,))
        encoder_hidden_states_ex = encoder_hidden_states.unsqueeze(1).repeat(1, num_beams, 1, 1)
        encoder_hidden_states_ex = encoder_hidden_states_ex.reshape(
            (batch_size * num_beams, -1, encoder_hidden_states_ex.size(-1)))
        encoder_mask_ex = encoder_mask.unsqueeze(1).repeat(1, num_beams, 1)
        encoder_mask_ex = encoder_mask_ex.reshape(
            (batch_size * num_beams, -1))

        while cur_len < max_length:
            outputs = self.dec(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states_ex,
                               encoder_attention_mask=encoder_mask_ex)

            next_token_logits = self.lm_head(outputs.last_hidden_state[:, -1, :])

            next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

            next_token_scores = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            probs = F.softmax(next_token_scores, dim=-1)

            next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)
            next_token_scores = torch.gather(next_token_scores, -1, next_tokens)

            next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
            next_tokens = torch.gather(next_tokens, -1, _indices)

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                add_score_ids=add_score_ids,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            if beam_scorer.is_done:
                break

            if stopping_criteria(input_ids, scores):
                break

        sequence_outputs = beam_scorer.finalize(
            input_ids, beam_scores, next_tokens, next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id
        )

        return sequence_outputs["sequences"]


class ChunkAlign_CLS_dec5_4_wo_chual_reasoning_beam(nn.Module):
    def __init__(self, global_enc, dec, enc_toker, dec_toker, num_labels, repeat_penalty, length_penalty,
                 constrained):
        super(ChunkAlign_CLS_dec5_4_wo_chual_reasoning_beam, self).__init__()
        self.oscar = global_enc
        self.enc_toker = enc_toker
        self.dec_toker = dec_toker
        self.dec = dec

        self.num_labels = num_labels
        self.classifier = nn.Linear(self.oscar.config.hidden_size, 2)
        self.cls_loss_fct = CrossEntropyLoss()

        self.vocab_num = self.dec.vocab_size
        self.lm_head = nn.Linear(self.dec.config.n_embd, self.dec.vocab_size, bias=False)
        self.gen_criterion = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=self.dec_toker.pad_token_id)

        self.beam_size = 5
        self.b_rtnl = torch.tensor(self.dec_toker.encode("<|b_rtnl|>")).cuda()
        self.repeat_penalty = repeat_penalty
        self.length_penalty = length_penalty
        self.constrained = constrained
        self.stop_words = ['a', 'is', 'am', 'are', 'the', 'an', 'it', 'has', 'very', 'and', 'his', 'her', 'there',
                           'she', 'they', 'what', 'this', 'that', 'these', 'those', 'do', 'does', 'why', 'for',
                           'with', 'how', 'when', 'where', 'woman', 'man', 'women', 'men', 'some', 'while',
                           'have', 'them', 'him', 'than', 'not', 'about']  # 1

    def test_beam(self, input_ids, img_feat, expl_ids, input_mask=None, label=None,
                  token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                  offsets=None, chunk_attention_mask=None, gather_index=None):
        hypo_len = input_ids.size(1)
        batch_size = input_ids.size(0)

        outputs = self.oscar(input_ids, img_feats=img_feat, attention_mask=input_mask,
                             position_ids=position_ids, token_type_ids=token_type_ids,
                             head_mask=head_mask,
                             encoder_history_states=encoder_history_states)
        global_output = outputs[0]
        global_CLS = outputs[1]
        attn_weight = outputs[2]

        add_score_ids = [[] for i in range(batch_size)]


        logits = self.classifier(global_CLS)
        logit_expl = binary_to_mp(logits)
        pre = logit_expl.max(dim=-1)[1]
        label = label.reshape(-1, 4)
        label = torch.argmax(label, -1)
        matched_0 = pre == label

        add_score_ids = [add_score_ids[i:i + 4] for i in range(0, len(add_score_ids), 4)]
        add_score_ids_chosen = []
        for index, ids in enumerate(add_score_ids):
            add_score_ids_chosen.append(ids[label[index]])

        encoder_hs = global_output[:, 1:hypo_len]
        encoder_mask = input_mask[:, 1:hypo_len]

        ques_num = label.size(0)
        add_tensor = torch.arange(ques_num, dtype=torch.int64).cuda()
        add_tensor = add_tensor * 4
        gather_index = add_tensor + label
        encoder_mask_chosen = torch.gather(encoder_mask, 0, gather_index.unsqueeze(-1).repeat(1, encoder_mask.size(-1)))
        encoder_hs_chosen = torch.gather(encoder_hs, 0,
                                         gather_index.unsqueeze(-1).unsqueeze(-1).repeat(1, encoder_hs.size(-2),
                                                                                         encoder_hs.size(-1)))
        expl_ids = expl_ids.reshape(ques_num, 4, -1)[:, 0, :]

        outputs = [''] * matched_0.size(0)
        batch_size = 1

        for i in range(matched_0.size(0)):
            if matched_0[i] == torch.tensor(True):
                logits_processor = LogitsProcessorList(
                    [RepetitionPenaltyLogitsProcessor(penalty=self.repeat_penalty),
                     # MinLengthLogitsProcessor(6, eos_token_id=self.e_rtnl)
                     ])
                # instantiate logits processors
                logits_warper = LogitsProcessorList([TopKLogitsWarper(32)])

                decoder_input = expl_ids[i]
                b_rtnl_index = torch.nonzero((decoder_input == self.b_rtnl.unsqueeze(0)).to(torch.int64))[0, -1]
                prompt_decoder_input = decoder_input[:b_rtnl_index]
                cur_len = prompt_decoder_input.size(0)
                max_gen_len = cur_len + 50
                beam_scorer = BeamSearchScorer_constrained(
                    batch_size=batch_size,
                    num_beams=self.beam_size,
                    max_length=max_gen_len,
                    device=torch.device('cuda'),
                    length_penalty=self.length_penalty,
                    constrained=self.constrained)
                beam_output = self.beam_sample(input_ids=prompt_decoder_input.unsqueeze(0), beam_scorer=beam_scorer,
                                               encoder_hidden_states=encoder_hs_chosen[i].unsqueeze(0),
                                               encoder_mask=encoder_mask_chosen[i].unsqueeze(0),
                                               logits_processor=logits_processor,
                                               logits_warper=logits_warper, add_score_ids=add_score_ids_chosen[i])
                outputs[i] = beam_output[0, b_rtnl_index + 1:]

        return outputs, matched_0, pre

    def beam_sample(
            self,
            input_ids: torch.LongTensor,
            beam_scorer,
            encoder_hidden_states,
            encoder_mask,
            add_score_ids=None,
            logits_processor=None,
            stopping_criteria=None,
            logits_warper=None,
            max_length=None,
            pad_token_id=None,
            eos_token_id=None,
            output_attentions=None,
            output_hidden_states=None,
            output_scores=None,
            return_dict_in_generate=None,
            **model_kwargs,

    ):
        # init values
        batch_beam_size, cur_len = input_ids.shape
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        max_length = cur_len + 50
        validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.dec_toker.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.dec_toker.eos_token_id
        # eos_token_id = eos_token_id if eos_token_id is not None else self.gpt_toker.convert_tokens_to_ids('<|e_exp|>')
        output_scores = True
        output_attentions = False
        output_hidden_states = (output_hidden_states)
        return_dict_in_generate = (return_dict_in_generate)

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        batch_size = encoder_hidden_states.size(0)
        num_beams = beam_scorer.num_beams

        input_ids = input_ids.unsqueeze(1).repeat(1, num_beams, 1).view(batch_size * num_beams, -1)

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores = beam_scores.view((batch_size * num_beams,))
        encoder_hidden_states_ex = encoder_hidden_states.unsqueeze(1).repeat(1, num_beams, 1, 1)
        encoder_hidden_states_ex = encoder_hidden_states_ex.reshape(
            (batch_size * num_beams, -1, encoder_hidden_states_ex.size(-1)))
        encoder_mask_ex = encoder_mask.unsqueeze(1).repeat(1, num_beams, 1)
        encoder_mask_ex = encoder_mask_ex.reshape(
            (batch_size * num_beams, -1))

        while cur_len < max_length:
            outputs = self.dec(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states_ex,
                               encoder_attention_mask=encoder_mask_ex)

            next_token_logits = self.lm_head(outputs.last_hidden_state[:, -1, :])

            next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

            next_token_scores = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            probs = F.softmax(next_token_scores, dim=-1)

            next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)
            next_token_scores = torch.gather(next_token_scores, -1, next_tokens)

            next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
            next_tokens = torch.gather(next_tokens, -1, _indices)

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                add_score_ids=add_score_ids,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            if beam_scorer.is_done:
                break

            if stopping_criteria(input_ids, scores):
                break

        sequence_outputs = beam_scorer.finalize(
            input_ids, beam_scores, next_tokens, next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id
        )

        return sequence_outputs["sequences"]
