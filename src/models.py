#!/usr/bin/env python3

# region         === importations ===         NOIGER #
from typing import Union
from typing import Tuple
from typing import Optional
from typing import List

import pandas as pd

import torch
from torch import nn
from torch import Tensor
from torch import FloatTensor
from torch import LongTensor
from torch.nn.modules.loss import L1Loss
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import BartModel
from transformers import BartForConditionalGeneration
from transformers import BartConfig
from transformers.models.bart.modeling_bart import BartEncoder, BartDecoder
from transformers import logging
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.modeling_outputs import Seq2SeqModelOutput
from transformers.modeling_outputs import CausalLMOutput
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers import get_linear_schedule_with_warmup
from transformers.generation_utils import GenerationMixin

import pytorch_lightning as pl

from .torch_cif import cif_function
from .utils import mask_generator
from .datasets import DataSetCollectorGeneral


logger = logging.get_logger("transformers.models.bart.modeling_bart")
import math
from transformers.models.bart.modeling_bart import BartLearnedPositionalEmbedding, BartEncoderLayer
from transformers.models.bart.modeling_bart import _expand_mask
import random

from dataclasses import dataclass
# endregion      === importations ===      NOIGERDNE #


# region       === aug_dataclasses ===        NOIGER #
@dataclass
class AugBaseModelOutput(BaseModelOutput):
    last_hidden_state: torch.FloatTensor = None
    out_attention_mask: Optional[torch.FloatTensor] = None
    length_loss: Optional[Tuple[torch.FloatTensor]] = None
    pred_word_lengths: Optional[torch.LongTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class AugSeq2SeqModelOutput(Seq2SeqModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_last_hidden_out_attention_mask: Optional[torch.FloatTensor] = None
    encoder_length_loss: Optional[Tuple[torch.FloatTensor]] = None
    encoder_pred_word_lengths: Optional[torch.LongTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class AugSeq2SeqLMOutput(Seq2SeqLMOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_last_hidden_out_attention_mask: Optional[torch.FloatTensor] = None
    encoder_length_loss: Optional[Tuple[torch.FloatTensor]] = None
    encoder_pred_word_lengths: Optional[torch.LongTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    masked_lm_loss: Optional[torch.FloatTensor] = None
    real_length_loss: Optional[torch.FloatTensor] = None
# endregion    === aug_dataclasses ===     NOIGERDNE #

class SentBartEncoder(BartEncoder):
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config, embed_tokens)
        embed_dim = config.d_model
        self.collapse_n = getattr(config, "collapse_n", 0)
        self.config.collapse_n = self.collapse_n
        if self.collapse_n == -1:
            self.skip_cif = True
        else:
            self.skip_cif = False
        
        self.post_initialization(embed_dim, word_extractor=cif_function)
        
    def post_initialization(self, embed_dim, word_extractor=cif_function):
        # == encoder == #
        self.alpha_predictor = nn.Linear(embed_dim, 1)  # TODO: check!
        self.word_extractor = word_extractor
        self.length_predictor = nn.Linear(embed_dim, 1)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        word_length_tensor: Optional[torch.LongTensor] = None,
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
      ) -> Union[Tuple, AugBaseModelOutput]:
        encoder_outputs = super().forward(
            input_ids,
            attention_mask,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_outputs[0],
            hidden_states=encoder_outputs[1] 
                if len(encoder_outputs) > 1 else None,
            attentions=encoder_outputs[2] 
                if len(encoder_outputs) > 2 else None,
        )

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if self.skip_cif:
            hidden_states = encoder_outputs.last_hidden_state
            out_attention_mask = attention_mask
            length_loss = (
                torch.zeros(len(hidden_states)).float(),
                torch.zeros(len(hidden_states)).long(),)
            pred_word_lengths = None
        else:
            (hidden_states, out_attention_mask, length_loss,
                          # out_attention_mask := (
                          #     encoder_output_attention_mask)
             pred_word_lengths,
             _, _) = self.sent_retriever(
                encoder_outputs.last_hidden_state,
                word_length_tensor=word_length_tensor,
                padding_mask=1 - attention_mask)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        if not return_dict:
            return tuple(v for v in [
                hidden_states, 
                out_attention_mask, 
                encoder_outputs.hidden_states,
                length_loss,
                pred_word_lengths,
                encoder_outputs.attentions,
            # ] if v is not None)
            ] if v is not None)
        return AugBaseModelOutput(
            last_hidden_state=hidden_states, 
            out_attention_mask=out_attention_mask, 
            length_loss=length_loss,
            pred_word_lengths=pred_word_lengths,
            hidden_states=encoder_outputs.hidden_states,
            # # attentions=encoder_outputs.attentions,
            # attentions=encoder_outputs.attentions,
        )

    def sent_retriever(self, 
        encoder__last_hidden_state, 
        word_length_tensor=None,
        padding_mask=None,
        return_all=False,
        return_original=False,
      ):
        alpha_values = self.alpha_predictor(encoder__last_hidden_state)
        alpha_values = alpha_values.squeeze(-1).sigmoid()  # B x S

        if word_length_tensor is None:
            # print("No given! self predict")
            word_length_tensor = alpha_values.sum(-1).long()
        else:
            # print("Wordlen given")
            # predicted_word_length_tensor = alpha_values.sum(-1).long()
            pass

        encoder__word_representations_CIF = (
            self.word_extractor(
                encoder__last_hidden_state,
                alpha=alpha_values,
                padding_mask=padding_mask,
                target_lengths=word_length_tensor,
            )
        )
        # TODO: Keep all CIF
        [encoder_word_representation] = encoder__word_representations_CIF['cif_out']
        [pred_word_lengths] = encoder__word_representations_CIF['alpha_sum']
        encoder_word_representation = encoder_word_representation.contiguous()
        # pred_word_lengths = pred_word_lengths.contiguous()
        # length_loss = 0.
        length_loss = ((pred_word_lengths, word_length_tensor,)
            if word_length_tensor is not None else
            (word_length_tensor, word_length_tensor,))
            # aliased as `encoder_word_representation`
            # FIXME: distributed problem!
            # TODO: add other CIF ouptuts!
        # length_loss = length_loss.contiguous()

        encoder_output_attention_mask = (
            # mask_generator(word_length_tensor) 
            mask_generator(word_length_tensor) 
            if word_length_tensor is not None else
            mask_generator(pred_word_lengths))
            # TODO: check length prediction!
            
        return (
            encoder_word_representation, 
            encoder_output_attention_mask, 
            length_loss,
            pred_word_lengths,
            encoder__word_representations_CIF if return_all else None,
            # encoder__last_hidden_state if return_original else None,
            encoder__last_hidden_state if return_original else None,
        )

class SentBart(BartModel):
    def __init__(self, config: BartConfig):
        super(BartModel, self).__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = SentBartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        word_length_tensor: Optional[torch.LongTensor] = None,
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, AugSeq2SeqModelOutput]:

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                word_length_tensor=word_length_tensor,
                # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # If the user passed a tuple for encoder_outputs, we wrap it in an AugBaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, AugBaseModelOutput):
            encoder_outputs = AugBaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                out_attention_mask=encoder_outputs[1],
                length_loss=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                pred_word_lengths=encoder_outputs[3] if len(encoder_outputs) > 3 else None,
                hidden_states=encoder_outputs[4] if len(encoder_outputs) > 4 else None,
                # attentions=encoder_outputs[5] if len(encoder_outputs) > 5 else None,
                    attentions=encoder_outputs[5] if len(encoder_outputs) > 5 else None,
            )
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            encoder_attention_mask=encoder_outputs[1],
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return AugSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_last_hidden_out_attention_mask=encoder_outputs.out_attention_mask,
            encoder_length_loss=encoder_outputs.length_loss,
            encoder_pred_word_lengths=encoder_outputs.pred_word_lengths,
            encoder_hidden_states=encoder_outputs.hidden_states,
            # # encoder_attentions=encoder_outputs.attentions,
            # encoder_attentions=encoder_outputs.attentions,
        )

    # region CHANGED >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def fix_encoder_(self, to_fix=True):
        self.encoder.requires_grad_(not to_fix)            
    # endregion <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

class SentBartForConditionalGeneration(BartForConditionalGeneration):
    def __init__(self, config: BartConfig):
        super(BartForConditionalGeneration, self).__init__(config)
        self.model = SentBart(config)
        self.register_buffer(
            "final_logits_bias", 
            torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(
            config.d_model, 
            self.model.shared.num_embeddings, 
            bias=False)

        self.weight_len = getattr(config, "weight_len", None)
        self.config.weight_len = self.weight_len

        # Initialize weights and apply final processing
        self.post_init()

    # region CHANGED >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def fix_encoder_(self, to_fix=True): 
        self.model.fix_encoder_(to_fix)
    # endregion <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        # region CHANGED >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        word_length_tensor: Optional[torch.LongTensor] = None,
        # endregion <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, AugSeq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        # print(self.model.encoder.layers[5].fc1.weight[0][:6] * 100)
        # print(self.model.encoder.alpha_predictor.weight[0][:6] * 100)
        # print(self.model.decoder.layers[0].fc1.weight[0][:6] * 100)
        
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            # region CHANGED >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            word_length_tensor=word_length_tensor,
            # endregion <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            # region CHANGED >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            # loss_fct = nn.CTCLoss(1)
            # masked_lm_loss = loss_fct(
            #     lm_logits, # $$$
            #     labels,
            #     attention_mask.sum(-1),
            #     (labels != 1).sum(-1))
            # breakpoint()
            # endregion <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        real_length_loss = None
        if word_length_tensor is not None and self.weight_len is not None:
            # region CHANGED >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            loss_fct = nn.L1Loss()
            # real_length_loss = torch.linalg.vector_norm(outputs.encoder_length_loss.float(), ord=1)
            real_length_loss = loss_fct(*(outputs.encoder_length_loss))
            # endregion <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        loss = ((masked_lm_loss if masked_lm_loss is not None else 0.)
              + ((self.weight_len if self.weight_len is not None else 0.
                 ) * (real_length_loss if real_length_loss is not None else 0.))
        ) if ((masked_lm_loss is not None) or (real_length_loss is not None)) else None

        if not return_dict:
            output = (lm_logits,) + outputs[1:] + (masked_lm_loss, real_length_loss,)
            return (
                ((loss,) + output) 
                if loss is not None else 
                output)

        return AugSeq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_last_hidden_out_attention_mask=outputs.encoder_last_hidden_out_attention_mask,
            encoder_length_loss=outputs.encoder_length_loss,
            encoder_pred_word_lengths=outputs.encoder_pred_word_lengths,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            masked_lm_loss=masked_lm_loss,
            real_length_loss=real_length_loss,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        word_length_tensor=None,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        output_dict = super().prepare_inputs_for_generation(
            decoder_input_ids,
            past,
            attention_mask,
            head_mask,
            decoder_head_mask,
            cross_attn_head_mask,
            use_cache,
            encoder_outputs,
            **kwargs
        )

        return {
            "word_length_tensor": word_length_tensor,
            **output_dict,
        }

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask=None,
        encoder_outputs=None,
        **model_kwargs,
    ):
        expanded_return_idx = (
            torch.arange(input_ids.shape[0])
                 .view(-1, 1)
                 .repeat(1, expand_size)
                 .view(-1)
                 .to(input_ids.device)
        )
        input_ids, model_kwargs = (
            GenerationMixin._expand_inputs_for_generation(
                input_ids,
                expand_size,
                is_encoder_decoder,
                attention_mask,
                encoder_outputs,
                **model_kwargs,
        ))
        if "word_length_tensor" in model_kwargs and model_kwargs.get("word_length_tensor") is not None:
            word_length_tensor = model_kwargs["word_length_tensor"]
            model_kwargs["word_length_tensor"] = word_length_tensor.index_select(0, expanded_return_idx)
        elif "word_length_tensor" in model_kwargs and model_kwargs.get("word_length_tensor") is None:
            model_kwargs.pop("word_length_tensor")
        # TODO: maybe check model arch?

        if "encoder_outputs" in model_kwargs:
            if "out_attention_mask" in encoder_outputs:
                out_attention_mask = model_kwargs["encoder_outputs"]["out_attention_mask"]
                model_kwargs["encoder_outputs"]["out_attention_mask"] = out_attention_mask.index_select(0, expanded_return_idx)

        return input_ids, model_kwargs
    # endregion <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
