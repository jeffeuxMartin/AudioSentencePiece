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
from .newutils import mask_generator
from .datasets import UnitDataset


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
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
# endregion    === aug_dataclasses ===     NOIGERDNE #

if """old""":
    # region             === old ===              NOIGER #
    class OldSentBartEncoder(BartEncoder):
        def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
            super().__init__(config)

            self.dropout = config.dropout
            self.layerdrop = config.encoder_layerdrop

            embed_dim = config.d_model
            self.padding_idx = config.pad_token_id
            self.max_source_positions = config.max_position_embeddings
            self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

            if embed_tokens is not None:
                self.embed_tokens = embed_tokens
            else:
                self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

            self.embed_positions = BartLearnedPositionalEmbedding(
                config.max_position_embeddings,
                embed_dim,
            )
            self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
            self.layernorm_embedding = nn.LayerNorm(embed_dim)

            self.gradient_checkpointing = False
            # Initialize weights and apply final processing
            self.post_init()
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
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            # retrieve input_ids and inputs_embeds
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
            elif input_ids is not None:
                input_shape = input_ids.size()
                input_ids = input_ids.view(-1, input_shape[-1])
            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")

            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

            embed_pos = self.embed_positions(input_shape)

            hidden_states = inputs_embeds + embed_pos
            hidden_states = self.layernorm_embedding(hidden_states)
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

            # expand attention_mask
            if attention_mask is not None:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

            encoder_states = () if output_hidden_states else None
            all_attentions = () if output_attentions else None

            # check if head_mask has a correct number of layers specified if desired
            if head_mask is not None:
                if head_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
                    )

            for idx, encoder_layer in enumerate(self.layers):
                if output_hidden_states:
                    encoder_states = encoder_states + (hidden_states,)
                # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
                dropout_probability = random.uniform(0, 1)
                if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                    layer_outputs = (None, None)
                else:
                    if self.gradient_checkpointing and self.training:

                        def create_custom_forward(module):
                            def custom_forward(*inputs):
                                return module(*inputs, output_attentions)

                            return custom_forward

                        layer_outputs = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(encoder_layer),
                            hidden_states,
                            attention_mask,
                            (head_mask[idx] if head_mask is not None else None),
                        )
                    else:
                        layer_outputs = encoder_layer(
                            hidden_states,
                            attention_mask,
                            layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                            output_attentions=output_attentions,
                        )

                    hidden_states = layer_outputs[0]

                if output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
                
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            (hidden_states, out_attention_mask,  
                            # attention_mask := encoder_output_attention_mask
            _, _) = self.sent_retriever(hidden_states, word_length_tensor)
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


            if not return_dict:
                return tuple(v for v in [
                    hidden_states, out_attention_mask, encoder_states, all_attentions
                    ] if v is not None)
            return AugBaseModelOutput(
                last_hidden_state=hidden_states, 
                out_attention_mask=out_attention_mask,
                hidden_states=encoder_states, 
                attentions=all_attentions
            )

        def sent_retriever(self, 
            encoder__last_hidden_state, 
            word_length_tensor=None,
            return_all=False,
            return_original=False,
        ):
            alpha_values = self.alpha_predictor(encoder__last_hidden_state)
            alpha_values = alpha_values.squeeze(-1).sigmoid()

            encoder__word_representations_CIF = (
                self.word_extractor(
                    encoder__last_hidden_state,
                    alpha=alpha_values,
                    padding_mask=None,
                    target_lengths=word_length_tensor,
                )
            )
            # TODO: Keep all CIF
            [encoder_word_representation] = encoder__word_representations_CIF['cif_out']
            encoder_word_representation = encoder_word_representation.contiguous()
                # aliased as `encoder_word_representation`
                # FIXME: distributed problem!
                # TODO: add other CIF ouptuts!

            encoder_output_attention_mask = (
                mask_generator(word_length_tensor) 
                if word_length_tensor is not None else
                None)
            
            
            return (
                encoder_word_representation, 
                encoder_output_attention_mask, 
                encoder__word_representations_CIF if return_all else None,
                encoder__last_hidden_state if return_original else None,
            )

    class OldSentBart(BartModel):
        def forward(   # Only 4 lines added
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            word_length_tensor: Optional[torch.LongTensor] = None,
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
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
                    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                    word_length_tensor=word_length_tensor,
                    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # If the user passed a tuple for encoder_outputs, we wrap it in an AugBaseModelOutput when return_dict=True
            elif return_dict and not isinstance(encoder_outputs, AugBaseModelOutput):
                encoder_outputs = AugBaseModelOutput(
                    last_hidden_state=encoder_outputs[0],
                    out_attention_mask=encoder_outputs[1],
                    hidden_states=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                    attentions=encoder_outputs[3] if len(encoder_outputs) > 3 else None,
                )
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_outputs[0],
                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                encoder_attention_mask=encoder_outputs[1],
                # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
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
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )

        # region CHANGED >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        def __init__(self, config: BartConfig):
            super().__init__(config)

            padding_idx, vocab_size = config.pad_token_id, config.vocab_size
            tgt_vocab_size = config.tgt_vocab_size if "tgt_vocab_size" in config.to_dict() else vocab_size
            config.tgt_vocab_size = tgt_vocab_size
            self.enc_emb = nn.Embedding(vocab_size, config.d_model, padding_idx)
            self.dec_emb = nn.Embedding(tgt_vocab_size, config.d_model, padding_idx)  # FIXME

            self.encoder = SentBartEncoder(config, self.enc_emb)
            self.decoder = BartDecoder(config, self.dec_emb)

            # Initialize weights and apply final processing
            self.post_init()

        def fix_encoder_(self, to_fix=True):
            _requires_grad_ = not to_fix
            self.encoder.requires_grad_(_requires_grad_)
            # self.alpha_predictor.requires_grad_(_requires_grad_)
            # self.length_predictor.requires_grad_(_requires_grad_)
            # self._word_extractor_fixer(to_fix)
        
        def _word_extractor_fixer(self, to_fix=True):  # FIXME: deprecated?
            _requires_grad_ = not to_fix
            if isinstance(self.word_extractor, cif_function):
                pass
            else:
                raise NotImplementedError  # if not CIF
                
        # endregion <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    class OldSentBartForConditionalGeneration(BartForConditionalGeneration):
        def __init__(self, config: BartConfig):  # Only 1 line added
            super().__init__(config)
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            self.model = SentBart(config)  # NOTE: lm_head change?
            self.register_buffer("final_logits_bias", 
                torch.zeros((1, self.model.dec_emb.num_embeddings)))
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            # Initialize weights and apply final processing
            self.post_init()
            self.lm_head = nn.Linear(  # FIXME??? 為什麼要 post_init 之後用？？？
                config.d_model, self.model.dec_emb.num_embeddings, bias=False)

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
                loss_fct = CrossEntropyLoss()  # TODO: ignore?
                masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.tgt_vocab_size), labels.view(-1))  # TODO:??? check 而已，應該 OK?
                # endregion <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            if not return_dict:
                output = (lm_logits,) + outputs[1:]
                return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

            return AugSeq2SeqLMOutput(
                loss=masked_lm_loss,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_last_hidden_out_attention_mask=outputs.encoder_last_hidden_out_attention_mask,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
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
            # cut decoder_input_ids if past is used
            if past is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]

            return {
                "input_ids": None,  # encoder_outputs is defined. input_ids not needed
                "word_length_tensor": word_length_tensor,
                "encoder_outputs": encoder_outputs,
                "past_key_values": past,
                "decoder_input_ids": decoder_input_ids,
                "attention_mask": attention_mask,
                "head_mask": head_mask,
                "decoder_head_mask": decoder_head_mask,
                "cross_attn_head_mask": cross_attn_head_mask,
                "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            }

        def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
            return shift_tokens_right(
                labels, 
                self.config.pad_token_id, 
                self.config.decoder_start_token_id)

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
                torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
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
            if "word_length_tensor" in model_kwargs:
                word_length_tensor = model_kwargs["word_length_tensor"]
                model_kwargs["word_length_tensor"] = word_length_tensor.index_select(0, expanded_return_idx)
            if "encoder_outputs" in model_kwargs:
                if "out_attention_mask" in encoder_outputs:
                    out_attention_mask = model_kwargs["encoder_outputs"]["out_attention_mask"]
                    model_kwargs["encoder_outputs"]["out_attention_mask"] = out_attention_mask.index_select(0, expanded_return_idx)
            return input_ids, model_kwargs
        # endregion <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    def pure_advanced_load_pretrained(
        pretrained_model, 
        model, 
        verbose=(lambda *a: None),
    ):
        pretrained_dict = dict(pretrained_model.state_dict())
        new_dict = model.state_dict()

        for key_src in pretrained_model.state_dict():
            val_src = pretrained_dict[key_src]
            val_tgt = new_dict.get(key_src, None)
            if val_tgt is not None:
                # 都有
                if val_src.shape != val_tgt.shape:
                    # 但重新更新了
                    verbose(f"{key_src} reshaped! {val_src.shape} | {val_tgt.shape}")
                    pretrained_dict.pop(key_src)
                else:
                    # OK
                    pass
            else:
                # 舊的有新的沒有
                verbose(f"{key_src} missing in new model! {val_src.shape}")
                pretrained_dict.pop(key_src)
        # 舊的沒有新的有，應該會被忽略！
        return model

    def advanced_load_pretrained(
        checkpoint_name, 
        model_class, 
        config_class, 
        verbose=print,
        **new_config_options,
    ):
        newconfig = config_class.from_pretrained(
            checkpoint_name, 
            **new_config_options)
        newconfig.update(new_config_options)  # FIXME: redundant???

        pretrained_model = model_class.from_pretrained(checkpoint_name)
        model = model_class(config=newconfig)
        
        model = pure_advanced_load_pretrained(
            pretrained_model=pretrained_model,
            model=model,
            verbose=verbose,
        )

        del pretrained_model
        return model

    class __PLSpeechToSemanticsNew(pl.LightningModule):
        def __init__(self, datasets, metric_cls, wandb_logger=None, *args, **kwargs):
            super().__init__()
            if "batch_size" in kwargs:
                self.batch_size = kwargs.get('batch_size', 9)
                kwargs.pop('batch_size')
            else:
                self.batch_size = 9

            self.my_hparams = {
                "lr": kwargs.get('lr', 2e-4),
                "weight_decay": 0.01,  # default value
            }

            if "lr" in kwargs:
                # self.lr = kwargs.get('lr', 9)
                kwargs.pop('lr')
            else:
                # self.lr = 9
                pass
            SpeechToSemantics = SentBartForConditionalGeneration
            self.model = SpeechToSemantics(*args, **kwargs)
            self.tokenizer = self.model.tokenizer
            self.wandb_logger = wandb_logger
            self.task_name = self.model.task_name
            self.trainset, self.valset = datasets
            
            # self.batch_size = 9 * 2  # TODO: check GPU numbers!
            # self.batch_size = 9  # TODO: check GPU numbers!
            self.demonstration_number = 3
            
            self.metrics = {
                "train": metric_cls(),
                "valid": metric_cls(),
                "validREAL": metric_cls(),
            }
            
            
        def forward(self, *args, **kwargs):
            return self.model(
                encoder_input_ids=kwargs["input_ids"],
                encoder_attention_mask=kwargs["attention_mask"],
                encoder_word_lengths_tensor=kwargs["word_length_tensor"],
                decoder_input_ids=kwargs["text_tokens"]["input_ids"],
                decoder_attention_mask=kwargs["text_tokens"]["attention_mask"],
            )
            
        def generate(self, *args, **kwargs):
            return self.model.generate(
                encoder_input_ids=kwargs["input_ids"],
                encoder_attention_mask=kwargs["attention_mask"],
                encoder_word_lengths_tensor=kwargs["word_length_tensor"],
                num_beams=kwargs["num_beams"],
                max_length=kwargs["max_length"],
            )
            
        def configure_optimizers__old(self):
            # Ref.: https://github.com/PyTorchLightning/pytorch-lightning/issues/328
            optimizer = torch.optim.AdamW(
                params=self.parameters(), 
                lr=self.my_hparams["lr"],
                weight_decay=self.my_hparams["weight_decay"],
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=5,
                verbose=True,
            )
            return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,  # Changed scheduler to lr_scheduler
            'monitor': 'valid_loss',
        }
        
        def setup(self, stage=None) -> None:
            if stage != "fit":
                return
            # Get dataloader by calling it - train_dataloader() is called after setup() by default
            train_loader = self.train_dataloader()

            # Calculate total steps
            tb_size = self.batch_size * max(1, self.trainer.num_devices)
            ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
            self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size

        def configure_optimizers(self):
            """Prepare optimizer and schedule (linear warmup and decay)"""
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.my_hparams["weight_decay"],
                },
                {
                    "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            optimizer = AdamW(
                optimizer_grouped_parameters, 
                lr=self.my_hparams["lr"],
                eps=self.my_hparams.get('adam_epsilon', 1e-8))

            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.my_hparams.get("warmup_steps", 500),
                num_training_steps=self.total_steps,
            )
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
            return [optimizer], [scheduler]
        
        def training_step(self, batch, batch_idx):
            outputs = self(**batch)
            assert self.training
            self.log(f"train_loss", outputs.loss, batch_size=self.batch_size, prog_bar=True) 
            
            predicted_ids = outputs.logits.argmax(-1)
            predicted_texts = self.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
            groundtruth_texts = batch["texts"]
        
            # ~~~ BUILD: demo dataframe ~~~ #
            
            return dict(
                loss=outputs.loss,
                preds=predicted_texts,
                target=groundtruth_texts,
            )

        def validation_step(self, batch, batch_idx):
            outputs = self(**batch)
            # print(outputs.logits.argmax(-1)[0])
            
            assert not self.training

            self.log(f"valid_loss", outputs.loss, batch_size=self.batch_size, prog_bar=True) 

            
            # real_predicted_ids = [self.generate(
            #     input_ids=batch["input_ids"][_id][None],
            #     attention_mask=batch["attention_mask"][_id][None],
            #     word_length_tensor=batch["word_length_tensor"][_id][None],
            #     num_beams=1,
            #     max_length=batch["word_length_tensor"].max().item(),
            # ) for _id in range(len(batch['input_ids']))]
            real_predicted_ids = [self.generate(
                input_ids=batch["input_ids"][0][None],
                attention_mask=batch["attention_mask"][0][None],
                word_length_tensor=batch["word_length_tensor"][0][None],
                num_beams=1,
                max_length=batch["word_length_tensor"].max().item(),
            )]
            real_predicted_texts = [
                self.tokenizer.batch_decode(s, skip_special_tokens=True)[0]
                for s in real_predicted_ids]
            
            predicted_ids = outputs.logits.argmax(-1)
            predicted_texts = self.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
            groundtruth_texts = batch["texts"]
            print()
            print('\033[01;34m')
            print("GT:", groundtruth_texts[0])
            print('\033[0m', end='')
            print('\033[01;35m', end='')
            print("PR:", predicted_texts[0])
            print('\033[0m', end='')
            print('\033[01;33m', end='')
            print("DE:", real_predicted_texts[0])
            print('\033[0m', end='')
            
            # ~~~ BUILD: demo dataframe ~~~ #

            return dict(
                loss=outputs.loss,
                preds=predicted_texts,
                reals=real_predicted_texts,
                reals_target=groundtruth_texts[0:1],
                target=groundtruth_texts,
            )
            
        def training_step_end(self, outputs):
            # return   # FIXME!!!
            mode = "train"
            metrics = self.metrics[mode]
            if 'preds' in outputs:
                if self.task_name == "ASR":
                    eval_result = metrics(
                        preds=outputs['preds'], 
                        target=list(outputs['target']))
                elif self.task_name == "ST":
                    eval_result = metrics(
                        preds=outputs['preds'], 
                        target=[[i] for i in outputs['target']])
                metric_name = {"ASR": "WER", "ST": "BLEU"}[self.task_name]
                self.log(f"{mode}_{metric_name}", eval_result, batch_size=self.batch_size, prog_bar=True)

        def validation_step_end(self, outputs):
            # return   # FIXME!!!
            mode = "valid"
            metrics = self.metrics[mode]
            metrics_REAL = self.metrics["validREAL"]
            if 'preds' in outputs:
                if self.task_name == "ASR":
                    eval_result = metrics(
                        preds=outputs['preds'], 
                        target=list(outputs['target']))
                    eval_resultREAL = metrics_REAL(
                        preds=outputs['reals'], 
                        target=list(outputs['reals_target']))
                elif self.task_name == "ST":
                    eval_result = metrics(
                        preds=outputs['preds'], 
                        target=[[i] for i in outputs['target']])
                    eval_resultREAL = metrics_REAL(
                        preds=outputs['reals'], 
                        target=[[i] for i in outputs['reals_target']])
                metric_name = {"ASR": "WER", "ST": "BLEU"}[self.task_name]
                self.log(f"{mode}_{metric_name}", eval_result, batch_size=self.batch_size, prog_bar=True)
                self.log(f"validREAL_{metric_name}", eval_resultREAL, batch_size=self.batch_size, prog_bar=True)
            
        def train_dataloader(self):
            return DataLoader(
                dataset=self.trainset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.trainset.num_workers,
                collate_fn=UnitDataset.tokenized_collate_fn(
                    self.tokenizer, 
                    padding_value=503, 
                    max_unit_length=1024, 
                    max_text_length=512,
                ),
            )
            
        def val_dataloader(self):
            return DataLoader(
                dataset=self.valset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.valset.num_workers,
                collate_fn=UnitDataset.tokenized_collate_fn(
                    self.tokenizer, 
                    padding_value=503, 
                    max_unit_length=1024, 
                    max_text_length=512,
                ),
            )
    # endregion          === old ===           NOIGERDNE #


class SentBartEncoder(BartEncoder):
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config, embed_tokens)
        embed_dim = config.d_model
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
        (hidden_states, out_attention_mask,  
                      # out_attention_mask := (
                      #     encoder_output_attention_mask)
         _, _) = self.sent_retriever(
             encoder_outputs.last_hidden_state, 
             word_length_tensor)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        if not return_dict:
            return tuple(v for v in [
                hidden_states, 
                out_attention_mask, 
                encoder_outputs.hidden_states,
                encoder_outputs.attentions,
            ] if v is not None)
        return AugBaseModelOutput(
            last_hidden_state=hidden_states, 
            out_attention_mask=out_attention_mask, 
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def sent_retriever(self, 
        encoder__last_hidden_state, 
        word_length_tensor=None,
        return_all=False,
        return_original=False,
      ):
        alpha_values = self.alpha_predictor(encoder__last_hidden_state)
        alpha_values = alpha_values.squeeze(-1).sigmoid()

        encoder__word_representations_CIF = (
            self.word_extractor(
                encoder__last_hidden_state,
                alpha=alpha_values,
                padding_mask=None,
                target_lengths=word_length_tensor,
            )
        )
        # TODO: Keep all CIF
        [encoder_word_representation] = encoder__word_representations_CIF['cif_out']
        encoder_word_representation = encoder_word_representation.contiguous()
            # aliased as `encoder_word_representation`
            # FIXME: distributed problem!
            # TODO: add other CIF ouptuts!

        encoder_output_attention_mask = (
            mask_generator(word_length_tensor) 
            if word_length_tensor is not None else
            None)
        
        
        return (
            encoder_word_representation, 
            encoder_output_attention_mask, 
            encoder__word_representations_CIF if return_all else None,
            encoder__last_hidden_state if return_original else None,
        )

class SentBart(BartModel):
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
                hidden_states=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                attentions=encoder_outputs[3] if len(encoder_outputs) > 3 else None,
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
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
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
            # endregion <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return AugSeq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_last_hidden_out_attention_mask=outputs.encoder_last_hidden_out_attention_mask,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
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
        if "word_length_tensor" in model_kwargs:
            word_length_tensor = model_kwargs["word_length_tensor"]
            model_kwargs["word_length_tensor"] = word_length_tensor.index_select(0, expanded_return_idx)

        if "encoder_outputs" in model_kwargs:
            if "out_attention_mask" in encoder_outputs:
                out_attention_mask = model_kwargs["encoder_outputs"]["out_attention_mask"]
                model_kwargs["encoder_outputs"]["out_attention_mask"] = out_attention_mask.index_select(0, expanded_return_idx)

        return input_ids, model_kwargs
    # endregion <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

