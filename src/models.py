#!/usr/bin/env python3

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

from transformers import BartModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_outputs import Seq2SeqModelOutput
from transformers.modeling_outputs import CausalLMOutput
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers import get_linear_schedule_with_warmup
import pytorch_lightning as pl

from .torch_cif import cif_function
from .newutils import mask_generator
from .datasets import UnitDataset

class WordLevelBartAutoEncoder(BartModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.encoder_hidden_dim = self.get_encoder().config.hidden_size
        self.decoder_hidden_dim = self.get_encoder().config.d_model
        self.cluster_num = 500 + 4  # FIXME: 把 cluster 數量設進 config!
                                    # NOTE: 加了 4 個 special tokens!
        
        self.post_initialization()
        
    def post_initialization(self):
        # == encoder == #
        self.alpha_predictor = nn.Linear(self.encoder_hidden_dim, 1)
        self.word_extractor = cif_function
        self.length_predictor = nn.Linear(self.encoder_hidden_dim, 1)
        
        # == decoder == #
        self.unit_reconstruction_lm_head = nn.Linear(self.decoder_hidden_dim, self.cluster_num, bias=False)
        
    def forward(self,
        input_ids: LongTensor = None,
        attention_mask: Optional[Tensor] = None,
        word_length_tensor: Optional[LongTensor] = None,
        decoder_input_ids: Optional[LongTensor] = None,
        decoder_attention_mask: Optional[LongTensor] = None,
        head_mask: Optional[Tensor] = None,
        decoder_head_mask: Optional[Tensor] = None,
        cross_attn_head_mask: Optional[Tensor] = None,
        encoder_outputs: Optional[List[FloatTensor]] = None,
        past_key_values: Optional[List[FloatTensor]] = None,
        inputs_embeds: Optional[FloatTensor] = None,
        decoder_inputs_embeds: Optional[FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
      ) -> Union[Tuple, Seq2SeqModelOutput]:
        
        # different to other models, Bart automatically creates decoder_input_ids from input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError("If no `decoder_input_ids` or `decoder_inputs_embeds` are passed, `input_ids` cannot be `None`. Please pass either `input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`.")

            decoder_input_ids = shift_tokens_right(
                input_ids, 
                self.config.pad_token_id, 
                self.config.decoder_start_token_id,
            )

        output_attentions = (output_attentions 
                             if output_attentions is not None else 
                             self.config.output_attentions)
        output_hidden_states = (output_hidden_states 
                                if output_hidden_states is not None else 
                                self.config.output_hidden_states)
        use_cache = (use_cache 
                     if use_cache is not None else 
                     self.config.use_cache)
        return_dict = (return_dict 
                       if return_dict is not None else 
                       self.config.use_return_dict)

        # ~ ~ ~ ~ ~ ~ ~ ~ ENCODER PASS ~ ~ ~ ~ ~ ~ ~ ~ #
        if encoder_outputs is None:
            encoder_outputs: BaseModelOutput = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            encoder__last_hidden_state = encoder_outputs.last_hidden_state
            
            alpha_values = self.alpha_predictor(encoder__last_hidden_state)
            alpha_values = alpha_values.squeeze(-1).softmax(-1)

            encoder__word_representations_CIF = (
                self.word_extractor(
                    encoder__last_hidden_state,
                    alpha=alpha_values,
                    padding_mask=None,
                    target_lengths=word_length_tensor,
                )
            )
            # TODO: Keep all CIF
            [encoder_outputs.last_hidden_state] = encoder__word_representations_CIF['cif_out']
            encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.contiguous()
                # aliased as `encoder_word_representation`
                # FIXME: distributed problem!
                # TODO: add other CIF ouptuts!

            encoder_output_attention_mask = mask_generator(word_length_tensor)
            
            
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # ~ ~ ~ ~ ~ ~ ~ ~ DECODER PASS ~ ~ ~ ~ ~ ~ ~ ~ #
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=encoder_output_attention_mask,
            
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        decoder_outputs.last_hidden_state = self.unit_reconstruction_lm_head(decoder_outputs.last_hidden_state).contiguous()  
                
        # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class SpeechToSemantics(nn.Module):
    def __init__(self, 
        word_level_encoder,
        representation_decoder,
        tokenizer,
        task_name,
        fixed_encoder=True,
    ):
      super().__init__()
      self.task_name = task_name
      self.fixed_encoder = fixed_encoder
      
      self.word_level_encoder = word_level_encoder
      if self.fixed_encoder:
          self.word_level_encoder.eval()
          for name, param in (
                  self.word_level_encoder
                      .named_parameters()): 
              param.requires_grad = False
      else:
          self.word_level_encoder.train()
      
      self.representation_decoder = representation_decoder
      self.tokenizer = tokenizer
    
    def _encoder_pass(self,
        encoder_input_ids,
        encoder_attention_mask=None,
        encoder_word_lengths_tensor=None,
      ):
        # === encoder pass === #
        if self.fixed_encoder:
            with torch.no_grad():  # FIXME: How to get fixed?
                word_level_outputs = self.word_level_encoder(
                    input_ids=encoder_input_ids,
                    attention_mask=encoder_attention_mask,
                    word_length_tensor=encoder_word_lengths_tensor,
                )
                word_representations = word_level_outputs.encoder_last_hidden_state
        else:
            word_level_outputs = self.word_level_encoder(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                word_length_tensor=encoder_word_lengths_tensor,
            )
            word_representations = word_level_outputs.encoder_last_hidden_state

        
        word_level_attention_mask = (
            mask_generator(encoder_word_lengths_tensor) 
            if encoder_word_lengths_tensor is not None else 
            None)
        return word_representations, word_level_attention_mask, word_level_outputs
      
    def forward(self,
        encoder_input_ids,
        encoder_attention_mask=None,
        encoder_word_lengths_tensor=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
      ):
        word_representations, word_level_attention_mask, _ = self._encoder_pass(
            encoder_input_ids=encoder_input_ids,
            encoder_attention_mask=encoder_attention_mask,
            encoder_word_lengths_tensor=encoder_word_lengths_tensor,
        )
        
        #@ print('\n\033[0;31m')
        #@ print(word_representations)
        #@ print('\033[0m\n')
            
        
        # === decoder pass === #
        if 0:  # CHECK: shifted? FIXME
            labels = shift_tokens_right(
                    decoder_input_ids, 
                    self.representation_decoder.config.pad_token_id, 
                    self.representation_decoder.config.decoder_start_token_id,
                ) if decoder_input_ids is not None else None
        else:
            labels = decoder_input_ids
        
        #@@ print('\n\033[0;32m')
        #@@ print(decoder_input_ids)
        #@@ print('\033[0m\n')

        decoded_predictions = self.representation_decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=word_representations, 
            encoder_attention_mask=word_level_attention_mask,
            labels=labels,
        )
        
        return CausalLMOutput(
            loss=decoded_predictions.loss,
            logits=decoded_predictions.logits,
        )
        
    def generate(self,
        encoder_input_ids,
        encoder_attention_mask,
        encoder_word_lengths_tensor,
        num_beams,
        max_length,
    ):
        word_representations, word_level_attention_mask, _ = self._encoder_pass(
            encoder_input_ids=encoder_input_ids,
            encoder_attention_mask=encoder_attention_mask,
            encoder_word_lengths_tensor=encoder_word_lengths_tensor,
        )
                    
        # === decoder pass === #
        decoded_predictions = self.representation_decoder.generate(
            encoder_hidden_states=word_representations, 
            encoder_attention_mask=word_level_attention_mask,
            num_beams=num_beams,
            max_length=max_length,
        )
    
        return decoded_predictions









from transformers import BartForCausalLM
class NewNewBartForCausalLM(BartForCausalLM):
    def __init__(self, config):
        config = copy.deepcopy(config)
        config.is_decoder = True
        config.is_encoder_decoder = True
        super().__init__(config)
        self.model = BartDecoderWrapper(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


    def prepare_inputs_for_generation(
        self,
        input_ids,
        encoder_hidden_states,
        encoder_attention_mask,
        # decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        # decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]  # MODIFIED XXX

        return {
            "input_ids": input_ids,  # encoder_outputs is defined. input_ids not needed
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "past_key_values": past,
            # "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            # "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }
        
    def prepare_inputs_for_generation1(
        self,
        decoder_input_ids,
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
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_inputs_for_generation2(
        self, 
        input_ids, 
        past=None, 
        attention_mask=None, 
        use_cache=None, 
        **kwargs):
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        if past:
            input_ids = input_ids[:, -1:]
        # first step, decoder_cached_states are empty
        return {
            "input_ids": input_ids,  # encoder_outputs is defined. input_ids not needed
            "attention_mask": attention_mask,
            "past_key_values": past,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)














import copy
from transformers import BartPretrainedModel
from transformers.models.bart.modeling_bart import BartDecoderWrapper
class BartAgainModel(BartPretrainedModel):
    def __init__(self, config):
        config = copy.deepcopy(config)
        config.is_decoder = True
        config.is_encoder_decoder = False
        super().__init__(config)
        self.model = BartDecoderWrapper(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ): pass




























class PLSpeechToSemantics(pl.LightningModule):
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











from transformers import BartConfig, BartForConditionalGeneration

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
    verbose=(lambda *a: None),
    **new_config_options,
  ):
    newconfig = config_class.from_pretrained(
        checkpoint_name, 
        **new_config_options)

    pretrained_model = model_class.from_pretrained(checkpoint_name)
    model = model_class(config=newconfig)
    
    model = pure_advanced_load_pretrained(
        pretrained_model=pretrained_model,
        model=model,
        verbose=verbose,
    )

    del pretrained_model
    return model
    

class WordLevelBartAutoEncoderNew(BartModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.encoder_hidden_dim = self.get_encoder().config.hidden_size
        self.decoder_hidden_dim = self.get_encoder().config.d_model
        self.cluster_num = 500 + 4  # FIXME: 把 cluster 數量設進 config!
                                    # NOTE: 加了 4 個 special tokens!
        
        self.post_initialization()
        
    def post_initialization(self):
        # == encoder == #
        self.alpha_predictor = nn.Linear(self.encoder_hidden_dim, 1)
        self.word_extractor = cif_function
        self.length_predictor = nn.Linear(self.encoder_hidden_dim, 1)
        
        # == decoder == #
        self.unit_reconstruction_lm_head = nn.Linear(self.decoder_hidden_dim, self.cluster_num, bias=False)
        
    def forward(self,
        input_ids: LongTensor = None,
        attention_mask: Optional[Tensor] = None,
        word_length_tensor: Optional[LongTensor] = None,
        decoder_input_ids: Optional[LongTensor] = None,
        decoder_attention_mask: Optional[LongTensor] = None,
        head_mask: Optional[Tensor] = None,
        decoder_head_mask: Optional[Tensor] = None,
        cross_attn_head_mask: Optional[Tensor] = None,
        encoder_outputs: Optional[List[FloatTensor]] = None,
        past_key_values: Optional[List[FloatTensor]] = None,
        inputs_embeds: Optional[FloatTensor] = None,
        decoder_inputs_embeds: Optional[FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
      ) -> Union[Tuple, Seq2SeqModelOutput]:
        
        # different to other models, Bart automatically creates decoder_input_ids from input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError("If no `decoder_input_ids` or `decoder_inputs_embeds` are passed, `input_ids` cannot be `None`. Please pass either `input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`.")

            decoder_input_ids = shift_tokens_right(
                input_ids, 
                self.config.pad_token_id, 
                self.config.decoder_start_token_id,
            )

        output_attentions = (output_attentions 
                             if output_attentions is not None else 
                             self.config.output_attentions)
        output_hidden_states = (output_hidden_states 
                                if output_hidden_states is not None else 
                                self.config.output_hidden_states)
        use_cache = (use_cache 
                     if use_cache is not None else 
                     self.config.use_cache)
        return_dict = (return_dict 
                       if return_dict is not None else 
                       self.config.use_return_dict)

        # ~ ~ ~ ~ ~ ~ ~ ~ ENCODER PASS ~ ~ ~ ~ ~ ~ ~ ~ #
        if encoder_outputs is None:
            encoder_outputs: BaseModelOutput = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            encoder__last_hidden_state = encoder_outputs.last_hidden_state
            
            alpha_values = self.alpha_predictor(encoder__last_hidden_state)
            alpha_values = alpha_values.squeeze(-1).softmax(-1)

            encoder__word_representations_CIF = (
                self.word_extractor(
                    encoder__last_hidden_state,
                    alpha=alpha_values,
                    padding_mask=None,
                    target_lengths=word_length_tensor,
                )
            )
            # TODO: Keep all CIF
            [encoder_outputs.last_hidden_state] = encoder__word_representations_CIF['cif_out']
            encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.contiguous()
                # aliased as `encoder_word_representation`
                # FIXME: distributed problem!
                # TODO: add other CIF ouptuts!

            encoder_output_attention_mask = mask_generator(word_length_tensor)
            
            
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # ~ ~ ~ ~ ~ ~ ~ ~ DECODER PASS ~ ~ ~ ~ ~ ~ ~ ~ #
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=encoder_output_attention_mask,
            
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        decoder_outputs.last_hidden_state = self.unit_reconstruction_lm_head(decoder_outputs.last_hidden_state).contiguous()  
                
        # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


model = advanced_load_pretrained(
    checkpoint_name='facebook/bart-base',
    model_class=WordLevelBartAutoEncoderNew, 
    config_class=BartConfig, 
    # new_config_options...
    max_position_embeddings=1024, 
    vocab_size=504)

model.get_encoder().requires_grad_(False);

from transformers import BartModel
from transformers import BartForConditionalGeneration
from transformers import BartConfig

class SentBart(BartModel):
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
      ) -> Union[Tuple, Seq2SeqModelOutput]:

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
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        (encoder_outputs, attention_mask,  
                        # attention_mask := encoder_output_attention_mask
         _, _) = self.sent_retriever(encoder_outputs, word_length_tensor)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
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

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    # region CHANGED >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.post_initialization(word_extractor=cif_function)

    def post_initialization(self, word_extractor=cif_function):
        # == encoder == #
        self.alpha_predictor = nn.Linear(self.encoder_hidden_dim, 1)
        self.word_extractor = word_extractor
        self.length_predictor = nn.Linear(self.encoder_hidden_dim, 1)

    def sent_retriever(self, 
        encoder_outputs, 
        word_length_tensor=None,
        return_all=False,
        return_original=False,
      ):
        encoder__last_hidden_state = encoder_outputs.last_hidden_state
        
        alpha_values = self.alpha_predictor(encoder__last_hidden_state)
        alpha_values = alpha_values.squeeze(-1).softmax(-1)

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
        
        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_word_representation,
            hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
            attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
        )
        
        return (
            encoder_outputs, 
            encoder_output_attention_mask, 
            encoder__word_representations_CIF if return_all else None,
            encoder__last_hidden_state if return_original else None,
        )

    def fix_encoder_(self, to_fix=True):
        _requires_grad_ = not to_fix
        self.encoder.requires_grad_(_requires_grad_)
        self.alpha_predictor.requires_grad_(_requires_grad_)
        self.length_predictor.requires_grad_(_requires_grad_)
        self._word_extractor_fixer(to_fix)
    
    def _word_extractor_fixer(self, to_fix=True):
        _requires_grad_ = not to_fix
        if isinstance(self.word_extractor, cif_function):
            pass
        else:
            raise NotImplementedError  # if not CIF
    # endregion <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

class SentBartForConditionalGeneration(BartForConditionalGeneration):
    def __init__(self, config: BartConfig):  # Only 1 line added
        super().__init__(config)
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self.model = SentBart(config)  # NOTE: lm_head change?
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    # region CHANGED >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def fix_encoder_(self, to_fix=True): 
        self.model.fix_encoder_(to_fix)
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
    verbose=(lambda *a: None),
    **new_config_options,
  ):
    newconfig = config_class.from_pretrained(
        checkpoint_name, 
        **new_config_options)

    pretrained_model = model_class.from_pretrained(checkpoint_name)
    model = model_class(config=newconfig)
    
    model = pure_advanced_load_pretrained(
        pretrained_model=pretrained_model,
        model=model,
        verbose=verbose,
    )

    del pretrained_model
    return model

class PLSpeechToSemanticsNew(pl.LightningModule):
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



