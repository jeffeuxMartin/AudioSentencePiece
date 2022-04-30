#!/usr/bin/env python3

from typing import Union, Tuple, Optional, List

import pandas as pd

import torch
from torch import nn
from torch import Tensor, FloatTensor, LongTensor
from torch.utils.data import DataLoader

from transformers import BartModel
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqModelOutput, CausalLMOutput
from transformers.models.bart.modeling_bart import shift_tokens_right

import pytorch_lightning as pl

from .torch_cif import cif_function  # FIXME later
from .utils import mask_generator
from .datasets import UnitDataset

class WordLevelBartAutoEncoder(BartModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # ~~~ model props ~~~ #
        self.encoder_hidden_dim = self.get_encoder().config.hidden_size
        self.decoder_hidden_dim = self.get_encoder().config.d_model
        self.cluster_num = 500 + 4  # FIXME: 把 cluster 數量設進 config!
                                    # NOTE: 加了 4 個 special tokens!
        
        # --- model inits --- #
        self.post_initialization()
        
    def post_initialization(self):
        # == encoder == #
        self.alpha_predictor = nn.Linear(self.encoder_hidden_dim, 1)
        self.word_extractor = cif_function  # FIXME: change different downsampling
        self.length_predictor = nn.Linear(self.encoder_hidden_dim, 1)  # TODO: 派上用場？
        
        # == decoder == #
        self.unit_reconstruction_lm_head = nn.Linear(self.decoder_hidden_dim, self.cluster_num, bias=False)
        
    # NOTE: Original decorators removed.
    def forward(self,
        input_ids: LongTensor = None,
        attention_mask: Optional[Tensor] = None,
        word_length_tensor: Optional[LongTensor] = None,  # <-- added
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

        output_attentions = (output_attentions if output_attentions is not None else 
                             self.config.output_attentions)
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else 
                                self.config.output_hidden_states)
        use_cache = (use_cache if use_cache is not None else self.config.use_cache)
        return_dict = (return_dict if return_dict is not None else self.config.use_return_dict)

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
        decoder_outputs.last_hidden_state = (  # FIXME: distributed problem!
            self.unit_reconstruction_lm_head(decoder_outputs.last_hidden_state)).contiguous()  
                
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
          # Done: How to fix the parameters?
      else:
          self.word_level_encoder.train()
      
      self.representation_decoder = representation_decoder
      self.tokenizer = tokenizer
      
    def forward(self,
        encoder_input_ids,
        encoder_attention_mask=None,
        encoder_word_lengths_tensor=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
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
                # Z = word_representations.clone().detach().cpu().numpy()
        else:
            word_level_outputs = self.word_level_encoder(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                word_length_tensor=encoder_word_lengths_tensor,
            )
            word_representations = word_level_outputs.encoder_last_hidden_state

        
        # FIXME: 為什麼已經被移到 cuda 上了？ cuda problem!
        word_level_attention_mask = (
            mask_generator(encoder_word_lengths_tensor) 
            if encoder_word_lengths_tensor is not None else 
            None)
            
        
        # === decoder pass === #
        labels = (shift_tokens_right(
                decoder_input_ids, 
                self.representation_decoder.config.pad_token_id, 
                self.representation_decoder.config.decoder_start_token_id,
            ) if decoder_input_ids is not None else # NOTE: since not shifted in model???
            None
        )
        
        decoded_predictions = self.representation_decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=word_representations, 
            encoder_attention_mask=word_level_attention_mask,
            labels=labels,  # TODO & FIXME! Shift???
        )
        
        return CausalLMOutput(
            loss=decoded_predictions.loss,
            logits=decoded_predictions.logits,
        )
        # ), Z
        
    def generate(self,
        encoder_input_ids,
        encoder_attention_mask,
        encoder_word_lengths_tensor,
        num_beams,
        max_length,
    ):
        # === encoder pass === #
        with torch.no_grad():  # FIXME: How to get fixed?
            word_level_outputs = self.word_level_encoder(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                word_length_tensor=encoder_word_lengths_tensor,
            )
            word_representations = word_level_outputs.encoder_last_hidden_state
            Z = word_representations.clone().detach().cpu().numpy()
        
        # FIXME: 為什麼已經被移到 cuda 上了？ cuda problem!
        word_level_attention_mask = (
            mask_generator(encoder_word_lengths_tensor) 
            if encoder_word_lengths_tensor is not None else 
            None)
            
        
        # === decoder pass === #
        decoded_predictions = self.representation_decoder.generate(
            # inputs=decoder_input_ids,
            # attention_mask=decoder_attention_mask,
            encoder_hidden_states=word_representations, 
            encoder_attention_mask=word_level_attention_mask,
            num_beams=num_beams,
            max_length=max_length,
        )
        
        return decoded_predictions
        # return decoded_predictions, Z
 
       
class PLSpeechToSemantics(pl.LightningModule):
    def __init__(self, datasets, metric_cls, wandb_logger=None, *args, **kwargs):
        super().__init__()
        if "batch_size" in kwargs:
            self.batch_size = kwargs.get('batch_size', 9)
            kwargs.pop('batch_size')
        else:
            self.batch_size = 9
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
            "valid": metric_cls()}
        
        self.my_hparams = {
            "lr": kwargs.get('lr', 2e-4),
            "weight_decay": 0.01,  # default value
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
        
    def configure_optimizers(self):
        # Ref.: https://github.com/PyTorchLightning/pytorch-lightning/issues/328
        optimizer = torch.optim.AdamW(
            params=self.parameters(), 
            lr=self.my_hparams["lr"],
            weight_decay=self.my_hparams["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=1,
            verbose=True,
        )
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler,  # Changed scheduler to lr_scheduler
           'monitor': 'valid_loss',
       }
       
    def step_unified(self, batch, batch_idx, mode="train"):
        # outputs, ZF = self(**batch)
        outputs = self(**batch)
        assert self.training == (mode == "train")
        if mode != "train":  # FIXME: 長度問題？ input 有沒有餵錯？
            # real_outputs, ZV = self.generate(
            real_outputs = self.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                word_length_tensor=batch["word_length_tensor"],
                num_beams=1,
                max_length=batch["word_length_tensor"].max().item(),
            )
            real_predicted_ids = real_outputs  
                # FIXME: 原來 generate 是直接 id 出來的！
            real_predicted_texts = self.tokenizer.batch_decode(
                real_predicted_ids, skip_special_tokens=True)
        self.log(f"{mode}_loss", outputs.loss, 
            batch_size=self.batch_size, prog_bar=True) 
        
        predicted_ids = outputs.logits.argmax(-1)
        predicted_texts = self.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        groundtruth_texts = batch["texts"]
        
        demonstration_df = pd.DataFrame.from_dict(
            dict(enumerate((list(zip(
                predicted_texts, 
                *([real_predicted_texts] if mode != "train" else []), 
                list(groundtruth_texts),
            )))[:self.demonstration_number]))).T
        demonstration_df.columns = ["predicted", 
            *(["real"] if mode != "train" else []), 
            "ground truth"]
        if self.wandb_logger is not None:
            self.wandb_logger.log_text(
                key=f"Prediction v.s Ground truth ({mode})",
                dataframe=demonstration_df,
            )
        else:
            pass
            # print(demonstration_df)
        return dict(
            loss=outputs.loss,
            preds=predicted_texts,
            target=groundtruth_texts,
        )
        
    def step_end_unified(self, outputs, mode="train"):
        # update and log
        metrics = self.metrics[mode]
        if self.task_name == "ASR":
            eval_result = metrics(
                preds=outputs['preds'], 
                target=list(outputs['target']))
        elif self.task_name == "ST":
            eval_result = metrics(
                preds=outputs['preds'], 
                target=[[i] for i in outputs['target']])
        metric_name = {"ASR": "WER", "ST": "BLEU"}[self.task_name]
        self.log(f"{mode}_{metric_name}", eval_result,
            batch_size=self.batch_size, prog_bar=True)
        
    def training_step(self, batch, batch_idx):
        return self.step_unified(batch, batch_idx, mode="train")
        
    def training_step_end(self, outputs):
        return self.step_end_unified(outputs, mode="train")

    def validation_step(self, batch, batch_idx):
        return self.step_unified(batch, batch_idx, mode="valid")

    def validation_step_end(self, outputs):
        return self.step_end_unified(outputs, mode="valid")
        
    def train_dataloader(self):
        print(f"\n"
            "\033[01;31m"
            f"{self.trainset.num_workers = }"
            "\033[0m"
        )
        return DataLoader(
            dataset=self.trainset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.trainset.num_workers,  # CHECK: should be careful!
            collate_fn=UnitDataset.tokenized_collate_fn(
                self.tokenizer, 
                padding_value=503, 
                max_unit_length=1024, 
                max_text_length=512,
            ),  # CHECK: should be customized!
        )
        
    def val_dataloader(self):
        print(f"\n"
            "\033[01;31m"
            f"{self.valset.num_workers = }"
            "\033[0m"
        )
        return DataLoader(
            dataset=self.valset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.valset.num_workers,  # CHECK: should be careful!
            collate_fn=UnitDataset.tokenized_collate_fn(
                self.tokenizer, 
                padding_value=503, 
                max_unit_length=1024, 
                max_text_length=512,
            ),  # CHECK: should be customized!
        )


