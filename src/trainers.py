#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Dict, Optional

import torch

import transformers
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers.trainer_callback import TrainerState
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_sagemaker_mp_enabled
from transformers.utils import is_apex_available
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp  # noqa
    from transformers.trainer_pt_utils import smp_forward_backward
    from transformers.trainer_pt_utils import smp_forward_only
    from transformers.trainer_pt_utils import smp_nested_concat
if is_apex_available():
    from apex import amp  # noqa
from transformers.trainer_pt_utils import nested_detach


@dataclass
class AugTrainerState(TrainerState):
    masked_lm_loss: Optional[torch.FloatTensor] = None
    real_length_loss: Optional[torch.FloatTensor] = None

class LogCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        pass

    def on_train_begin(self, args, state, control, **kwargs):
        return super().on_train_begin(
            args, AugTrainerState(**(vars(state))), control, **kwargs)

    def on_step_end(self, args, state, control, **kwargs):
        # print(dict(
        #     masked_lm_loss=state.masked_lm_loss,
        #     real_length_loss=state.real_length_loss,
        # ))
        pass

    def on_log(self, args, state, control, **kwargs):
        pass


class AugTrainer(Trainer):
    def log(self, logs):
        # return super().log(logs)
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)
        if getattr(self.state, "masked_lm_loss", None) is not None:
            logs["CE loss"] = round(self.state.masked_lm_loss, 3)
        if getattr(self.state, "real_length_loss", None) is not None:
            logs["real_length_loss"] = round(self.state.real_length_loss, 3)
            
        if "learning_rate" in logs:
            logs["learning_rate"] = eval("%.3e" % (logs["learning_rate"]))
        output = {
            **logs, 
            **{"step": self.state.global_step},
        }
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(
            self.args, self.state, self.control, logs)

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            scaler = self.scaler if self.do_grad_scaling else None
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.autocast_smart_context_manager():
            loss, other_outputs = self.compute_loss(model, inputs, return_outputs=True)
            if 'masked_lm_loss' in other_outputs:
                self.state.masked_lm_loss = other_outputs.get('masked_lm_loss').detach().item()
            if 'real_length_loss' in other_outputs:
                self.state.real_length_loss = other_outputs.get('real_length_loss').detach().item()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
      ):
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():  # nottodo~~
                raw_outputs = smp_forward_only(model, inputs)  # noqa
                if has_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)  # noqa
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)  # noqa
            else:
                if has_labels:
                    with self.autocast_smart_context_manager():
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()
                    # @@
                    self.state.masked_lm_loss = outputs.masked_lm_loss.detach().item()
                    self.state.real_length_loss = outputs.real_length_loss.detach().item()
                    # $$

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.autocast_smart_context_manager():
                        outputs = model(**inputs)
                    # @@
                    self.state.masked_lm_loss = outputs.masked_lm_loss.detach().item()
                    self.state.real_length_loss = outputs.real_length_loss.detach().item()
                    # $$
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

class AugSeq2SeqTrainer(Seq2SeqTrainer, AugTrainer):
    def prediction_step(self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
      ):
        if not self.args.predict_with_generate or prediction_loss_only:
            return super(Seq2SeqTrainer, self).prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
        }

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
        if "global_attention_mask" in inputs:
            gen_kwargs["global_attention_mask"] = inputs.get("global_attention_mask", None)
        if "word_length_tensor" in inputs:
            gen_kwargs["word_length_tensor"] = inputs.get("word_length_tensor", None)

        # prepare generation inputs
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        generated_tokens = self.model.generate(
            generation_inputs,
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            with self.autocast_smart_context_manager():
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        else:
            labels = None

        return (loss, generated_tokens, labels)
