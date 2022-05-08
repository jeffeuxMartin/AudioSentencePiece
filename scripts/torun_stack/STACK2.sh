python3 AudioSentencePiece/main.py -b 6 -B 4 --scratch --eval_steps 50 --metric_batch 4 --verbose_batch 0 --generation_max_length 512 --eval_accumulation_steps 10
python3 AudioSentencePiece/main.py -b 6 -B 4 --scratch --eval_steps 50 --metric_batch 4 --verbose_batch 0 --generation_max_length 512 --eval_accumulation_steps 10 --dev_split dummy --nowandb

OUTPUTDIR_PREFIX=exp/hf_ckpts/focused
batchsize=6

python3 AudioSentencePiece/main.py \
  --output_dir "$OUTPUTDIR_PREFIX/$(date +%Y%m%d_%H%M%S)"
  ` # setups ` ` # ~~~ # ` \
  --task ASR \
  --datapath data/LibriSpeechUnits \
  --proj_name HuggingFaceSent \
  `     # --nolower  # ASR ` \
  \
  --train_split train-clean-100 \
  --dev_split dev-clean \
  ` # exp settings ` ` # --- # ` \
  --scratch \
  ` # fixed ----------- ` \
  -b $batchsize \
  ` # not important ` \
  --metric_batch $(($batchsize * 2)) \
  --verbose_batch 0
