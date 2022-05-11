#!sh
python3 \
AudioSentencePiece/main_pl.py \
  --pretrained_name ./AEmodel --fix_encoder \
  -b 8 \
  -B 8 \
  --eval_steps 3000 \
  --metric_batch 300 \
  --verbose_batch 600 \
  \
  --train_split train-clean-100 \
  --dev_split dev-clean \
  --test_split test-clean \
  \
  -e 10 \
  --lr 2e-4 \
  --gacc 2 \
  --proj_name HuggingFacePLSent \
  --task ASR \
  \
  --run_name New \
  --datapath data/LibriSpeechUnits \
  --mode train
