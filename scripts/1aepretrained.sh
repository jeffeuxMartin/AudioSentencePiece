#!sh
conda activate revived
cd ~/Audio*            
cd A*; git pull; cd ..
hrun -c 8 -m 16 -s \
-GG -g 2080Ti \
`which python` \
AudioSentencePiece/main_pl.py \
  --pretrained_name selfpret/AEmodel --fix_encoder \
  -b 9 \
  -B 9 \
  --eval_steps 300 \
  --metric_batch 150 \
  --verbose_batch 300 \
  \
  --train_split train-clean-100 \
  --dev_split dev-clean \
  --test_split test-clean \
  \
  -e 10 \
  --lr 2e-4 \
  --gacc 1 \
  --proj_name HuggingFacePLSent \
  \
  --mode train \
  ` # --task AE --autoencoder ` \
  --task ASR \
  ` # --task ST ` \
  \
  --run_name NewASR \
  --datapath data/LibriSpeechUnits \
  