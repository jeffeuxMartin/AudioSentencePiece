#!/bin/sh
sleep 3
cd A*; git pull; cd ..;
hrun -c 12 -m 24 \
  -GG -g 2080Ti \
  zsh AudioSentencePiece/scripts/run_battleship_header.sh \
  AudioSentencePiece/main_pl.py \
    --proj_name HuggingFacePLSent \
    -e 10 -b 9 --gacc 1 -B 12 --lr 2e-4 \
    \
    --eval_steps 500 \
    --metric_batch 60 \
    --verbose_batch 60  \
    \
    --task ST \
    --datapath data/CoVoSTUnits \
    --train_split train \
    --dev_split dev  
