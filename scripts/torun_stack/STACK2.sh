# python3 AudioSentencePiece/main.py -b 6 -B 4 --scratch --eval_steps 50 --metric_batch 4 --verbose_batch 0 --generation_max_length 512 --eval_accumulation_steps 10
# python3 AudioSentencePiece/main.py -b 6 -B 4 --scratch --eval_steps 50 --metric_batch 4 --verbose_batch 0 --generation_max_length 512 --eval_accumulation_steps 10 --dev_split dummy --nowandb


#!sh
# ============== constant ============== #
WORKDIR=$HOME/AudioWords
APROOT=$WORKDIR/AudioSentencePiece
RUNNING=$APROOT/scripts/run_battleship.sh

OUTPUTDIR_PREFIX=exp/hf_ckpts/focused
GPUType=2080Ti  # 6
# GPUType=3090    # 9
GPUCount=3

# ---------------- main ---------------- #
batchsize=$((18 / $GPUCount))
cd $WORKDIR
cd AudioSentencePiece && git pull && cd ..
mkdir -p $OUTPUTDIR_PREFIX
hrun \
  -c 8 -m 16 \
  -$(printf -- 'G%.0s' $(seq 1 $GPUCount)) -g $GPUType \
  zsh $RUNNING \
` # python3 AudioSentencePiece/main.py ` \
  --output_dir "$OUTPUTDIR_PREFIX/$(date +%Y%m%d_%H%M%S)" \
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

