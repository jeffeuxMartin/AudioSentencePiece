# ALLbatch=18 GPUType=3090 GPUCount=3 zsh ~/AudioWords/AudioSentencePiece/scripts/exp_queue/collunit_noCIF_ST.sh                          # full unit --> batchsz smaller
# ALLbatch=9
# GPUType=2080Ti    # 9                      
# GPUCount=3
#!sh
# ============== constant ============== #                                     
WORKDIR=$HOME/AudioWords
APROOT=$WORKDIR/AudioSentencePiece
RUNNING=$APROOT/scripts/run_battleship.sh

OUTPUTDIR_PREFIX=exp/hf_ckpts/focused

# ---------------- main ---------------- #
batchsize=$(($ALLbatch / $GPUCount))
cd $WORKDIR
cd AudioSentencePiece && git pull && cd ..
mkdir -p $OUTPUTDIR_PREFIX                
hrun $SPOT \
  -c 8 -m 16 \
  -$(printf -- 'G%.0s' $(seq 1 $GPUCount)) -g $GPUType \
  zsh $RUNNING \
` # python3 AudioSentencePiece/main.py ` \
  --run_name 'collunit_CIF_ST' \
  --output_dir "$OUTPUTDIR_PREFIX/$(date +%Y%m%d_%H%M%S)" \
  ` # setups ` ` # ~~~ # ` \
  --task ST \
  --datapath data/CoVoSTUnits \
  --proj_name HuggingFaceSent \
  --nolower ` # ST ` \
  \
  --train_split train \
  --dev_split dev \
  ` # exp settings ` ` # --- # ` \
  --scratch \
  ` # --notcoll ` \
  ` # fixed ----------- ` \
  -b $batchsize -B $batchsize \
  ` # not important ` \
  --metric_batch $(($batchsize * 1)) \
  --verbose_batch 0                                                                                        
  

# parser.add_argument("--task", type=str, default='ASR')
# parser.add_argument("--datapath", type=str, default="data/LibriSpeechUnits")
# parser.add_argument("--proj_name", type=str, default="HuggingFaceSentASR_May08")

# parser.add_argument("--train_split", type=str, default='train-clean-100')
# parser.add_argument("--dev_split", type=str, default='dev-clean')
# parser.add_argument("--test_split", type=str, default=None)
# parser.add_argument("--pretrained_name", type=str, default='facebook/bart-base')
# parser.add_argument("--output_dir", type=str, default=(
#     EXP_PREFIX / "hf_ckpts/basic_trial1" / Path(strftime(now(), r'%Y%m%d_%H%M%S'))))
    
# # @@@ exp setups!
# parser.add_argument("--run_name", type=str, default=None)
# parser.add_argument("--weight_len", type=float, default=None)
# parser.add_argument("--notcoll", action='store_false', dest='coll'); parser.set_defaults(coll=True)
# parser.add_argument('--autoencoder', action='store_true')
# parser.add_argument('--fix_encoder', action='store_true')
# parser.add_argument('--original', action='store_true')
# parser.add_argument("--collapse_n", type=int, default=0)
# parser.add_argument('--scratch', action='store_true')
# parser.add_argument("--nolower", action='store_false', dest='lower'); parser.set_defaults(lower=True)
# # !!! better fixed with care !!!!!!!!!!!!!!!!!!!1
# parser.add_argument("-e", "--epochs", type=int, default=10)
# parser.add_argument("-b", "--batch_size", type=int, default=6)
# parser.add_argument("-lr", "--lr", type=float, default=2e-4)
# parser.add_argument("--eval_accumulation_steps", type=int, default=25)
