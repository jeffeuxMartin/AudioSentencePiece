#!/bin/sh

Task="${Task:=ASR}"
Battleship="${Battleship:=true}"
USE_AE="${USE_AE:=false}"
FULLUNIT="${FULLUNIT:=false}"

echo '如果是用沒 CIF 記得加 --original (will overwrite `collapse_n`!)'
echo '如果是用 not collapsed unit 記得加 --notcoll 並用 FULLUNIT=true'
###############################
export PATH=`
    `/home/jeffeuxmartin/miniconda3/bin:`
    `/home/jeffeuxmartin/.local/bin:`
    `:"$PATH"


# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/jeffeuxmartin/miniconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/jeffeuxmartin/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/jeffeuxmartin/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/jeffeuxmartin/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<


cd /home/jeffeuxmartin/AudioWords
conda activate revived
#############################33

if [ $FULLUNIT = false ]; then
  case $Task in  # coll
    AE)  bsz=`   `16; evalbsz=`    `8;;
    ASR) bsz=`   `18; evalbsz=`   `24;;
    ST)  bsz=`   `18; evalbsz=`   `18;;
    *)   bsz=`    `6; evalbsz=`    `6;;
  esac
else
  case $Task in  # full
    AE)  bsz=`   `12; evalbsz=`    `4;;
    ASR) bsz=`   `18; evalbsz=`   `24;;
    ST)  bsz=`   `18; evalbsz=`   `18;;
    *)   bsz=`    `6; evalbsz=`    `6;;
  esac
fi

# region
function calculate_batch0 () {
  CALCULATE_SCRIPT="$(python3 -c '
import torch;
vram_size = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2)
gpu_count = (torch.cuda.device_count())
print(f''"""''
GPU_COUNTS={gpu_count}
VRAM_SIZE={vram_size}
''"""'')')"

  eval "$CALCULATE_SCRIPT"
  echo "你拿到 $GPU_COUNTS 張 GPU, 每張 VRAM = $VRAM_SIZE GB..."

  case Task in
    AE ) datasize=6;;
    ASR) datasize=4;;
    ST ) datasize=2;;
    *  ) datasize=2;;
  esac

  if [[ $VRAM_SIZE -ge 40 ]]; then
    GPU_maxbatch=$((4 * 24 / $datasize))
  elif [[ $VRAM_SIZE -ge 20 ]]; then
    GPU_maxbatch=$((2 * 24 / $datasize))
  elif [[ $VRAM_SIZE -ge 10 ]]; then
    GPU_maxbatch=$((1 * 24 / $datasize))
  else
    echo 'No GPU!'
    exit
  fi
  
  export GPU_maxbatch=$GPU_maxbatch
  export GPU_COUNTS=$GPU_COUNTS
}

function calculate_batch () {
  evalbatchsize=$2
  evalbatchsize="${evalbatchsize:=$1}"
  calculate_batch0
  eval $(python -c '
def ceil(x): return 1 + int(x - 0.0001)
max_batch_per_gpu = '$GPU_maxbatch'
gpu_count = '$GPU_COUNTS'
batchsize = '$1'
averaged_batchsize = int(batchsize / gpu_count)
grad_acc = ceil(averaged_batchsize / max_batch_per_gpu)
real_batch = averaged_batchsize // grad_acc

evalbatchsize = '$evalbatchsize'
evalaveraged_batchsize = int(evalbatchsize / gpu_count)
evalgrad_acc = ceil(evalaveraged_batchsize / max_batch_per_gpu)
evalreal_batch = evalaveraged_batchsize // evalgrad_acc
print(f"""
real_batch={real_batch}
evalreal_batch={evalreal_batch}
gacc={grad_acc}""")
')
  export real_batch=$real_batch
  export evalreal_batch=$evalreal_batch
  export gacc=$gacc
  export GPU_COUNTS=$GPU_COUNTS
}
# endregion

calculate_batch $bsz $evalbsz

IFS=' ' \
AEargs=(
  -b $real_batch -B $evalreal_batch --gacc $gacc 
  --lr 2e-4 -e 3
  --autoencoder
)

IFS=' ' \
ASRargs=(
  -b $real_batch -B $evalreal_batch --gacc $gacc 
  --lr 2e-4 -e 10
)

IFS=' ' \
STargs=(
  -b $real_batch -B $evalreal_batch --gacc $gacc 
  --lr 2e-4 -e 10
)


# region --- def
IFS=' ' \
ASRdata=(
  --datapath     data/LibriSpeechUnits
  --train_split 'train-clean-100'
  --dev_split   'dev-clean'
  --test_split  'test-clean'
)

IFS=' ' \
STdata=(
  --datapath     data/CoVoSTUnits
  --train_split 'train'
  --dev_split   'dev'
  --test_split  'test'
  --nolower
)


IFS=' ' \
AEdemo=(
  --eval_steps 300 \
  --metric_batch 150 \
  --verbose_batch 300 \
)

IFS=' ' \
ASRdemo=(
  --eval_steps 300 \
  --metric_batch 150 \
  --verbose_batch 300 \
)

IFS=' ' \
STdemo=(
  --eval_steps 300 \
  --metric_batch 150 \
  --verbose_batch 300 \
)
# endregion --- def


case $Task in
  AE)
    IFS=' ' Taskargs=($AEargs)
    IFS=' ' Taskdata=($ASRdata)
    IFS=' ' Taskdemo=($AEdemo)
      ;;
  ASR)
    IFS=' ' Taskargs=($ASRargs)
    IFS=' ' Taskdata=($ASRdata)
    IFS=' ' Taskdemo=($ASRdemo)
      ;;
  ST)
    IFS=' ' Taskargs=($STargs)
    IFS=' ' Taskdata=($STdata)
    IFS=' ' Taskdemo=($STdemo)
      ;;
  *)
      ;;
esac

if [ $USE_AE = true ]; then
    PretrainedAE="${PretrainedAE:=selfpret/AEmodel}"
    IFS=' ' PretrainedArgs=(
      --pretrained_name "$PretrainedAE" 
      --fix_encoder)
  else
    PretrainedArgs=''
fi

# conda activate revived
# cd ~/Audio*            
# cd A*; git pull; cd ..
# hrun -c 8 -m 16 -t 1-0 \
# -G -g 2080Ti \
# `which python` \

debug=0
if [ debug = '1' ]; then
    DEB='python3 -m pdb -c continue'
else
    DEB="python3"
fi


if [ $Battleship = true ]; then
  if [ $GPU_COUNTS -ge 2 ]; then
      OMP_NUM_THREADS=$GPU_COUNTS \
          $DEB -m torch.distributed.launch \
                  --nproc_per_node=$GPU_COUNTS \
          AudioSentencePiece/main_pl.py \
            --task $Task \
            ${Taskargs[@]} ${Taskdata[@]} ${Taskdemo[@]} \
            ${PretrainedArgs[@]} \
            $@
  else
      $DEB \
      AudioSentencePiece/main_pl.py \
      --task $Task \
      ${Taskargs[@]} ${Taskdata[@]} ${Taskdemo[@]} \
      ${PretrainedArgs[@]} \
      $@
  fi
  exit
else
  python \
  AudioSentencePiece/main_pl.py \
    --task $Task \
    ${Taskargs[@]} ${Taskdata[@]} ${Taskdemo[@]} \
    ${PretrainedArgs[@]} \
    $@
  exit
fi
