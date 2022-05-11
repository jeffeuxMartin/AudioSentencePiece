#!zsh

Task="${Task:=ASR}"
USE_AE="${USE_AE:=False}"




bszAEfull=`    `6; evalbszAEfull=`    `6
bszAEcoll=`    `6; evalbszAEcoll=`    `6
bszASRfull=`   `6; evalbszASRfull=`   `6
bszASRcoll=`   `6; evalbszASRcoll=`   `6
bszSTfull=`    `6; evalbszSTfull=`    `6
bszSTcoll=`    `6; evalbszSTcoll=`    `6

function calculate_batch () {
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
    AE ) datasize=4;;
    ASR) datasize=2;;
    ST ) datasize=1;;
    *  ) datasize=1;;
  esac

  if [ $VRAM_SIZE -ge 40 ]; then
    GPU_maxbatch=$((4 * 12 / $datasize))
  elif [ $VRAM_SIZE -ge 20 ]; then
    GPU_maxbatch=$((2 * 12 / $datasize))
  elif [ $VRAM_SIZE -ge 10 ]; then
    GPU_maxbatch=$((1 * 12 / $datasize))
  else
    echo 'No GPU!'
  fi

  evalbatchsize="${evalbatchsize:=$1}"
  python -c '
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
'
}

IFS=' ' \
AEargs=(
  -b 16 -B 16 --gacc 1 
  --lr 2e-4 -e 3
  --autoencoder
)

IFS=' ' \
ASRargs=(
  -b 18 -B 18 --gacc 1 
  --lr 2e-4 -e 10
)

IFS=' ' \
STargs=(
  -b 18 -B 18 --gacc 1 
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

if [ $USE_AE = true ];
    PretrainedAE=selfpret/AEmodel
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
python \
AudioSentencePiece/main_pl.py \
  --task $Task \
  ${Taskargs[@]} ${Taskdata[@]} ${Taskdemo[@]} \
  ${PretrainedArgs[@]} \
  --proj_name HuggingFacePLSent \
  --run_name NewASR \
  --mode train \
  \
  --nowandb
