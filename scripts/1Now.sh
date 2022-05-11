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

  if [[ $VRAM_SIZE -ge 40 ]]; then
    GPU_maxbatch=$((4 * 12 / $datasize))
  elif [[ $VRAM_SIZE -ge 20 ]]; then
    GPU_maxbatch=$((2 * 12 / $datasize))
  elif [[ $VRAM_SIZE -ge 10 ]]; then
    GPU_maxbatch=$((1 * 12 / $datasize))
  else
    echo 'No GPU!'
    exit()
  fi
  
  export GPU_maxbatch=$GPU_maxbatch
  export GPU_COUNTS=$GPU_COUNTS
}

function calculate_batch2 () {
  evalbatchsize=$2
  evalbatchsize="${evalbatchsize:=$1}"
  calculate_batch
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
  export real_batch=$real_batch
  export evalreal_batch=$evalreal_batch
  export gacc=$gacc
}

calculate_batch2 16
echo $real_batch
echo $evalreal_batch
echo $gacc
