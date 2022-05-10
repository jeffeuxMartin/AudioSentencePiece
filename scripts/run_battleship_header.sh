#!/bin/sh

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

debug=0
if [ debug = '1' ]; then
    DEB='python3 -m pdb -c continue'
else
    DEB="python3"
fi

echo '============ START! ============'
if [ $GPU_COUNTS -ge 2 ]; then
# python3 -i
# python3 -m pdb -c continue -m torch.distributed.launch --nproc_per_node=2 `
    OMP_NUM_THREADS=$GPU_COUNTS \
        $DEB -m torch.distributed.launch \
                --nproc_per_node=$GPU_COUNTS \
                $@
                # AudioSentencePiece/main.py $@ 
else
    # $DEB new_ver_0425.py $@ 
    $DEB $@
    # $DEB AudioSentencePiece/main.py $@
fi
exit
