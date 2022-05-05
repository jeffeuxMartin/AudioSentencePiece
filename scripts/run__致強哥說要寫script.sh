#!/bin/sh

# hrun -s -GGG -c 8 -m 32 zsh \
#    AudioSentencePiece/scripts/run__致強哥說要寫script.sh

MAINFILE=AudioSentencePiece/nuevo_main.py

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
echo '============ START! ============'

GPU_COUNTS=$(
    python3 -c '
import torch;
print(torch.cuda.device_count())'
)

# echo $GPU_COUNTS

debug=1
if [ debug = '1' ]; then
    DEB='python3 -m pdb -c continue'
else
    DEB="python3"
fi

if [ $GPU_COUNTS -ge 1 ]; then
# python3 -i
# python3 -m pdb -c continue -m torch.distributed.launch --nproc_per_node=2 `
    OMP_NUM_THREADS=$GPU_COUNTS \
        $DEB -m torch.distributed.launch \
                --nproc_per_node=$GPU_COUNTS \
                $MAINFILE $@ 
                # AudioSentencePiece/main.py $@ 
else
    # $DEB new_ver_0425.py $@ 
    $DEB $MAINFILE $@
    # $DEB AudioSentencePiece/main.py $@
fi
exit

