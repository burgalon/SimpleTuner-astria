#!/bin/bash
set -ue

trap "pkill -f sleep" term
#export WANDB_API_KEY=d185f8b94689d8735dd87a082f0c9cf2f416fcd9
#export TR_PREPROCESSING=2
# Need to test CAPTION_STRATEGY=textfile

# Dress
#TR_LORA_RANK=32 TR_LORA_ALPHA=32 TR_FLUX_LORA_TARGET=product TR_BLACK_MASK=1 TR_REPORT_TO=wandb python3 astria/train.py 1781064
#DEBUG=1 python3 astria/infer.py 19646495 19646460 19646550


# Irit
#TR_FLUX_LORA_TARGET=84 TR_SEGMENTATION=1 TR_REPORT_TO=wandb python3 astria/train.py 1587148
#DEBUG=1 python3 astria/infer.py 19462605 18258992 18258991 18258990

## Trump - bad training probably because of the background
#XFLUX_LORA_TARGET=84 XXX_SEGMENTATION=1 TR_REPORT_TO=wandb python3 astria/train.py 1782432
#DEBUG=1 python3 astria/infer.py 19562991 19562843 19562804 19562803 19562793

# 1587148 1649898 1815518 1782432


## Anat - 1 bad image long shots, and 7 good portraits - simpletuner_v0
python3 astria/train.py 1649898 && python3 astria/infer.py 19620098 18551264 18551263 18551261


# Bald + plain background overfitting
python3 astria/train.py 1815518 && DEBUG=1 python3 astria/infer.py 19530314 19530312 19530311

# Irit new training
python3 astria/train.py 1858416 && DEBUG=1 python3 astria/infer.py 19926288 19926287 19926286 19926284

# Irit product training simpletuner_v0
python3 astria/train.py 1689634 && DEBUG=1 python3 astria/infer.py 18730837 18730836 18730835
