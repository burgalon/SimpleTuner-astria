#!/bin/bash

# Check if RUN_ID and TRACKER_PROJECT are provided as arguments
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Error: Both RUN_ID and TRACKER_PROJECT are required."
  echo "Usage: $0 <RUN_ID> <TRACKER_PROJECT>"
  exit 1
fi

# Assign the provided arguments to variables
export RUN_ID=$1
export TRACKER_PROJECT=$2

export TRAIN_STEPS=301
export DEBUG=1
export MOCK_SERVER=1
export TR_REPORT_TO="wandb"
export INSTANCE_PROMPT="!TOKEN, a confident and professional individual with a calm demeanor."
export PREPROCESSING="2"
export FOLDER_SUFFIX="sweeprank"
export VALIDATION_STEPS="50"

TRACKER_NAME="rank=1" LORA_RANK="1" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="rank=2" LORA_RANK="2" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="rank=4" LORA_RANK="4" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="rank=8" LORA_RANK="8" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="rank=16" LORA_RANK="16" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="rank=32" LORA_RANK="32" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="rank=64" LORA_RANK="64" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="rank=128" LORA_RANK="128" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="rank=256" LORA_RANK="256" python3 astria/train.py "$RUN_ID"
TRACKER_NAME="rank=512" LORA_RANK="512" python3 astria/train.py "$RUN_ID"