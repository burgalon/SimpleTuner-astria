#!/bin/bash

# Number of GPUs available
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Function to start a worker
start_worker() {
  GPU_ID=$1
  SESSION_NAME="worker_gpu_$GPU_ID"

  # Kill any existing session with the same name
  tmux kill-session -t "$SESSION_NAME" 2>/dev/null

  # Start a new tmux session
#  tmux new-session -d -s "$SESSION_NAME" \
#    "CUDA_VISIBLE_DEVICES=$GPU_ID echo running with $GPU_ID; sleep 60; bash"
  tmux new-session -d -s "$SESSION_NAME" \
    "CUDA_VISIBLE_DEVICES=$GPU_ID python3 astria/infer.py; sleep 5"

  echo "Started worker for GPU $GPU_ID in tmux session: $SESSION_NAME"
}

# Start workers for each GPU
#for GPU_ID in $(seq 7 $((NUM_GPUS - 1))); do
#  start_worker "$GPU_ID"
#done

# Monitor workers and restart if needed
while true; do
  for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    SESSION_NAME="worker_gpu_$GPU_ID"
    if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      start_worker "$GPU_ID"
      sleep 10
      break
    fi
  done
  sleep 5
done
