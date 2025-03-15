#!/bin/bash

# Number of GPUs available
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Function to start a worker
start_worker() {
  GPU_ID=$1
  SESSION_NAME="worker_gpu_$GPU_ID"
  LOG_FILE="/data/cache/${SESSION_NAME}.log"

  # Kill any existing session with the same name
  tmux kill-session -t "$SESSION_NAME" 2>/dev/null

  # Remove old log file if it exists (optional)
  rm -f "$LOG_FILE"

  # Start a new tmux session
  tmux new-session -d -s "$SESSION_NAME" \
    "DISABLE_INFERENCE=$DISABLE_INFERENCE DISABLE_TRAINING=$DISABLE_TRAINING CUDA_VISIBLE_DEVICES=$GPU_ID python3 astria/infer.py 2>&1 | tee \"$LOG_FILE\"; sleep 5"

  echo "Started worker for GPU $GPU_ID in tmux session: $SESSION_NAME $(date)"
}

# Flag to track first-time initialization
FIRST_RUN=true

# Monitor workers and restart if needed
while true; do
  for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    SESSION_NAME="worker_gpu_$GPU_ID"
    LOG_FILE="/data/cache/${SESSION_NAME}.log"

    # If there's no tmux session for this GPU, it means the process ended or was never started
    if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then

      # If a log file exists, show the last 50 lines for debugging
      if [ -f "$LOG_FILE" ]; then
        echo "================================================================"
        echo "Session $SESSION_NAME exited. Showing last 50 lines of $LOG_FILE"
        echo "================================================================"
        tail -n 50 "$LOG_FILE"
        echo "================================================================"
      else
        echo "No log file found for $SESSION_NAME."
      fi

      # Start a new session
      start_worker "$GPU_ID"

      # Sleep longer on the very first restart, shorter on subsequent ones
      if [ "$FIRST_RUN" = true ]; then
        sleep 15
        FIRST_RUN=false
      else
        sleep 10
      fi

      # After starting (and sleeping), break from the for-loop to re-check from GPU 0
      break
    fi
  done
  # Check again after a short delay
  sleep 5
done
