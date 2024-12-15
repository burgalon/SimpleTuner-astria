#!/usr/bin/env bash
#
# Usage: ./run_all_sweeps_tmux.sh <RUN_ID> <TRACKER_PROJECT>
#
# This script creates a new tmux session called "sweeps" with two windows,
# each split into 4 panes, for a total of 8 sweeps across 8 GPUs.
#

if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Error: Both RUN_ID and TRACKER_PROJECT are required."
  echo "Usage: $0 <RUN_ID> <TRACKER_PROJECT>"
  exit 1
fi

RUN_ID="$1"
TRACKER_PROJECT="$2"

# Create a new detached tmux session named "sweeps"
tmux new-session -d -s sweeps -n Window1

################################################################################
# First window (Window1) split into 4 panes
################################################################################
# By default, we start in pane 0 of Window1

# Split horizontally (Window1 now has pane 0 and pane 1)
tmux split-window -h -t sweeps:Window1

# Vertically split each of the two panes again, so total 4 panes
tmux split-window -v -t sweeps:Window1.0
tmux split-window -v -t sweeps:Window1.1

# Now we have 4 panes: indices 0,1,2,3 in Window1
# Assign each sweep to a different GPU in each pane

tmux send-keys -t sweeps:Window1.0 "cd astria/sweeps && CUDA_VISIBLE_DEVICES=0 ./batch_size_sweep.sh $RUN_ID $TRACKER_PROJECT" C-m
tmux send-keys -t sweeps:Window1.1 "cd astria/sweeps && CUDA_VISIBLE_DEVICES=1 ./caption_dropout_sweep.sh $RUN_ID $TRACKER_PROJECT" C-m
tmux send-keys -t sweeps:Window1.2 "cd astria/sweeps && CUDA_VISIBLE_DEVICES=2 ./lora_rank_sweep.sh $RUN_ID $TRACKER_PROJECT" C-m
tmux send-keys -t sweeps:Window1.3 "cd astria/sweeps && CUDA_VISIBLE_DEVICES=3 ./lr_sweep.sh $RUN_ID $TRACKER_PROJECT" C-m

################################################################################
# Second window (Window2) split into 4 panes
################################################################################
tmux new-window -t sweeps -n Window2

# Split horizontally (Window2 now has pane 0 and pane 1)
tmux split-window -h -t sweeps:Window2

# Vertically split each of the two panes again, so total 4 panes
tmux split-window -v -t sweeps:Window2.0
tmux split-window -v -t sweeps:Window2.1

# Now we have 4 panes: indices 0,1,2,3 in Window2
# Assign each sweep to a different GPU in each pane

tmux send-keys -t sweeps:Window2.0 "cd astria/sweeps && CUDA_VISIBLE_DEVICES=4 ./prompt_sweep.sh $RUN_ID $TRACKER_PROJECT" C-m
tmux send-keys -t sweeps:Window2.1 "cd astria/sweeps && CUDA_VISIBLE_DEVICES=5 ./resolution_sweep.sh $RUN_ID $TRACKER_PROJECT" C-m
tmux send-keys -t sweeps:Window2.2 "cd astria/sweeps && CUDA_VISIBLE_DEVICES=6 ./shift_sweep.sh $RUN_ID $TRACKER_PROJECT" C-m
tmux send-keys -t sweeps:Window2.3 "cd astria/sweeps && CUDA_VISIBLE_DEVICES=7 ./token_sweep.sh $RUN_ID $TRACKER_PROJECT" C-m

################################################################################
# Attach to the session so you can watch everything
################################################################################
tmux attach-session -t sweeps
