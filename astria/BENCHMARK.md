# Performance
A40  $0.35/hr res=512 train_batch=1 target=tiny - 1.37s/it
A40  $0.35/hr res=512 train_batch=1 target=default optimizer=optimi-stableadamw - 1.36s/it 23minutes for 1k steps
A40  $0.35/hr res=512 train_batch=4 target=default optimizer=optimi-stableadamw - 4.92s/it
H100 $2.89/hr res=512 train_batch=4 target=default optimizer=optimi-stableadamw - 1.78s/it - X2.76 faster. Replicate $5.5008/h
H100 $2.89/hr res=512 train_batch=1 target=default optimizer=optimi-stableadamw - 0.48s/it 8minutes for 1k steps - 2.08it/s

## irit 1858416 
1.14s/it old v0 simpletuner - BASELINE
1.42s/it preset=flux-lora-fast learning_rate=5e-4 lora_rank=16 lora_alpha=16 train_batch=4 preprocessing=2 lr_scheduler=polynomial XXflux_lora_target=portrait XXsegmentation=1
1.6s/it preset=flux-lora-fast learning_rate=5e-4 lora_rank=16 lora_alpha=16 train_batch=4 preprocessing=2 lr_scheduler=polynomial XXflux_lora_target=portrait segmentation=1
1.8s/it preset=flux-lora-fast learning_rate=5e-4 lora_rank=16 lora_alpha=16 train_batch=4 preprocessing=2 lr_scheduler=polynomial flux_lora_target=portrait segmentation=1
