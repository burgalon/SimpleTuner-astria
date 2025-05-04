# Performance
A40  $0.35/hr res=512 train_batch=1 target=tiny - 1.37s/it
A40  $0.35/hr res=512 train_batch=1 target=default optimizer=optimi-stableadamw - 1.36s/it 23minutes for 1k steps
A40  $0.35/hr res=512 train_batch=4 target=default optimizer=optimi-stableadamw - 4.92s/it
H100 $2.89/hr res=512 train_batch=4 target=default optimizer=optimi-stableadamw - 1.78s/it - X2.76 faster. Replicate $5.5008/h
H100 $2.89/hr res=512 train_batch=1 target=default optimizer=optimi-stableadamw - 0.48s/it 8minutes for 1k steps - 2.08it/s

