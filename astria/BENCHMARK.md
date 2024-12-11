# Performance
A40  $0.35/hr res=512 train_batch=1 target=tiny - 1.37s/it
A40  $0.35/hr res=512 train_batch=1 target=default optimizer=optimi-stableadamw - 1.36s/it 23minutes for 1k steps
A40  $0.35/hr res=512 train_batch=4 target=default optimizer=optimi-stableadamw - 4.92s/it
H100 $2.89/hr res=512 train_batch=4 target=default optimizer=optimi-stableadamw - 1.78s/it - X2.76 faster. Replicate $5.5008/h
H100 $2.89/hr res=512 train_batch=1 target=default optimizer=optimi-stableadamw - 0.48s/it 8minutes for 1k steps - 2.08it/s

## irit 
1.55-1.6s/it new simpletuner
Optimizer arguments={'lr': 0.0005, 'betas': (0.9, 0.999), 'weight_decay': 0.01, 'eps': 1e-06}

1.45 new simpletuner without flashattention

1.34 s/it old simpletuner
Optimizer arguments, weight_decay=0.01 eps=1e-08, extra_arguments={'weight_decay': 0.01, 'eps': 1e-08, 'betas': (0.9, 0.999), 'lr': 0.0005}
